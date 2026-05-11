import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import asyncio
from fastapi.staticfiles import StaticFiles
import uvicorn
import concurrent.futures
from tracker import run_tracking_inference, render_pose_video
from movement import extract_movement_features, render_movement_video

app = FastAPI(title="Badminton Pose Detection API")

# Ensure temp directory exists for outputs
TEMP_DIR = "temp_exports"
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

progress_store = {}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Placeholder model paths
POSE_DETECTION_MODEL_PATH = "yolov8n-pose.pt" # Replace with your actual local model path later
CUSTOM_CLASSIFICATION_MODEL_PATH = "best.pt"  # Fine-tuned model

def process_video_task(task_id, video_path, mp4_path, json_path, csv_path):
    def progress_cb(current, total):
        pct = int((current / total) * 100) if total > 0 else 0
        progress_store[task_id] = {"status": "processing", "progress": pct}

    try:
        metrics = run_tracking_inference(
            video_path=video_path,
            out_mp4=mp4_path,
            out_json=json_path,
            out_csv=csv_path,
            model_path=POSE_DETECTION_MODEL_PATH,
            custom_model_path=CUSTOM_CLASSIFICATION_MODEL_PATH,
            progress_callback=progress_cb
        )

        movement_json_path = os.path.join(TEMP_DIR, "movement_metrics.json")
        movement_csv_path = os.path.join(TEMP_DIR, "movement_features.csv")
        movement_mp4_path = os.path.join(TEMP_DIR, "movement_output.mp4")
        
        movement_metrics = extract_movement_features(
            input_json_path=json_path,
            output_json_path=movement_json_path,
            output_csv_path=movement_csv_path,
            output_video_path=None,
            original_video_path=None
        )

        progress_store[task_id]["progress"] = 90
        
        # Step 3: Render both videos simultaneously using multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(render_pose_video, video_path, mp4_path, json_path)
            f2 = executor.submit(render_movement_video, movement_json_path, video_path, movement_mp4_path)
            concurrent.futures.wait([f1, f2])

        progress_store[task_id] = {
            "status": "completed", 
            "progress": 100, 
            "metrics": metrics,
            "movement_metrics": movement_metrics,
            "exports": {
                "json_url": "/api/download/output.json",
                "csv_url": "/api/download/output.csv",
                "mp4_url": "/api/download/output.mp4",
                "movement_csv_url": "/api/download/movement_features.csv",
                "movement_json_url": "/api/download/movement_metrics.json",
                "movement_mp4_url": "/api/download/movement_output.mp4"
            }
        }
    except Exception as e:
        progress_store[task_id] = {"status": "failed", "error": str(e)}

@app.post("/api/process-video")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_path = f"temp_exports/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    task_id = str(uuid.uuid4())
    progress_store[task_id] = {"status": "starting", "progress": 0}
    
    json_path = os.path.join(TEMP_DIR, "output.json")
    csv_path = os.path.join(TEMP_DIR, "output.csv")
    mp4_path = os.path.join(TEMP_DIR, "output.mp4")
    
    background_tasks.add_task(process_video_task, task_id, video_path, mp4_path, json_path, csv_path)
    
    return {"task_id": task_id}

@app.get("/api/progress/{task_id}")
def get_progress(task_id: str):
    return progress_store.get(task_id, {"status": "not_found", "progress": 0})

@app.get("/api/progress-stream/{task_id}")
async def progress_stream(task_id: str):
    import json
    async def event_generator():
        last_progress = -1
        last_status = None
        while True:
            data = progress_store.get(task_id, {"status": "not_found", "progress": 0})
            
            if data["progress"] != last_progress or data["status"] != last_status:
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = data.get("progress")
                last_status = data.get("status")
                
            if data["status"] in ["completed", "failed", "not_found"]:
                break
                
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        media_type = "video/mp4" if filename.endswith(".mp4") else None
        return FileResponse(file_path, filename=filename, media_type=media_type)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


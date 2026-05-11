import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from tracker import run_tracking_inference

app = FastAPI(title="Badminton Pose Detection API")

# Ensure temp directory exists for outputs
TEMP_DIR = "temp_exports"
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

progress_store = {}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
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
        progress_store[task_id] = {
            "status": "completed", 
            "progress": 100, 
            "metrics": metrics,
            "exports": {
                "json_url": "/api/download/output.json",
                "csv_url": "/api/download/output.csv",
                "mp4_url": "/api/download/output.mp4"
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

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        media_type = "video/mp4" if filename.endswith(".mp4") else None
        return FileResponse(file_path, filename=filename, media_type=media_type)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


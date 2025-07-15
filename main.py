# main.py
from fastapi import FastAPI, Depends, HTTPException, Query, Form, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime, timedelta
import uvicorn
import cv2
import numpy as np
import torch
import logging
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Optional, Dict
from jose import jwt
import json

app = FastAPI()

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database models
class User(BaseModel):
    username: str
    password: str

class Camera(BaseModel):
    id: int
    name: str
    type: str  # "usb" or "rtsp"
    source: str  # device index or RTSP URL
    status: str = "inactive"

class AttendanceRecord(BaseModel):
    user_id: str
    timestamp: datetime
    direction: str  # "in" or "out"

# Mock database
users_db = {
    "admin": {"password": "adminpass"}
}

cameras_db = {}
attendance_records = []

# Dictionary to store the last recorded time and direction for each user
last_record_info: Dict[str, Dict[str, datetime or str]] = {}

# Face recognition setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(device=device)
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "face_embeddings_collection"
SIMILARITY_THRESHOLD = 0.7 # User has set threshold to 0.7

# Authentication endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

from jose import jwt

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Camera management endpoints
@app.post("/cameras", response_model=Camera)
async def add_camera(camera: Camera):
    cameras_db[camera.id] = camera
    return camera

@app.get("/cameras", response_model=List[Camera])
async def get_cameras():
    return list(cameras_db.values())

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: int):
    if camera_id in cameras_db:
        del cameras_db[camera_id]
        return {"message": "Camera removed"}
    raise HTTPException(status_code=404, detail="Camera not found")

# Video streaming endpoint
def generate_frames(camera_id: int):
    camera = cameras_db.get(camera_id)
    if not camera:
        yield b"Camera not found"
        return
    
    if camera.type == "usb":
        cap = cv2.VideoCapture(int(camera.source))
    else:  # RTSP
        cap = cv2.VideoCapture(camera.source)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = rgb_frame[y1:y2, x1:x2]
# Skip invalid or too small crops
                if face_img.size == 0 or (y2 - y1 < 20 or x2 - x1 < 20):
                    continue

                try:
                    face_tensor = mtcnn(face_img)
                    if face_tensor is None:
                        continue
                except Exception as e:
                    print(f"mtcnn failed: {e}")
                    continue
                
                if len(face_tensor.shape) == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                
                embedding = model(face_tensor.to(device)).detach().cpu().numpy()
                if embedding.shape[0] == 1:
                    embedding = embedding[0]
                
                # Search in Qdrant
                search_result = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=embedding.tolist(),
                    limit=1,
                    with_payload=True
                )
                
                if search_result and search_result[0].score > SIMILARITY_THRESHOLD:
                    user_id = search_result[0].payload["user_id"]
                    current_time = datetime.now()
                    
                    # Determine attendance direction (in/out)
                    last_info = last_record_info.get(user_id)
                    direction = "in" # Default direction
                    
                    if last_info:
                        last_timestamp = last_info["timestamp"]
                        last_direction = last_info["direction"]
                        
                        # If more than 2 minutes passed since last record, toggle direction
                        if (current_time - last_timestamp) > timedelta(minutes=2):
                            direction = "out" if last_direction == "in" else "in"
                        else:
                            # If within 2 minutes, keep the same direction as the last record
                            direction = last_direction
                    
                    # Record attendance only if it's a new "in" or "out" event after 2 minutes
                    # Or if it's the very first record for the user
                    if not last_info or (current_time - last_info["timestamp"]) > timedelta(minutes=2) or (last_info["direction"] != direction):
                        record = AttendanceRecord(
                            user_id=user_id,
                            timestamp=current_time,
                            direction=direction
                        )
                        attendance_records.append(record)
                        last_record_info[user_id] = {"timestamp": current_time, "direction": direction}
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{user_id} ({direction})", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Attendance endpoints
@app.get("/attendance", response_model=List[AttendanceRecord])
async def get_attendance(
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    direction: Optional[str] = Query(None)
):
    filtered_records = attendance_records
    
    if user_id:
        filtered_records = [r for r in filtered_records if r.user_id == user_id]
    
    if start_date:
        filtered_records = [r for r in filtered_records if r.timestamp >= start_date]
    
    if end_date:
        filtered_records = [r for r in filtered_records if r.timestamp <= end_date]
    
    if direction:
        filtered_records = [r for r in filtered_records if r.direction == direction]
    
    return filtered_records

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/attendance_view", response_class=HTMLResponse)
async def attendance_page(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

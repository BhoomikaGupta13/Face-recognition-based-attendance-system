# Face Recognition Based Attendance System

This project is a real-time, intelligent attendance system based on **face recognition**, developed using **FastAPI**, **FaceNet**, **Qdrant**, and **Docker**. It automatically identifies individuals using a live **webcam** or **IP camera** feed and logs their attendance with timestamps and direction (IN/OUT), all through a clean and interactive web interface.

Designed with scalability in mind, the system integrates deep learning for facial feature extraction and a vector database for efficient face matching. A fully functional **dashboard** allows admins to manage cameras, view attendance records, and filter logs by user, date, or direction.

> ✅ **Demonstration video included** – showing the complete working system in action.

---
## 🚀 Features

- 🧠 Face recognition using **FaceNet (InceptionResnetV1)** and **MTCNN**
- 📦 Vector database storage with **Qdrant**
- 📹 Real-time camera feed using **OpenCV**
- 📊 Attendance dashboard with **filters**
- 🔐 Secure login via token-based auth
- 🌐 Web interface using **FastAPI + Jinja2 + Tailwind CSS**
- 🌍 Supports **Webcam** or **IP Camera (RTSP)** sources

---
## 🔧 Setup Instructions

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/BhoomikaGupta13/Face-recognition-based-attendance-system.git
cd Face-recognition-based-attendance-system
```

### 2. 🐍 Install Python Dependencies
Make sure Python ≥ 3.9 is installed.
```bash
pip install -r requirements.txt
```
### 3. 🐳 Start Qdrant with Docker
Make sure Docker Desktop is installed and running.
```bash
docker run -p 6333:6333 -p 6334:6334 -d qdrant/qdrant
```
Check Qdrant dashboard: http://localhost:6333

### 4. 🖼 Prepare Your Dataset
Place your dataset inside a folder named filtered_dataset/:
```bash
kaggle_filtered_dataset/
├── user1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── user2/
│   ├── img1.jpg
│   ├── ...
```
⚠️ Atleast images per user are recommended for better recognition.

### 5. 🔐 Generate Face Embeddings
```bash
python generate_embeddings.py
```
This will extract face embeddings and upload them to Qdrant.

### 6. 🚀 Run the Backend Server
```bash
python main.py
```
Open in browser: http://localhost:8000

Use:
Username: admin
Password: adminpass

### 7. 📹 Start Attendance via Webcam or IP Camera

Once you're logged in:

- Add a **USB camera** or **IP camera** from the **Dashboard**
- Click **"Start Camera"** to begin the live feed
- The system will automatically detect faces and register **"IN"** or **"OUT"** based on timing logic

---

### 🎯 IP Camera Support

You can also connect any **IP Camera (RTSP stream)** through the dashboard.  
This makes the system ideal for deployment in **schools**, **offices**, or **institutions** where **multiple remote cameras** are required.

---

### 📈 Filter & View Attendance

Navigate to **"View Attendance"** from the dashboard to:

- 🔍 **Filter by User ID**
- 📅 **Filter by Date**
- 🔄 **Filter by Direction** (IN / OUT)

✅ Optional: Run Face Recognition Standalone
For quick webcam testing:

```bash
python recognize_faces.py
```
Requires haarcascade_frontalface_default.xml in the same folder.

## 🧪 Tech Stack

| Technology          | Purpose                                         |
|---------------------|-------------------------------------------------|
| **FastAPI**         | Backend REST APIs and template rendering        |
| **FaceNet (facenet-pytorch)** | Face embeddings via deep learning         |
| **MTCNN**           | Face detection and alignment                    |
| **Qdrant**          | Vector database for fast similarity search      |
| **OpenCV**          | Camera capture and real-time image processing   |
| **Docker**          | Containerized Qdrant deployment                 |
| **Tailwind CSS**    | Modern frontend styling                         |
| **JavaScript**      | UI logic and interactivity                      |

## ✅ Final Thoughts

This project demonstrates how deep learning and modern backend technologies can be combined to build practical, real-time systems. Whether for institutions, offices, or classrooms, this face recognition-based attendance system offers a scalable and intelligent solution for seamless identity verification and logging.

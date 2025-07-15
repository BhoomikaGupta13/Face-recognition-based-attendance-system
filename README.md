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

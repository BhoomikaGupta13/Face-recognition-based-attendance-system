import os
import cv2
import numpy as np
import torch
import time
import logging
from facenet_pytorch import InceptionResnetV1, MTCNN
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration ---
DATASET_PATH = 'kaggle_filtered_dataset'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333  # Default Qdrant HTTP port
QDRANT_COLLECTION_NAME = 'face_embeddings_collection'
EMBEDDING_DIMENSION = 512  # FaceNet's default embedding size
QDRANT_TIMEOUT = 300  # seconds (5 minutes)
UPLOAD_BATCH_SIZE = 1  # Set to 1 for immediate upload; increase for micro-batching if desired

# --- Logging Setup ---
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'embedding_generation.log')),
        logging.StreamHandler()
    ]
)

logging.info("Starting embedding generation process...")

# --- 1. Initialize FaceNet Model ---
logging.info("Loading FaceNet model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
logging.info(f"FaceNet model loaded on device: {device}")

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)
logging.info("MTCNN face detector initialized.")

# --- 2. Initialize Qdrant Client ---
logging.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
logging.info("Qdrant client initialized.")

# --- 3. Create Qdrant Collection (if it doesn't exist) ---
logging.info(f"Checking for Qdrant collection: '{QDRANT_COLLECTION_NAME}'...")
try:
    client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Skipping creation.")
except Exception:
    logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist. Creating it...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
    logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' created with dimension {EMBEDDING_DIMENSION} and Cosine distance.")

# --- 4. Generate Embeddings and Upload Immediately to Qdrant ---
point_id_counter = 0
upload_buffer = []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_points(points):
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True,
        points=points
    )

logging.info(f"Scanning dataset from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    logging.error(f"Dataset path '{DATASET_PATH}' not found. Please ensure your dataset is in the correct location.")
    exit()

for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_dir):
        continue
    logging.info(f"Processing images for user: {person_name}")
    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 3:
        logging.warning(f"User '{person_name}' has less than 3 images ({len(image_files)}). Skipping for robust recognition.")
        continue
    for image_name in image_files:
        image_path = os.path.join(person_dir, image_name)
        logging.info(f" Processing image: {image_path}")
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f" Could not read image: {image_path}. Skipping.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            start_face_detection_time = time.time()
            face_tensor = mtcnn(img_rgb)
            end_face_detection_time = time.time()
            logging.info(f" Face detection time: {end_face_detection_time - start_face_detection_time:.4f} seconds")
            if face_tensor is None:
                logging.warning(f" No face detected in {image_name}. Skipping embedding generation.")
                continue
            if len(face_tensor.shape) == 3:
                face_tensor = face_tensor.unsqueeze(0)
            elif len(face_tensor.shape) == 4 and face_tensor.shape[0] > 1:
                logging.warning(f" Multiple faces detected in {image_name}. Using the first detected face.")
                face_tensor = face_tensor[0].unsqueeze(0)
            if face_tensor is None or face_tensor.nelement() == 0:
                logging.warning(f" MTCNN returned an empty or invalid tensor for {image_name}. Skipping embedding generation.")
                continue
            start_embedding_time = time.time()
            face_embedding = model(face_tensor.to(device)).detach().cpu().numpy()
            end_embedding_time = time.time()
            logging.info(f" Embedding generation time: {end_embedding_time - start_embedding_time:.4f} seconds")
            if face_embedding.shape[0] == 1:
                face_embedding = face_embedding[0]
            point = PointStruct(
                id=point_id_counter,
                vector=face_embedding.tolist(),
                payload={
                    "user_id": person_name,
                    "image_name": image_name,
                    "image_path": image_path
                }
            )
            upload_buffer.append(point)
            point_id_counter += 1

            # Upload as soon as buffer is full (or immediately if UPLOAD_BATCH_SIZE=1)
            if len(upload_buffer) >= UPLOAD_BATCH_SIZE:
                upload_points(upload_buffer)
                logging.info(f"Uploaded {len(upload_buffer)} embedding(s) to Qdrant.")
                upload_buffer = []
        except Exception as e:
            logging.error(f" Error processing {image_path}: {e}")

# Upload any remaining points in the buffer
if upload_buffer:
    upload_points(upload_buffer)
    logging.info(f"Uploaded final {len(upload_buffer)} embedding(s) to Qdrant.")

logging.info("Embedding generation and upload process completed.")












# docker run -p 6333:6333 -p 6334:6334 -d qdrant/qdrant
# http://localhost:6333/dashboard#/collections/face_embeddings_collection

# https://gemini.google.com/app/0e5faa2c62a2d9e1
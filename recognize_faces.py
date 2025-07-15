import cv2
import numpy as np
import torch
import time
import logging
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

# --- Configuration ---
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = 'face_embeddings_collection'
SIMILARITY_THRESHOLD = 0.50  # <<< IMPORTANT: ADJUST THIS VALUE based on your testing (0.7 to 0.85 usually)
WEBCAM_INDEX = 0 # 0 is usually the default webcam

# Path to OpenCV's Haar Cascade XML file for face detection
# Assuming 'haarcascade_frontalface_default.xml' is downloaded
# and placed directly in the same folder as this script.
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'


# --- Logging Setup ---
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'face_recognition_realtime.log')),
        logging.StreamHandler() # Also print to console
    ]
)

logging.info("Starting real-time face recognition process...")

# --- 1. Initialize FaceNet Model and MTCNN ---
logging.info("Loading FaceNet model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
logging.info(f"FaceNet model loaded on device: {device}")

# Initialize MTCNN for precise face alignment for FaceNet
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)
logging.info("MTCNN face detector initialized for FaceNet preprocessing.")

# --- 2. Initialize Qdrant Client ---
logging.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Alternative: In-memory Qdrant (uncomment next line and comment above line if not using Docker) ---
# client = QdrantClient(":memory:")

try:
    # Check if the collection exists before proceeding
    client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    logging.info(f"Connected to Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
except Exception:
    logging.error(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist in Qdrant. "
                  "Please run 'generate_embeddings.py' first to populate the database.")
    exit()

# --- 3. Initialize OpenCV Face Detector (Haar Cascade) ---
logging.info(f"Loading Haar Cascade face detector from: {HAARCASCADE_PATH}")
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    logging.error(f"Could not load Haar Cascade classifier from {HAARCASCADE_PATH}. "
                  "Please check the path to haarcascade_frontalface_default.xml.")
    exit()
logging.info("Haar Cascade face detector loaded.")

# --- 4. Initialize Webcam ---
logging.info(f"Accessing webcam (index: {WEBCAM_INDEX})...")
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    logging.error(f"Could not open webcam at index {WEBCAM_INDEX}. Please check if camera is connected and available.")
    exit()
logging.info("Webcam successfully opened.")

# --- 5. Main Recognition Loop ---
logging.info(f"Starting video stream. Similarity Threshold: {SIMILARITY_THRESHOLD}. Press 'q' to quit.")

frame_count = 0
while True:
    start_frame_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to grab frame from webcam. Exiting...")
        break

    # Convert frame to grayscale for Haar Cascade (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Detection using Haar Cascade
    start_detection_time = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    end_detection_time = time.time()
    logging.info(f"Frame {frame_count}: OpenCV Haar Cascade detection time: {end_detection_time - start_detection_time:.4f} seconds")

    for (x, y, w, h) in faces:
        # Extract the face region (crop with a bit of margin)
        # Ensure the crop coordinates are within image bounds
        y1, y2, x1, x2 = max(0, y), min(frame.shape[0], y + h), max(0, x), min(frame.shape[1], x + w)
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0: # Check if the cropped image is empty
            logging.warning(f"Empty face image extracted at ({x},{y},{w},{h}). Skipping.")
            continue

        try:
            # Convert to RGB for FaceNet/MTCNN
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Use MTCNN to get the aligned face tensor for FaceNet
            start_face_alignment_time = time.time()
            face_tensor = mtcnn(face_img_rgb)
            end_face_alignment_time = time.time()
            logging.info(f"Frame {frame_count}: MTCNN alignment time: {end_face_alignment_time - start_face_alignment_time:.4f} seconds")


            if face_tensor is None:
                logging.warning(f"Frame {frame_count}: MTCNN failed to detect or align face. Skipping embedding generation.")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red for undetected
                cv2.putText(frame, "No Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            # Ensure it's a single face tensor for model input (unsqueeze if needed)
            if len(face_tensor.shape) == 3: # If MTCNN returned one face as [C, H, W]
                face_tensor = face_tensor.unsqueeze(0) # Add batch dimension [1, C, H, W]

            # Generate embedding
            start_embedding_time = time.time()
            face_embedding = model(face_tensor.to(device)).detach().cpu().numpy()
            end_embedding_time = time.time()
            logging.info(f"Frame {frame_count}: FaceNet embedding generation time: {end_embedding_time - start_embedding_time:.4f} seconds")

            # Flatten the embedding if it's a 2D array from batch_size=1
            if face_embedding.shape[0] == 1:
                face_embedding = face_embedding[0]

            # Query Qdrant
            start_matching_time = time.time()
            search_result = client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=face_embedding.tolist(),
                limit=1, # Get only the top 1 result
                query_filter=None, # No additional filters
                with_payload=True # Get the metadata (user_id)
            )
            end_matching_time = time.time()
            logging.info(f"Frame {frame_count}: Qdrant matching time: {end_matching_time - start_matching_time:.4f} seconds")

            recognized_name = "Unknown"
            box_color = (0, 0, 255) # Red for Unknown

            # --- ADDED: Logging for actual similarity score ---
            if search_result: # Check if Qdrant returned any result
                actual_score = search_result[0].score
                best_match_id = search_result[0].payload['user_id']
                logging.info(f"Frame {frame_count}: Best match for '{best_match_id}' with score: {actual_score:.4f} (Threshold: {SIMILARITY_THRESHOLD})")

                if actual_score >= SIMILARITY_THRESHOLD:
                    # We found a match above the threshold!
                    recognized_name = best_match_id
                    box_color = (0, 255, 0) # Green for known user
                else:
                    logging.info(f"Frame {frame_count}: Best match ({best_match_id}) score {actual_score:.4f} is below threshold {SIMILARITY_THRESHOLD}. Labeling as 'Unknown'.")
            else:
                logging.info(f"Frame {frame_count}: No matches found in Qdrant for current face.")

            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        except Exception as e:
            logging.error(f"Frame {frame_count}: Error processing detected face at ({x},{y},{w},{h}): {e}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red for error
            cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    end_frame_time = time.time()
    logging.info(f"Frame {frame_count}: Total frame processing time: {end_frame_time - start_frame_time:.4f} seconds\n")
    frame_count += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info(" 'q' pressed. Exiting video stream.")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
logging.info("Real-time face recognition process finished.")
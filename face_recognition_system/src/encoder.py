import face_recognition
import pickle
import os
from . import config

class FaceTrainer:
    def train(self):
        print("[INFO] Starting training...")
        known_encodings = []
        known_names = []

        if not os.path.exists(config.RAW_IMG_DIR):
            print("[ERROR] No data directory found.")
            return

        person_dirs = [d for d in os.listdir(config.RAW_IMG_DIR) 
                       if os.path.isdir(os.path.join(config.RAW_IMG_DIR, d))]

        for person in person_dirs:
            person_folder = os.path.join(config.RAW_IMG_DIR, person)
            images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png'))]
            
            print(f"[INFO] Processing {person} ({len(images)} images)...")

            for img_name in images:
                img_path = os.path.join(person_folder, img_name)
                image = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(image, model="hog")
                encodings = face_recognition.face_encodings(image, boxes)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person)

        print("[INFO] Serializing encodings...")
        data = {"encodings": known_encodings, "names": known_names}
        
        with open(config.ENCODINGS_PATH, "wb") as f:
            f.write(pickle.dumps(data))
        
        print(f"[SUCCESS] Model saved to {config.ENCODINGS_PATH}")
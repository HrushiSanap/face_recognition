import cv2
import os
import face_recognition
from . import config

class FaceCapture:
    def __init__(self):
        self.cap = None

    def capture_faces(self, person_name):
        person_path = os.path.join(config.RAW_IMG_DIR, person_name)
        
        if os.path.exists(person_path):
            print(f"Warning: '{person_name}' already exists.")
            if input("Overwrite? (y/n): ").lower() != 'y':
                return

        os.makedirs(person_path, exist_ok=True)
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        count = 0
        print(f"\n[INFO] capturing images for {person_name}...")
        print("Please look at the camera and turn your head slightly.")

        while count < config.TOTAL_PHOTOS_TO_CAPTURE:
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame, model="hog")
            display_frame = frame.copy()

            if len(face_locs) == 1:
                top, right, bottom, left = face_locs[0]
                # Add padding
                padding = 20
                h, w, _ = frame.shape
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(h, bottom + padding)
                right = min(w, right + padding)
                
                face_img = frame[top:bottom, left:right]
                
                if face_img.size != 0:
                    cv2.imwrite(os.path.join(person_path, f"{count+1}.jpg"), face_img)
                    count += 1
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(display_frame, f"Captured: {count}/{config.TOTAL_PHOTOS_TO_CAPTURE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Capture Mode", display_frame)

            if cv2.waitKey(150) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Saved {count} images to {person_path}")
import cv2
import face_recognition
import pickle
import os
import time
import numpy as np
from scipy.spatial import distance as dist
from . import config

class SmartFaceSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        
        # State Trackers
        # Stores: {face_index: {"live": False, "live_expiry": 0, "blink_counter": 0, "last_nose_pos": None}}
        self.liveness_states = {} 
        
        self.frame_count = 0
        self.load_encodings()

    def load_encodings(self):
        if not os.path.exists(config.ENCODINGS_PATH):
            print(f"[ERROR] Model not found at {config.ENCODINGS_PATH}. Train first.")
            return False
        
        print("[INFO] Loading model...")
        with open(config.ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        return True

    def calculate_ear(self, eye):
        """Calculates Eye Aspect Ratio to detect blinking."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        # Set resolution explicitly to ensure consistent pixel measurements
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        print("[INFO] System Started. Nod your head OR blink to verify.")

        current_names = []
        current_locs = []

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_SCALE, fy=config.FRAME_RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # --- 1. Detection ---
            current_locs = face_recognition.face_locations(rgb_small)

            # --- 2. Recognition (Optimization: Run every N frames) ---
            if self.frame_count % config.RECOGNITION_INTERVAL == 0:
                current_names = []
                encodings = face_recognition.face_encodings(rgb_small, current_locs)
                for encoding in encodings:
                    matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.5)
                    name = "Unknown"
                    face_dists = face_recognition.face_distance(self.known_encodings, encoding)
                    if len(face_dists) > 0:
                        best_match = np.argmin(face_dists)
                        if matches[best_match]:
                            name = self.known_names[best_match]
                    current_names.append(name)
            
            # --- 3. Liveness Detection (Every Frame) ---
            landmarks_list = face_recognition.face_landmarks(rgb_small, current_locs)
            
            # Reset states if no faces found to save memory (optional)
            if not current_locs:
                self.liveness_states = {}

            for i, landmarks in enumerate(landmarks_list):
                # Initialize state for this face index if new
                if i not in self.liveness_states:
                    self.liveness_states[i] = {
                        "live": False, 
                        "live_expiry": 0, 
                        "blink_counter": 0,
                        "last_nose_pos": None
                    }
                
                state = self.liveness_states[i]

                # --- A. BLINK DETECTION ---
                left_ear = self.calculate_ear(landmarks['left_eye'])
                right_ear = self.calculate_ear(landmarks['right_eye'])
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < config.EYE_AR_THRESH:
                    state["blink_counter"] += 1
                else:
                    if state["blink_counter"] >= config.EYE_AR_CONSEC_FRAMES:
                        # Success: Blink detected
                        state["live_expiry"] = time.time() + config.LIVENESS_TIMEOUT
                    state["blink_counter"] = 0

                # --- B. NOSE MOVEMENT DETECTION ---
                # Get Nose Tip (Point index 2 in the list of 5 nose_tip points)
                nose_tip_pts = landmarks['nose_tip']
                current_nose_pos = np.array(nose_tip_pts[2]) 

                if state["last_nose_pos"] is not None:
                    # Calculate movement distance (Euclidean)
                    move_dist = dist.euclidean(current_nose_pos, state["last_nose_pos"])
                    
                    # If movement is significant, refresh liveness
                    if move_dist > config.MOVEMENT_THRESHOLD:
                         state["live_expiry"] = time.time() + config.LIVENESS_TIMEOUT
                
                # Update history
                state["last_nose_pos"] = current_nose_pos

                # --- C. FINAL LIVENESS STATUS ---
                state["live"] = time.time() < state["live_expiry"]

            # --- 4. Visualization ---
            for i, (top, right, bottom, left) in enumerate(current_locs):
                # Scale coordinates back up
                scale = int(1/config.FRAME_RESIZE_SCALE)
                top *= scale; right *= scale; bottom *= scale; left *= scale
                
                name = current_names[i] if i < len(current_names) else "Unknown"
                
                # Get liveness state
                is_live = False
                if i in self.liveness_states:
                    is_live = self.liveness_states[i]["live"]
                
                # Color Logic
                if name == "Unknown":
                    color = (0, 0, 255) # Red
                    label = "Unknown"
                elif is_live:
                    color = (0, 255, 0) # Green
                    label = f"{name} (LIVE)"
                else:
                    color = (255, 255, 0) # Cyan/Yellow
                    label = f"{name} (Move/Blink)"

                # 1. Draw the Bounding Box (Thinner, cleaner)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # 2. Draw Floating Text with Outline (Stroke) for readability
                # We place text slightly ABOVE the face box
                text_pos = (left, top - 10)
                
                # Draw black outline (thick)
                cv2.putText(frame, label, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                
                # Draw colored text (thin) on top
                cv2.putText(frame, label, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('FaceGuard - Modular', frame)
            self.frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
import argparse
import sys
from src.utils import setup_directories, check_dependencies
from src.data_loader import FaceCapture
from src.encoder import FaceTrainer
from src.recognition import SmartFaceSystem

def main():
    # 1. Setup
    setup_directories()
    check_dependencies()

    # 2. Parse Arguments
    parser = argparse.ArgumentParser(description="FaceGuard: Modular Face Recognition System")
    parser.add_argument("mode", choices=["capture", "train", "run"], 
                        help="Select mode: 'capture' new user, 'train' model, or 'run' recognition")
    parser.add_argument("--name", help="Name of the person to capture (required for 'capture' mode)")

    args = parser.parse_args()

    # 3. Execute
    if args.mode == "capture":
        if not args.name:
            print("[ERROR] Please provide a name using --name 'JohnDoe'")
            return
        capture = FaceCapture()
        capture.capture_faces(args.name)

    elif args.mode == "train":
        trainer = FaceTrainer()
        trainer.train()

    elif args.mode == "run":
        system = SmartFaceSystem()
        system.run()

if __name__ == "__main__":
    main()
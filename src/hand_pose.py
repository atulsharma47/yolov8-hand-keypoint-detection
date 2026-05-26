from pathlib import Path
import cv2
import json
from ultralytics import YOLO


# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = BASE_DIR / "models" / "yolov8n-pose.pt"
IMAGE_PATH = BASE_DIR / "assets" / "test_image_2.jpg"

OUTPUT_IMAGE = BASE_DIR / "outputs" / "annotated_image.jpg"
OUTPUT_JSON = BASE_DIR / "outputs" / "hand_keypoints1.json"


def load_model():
    """
    Load YOLOv8 pose model.
    """
    try:
        model = YOLO(str(MODEL_PATH))
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def detect_hands(model, image_path):
    """
    Detect human pose and extract hand keypoints.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Unable to load image: {image_path}")
        return [], None

    results = model(image)

    hands_data = []

    for result in results:

        if hasattr(result, "keypoints") and result.keypoints is not None:

            keypoints = result.keypoints.xy.cpu().numpy()

            for kp in keypoints:

                # YOLO pose keypoints index:
                # 9  -> left wrist
                # 10 -> right wrist

                if len(kp) > 10:

                    left_hand = kp[9]
                    right_hand = kp[10]

                    hands_data.append({
                        "left_hand": left_hand.tolist(),
                        "right_hand": right_hand.tolist()
                    })

                    # Draw keypoints
                    cv2.circle(
                        image,
                        tuple(map(int, left_hand)),
                        10,
                        (0, 255, 0),
                        -1
                    )

                    cv2.circle(
                        image,
                        tuple(map(int, right_hand)),
                        10,
                        (0, 0, 255),
                        -1
                    )

    # Save annotated image
    cv2.imwrite(str(OUTPUT_IMAGE), image)

    return hands_data, OUTPUT_IMAGE


def save_results(image_name, hands_data):
    """
    Save detected hand keypoints to JSON.
    """

    output_data = {
        "image_name": str(image_name),
        "detections": hands_data
    }

    with open(OUTPUT_JSON, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"[INFO] Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":

    print("[INFO] Loading YOLOv8 pose model...")

    model = load_model()

    if model:

        print("[INFO] Running hand keypoint detection...")

        hands_data, output_image_path = detect_hands(
            model,
            IMAGE_PATH
        )

        if hands_data:

            save_results(IMAGE_PATH.name, hands_data)

            print(f"[INFO] Annotated image saved to: {output_image_path}")

        else:
            print("[INFO] No hand keypoints detected.")
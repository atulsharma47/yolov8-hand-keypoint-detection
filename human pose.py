import cv2
import json
import numpy as np
from ultralytics import YOLO

def load_model():
    """Load the pre-trained YOLO model for human pose detection."""
    try:
        model = YOLO('yolov8n-pose.pt')  
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def detect_hands(model, image_path):
    """Detect human poses and extract hand keypoints."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return [], ""
    
    results = model(image)
    hands_data = []
    
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Extract keypoints
            
            for kp in keypoints:
                if len(kp) > 10:
                    left_hand = kp[9]  # Left hand keypoint
                    right_hand = kp[10]  # Right hand keypoint
                    hands_data.append({"hand1": left_hand.tolist(), "hand2": right_hand.tolist()})
                    
                    # Draw keypoints on the image
                    cv2.circle(image, tuple(map(int, left_hand)), 20, (0, 255, 0), -1)
                    cv2.circle(image, tuple(map(int, right_hand)), 20, (0, 0, 255), -1)
    
    output_image_path = "annotated_image.jpg"
    cv2.imwrite(output_image_path, image)
    
    return hands_data, output_image_path

def save_results(image_name, hands_data):
    """Save hand keypoints to a JSON file."""
    output_json = {"hands": [{"imagename": image_name, **hand} for hand in hands_data]}
    
    with open("hand_keypoints1.json", "w") as json_file:
        json.dump(output_json, json_file, indent=4)
    
    print("Results saved to hand_keypoints1.json")

if __name__ == "__main__":
    image_path = "test_image_2.jpg"  
    model = load_model()
    if model:
        hands_data, output_image_path = detect_hands(model, image_path)
        if hands_data:
            save_results(image_path, hands_data)
            print(f"Annotated image saved as {output_image_path}")
        else:
            print("No hand keypoints detected.")

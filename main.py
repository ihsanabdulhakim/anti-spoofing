import torch
from torchvision import models, transforms
import torch.nn.functional as F
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
import os
import shutil
from face_detection import YuNet  # Import YuNet model class

# Define class labels and their corresponding colors
CLASS_LABELS = ['realhuman', 'spoofhuman']  # Use only two labels
CLASS_COLORS = {
    'realhuman': (0, 255, 0),  # Green
    'spoofhuman': (0, 0, 255)  # Red
}

# Load the trained model
model_path = './model/anti-spoofing-resnet50.pt'
device = torch.device('cpu')  # Use CPU

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_LABELS))  

# Load the saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Initialize YuNet face detector
yunet_model_path = './model/face_detection_yunet_2023mar.onnx'
yunet = YuNet(
    modelPath=yunet_model_path,
    inputSize=[320, 320],
    confThreshold=0.9,
    nmsThreshold=0.3,
    topK=5000,
    backendId=cv2.dnn.DNN_BACKEND_OPENCV,
    targetId=cv2.dnn.DNN_TARGET_CPU
)

# Define transformations for the face image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to match training input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Match training normalization
])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.9

# Create the directory to save 'realhuman' images, clear it if it already exists
save_dir = 'realhuman_saved'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)  # Remove all files in the folder
os.makedirs(save_dir, exist_ok=True)  # Recreate the folder

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    h, w, _ = frame.shape
    yunet.setInputSize([w, h])

    # Inference
    results = yunet.infer(frame)  # Get face detection results

    if results is not None:  # Check if any faces were detected
        for det in results:
            bbox = det[0:4].astype(np.int32)
            conf = det[-1]

            # Only process if confidence is above the threshold
            if conf >= CONFIDENCE_THRESHOLD:
                x, y, w, h = bbox
                face = frame[y:y + h, x:x + w]

                # Check if face extraction is successful
                if face.size == 0:
                    print("Extracted face is empty. Skipping this face.")
                    continue  # Skip processing if face extraction failed

                # Convert to RGB and apply transformations
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_tensor = transform(face_pil).unsqueeze(0)

                with torch.no_grad():
                    face_tensor = face_tensor.to(device)
                    outputs = model(face_tensor)

                    # Apply Softmax to get probabilities
                    probabilities = F.softmax(outputs, dim=1)
                    confidences, predictions = torch.max(probabilities, 1)
                    confidence = confidences.item()
                    class_idx = predictions.item()

                    # Determine the label
                    label = CLASS_LABELS[class_idx]

                    # Check if the label is 'realhuman' and meets the confidence threshold
                    if label == 'realhuman':
                        color = CLASS_COLORS[label]
                        text_label = f"{label}: {confidence:.4f}"

                        # Print the confidence value to the terminal
                        #print(f"Detected {label} with confidence: {confidence:.4f}") #uncomment this if needed

                        # Draw bounding box and label text for 'realhuman' only
                        cv2.putText(frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        # Save the face if confidence is bigger than 0.99
                        if round(confidence, 4) > 0.99:
                            # Generate a unique filename using timestamp
                            filename = os.path.join(save_dir, f"realhuman_{x}_{y}_{w}_{h}.jpg")
                            try:
                                cv2.imwrite(filename, face)  # Save the cropped face image
                                print(f"Saved 'realhuman' image: {filename}")
                            except Exception as e:
                                print(f"Failed to save image: {e}")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

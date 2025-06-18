import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys

# Debug flag
DEBUG = True

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    """Calculate the angle in degrees between three points."""
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)
    
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    angle = math.degrees(math.acos(min(1.0, max(-1.0, dot_product / (mag1 * mag2)))))
    return angle

# Function to detect and classify pose
def detect_and_classify_pose(frame, pose, is_static=False):
    """Detect pose landmarks and classify the yoga pose."""
    if DEBUG:
        print("Entering detect_and_classify_pose...")
    
    output_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    label = "Unknown"
    color = (0, 0, 255)  # Red for unknown
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            output_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        height, width, _ = frame.shape
        landmarks = [(landmark.x * width, landmark.y * height, landmark.z * width) 
                     for landmark in results.pose_landmarks.landmark]
        
        # Extract key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        if DEBUG:
            print(f"Angles - Left Elbow: {left_elbow_angle:.2f}, Right Elbow: {right_elbow_angle:.2f}")
            print(f"        Left Shoulder: {left_shoulder_angle:.2f}, Right Shoulder: {right_shoulder_angle:.2f}")
            print(f"        Left Hip: {left_hip_angle:.2f}, Right Hip: {right_hip_angle:.2f}")
            print(f"        Left Knee: {left_knee_angle:.2f}, Right Knee: {right_knee_angle:.2f}")
        
        # Pose classification logic
        if (160 < left_elbow_angle < 190 and 160 < right_elbow_angle < 190 and
            left_shoulder_angle < 30 and right_shoulder_angle < 30):
            label = "T-Pose"
        elif ((160 < left_knee_angle < 190 and right_knee_angle < 60) or
              (160 < right_knee_angle < 190 and left_knee_angle < 60)):
            label = "Tree Pose"
        elif (160 < left_elbow_angle < 190 and 160 < right_elbow_angle < 190 and
              70 < left_shoulder_angle < 110 and 70 < right_shoulder_angle < 110 and
              80 < left_knee_angle < 120 and 160 < right_knee_angle < 190):
            label = "Warrior Pose"
        
        if label != "Unknown":
            color = (0, 255, 0)  # Green for recognized pose
    
    cv2.putText(output_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return output_frame, label

# Function to resize image with zoom factor
def zoom_image(image, zoom_factor):
    """Resize the image based on zoom factor."""
    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    if new_height > 0 and new_width > 0:
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return image

# Static image mode with zoom
def process_static_image(image_path):
    """Process a static image for pose detection with zoom."""
    if DEBUG:
        print("Entering process_static_image...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}. Please check the file name and path.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir(os.path.dirname(image_path))}")
        sys.exit(1)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}. File might be corrupted or not an image.")
        sys.exit(1)
    
    print("Processing static image...")
    print("Press '+' to zoom in, '-' to zoom out, 'q' to quit.")
    
    zoom_factor = 1.0
    step = 0.1  # Zoom step size
    
    while True:
        zoomed_image = zoom_image(image, zoom_factor)
        output_image, label = detect_and_classify_pose(zoomed_image, pose, is_static=False)
        
        cv2.imshow("Static Yoga Pose Detection", output_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('+') and zoom_factor < 2.0:
            zoom_factor += step
            print(f"Zoom factor: {zoom_factor:.1f}")
        elif key == ord('-') and zoom_factor > 0.5:
            zoom_factor -= step
            print(f"Zoom factor: {zoom_factor:.1f}")
        elif key == ord('q') or cv2.getWindowProperty("Static Yoga Pose Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    print(f"Detected Pose: {label}")
    print("Static mode completed.")

# Live webcam mode with zoom
def process_live_video():
    """Process live video feed for pose detection with zoom."""
    if DEBUG:
        print("Entering process_live_video...")
    
    print("Starting live yoga pose detection...")
    print("Press '+' to zoom in, '-' to zoom out, 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if it's connected or try a different index (e.g., 1).")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened successfully. Starting video feed...")
    
    zoom_factor = 1.0
    step = 0.1  # Zoom step size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Webcam may be disconnected.")
            break
        
        frame = cv2.flip(frame, 1)
        zoomed_frame = zoom_image(frame, zoom_factor)
        output_frame, label = detect_and_classify_pose(zoomed_frame, pose, is_static=False)
        
        cv2.imshow("Live Yoga Pose Detection", output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('+') and zoom_factor < 2.0:
            zoom_factor += step
            print(f"Zoom factor: {zoom_factor:.1f}")
        elif key == ord('-') and zoom_factor > 0.5:
            zoom_factor -= step
            print(f"Zoom factor: {zoom_factor:.1f}")
        elif key == ord('q') or cv2.getWindowProperty("Live Yoga Pose Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Live detection stopped.")

# Main execution
if __name__ == "__main__":
    if DEBUG:
        print("Script started...")
    
    working_dir = r"C:\Users\sagar_c7otrfh\OneDrive\Documents\free-yoga-website-template_new\free-yoga-website-template"
    image_path = os.path.join(working_dir, "unknown1.jpg")
    
    print("Choose mode (1 for static image, 2 for live video): ", end="")
    try:
        mode = input().strip()
        if DEBUG:
            print(f"User entered mode: '{mode}'")
        
        if mode == "1":
            process_static_image(image_path)
        elif mode == "2":
            process_live_video()
        else:
            print("Invalid choice. Enter 1 for static image or 2 for live video.")
            sys.exit(1)
    except EOFError:
        print("Error: No input provided. Please run in a terminal and enter 1 or 2.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
    finally:
        pose.close()
        print("Script completed.")
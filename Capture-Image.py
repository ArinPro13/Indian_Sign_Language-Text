import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set destination folder for captured frames
dest_folder = "/Users/arinpramanick/Desktop/SIH-Project/Cropped frames/Test_image"
os.makedirs(dest_folder, exist_ok=True)

# Function to crop the image to hands and handle potential errors
def crop_image_to_hands(image, hand_landmarks_list):
    if not hand_landmarks_list:
        return None

    try:
        x_min = min([min([landmark.x for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
        y_min = min([min([landmark.y for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
        x_max = max([max([landmark.x for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
        y_max = max([max([landmark.y for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])

        h, w, _ = image.shape
        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    except:
        return None

# Frame counter and capture flag
frame_count = 0
capturing = False

# Create a separate window for the cropped image
cropped_window_name = "Cropped Hands"
cv2.namedWindow(cropped_window_name)

while True:
    success, image = cap.read()
    if not success:
        break

    # Process the image
    results = hands.process(image)

    # Draw hand landmarks and annotations
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cropped_image = crop_image_to_hands(image.copy(), results.multi_hand_landmarks)
            if cropped_image is not None:
                # Create a white background for the cropped image
                white_bg = np.ones((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8) * 255

                # Overlay the cropped image onto the white background
                hands_only = white_bg.copy()
                non_white_pixels = np.where(cropped_image != 0)
                hands_only[non_white_pixels] = cropped_image[non_white_pixels]

                # Capture frames within the destination folder if capturing is enabled
                if capturing and frame_count < 100:
                    frame_count += 1
                    cv2.imwrite(f"{dest_folder}/Image_{time.time()}.jpg", hands_only)
                    print("Frame Count:", frame_count)

                # Display the cropped image with white background in the separate window
                cv2.imshow(cropped_window_name, hands_only)

    # Display the original image with hand landmarks
    cv2.imshow('Hand Detection with Rectangles', image)

    # Check for keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Toggle capturing
        capturing = not capturing
        if capturing:
            frame_count = 0  # Reset the frame counter when starting to capture
        print("Capturing:", capturing)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

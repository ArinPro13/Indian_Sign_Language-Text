import cv2
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
import json

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load the pre-trained model
model = load_model("/Users/arinpramanick/Desktop/SIH-Project/Project/ISL_Predict.h5")  # Load the model using Keras directly

# Load class labels from the JSON file
with open("/Users/arinpramanick/Desktop/SIH-Project/Project/class_indices_letter.json", "r") as f:
    labels_dict = json.load(f)

# Ensure labels are in the right order (assuming indices are in ascending order)
labels = [labels_dict[str(i)] for i in range(len(labels_dict))]

# Capture video from webcam
cap = cv2.VideoCapture(0)

def crop_image_to_hands(image, hand_landmarks_list):
    """
    Crop the image to include both hands.
    """
    x_min = min([min([landmark.x for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
    y_min = min([min([landmark.y for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
    x_max = max([max([landmark.x for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])
    y_max = max([max([landmark.y for landmark in hand_landmarks.landmark]) for hand_landmarks in hand_landmarks_list])

    # Convert normalized coordinates to absolute pixel values
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

    # Add some padding around the cropped area
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

while True:
    success, image = cap.read()

    # Process the image
    results = hands.process(image)

    # Draw hand landmarks and annotations
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Crop the image to include both hands
        cropped_image = crop_image_to_hands(image, results.multi_hand_landmarks)

        # Resize the cropped image to match the input size of the model
        resized_cropped_image = cv2.resize(cropped_image, (224, 224))  # Resize to the input size expected by your model
        resized_cropped_image = np.expand_dims(resized_cropped_image, axis=0)  # Add batch dimension
        resized_cropped_image = resized_cropped_image / 255.0  # Normalize the image

        # Prediction with the model
        prediction = model.predict(resized_cropped_image)  # Get the prediction probabilities
        predicted_index = np.argmax(prediction)  # Find the class with the highest score
        predicted_label = labels[predicted_index]  # Get the corresponding label

        print(f"Prediction: {predicted_label}")
        cv2.putText(image, f"Prediction: {predicted_label}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

        # Show the cropped and processed image
        # cv2.imshow('Cropped Hand Detection', cropped_image)

    # Display the original image with hand landmarks
    cv2.imshow('Hand Detection with Rectangles', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("/Users/arinpramanick/Desktop/SIH-Project/Model/Model_Keras/keras_model.h5", "")
offset = 20
imgSize = 300
labels = ['H', 'E', 'L', 'O', 'W', 'R', 'D']

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Resize based on smaller dimension
            scale = min(imgSize / w, imgSize / h)
            new_w, new_h = int(w * scale), int(h * scale)
            imgResize = cv2.resize(imgCrop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create white background with padding
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            pad_x = (imgSize - new_w) // 2
            pad_y = (imgSize - new_h) // 2
            imgWhite[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = imgResize

            # Predict the letter
            prediction, index = classifier.getPrediction(imgWhite)
            print(f"Predicted letter: {prediction} ({labels[index]})")

    # Display the original image and the cropped hand image
    cv2.imshow("Original Image", img)
    cv2.imshow("Cropped Hand", imgCrop)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
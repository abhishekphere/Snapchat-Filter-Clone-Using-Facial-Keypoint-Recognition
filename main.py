"""
This is the main program that implements Face detection, Facial recognition and maps filters on the face.
"""

import cv2
import numpy as np

from helper import *

# Load the model built in the previous step
my_model = load_my_CNN_model('my_model')

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Define filters
filters = [ 'images/sunglasses_4.png', 'images/sunglasses_6.png', 'images/sunglasses_3.png', 'images/sunglasses_2.png', 'images/sunglasses.png', 'images/sunglasses_5.jpg']
filterIndex = 0

# Load the video - from webcam input
camera = cv2.VideoCapture(0)
index = 0
while True:
    (grabbed, frame) = camera.read()

    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)

    # Convert to HSV and GRAY for convenience
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haar cascade object
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    center = None

# Loop over all the faces found in the frame
    for (x, y, w, h) in faces:
        # Make the faces ready for the model (normalize, resize and stuff)
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]    

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape # A Copy for future reference
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predict the keypoints using the model
        keypoints = my_model.predict(face_resized)

        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the original image
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # Pair the keypoints together - (x1, y1)
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0] - points[9][0]) * 1.1)
        sunglass_height = int((points[10][1] - points[8][1]) / 1.1)
        sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_CUBIC)
        transparent_region = sunglass_resized[:, :, :3] != 0
        face_resized_color[int(points[9][1]):int(points[9][1]) + sunglass_height,
        int(points[9][0]):int(points[9][0]) + sunglass_width, :][transparent_region] = sunglass_resized[:, :, :3][
            transparent_region]

        # Map the face with shades back to its original shape
        frame[y:y + h, x:x + w] = cv2.resize(face_resized_color, original_shape, interpolation=cv2.INTER_CUBIC)

        # Add KEYPOINTS to the frame2
        for keypoint in points:
            cv2.circle(face_resized_color2, keypoint, 1, (0, 255, 0), 1)

        # Map the face with keypoints back to the original image (a separate one)
        frame2[y:y + h, x:x + w] = cv2.resize(face_resized_color2, original_shape, interpolation=cv2.INTER_CUBIC)

        # Show the frame and the frame2
        cv2.imshow("Selfie Filters", frame)
        cv2.imshow("Facial Keypoints", frame2)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
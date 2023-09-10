import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model('AI_Fish\model.h5')

# Define the class labels
class_labels = ['Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Stripped Red Mullet', 'Trout']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the object tracker
tracker = None
detected_object = False
x, y, w, h = 0, 0, 0, 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess the frame for prediction
    input_image = cv2.resize(frame, (224, 224))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

    # Make predictions
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
        detected_object = True
    else:
        predicted_class_label = "Unknown class"
        detected_object = False

    # Object tracking
    if detected_object:
        if tracker is None:
            # Initialize the tracker when an object is first detected
            x, y, w, h = cv2.boundingRect(np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]]))
            tracker = cv2.legacy.TrackerCSRT_create()  # Use cv2.TrackerMOSSE_create() instead of cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
        else:
            # Update the position of the rectangle to follow the tracked object
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(i) for i in box]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)  # Green color

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, predicted_class_label, (x, y - 10), font, font_scale, color, thickness)

    # Show the frame
    cv2.imshow('Camera', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

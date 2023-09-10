import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Make predictions
input_image = image.load_img('test_data\Gilt-Head.png', target_size=(224, 224))
input_image = image.img_to_array(input_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions)

class_labels = ['Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Stripped Red Mullet', 'Trout']

if predicted_class_index < len(class_labels):
    predicted_class_label = class_labels[predicted_class_index]
    print("Predicted class label:", predicted_class_label)
else:
    print("Unknown class label")

import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Make predictions
input_image = image.load_img('test_data/sea Bass.png', target_size=(224, 224))
input_image = image.img_to_array(input_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions)

class_labels = ['Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Stripped Red Mullet', 'Trout']

# Calculate probabilities and normalize them to sum up to 100
probabilities = tf.nn.softmax(predictions)
normalized_probs = probabilities.numpy()[0] * 100 / np.sum(probabilities)

# Sort the probabilities and class labels in descending order
sorted_indices = np.argsort(normalized_probs)[::-1]
sorted_labels = [class_labels[i] for i in sorted_indices]
sorted_probs = normalized_probs[sorted_indices]

# Display the results
print("Predicted class probabilities:")
for label, prob in zip(sorted_labels, sorted_probs):
    print(f"{label}: {prob:.2f}%")

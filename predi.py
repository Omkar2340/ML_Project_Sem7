import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model('saved_model_main.keras')

# Define the path to your uploads folder
uploads_folder = r"C:\Users\omkar\OneDrive\Desktop\Kaggle_lung\uploads" # Update with the actual path

# Define the target size for the model
target_size = (150, 150)  # This matches the model's input size

# Get the list of image files in the uploads folder
image_files = [f for f in os.listdir(uploads_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to predict images
def predict_images(model, image_files, uploads_folder):
    predictions = []
    for image_file in image_files:
        # Load and preprocess the image
        img_path = os.path.join(uploads_folder, image_file)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        
        # Assuming categorical output, get the class label (update as necessary)
        class_label = np.argmax(prediction, axis=-1)[0]  # Get the class index

        predictions.append((image_file, class_label))
    
    return predictions

# Make predictions
predictions = predict_images(model, image_files, uploads_folder)

# Print predictions
for image_file, class_label in predictions:
    print(f"Image: {image_file}, Predicted Class: {class_label}")

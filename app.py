import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_prep_image(file, img_shape=256):
    # Convert BytesIO to numpy array using PIL
    img = Image.open(file).convert("RGB")
    img = img.resize((img_shape, img_shape))

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0

    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Function to make a prediction
def pred_and_plot(model, img, class_names):
    pred = model.predict(img)

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    return pred_class

# Load models
model_paths = {
    'Cassava': "Cassava.h5",
    'Rice': "Rice.h5",
    'apple': "apple.h5",
    'grape': "grape.h5",
    'Plant': "Plant_disease.h5",
}

class_names = {
    'Plant': ['Cassava', 'Rice', 'apple', 'cheery', 'corn', 'grape', 'orange', 'peach', 'pepper,bell', 'potato',
              'squash', 'strawberry', 'tomato'],
    'Cassava': ['Bacterial Blight', 'Brown Streak Disease', 'Green Mottle', 'Healthy', 'Mosaic Disease'],
    'Rice': ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast'],
    'apple': ['apple scab', 'black rot', 'cedar apple rust', 'healthy'],
    'grape': ['black rot', 'esca', 'healthy', 'leaf blight'],
}

# Load the main plant disease model
leaf_detector = tf.keras.models.load_model(model_paths['Plant'])

# Streamlit App
st.title("Plant Disease Classifier and Diagnosis")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Disease solutions dictionary
disease_solutions = {
    'Bacterial Blight': 'Apply antibacterial spray.',
    'Brown Streak Disease': 'Isolate infected plants and apply appropriate fungicides.',
    'Green Mottle': 'Ensure proper nutrition and water management.',
    'Healthy': 'No action required for healthy plants.',
    'Mosaic Disease': 'Isolate infected plants and destroy them to prevent further spread.',
    'BrownSpot': 'Use disease-resistant rice varieties and practice proper field hygiene.',
    'Hispa': 'Apply insecticides and use cultural control methods.',
    'LeafBlast': 'Use resistant varieties and apply fungicides when needed.',
    'apple scab': 'Prune infected branches and apply fungicides.',
    'black rot': 'Remove and destroy infected plants. Apply fungicides as a preventive measure.',
    'cedar apple rust': 'Prune infected branches and apply fungicides.',
    'esca': 'Prune infected parts and apply fungicides. Ensure proper vineyard hygiene.',
    'leaf blight': 'Apply fungicides and practice good garden hygiene.',
}

if uploaded_file is not None:
    # Display the uploaded image
    image = load_and_prep_image(uploaded_file)
    st.image(image[0], caption="Uploaded Image.", use_column_width=True)

    # Make a prediction with the main model
    pred_class = pred_and_plot(leaf_detector, image, class_names['Plant'])

    st.write(f"**Predicted Plant:** {pred_class}")

    # Load the specific model based on the predicted class
    if pred_class in model_paths:
        specific_model_path = model_paths[pred_class]
        specific_model = tf.keras.models.load_model(specific_model_path)

        # Get the disease classes for the specific plant
        specific_diseases = class_names[pred_class]

        # Make a prediction with the specific model
        disease_prediction = pred_and_plot(specific_model, image, class_names[pred_class])
        st.write(f"**Predicted Disease:** {disease_prediction}")

        # Display solution for the predicted disease
        if disease_prediction in disease_solutions:
            st.write(f"**Solution:** {disease_solutions[disease_prediction]}")
        else:
            st.warning("No solution found for the predicted disease.")

    else:
        st.warning(f"No specific model found for plant: {pred_class}")

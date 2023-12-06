# Import necessary libraries
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Function to process the input image for prediction
def process_image(image_path):
    try:
        im = Image.open(image_path)
        test_image = np.asarray(im)
        image = tf.cast(test_image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image /= 255
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        exit()

# Function to load class names from a JSON file
def get_class_names(json_path):
    try:
        with open(json_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        exit()

# Function to make predictions using the model
def predict(processed_image, model, top_k):
    try:
        ps = model.predict(np.expand_dims(processed_image, axis=0))
        probabilities = ps[0]
        if top_k is None:
            classes = probabilities.argsort()[::-1][:]
            probs = probabilities[classes]
        else:
            classes = probabilities.argsort()[::-1][:top_k]
            probs = probabilities[classes]
        return probs, classes
    except Exception as e:
        print(f"Error during prediction: {e}")
        exit()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('image_path', type=str, help='Specify path to the image')
parser.add_argument('model', type=str, help='Specify the path to the model to use')
parser.add_argument('--top_k', type=int, help='To return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
args = parser.parse_args()

# Main script logic
image_path = args.image_path
model_path = args.model
top_k = args.top_k
category_names = args.category_names

# Process the image
print("Processing image...")
image = process_image(image_path)

# Load the model
print(f"Loading model from {model_path}...")
try:
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Make prediction
print("Making prediction...")
probs, classes = predict(image, loaded_model, top_k)

# Output results
if category_names is not None:
    print("Mapping class names...")
    class_names = get_class_names(category_names)
    names_class = []
    for i in classes:
        names_class.append(class_names.get(str(i + 1)))
    for i in range(len(probs)):
        print(f"Class: {names_class[i]}, Label: {classes[i]}, Probability: {probs[i]}")
else:
    for i in range(len(probs)):
        print(f"Label: {classes[i]}, Probability: {probs[i]}")

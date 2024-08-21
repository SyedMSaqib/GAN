import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the pre-trained discriminator model
discriminator = load_model('discriminator.h5')

def classify_image(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to (32, 32) as expected by the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # If preprocessing was applied during training, include it here
    img_array = img_array / 255.0  # Uncomment if normalization was used

    # Predict
    prediction = model.predict(img_array)

    # Interpret the result
    threshold = 0.5
    if prediction[0][0] > threshold:
        return "\033[1mReal\033[0m"
    else:
        return "\033[1mFake\033[0m"

# Path to the image you want to test
img_path = 'putin.jpeg'
result = classify_image(img_path, discriminator)
print(f"The image is: {result}")

# Disable Matplotlib warning for non-GUI backend
plt.switch_backend('agg')

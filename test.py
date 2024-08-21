import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the pre-trained discriminator model
discriminator = load_model('discriminator.h5')

def preprocess_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to (32, 32) as expected by the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image to [-1, 1], same as during training
    img_array = (img_array - 127.5) / 127.5
    
    return img_array

def classify_image(img_path, model):
    img_array = preprocess_image(img_path)

    # Predict
    prediction = model.predict(img_array)
    print(f"Discriminator raw output: {prediction}")

    # Interpret the result
    threshold = 0.5
    if prediction[0][0] > threshold:
        return "Real"
    else:
        return "Fake"

# Path to the image you want to test
img_path = 'ganPanda.png'
result = classify_image(img_path, discriminator)
print(f"The image is: {result}")

# Display the image
img = image.load_img(img_path, target_size=(32, 32))  # Resize to (32, 32) for display
plt.imshow(img)
plt.title(f"Classified as: {result}")
plt.show()

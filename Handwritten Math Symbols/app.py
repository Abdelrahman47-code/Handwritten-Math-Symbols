from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('handwritten_math_symbols_model.keras')
symbol_folders = ['-', '!', '(', ')', ',', '+', '0']

# Initialize Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling image upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Create the 'temp' directory if it doesn't exist
    temp_dir = 'static/temp'  # Adjusted path
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Get the uploaded image file
    img_file = request.files['image']

    # Save the image to a temporary folder
    img_path = os.path.join(temp_dir, 'temp_image.jpg')
    img_file.save(img_path)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    prediction_index = np.argmax(predictions)
    predicted_label = symbol_folders[prediction_index]
    confidence = predictions[0][prediction_index]

    # Set a threshold confidence level (adjust as needed)
    threshold_confidence = 0.5

    if confidence >= threshold_confidence and predicted_label in symbol_folders:
        # Render the result page with the predicted label
        return render_template('result.html', predicted_label=predicted_label)
    else:
        # Render a page indicating that no symbol was recognized
        return render_template('no_symbol.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

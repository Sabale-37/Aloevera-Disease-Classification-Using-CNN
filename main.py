from flask import Flask, render_template, request, url_for
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# =============================
# Load the Model
# =============================
MODEL_PATH = os.path.join(os.path.dirname(_file_), 'aloe_vera_model.h5')  # Update this path if needed
model = load_model(MODEL_PATH)
print("Model Loaded Successfully")

# =============================
# Utility Function for Prediction
# =============================
def predict_aloe_disease(image_path):
    # Load and preprocess the image
    test_image = load_img(image_path, target_size=(128, 128))
    print("@@ Got Image for Prediction")
    
    test_image = img_to_array(test_image) / 255.0  # Normalize the image
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    
    # Predict the class
    result = model.predict(test_image)
    print('@@ Raw Result =', result)
    
    pred_class = np.argmax(result, axis=1)[0]
    print("@@ Predicted Class =", pred_class)
    
    # Map predicted class to disease name and HTML page
    class_map = {
        0: ("Healthy Aloe Vera", 'aloevera_healthy.html'),
        1: ("Aloe Vera - Rot Disease", 'aloevera_rot.html'),
        2: ("Aloe Vera - Rust Disease", 'aloevera_rust.html')
    }
    
    return class_map.get(pred_class, ("Unknown Condition", 'index.html'))

# =============================
# Flask App Initialization
# =============================
app = Flask(_name_)

# =============================
# Home Route
# =============================
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# =============================
# Prediction Route
# =============================
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded image
        file = request.files['image']
        filename = file.filename
        print("@@ Input Posted =", filename)
        
        # Save uploaded file
        upload_dir = 'static/upload/'  # Relative path
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)  # Ensure the upload directory exists
        
        file_path = os.path.join(upload_dir, filename)
        try:
            file.save(file_path)
        except Exception as e:
            print("Error saving file:", e)
            return "Failed to upload image", 500
        
        print("@@ Predicting Class...")
        disease_name, output_page = predict_aloe_disease(file_path)
        
        # Generate URL for the uploaded image
        file_url = url_for('static', filename='upload/' + filename)
        
        # Render the output HTML page with prediction result
        return render_template(output_page, pred_output=disease_name, user_image=file_url)

# =============================
# Main Execution
# =============================
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

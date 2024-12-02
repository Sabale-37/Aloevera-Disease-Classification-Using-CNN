---

# 🌿 Aloe Vera Disease Classification Project

This project focuses on detecting and classifying diseases in aloe vera plants using a Convolutional Neural Network (CNN). The system identifies three conditions—**Healthy**, **Rot**, and **Rust**—to support early detection, promote sustainable farming practices, and aid farmers in maintaining healthy crops.

---

## 🚀 Project Overview

### Objectives:
- Develop an automated system to detect and classify aloe vera diseases.
- Provide early disease detection to improve agricultural outcomes.
- Create a user-friendly web application for real-time classification.

### Key Features:
- **Disease Detection**: Identifies Healthy, Rot, and Rust conditions.
- **Augmentation Support**: Enhances model robustness with augmented data.
- **Background Removal**: Improves model accuracy by focusing on essential features.
- **Web Application**: Deployable Flask-based web app for real-time disease classification.

---

## 📊 Dataset

### Structure:
The dataset contains **9000 images** categorized into three classes:

  ## Classes:
   1. **Healthy**
   2. **Rot**
   3. **Rust**

---

## 🛠️ Technology Stack

### **Hardware**:
- **Processor**: AMD Ryzen 5 Hexa Core 5600H
- **RAM**: 8 GB
- **Storage**: 512 GB SSD
- **Graphics**: NVIDIA GeForce RTX 3050 Ti (4 GB)

### **Software**:
- **AI/ML Frameworks**: TensorFlow, Keras
- **Front-end**: HTML, CSS, JavaScript
- **Back-end**: Flask
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib
- **Deployment**: Flask Web Application

---

## ⚙️ How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sabale-37/Aloevera-Disease-Classification-Using-CNN
   cd Aloevera-Disease-Classification-Using-CNN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```bash
   python main.py
   ```

4. **Open the web interface**:
   - Navigate to `http://localhost:5000` in your web browser.

---

## 📈 Model Training

### Steps:
1. **Data Preprocessing**:
   - Normalize images.
   - Data augmentation to enhance model performance.

2. **Model Architecture**:
   - Implemented a Convolutional Neural Network (CNN) with Keras.
   - Configured for classification with three output classes.

3. **Training**:
   - Used cross-entropy loss and the Adam optimizer.
   - Evaluated using accuracy, precision, and recall metrics.

---

## 🌟 Future Enhancements
- Integrate real-time detection using a mobile app.
- Enhance model accuracy with more diverse datasets.
- Implement a feedback mechanism for continuous model improvement.

---

## 🤝 Contribution
Contributions are welcome! Feel free to fork this repository, submit issues, or pull requests.

---


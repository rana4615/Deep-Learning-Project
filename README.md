# 🫁 Chest X-Ray Medical Diagnosis using Deep Learning
This project uses Convolutional Neural Networks (CNNs) to classify chest X-ray images into normal and abnormal (potentially pneumonia-infected) categories. Leveraging deep learning with TensorFlow/Keras, this diagnostic model demonstrates how AI can assist in automating preliminary radiological analysis.

## 📌 Project Highlights
- ✅ Image classification of chest X-rays using CNN
- 🧠 Applied data preprocessing and augmentation
- 📊 Trained and validated using Keras & TensorFlow
- 📈 Achieved high accuracy and strong model performance on validation data
- 📁 Dataset: Chest X-ray dataset (e.g., [Kaggle's Chest X-ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia))

## 🧠 Model Architecture
- **Input layer**: Resized grayscale X-ray images
- **Convolutional layers**: Feature extraction
- **MaxPooling**: Dimensionality reduction
- **Dropout**: Regularization to prevent overfitting
- **Dense layers**: Final classification
- **Activation**: ReLU and Sigmoid

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/chest-xray-diagnosis.git
pip install -r requirements.txt
jupyter notebook ChestXRay_Medical_Diagnosis_Deep_Learning.ipynb

📊 Results

Training accuracy: ~97%,
Validation accuracy: ~94%,
Loss curves show good convergence with minimal overfitting.

⚙️ Technologies Used

Python,
TensorFlow / Keras,
OpenCV / PIL,
Matplotlib / Seaborn,
NumPy / Pandas,
Jupyter Notebook,

✅ Use Cases

Medical imaging diagnostics,
Radiology assistance tools,
AI in healthcare research,
Early screening applications

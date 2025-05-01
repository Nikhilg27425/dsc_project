
# 🩺 Pneumonia Detection from Chest X-Ray Images

This project utilizes deep learning techniques to detect pneumonia from chest X-ray images. By leveraging Convolutional Neural Networks (CNNs) built with TensorFlow and Keras, the model aims to assist in the early diagnosis of pneumonia, thereby improving patient outcomes.

## 📖 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Deployment](#deployment)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## 📝 Overview

Pneumonia is a significant health concern worldwide, and early detection is crucial for effective treatment. This project focuses on automating the detection process using deep learning, reducing the reliance on manual interpretation of X-ray images.

## 📊 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which includes:

- **Normal**: Chest X-ray images without pneumonia.
- **Pneumonia**: Chest X-ray images showing signs of pneumonia (both bacterial and viral).

## 🧠 Model Architecture

The CNN model comprises:

- **Input Layer**: Accepts grayscale images resized to 224x224 pixels.
- **Convolutional Layers**: Extract features using filters.
- **Pooling Layers**: Reduce spatial dimensions.
- **Fully Connected Layers**: Interpret features to make predictions.
- **Output Layer**: Binary classification indicating presence or absence of pneumonia.

## 🏋️ Training Details

- **Framework**: TensorFlow and Keras.
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam.
- **Metrics**: Accuracy, Precision, Recall.
- **Epochs**: 10.
- **Batch Size**: 32.
- **Data Augmentation**: Applied to enhance model generalization.

## 📈 Evaluation Metrics

The model's performance is evaluated using:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Correct positive predictions over total positive predictions.
- **Recall (Sensitivity)**: Correct positive predictions over actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

## 🚀 Deployment

A Streamlit web application is developed for user interaction:

1. **Upload**: Users can upload chest X-ray images.
2. **Prediction**: The model processes the image and predicts the likelihood of pneumonia.
3. **Result**: Displays the prediction with confidence score.

To run the app:

```bash
streamlit run app.py
```

## ⚙️ Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model**:

   The trained model is hosted on Google Drive. Use the following code to download:

   ```python
   import gdown

   url = 'https://drive.google.com/uc?id=your_file_id'
   output = 'pneumonia_model_best.h5'
   gdown.download(url, output, quiet=False)
   ```

4. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

## 📊 Results

The model achieved:

- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 94%
- **F1 Score**: 92%

These results indicate the model's effectiveness in detecting pneumonia from chest X-ray images.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- TensorFlow and Keras for the deep learning frameworks.
- Streamlit for the web application deployment.

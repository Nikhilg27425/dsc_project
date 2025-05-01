

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.graph_objects as go
import pandas as pd

# Set page configuration nnn
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #5c88da;
        margin-bottom: 1rem;
    }
    .prediction-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .prediction-pneumonia {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Set constants
IMG_WIDTH, IMG_HEIGHT = 224, 224  # as per model input
MODEL_PATH = "pneumonia_model_best.h5"

# Load model once
@st.cache_resource
def load_cnn_model():
    try:
        return load_model(MODEL_PATH)
    except:
        # Fallback path in case the first one doesn't work
        return load_model('pneumonia_detection_model.h5')

# Navigation bar
def create_navbar():
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.markdown('<div class="main-header">Pneumonia Detection from Chest X-ray</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Home", "About Pneumonia", "How It Works", "Dataset Information"])
    
    return tabs

# Sidebar content
def create_sidebar():
    #st.sidebar.image("https://via.placeholder.com/150x150.png?text=X-ray+AI", width=150)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("", ["Upload & Detect", "Project Information", "Technical Details"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About This Project")
    st.sidebar.markdown("""
    This application uses a deep learning model to detect pneumonia from chest X-ray images.
    
    The model was trained on thousands of X-ray images and achieves high accuracy in distinguishing between normal lungs and those affected by pneumonia.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## References")
    st.sidebar.markdown("[Read the full blog post](https://medium.com/@23ucs752/detecting-pneumonia-from-chest-x-rays-using-deep-learning-and-scikit-learn-dd37da4e9720)")
    
    return page

# Main page function
def main():
    # Load the model
    try:
        model = load_cnn_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Create navbar and sidebar
    tabs = create_navbar()
    page = create_sidebar()
    
    # Home Tab Content
    with tabs[0]:
        if page == "Upload & Detect":
            st.markdown('<div class="sub-header">Upload a chest X-ray image for analysis</div>', unsafe_allow_html=True)
            
            # Information box
            st.markdown("""
            <div class="info-box">
                <p>This application uses a trained convolutional neural network to analyze chest X-ray images
                and predict whether they show signs of pneumonia or appear normal.</p>
                <p>For best results, upload a clear, frontal chest X-ray image in JPG, JPEG, or PNG format.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # File uploader
            uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
            
            col1, col2 = st.columns([1, 1])
            
            # If file is uploaded
            if uploaded_file is not None:
                with col1:
                    # Load and display the image
                    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
                    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
                
                with col2:
                    # Process and predict
                    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = img_to_array(image_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(img_array)[0][0]
                    label = "Pneumonia" if prediction > 0.5 else "Normal"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    
                    # Create gauge chart for visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=float(prediction),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Prediction Score", 'font': {'size': 24}},
                        gauge={
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.5], 'color': 'lightgreen'},
                                {'range': [0.5, 1], 'color': 'lightcoral'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction result with styled box
                    if label == "Normal":
                        st.markdown(f'<div class="prediction-normal">Prediction: {label} (Confidence: {confidence:.2f})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-pneumonia">Prediction: {label} (Confidence: {confidence:.2f})</div>', unsafe_allow_html=True)
                    
                    # Add interpretation
                    st.markdown("### Interpretation:")
                    if label == "Normal":
                        st.write("The model predicts that this X-ray shows no significant signs of pneumonia. The lungs appear to be within normal parameters.")
                    else:
                        st.write("The model predicts that this X-ray shows signs consistent with pneumonia. There appear to be areas of opacity or consolidation in the lung fields.")
                    
                    st.warning("Note: This tool is for educational purposes only and should not replace professional medical diagnosis.")
            
            else:
                # Sample images section when no upload
                st.markdown("### Sample X-rays")
                st.write("Below are examples of normal and pneumonia-affected chest X-rays:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image("normal.jpg", caption="Example of Normal X-ray")
                with col2:
                    st.image("pneumonia.jpg", caption="Example of Pneumonia X-ray")
        
        elif page == "Project Information":
            st.markdown("## Pneumonia Detection Project")
            st.markdown("""
            ### What is Pneumonia?
            
            Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. 
            Symptoms typically include some combination of productive or dry cough, chest pain, fever, and difficulty breathing.
            
            ### Why Deep Learning for Pneumonia Detection?
            
            Deep learning models, particularly Convolutional Neural Networks (CNNs), have shown remarkable success in medical image analysis.
            The ability of these models to automatically learn features from raw data makes them particularly well-suited for radiological applications.
            
            ### Project Goals
            
            1. To develop an accurate model for pneumonia detection from chest X-rays
            2. To create an accessible tool that can assist healthcare professionals
            3. To demonstrate the potential of AI in medical diagnostics
            """)
            
            # Add more project information here
            st.markdown("### Model Performance")
            
            metrics = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [0.92, 0.94, 0.91, 0.93]
            }
            
            st.dataframe(pd.DataFrame(metrics), hide_index=True)
            
        elif page == "Technical Details":
            st.markdown("## Technical Implementation")
            st.markdown("""
            ### Model Architecture
            
            This application uses a Convolutional Neural Network (CNN) trained from scratch.
            The model was trained on the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
            
            ```
            Model Architecture:
            - Input Layer (224x224x1)
            - Multiple Convolutional blocks with MaxPooling
            - Dropout layers for regularization
            - Dense layers with ReLU activation
            - Output layer with sigmoid activation for binary classification
            ```
            
            ### Training Process
            
            The model was trained using:
            - Adam optimizer with learning rate scheduling
            - Binary cross-entropy loss function
            - Data augmentation to improve generalization
            - Early stopping to prevent overfitting
            
            ### Preprocessing Pipeline
            
            1. Resize images to 224x224 pixels
            2. Convert to grayscale if needed
            3. Normalize pixel values to range [0,1]
            4. Apply data augmentation during training
            5. Applying gaussian filters to remove noise
            6. Applying histogram equalizer to improve contrast of the image
            """)
            
            
    
    # About Pneumonia Tab
    with tabs[1]:
        st.markdown("## About Pneumonia")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### What is Pneumonia?
            
            Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.
            
            ### Causes
            
            Pneumonia can be caused by:
            - Bacteria
            - Viruses
            - Fungi
            - Aspiration of food, liquid, or vomit
            
            ### Risk Factors
            
            - Age (very young and elderly)
            - Weakened immune system
            - Chronic diseases
            - Smoking
            - Hospitalization
            """)
        
        with col2:
            st.markdown("""
            ### Symptoms
            
            Common symptoms of pneumonia include:
            - Chest pain when breathing or coughing
            - Confusion or changes in mental awareness (in adults aged 65 and older)
            - Cough, which may produce phlegm
            - Fatigue
            - Fever, sweating and shaking chills
            - Lower than normal body temperature (in adults older than 65 and people with weak immune systems)
            - Nausea, vomiting or diarrhea
            - Shortness of breath
            
            ### Diagnosis
            
            Pneumonia is typically diagnosed through:
            - Physical examination
            - Chest X-rays
            - Blood tests
            - Sputum tests
            - Pulse oximetry
            """)
    
    # How It Works Tab
    with tabs[2]:
        st.markdown("## How the AI Detection Works")
        
        st.markdown("""
        ### Deep Learning for Pneumonia Detection
        
        This application uses a Convolutional Neural Network (CNN) to analyze chest X-ray images and detect signs of pneumonia.
        
        #### The Process:
        
        1. **Image Upload**: User uploads a chest X-ray image
        2. **Preprocessing**: The image is resized to 224x224 pixels and normalized
        3. **Feature Extraction**: The CNN extracts relevant features from the image
        4. **Classification**: The model predicts whether pneumonia is present
        5. **Result Display**: The prediction and confidence score are displayed
        
        ### Why CNNs Work Well for Medical Imaging
        
        CNNs are particularly effective for medical image analysis because:
        
        - They automatically learn hierarchical features from the data
        - They can detect subtle patterns that might be missed by the human eye
        - They maintain spatial relationships between pixels
        - They can be trained to be robust to variations in image quality and positioning
        """)
        
        # Add explanation of model architecture
        st.image("cnn_arch.png", caption="Simplified CNN Architecture for Pneumonia Detection")
    
    # Dataset Information Tab
    with tabs[3]:
        st.markdown("## Dataset Information")
        
        st.markdown("""
        ### Chest X-Ray Dataset
        
        This model was trained on the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which contains:
        
        - 5,863 X-Ray images (JPEG)
        - 2 categories: Pneumonia and Normal
        - Pediatric patients aged 1-5 years from Guangzhou Women and Children's Medical Center
        
        ### Data Distribution
        
        - **Training set**: 5,216 images
        - **Validation set**: 16 images
        - **Test set**: 624 images
        
        ### Class Distribution
        
        - **Normal**: 1,583 images
        - **Pneumonia**: 4,273 images (2,780 bacterial, 1,493 viral)
        """)
        
        # Add visualization of dataset distribution
        data = pd.DataFrame({
            'Category': ['Normal',  'Viral Pneumonia'],
            'Count': [3875, 1341]
        })
        
        fig = go.Figure([go.Bar(x=data['Category'], y=data['Count'], marker_color=['#4CAF50', '#FF5722', '#2196F3'])])
        fig.update_layout(
            title="Dataset Distribution",
            xaxis_title="Category",
            yaxis_title="Number of Images",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Pneumonia Detection App | Created with Streamlit | 2025</p>
        <p>Disclaimer: This application is for educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
### Wheat Leaf Disease Detection using Deep Learning (VGG19)
## Project Overview
This project focuses on the automated detection and classification of wheat leaf diseases using Convolutional Neural Networks (CNNs) with Transfer Learning (VGG19). It supports smart agriculture by enabling early disease diagnosis, reducing manual errors, and promoting sustainable farming.

### Problem Statement
Manual identification of crop diseases is often inaccurate, slow, and not scalable. This project addresses the challenge by developing a CNN-based model capable of classifying wheat leaf diseases from images taken in real-world conditions.

### Key Concepts
CNN (Convolutional Neural Networks) for automatic feature extraction

Transfer Learning with VGG19 pretrained on ImageNet

Data Augmentation to simulate diverse real-world environments

Image Preprocessing and Model Evaluation using standard metrics

### Tech Stack
Language: Python

Libraries/Frameworks: TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn

Deployment Tool: Streamlit

Development Platform: Google Colab / Jupyter Notebook

### Dataset
    ~3,700 labeled images (Healthy, Leaf Rust, Yellow Rust, Crown Rot)

Data sourced from Kaggle

Augmented to ~9,000+ samples using:

Rotation (±30°)

Horizontal/Vertical Flip

Zoom & Brightness adjustment

Noise addition

### Model Architecture
    Base Model: VGG19 (include_top=False)

    Custom Classifier Head:

    AveragePooling2D

    Flatten

    Dense(512, activation='relu')

    Dropout(0.2)

    Dense(4, activation='softmax')

    Loss Function: Categorical Crossentropy

    Optimizer: Adam (lr=1e-3)

    Callbacks: EarlyStopping, ReduceLROnPlateau

### Performance Metrics
    Validation Accuracy: ~95%

    F1-Score: High across all four classes

    AUC-ROC: > 0.95

    Confusion Matrix: Clear separation with minimal misclassifications

### Deployment
The model is deployed using Streamlit, offering:

Simple UI for uploading wheat leaf images

Real-time prediction and confidence score

Accessible via web browser (desktop & mobile)

### Future Scope
Deploy on mobile apps and edge devices (Raspberry Pi, Android)
Expand to other crops like rice, maize, and vegetables
Integrate with IoT and drone surveillance
Implement Explainable AI (Grad-CAM)
Add multilingual, farmer-friendly interfaces

### Team Members
Aruhi Pallavi
Akash
Abhishek Kushwaha
Supervisor: Mr. Vikas Jalodia, Department of CSE, GCET

# Wheat Leaf Disease Detection using Deep Learning (VGG19)
📌Project Overview
This project focuses on the automated detection and classification of wheat leaf diseases using Convolutional Neural Networks (CNNs) with Transfer Learning (VGG19). It aims to support smart agriculture by enabling early disease diagnosis, minimizing manual errors, and promoting sustainable farming practices.

We classified wheat leaves into four categories:

Healthy

Leaf Rust

Yellow Rust

Crown Rot

🎯 Problem Statement
Manual identification of crop diseases is time-consuming, labor-intensive, and often inaccurate. This project addresses the challenge by building a deep learning model capable of accurately classifying wheat leaf diseases from images—even under varying environmental conditions.

🧠 Key Concepts
CNNs (Convolutional Neural Networks) for feature extraction and classification

Transfer Learning using VGG19 pre-trained on ImageNet

Data Augmentation to simulate real-world conditions

Explainable AI with Grad-CAM (optional)

🛠️ Tech Stack
Language: Python

Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn

Deployment: Streamlit (for building an interactive web app)

Development Environment: Google Colab & Jupyter Notebooks

📂 Dataset
~3,700 images (Healthy, Leaf Rust, Yellow Rust, Crown Rot)

Sourced from Plant Village, Kaggle, and custom real-world images

Augmented to ~9,000+ images using:

Rotation

Horizontal Flip

Zoom

Brightness Variation

Noise Injection

🔧 Model Architecture
Base: VGG19 (pre-trained, include_top=False)

Custom Head:

AveragePooling2D

Flatten

Dense (512, ReLU)

Dropout (0.2)

Dense (4, Softmax)

Loss: Categorical Crossentropy

Optimizer: Adam (lr=1e-3)

Callbacks: EarlyStopping, ReduceLROnPlateau

📊 ### Results
Validation Accuracy: ~95%

F1-Score: High across all classes

AUC-ROC: > 0.95

Confusion Matrix showed minimal class overlap

Successfully generalized to real-world wheat leaf images

🌐 Deployment
The model was deployed using Streamlit, allowing users to:

Upload an image of a wheat leaf

Receive real-time classification results

View the predicted class and confidence score

💡 Future Scope
Integration with drones and IoT devices for real-time field monitoring

Model optimization for edge devices (mobile phones)

Expansion to multiple crops and regional disease databases

Incorporation of Explainable AI (Grad-CAM)

Localization with multilingual user interfaces for farmers

🙌 Team
Aruhi Pallavi

Akash

Abhishek Kushwaha
Guided by: Mr. Vikas Jalodia, Department of CSE, GCET

# Lung-Cancer-Classification
A deep learning-based lung cancer classification project using CNN, Transfer Learning (VGG16, ResNet50, DenseNet121, AlexNet), and K-Fold cross-validation on histopathological images.

# 🫁 Lung Cancer Classification with Deep Learning
This project focuses on the classification of lung cancer cell images using advanced deep learning techniques. Various CNN architectures and transfer learning models were trained, evaluated, and compared to determine the most effective approach for accurate diagnosis of lung cancer.

## 📂 Dataset
The dataset consists of histopathological images categorized into 4 classes :
- **Benign**
- **Adenocarcinoma**
- **Squamous Cell Carcinoma**
- **Unknown**

A total of **6100** images were used, equally distributed among the four classes.

- Dataset Link : https://academictorrents.com/details 7a638ed187a6180fd6e464b3666a6ea0499af4af

## 🧠 Models Used
### ✅ Custom CNN Model
A specially designed convolutional neural network architecture developed from scratch for baseline comparison.

### ✅ VGG16 (Transfer Learning)
Used ImageNet-pretrained VGG16 as a base model with custom classification layers on top.

### ✅ ResNet50 (Transfer Learning)
Employed the powerful ResNet50 architecture with fine-tuned layers for cancer cell image classification.

### ✅ AlexNet
An early but influential deep CNN architecture adapted for this classification task.

### ✅ DenseNet121
Utilized DenseNet121 for its efficient parameter usage and feature reuse.

### ✅ K-Fold Cross Validation
Implemented K-Fold cross-validation (K=5) to evaluate the generalizability of models.

## 📊 Evaluation Metrics
For all models, the following metrics were computed :
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- ROC-AUC (per class + macro average)
- Model Summary

## 🖥️ Project Structure
📁 Procets/
│ ├── AlexNet_Model
│ ├── CNN_Model
│ ├── Datasets
│ ├── DenseNet121_Model
│ ├── KFold_Model
│ ├── ResNet50_Model
│ └── VGG16_Model
├── .gitignore
├── LICENSE (MIT)
└── README.md


## 🧪 Technologies Used
- Python 3.10+
- TensorFlow / Keras
- Matplotlib / Seaborn / scikit-learn
- NumPy / PIL / OpenCV
- VsCode + CUDA destekli eğitim

## 💻 Interface
The project has a separate interface (GUI) for each model. Users can predict the class of the cell image they have uploaded.


## 🚀 Getting Started
1. Clone the repository :
   git clone https://github.com/<aliiball>/Lung-Cancer-Classification.git
   cd Lung-Cancer-Classification
   
2. Create a virtual environment and activate it :
   python -m venv env
   source env/bin/activate  # Linux/macOS
   .\env\Scripts\activate   # Windows

3. Install dependencies :
   pip install -r requirements.txt

4. Run a training script :
   python app/vgg16_model.py

5. Launch the GUI interface :
   python app/main.py

## ✨ Contributors
👨‍💻 Muhammet Ali BAL – Project Developer & Researcher

## 📜 License:
This project was created by Muhammet Ali BAL for his bachelor's degree thesis and is intended for academic and educational purposes only. Permission is required for commercial use.
# 🧠🩺 Brain & Breast Tumor Detection and Segmentation

This project presents a comprehensive deep learning system for the **detection** and **segmentation** of tumors in **brain** and **breast** medical images. The system uses five distinct models, organized in a modular pipeline, and is deployed using an interactive **Streamlit** web application. This work aims to support medical diagnosis by offering an intelligent, automated solution that enhances accuracy and efficiency in radiological workflows.

---

## 📌 Project Objectives

- ✅ Automatically classify whether the uploaded image is a **brain** or **breast** scan.
- ✅ Detect the presence of a **tumor in brain scans**.
- ✅ Perform **segmentation of brain tumors**, if present.
- ✅ Classify **breast scans** into **normal**, **benign**, or **malignant** categories.
- ✅ Segment **breast tumors** for benign or malignant cases.
- ✅ Provide an **interactive web interface** to support medical practitioners.

---

## 🧱 System Architecture

The system follows a conditional multi-model pipeline based on the input type:

| Model # | Task                          | Input        | Output                        | Type         |
|---------|-------------------------------|--------------|-------------------------------|--------------|
| Model 1 | Brain vs Breast Classification | Image        | `"Brain"` / `"Breast"`        | CNN          |
| Model 2 | Brain Tumor Detection          | Brain Image  | `"Tumor"` / `"No Tumor"`      | Xception     |
| Model 3 | Brain Tumor Segmentation       | Brain Image  | Tumor Segmentation Mask       | U-Net        |
| Model 4 | Breast Tumor Classification    | Breast Image | `"Normal"`, `"Benign"`, `"Malignant"` | Xception     |
| Model 5 | Breast Tumor Segmentation      | Breast Image | Tumor Segmentation Mask       | U-Net        |

---

## 📚 Dataset Description

### 🧠 Brain Tumor Dataset
- **Source:** [Kaggle - Brain & Breast Tumor](https://www.kaggle.com/datasets/khaledhellmy/brain-breast-tumor)
- **Type:** MRI
- **Labels:** Tumor / No Tumor
- **Includes:** Pixel-wise segmentation masks

### 🩺 Breast Tumor Dataset
- **Source:** [Kaggle - Breast Ultrasound Images](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)
- **Labels:** Normal / Benign / Malignant
- **Includes:** Segmentation masks


---

## 🧠 Model Details

### 🔹 Model 1: Brain vs Breast Image Classification
- Type: Binary CNN
- Architecture: Custom CNN
- Output: Brain or Breast

    ![alt text](Images\image.png)

### 🔹 Model 2: Brain Tumor Detection
- Type: Binary Classifier
- Architecture: Xception
- Metrics: Accuracy, Precision, Recall

    ![alt text](Images\image-1.png)

### 🔹 Model 3: Brain Tumor Segmentation
- Type: Semantic Segmentation
- Architecture: U-Net
- Loss Function: Dice + Binary Cross-Entropy

    ![alt text](Images\image-2.png)

### 🔹 Model 4: Breast Tumor Classification
- Type: Multiclass Classifier
- Architecture: Xception
- Classes: Normal, Benign, Malignant
- Imbalance Handling: Random Oversampling

    ![alt text](Images\image-3.png)

### 🔹 Model 5: Breast Tumor Segmentation
- Type: Semantic Segmentation
- Architecture: U-Net
- Loss Function: Dice + Categorical Cross-Entropy

    ![alt text](Images\image-4.png)
---

## ⚙️ Implementation Details

- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Environment:** Google Colab, Kaggle Notebooks
- **Libraries:** OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Streamlit

### 🧪 Training Configuration
- Optimizer: Adam
- Batch Size: 8–32
- Early Stopping and Learning Rate Scheduling applied

---

## 📊 Results & Evaluation

| Model                          | Training Accuracy | Validation Accuracy |
|-------------------------------|-------------------|---------------------|
| Brain vs Breast Classification | 100%              | 100%                |
| Brain Tumor Detection          | 99.44%            | 94.00%              |
| Brain Tumor Segmentation       | 98.90%            | 95.32%              |
| Breast Tumor Classification    | 96.26%            | 89.31%              |
| Breast Tumor Segmentation      | 98.27%            | 95.71%              |

Visual results include:
- Classification labels
- Tumor segmentation masks overlaid on input images

---

## 🚀 Deployment: Streamlit App

The entire pipeline is deployed using **Streamlit**, enabling easy use for doctors, researchers, and healthcare workers.

### 🔹 App Features
- Upload medical image (brain or breast)
- Automatic classification using Model 1
- Conditional pipeline execution:
  - Brain scans → Detection → Segmentation
  - Breast scans → Classification → Segmentation
- Visual display of:
  - Classification result
  - Segmentation overlay (if applicable)

---
## 📦 Requirements

To run this project locally, install the following dependencies:

```bash
pip install -r requirements.txt
```

## 👥 Contributors

- Khaled Zakarya Zyada 
- Khaled Ahmed Helmy  




---

## 📬 Contact

For any questions or feedback, please contact:  
- `[khaledzzyadaa@gmail.com]`
- `[khhelmy654@gmail.com]`  
or open an issue in this repository.
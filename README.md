<div align="center">
  <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
    <img src="images/cover.png" alt="Logo" width="" height="200">
  </a>

<h1 align="center">Medica Image Segmentation</h1>
</div>
    
## 1. Problem Statement
Medical image segmentation is a crucial part of artificial intelligence. It involves tasks like identifying cancer in X-ray images. Our goal is to develop a model that can automatically outline the stomach and intestines in MRI scans. To achieve this, we will use a popular image segmentation algorithm.
<div align="center">
  <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
    <img src="images/image1.jpg" alt="Logo" width="" height="200">
  </a>
</div>

In this figure, the tumor (pink thick line) is close to the stomach (red thick line). High doses of radiation are directed to the tumor while avoiding the stomach. The dose levels are represented by the rainbow of outlines, with higher doses represented by red and lower doses represented by green.

However, this task presents challenges, especially in cancer detection, which can be difficult even for experts. Given the sensitivity and importance of this field, failing to detect cancer in a patient can lead to significant problems. On the other hand, incorrectly diagnosing someone with cancer can also cause substantial distress for the patient and their family.
Our model needs to meet the following criteria:

* High accuracy is essential, even if it sacrifices speed.
* The model should provide clear explanations because of the sensitive nature of this field.
* Thorough documentation is crucial for our project's success.

## 2. Related Works
there are several way to impelement segmentation algorithm for medical image classification. but we divide this into 2 group:
* Convloutional methods:
  * U-Net: The architecture consists of this model contracting path to capture context and a symmetric expanding path that enables precise localization.
  * V-net: V-Net extends U-Net by incorporating volumetric convolutions to handle 3D medical image segmentation.
  * DeepLab V3+: DeepLab V3+ employs atrous spatial pyramid pooling and encoder-decoder architecture with depthwise separable convolutions for dense prediction tasks.
    
* Transformer-Based Methods:
  * Vision Transformer (ViT): ViT is a transformer architecture initially proposed for image classification tasks. It divides the input image into fixed-size patches, linearly embeds them, and processes them using transformer blocks. Despite being designed for classification, ViT and its variants have been adapted for segmentation tasks in medical imaging.

  * SegFormer: SegFormer extends the transformer architecture for semantic segmentation tasks. It introduces a hybrid CNN-transformer backbone and employs a hierarchical transformer decoder to generate segmentation masks.

  * TransUNet: TransUNet combines the strengths of convolutional neural networks (CNNs) and transformers. It uses a CNN backbone to extract spatial features and a transformer decoder to capture global contextual information for semantic segmentation.

## 3. The Proposed Method
In this work, we propose the utilization of U-Net for our task. Our decision is supported by several reasons:

* Simplicity and Efficiency: U-Net offers a simple architecture with fewer parameters, facilitating easier training and optimization.

* Proven Performance: U-Net has demonstrated its effectiveness across various tasks in medical image analysis, consistently achieving state-of-the-art results.

* Versatility and Adaptability: The modular design of U-Net allows for easy customization and adaptation to different imaging modalities and applications.

* Robustness and Generalization: Due to its architectural characteristics, U-Net tends to generalize well to diverse datasets and imaging conditions.

* Community Support and Resources: U-Net enjoys widespread adoption in the research community, providing access to pre-trained models, implementation libraries, and extensive documentation
  
![architecture of proposed model](images/im2g.jpg)

This figure illustrates the architecture of our designed U-Net model, which consists of a contracting path to capture contextual information and an expanding path for precise localization.

## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://drive.google.com/file/d/1-2ggesSU3agSBKpH-9siKyyCYfbo3Ixm/view?usp=sharing)

### 4.2. Model
In this subsection, the architecture and specifics of the deep learning model employed for the segmentation task are presented. It describes the model's layers, components, libraries, and any modifications made to it.

### 4.3. Configurations
This part outlines the configuration settings used for training and evaluation. It includes information on hyperparameters, optimization algorithms, loss function, metric, and any other settings that are crucial to the model's performance.

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.

### 4.5. Evaluate
In the evaluation section, the methods and metrics used to assess the model's performance are detailed. It explains how the model's segmentation results are quantified and provides insights into the model's effectiveness.

# 🖼️ COCO Image Caption Generator

This is a deep learning project that generates descriptive captions for images using a CNN-RNN architecture. Built with PyTorch and deployed using Streamlit, the application allows users to upload an image and receive an AI-generated caption trained on a vocabulary inspired by the COCO dataset.

---

## 🚀 Demo

![Image Caption Generator Demo](https://github.com/yourusername/your-repo-name/assets/demo.gif)  
*Optional: Include a short gif/screenshot of your app in action.*

---

## 🧠 Model Overview

- **Encoder**: ResNet-50 CNN extracts image features.
- **Decoder**: LSTM RNN generates natural language captions.
- **Vocabulary**: Demo uses a dummy vocab. Replace with COCO-trained vocab for full performance.

---

## 🛠️ Features

- 📤 Upload images (JPG, PNG, JPEG)
- 🗂️ Optionally use sample images from `train2014` folder
- 🤖 Generates descriptive captions
- 💻 Compatible with both CPU and GPU

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
pip install -r requirements.txt
```

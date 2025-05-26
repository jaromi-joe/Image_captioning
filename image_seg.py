# image_segmentation_app.py

import streamlit as st
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Apply segmentation
def segment_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)["out"]
    prediction = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    return prediction

# Overlay segmentation mask
def decode_segmap(mask, labels=None):
    label_colors = np.array([
        (0, 0, 0),       # 0=background
        (128, 0, 0),     # 1=aeroplane
        (0, 128, 0),     # 2=bicycle
        (128, 128, 0),   # 3=bird
        (0, 0, 128),     # 4=boat
        (128, 0, 128),   # 5=bottle
        (0, 128, 128),   # 6=bus
        (128, 128, 128), # 7=car
        (64, 0, 0),      # 8=cat
        (192, 0, 0),     # 9=chair
        (64, 128, 0),    # 10=cow
        (192, 128, 0),   # 11=diningtable
        (64, 0, 128),    # 12=dog
        (192, 0, 128),   # 13=horse
        (64, 128, 128),  # 14=motorbike
        (192, 128, 128), # 15=person
        (0, 64, 0),      # 16=potted plant
        (128, 64, 0),    # 17=sheep
        (0, 192, 0),     # 18=sofa
        (128, 192, 0),   # 19=train
        (0, 64, 128)     # 20=tv/monitor
    ])
    
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for label in np.unique(mask):
        idx = mask == label
        r[idx] = label_colors[label][0]
        g[idx] = label_colors[label][1]
        b[idx] = label_colors[label][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Streamlit UI
def main():
    st.title("🧠 Image Segmentation using DeepLabV3 + COCO")
    st.write("Upload an image and see semantic segmentation in action!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Segmenting... Please wait."):
            model = load_model()
            image_tensor = preprocess(image)
            mask = segment_image(model, image_tensor)
            seg_image = decode_segmap(mask)

            st.subheader("🎯 Segmentation Output")
            st.image(seg_image, caption="Segmented Output", use_column_width=True)

            if st.checkbox("Show raw mask array"):
                st.write(mask)

if __name__ == "__main__":
    main()

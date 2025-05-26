import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import pickle

# -------------------- SETUP --------------------
st.set_page_config(page_title="Image Caption Generator", layout="wide")
st.title("🖼️ COCO Image Caption Generator")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- VOCAB DUMMY EXAMPLE --------------------
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "a", 5: "dog", 6: "on", 7: "beach"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

vocab = Vocabulary()

# -------------------- TRANSFORMS --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# -------------------- MODELS --------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        output_ids = []
        states = None
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            output_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
            if predicted.item() == vocab.stoi["<EOS>"]:
                break
        return output_ids

# -------------------- LOAD MODELS --------------------
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

encoder.eval()
decoder.eval()

# Optionally load pre-trained weights if available
# encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
# decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

# -------------------- CAPTION GENERATOR --------------------
def generate_caption(image, encoder, decoder, vocab, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)
        output_ids = decoder.sample(features)
    words = []
    for idx in output_ids:
        word = vocab.itos.get(idx, "<UNK>")
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)
    return ' '.join(words)



# -------------------- STREAMLIT UI --------------------
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "png", "jpeg"])
default_images = os.listdir("train2014") if os.path.exists("train2014") else []

use_sample = False
if not uploaded_file and default_images:
    use_sample = st.checkbox("Use a sample image from COCO?", value=True)
    if use_sample:
        sample_image_name = st.selectbox("Choose sample image:", default_images)
        image_path = os.path.join("train2014", sample_image_name)
        uploaded_file = image_path

if uploaded_file:
    if isinstance(uploaded_file, str):
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Selected Image", use_column_width=True)
    st.write("🔍 Generating caption...")
    caption = generate_caption(image, encoder, decoder, vocab, transform, device)
    st.success(f"📝 Caption: {caption}")

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.deconvolution_model import DeconvolutionModel  # Import your Deconvolution model
from utils.gan_model import Generator  # Import your GAN generator (assuming it's in gan_model.py)
import torch.optim as optim

# Function to load model checkpoint
def load_checkpoint(model, filename, device):
    # Load the checkpoint
    checkpoint = torch.load(filename, map_location=device)
    
    # Extract the model state dict (ignoring the optimizer state dict)
    model_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Check if there are missing or unexpected keys, and handle them
    try:
        model.load_state_dict(model_state_dict)
        print(f"Checkpoint loaded from {filename}")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
    
    return model

# Set device for model inference (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Deconvolution model and GAN Generator model
deconv_model = DeconvolutionModel().to(device)  # Move Deconvolution model to the correct device
gan_generator = Generator().to(device)  # Move GAN generator model to the correct device

# Load checkpoint for both models
deconv_model = load_checkpoint(deconv_model, "checkpoints/checkpoint_epoch10.pth", device)
gan_generator = load_checkpoint(gan_generator, "GAN_checkpoints_10_epochs/best_generator.pth", device)

# Set models to eval mode
deconv_model.eval()
gan_generator.eval()

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Streamlit UI
st.title("Image Enhancement with AI Models")
st.write("Upload an image to enhance using either the Deconvolution Model or GAN.")

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Model selection
    model_choice = st.selectbox("Choose Enhancement Model", ["Deconvolution Model", "GAN"])

    # Prepare input image (move it to the correct device)
    input_image_tensor = transform(input_image).unsqueeze(0).to(device)

    if model_choice == "Deconvolution Model":
        with torch.no_grad():
            # Run inference with Deconvolution model
            deconv_output = deconv_model(input_image_tensor)
            deconv_output = deconv_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            deconv_output = (deconv_output * 0.5 + 0.5) * 255  # Denormalize

            st.image(deconv_output.astype("uint8"), caption="Enhanced by Deconvolution Model", use_container_width=True)
    else:
        with torch.no_grad():
            # Run inference with GAN Generator
            gan_output = gan_generator(input_image_tensor)
            gan_output = gan_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gan_output = (gan_output * 0.5 + 0.5) * 255  # Denormalize

            st.image(gan_output.astype("uint8"), caption="Enhanced by GAN", use_container_width=True)

# Run this Streamlit app with: streamlit run app.py

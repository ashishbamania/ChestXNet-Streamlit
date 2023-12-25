import torch
import torchvision.models as models
import streamlit as st
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image_path, image_size=224):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Define the preprocessing pipeline
    preprocess_pipeline = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image and add a batch dimension
    preprocessed_image = preprocess_pipeline(image).unsqueeze(0)
    
    return preprocessed_image

def predict_image(model, input_tensor):

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass through the model
    with torch.no_grad():
        output_logits = model(input_tensor)

    # Convert logits to probabilities
    probabilities = F.softmax(output_logits, dim=1)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities



# Initialize the model
model = models.densenet121()
checkpoint_best = torch.load('checkpoint', map_location=torch.device('cpu'))
model = checkpoint_best['model']

# Set up the Streamlit app title
st.set_page_config(page_title='Radiology With AI', initial_sidebar_state = 'auto')

st.title("🩺 Chest X-Ray Reported In Seconds")
subtitle = '<p style="font-size: 18px;font-weight: 800;">By Dr. Ashish Bamania inspired by <a href="https://github.com/jrzech/reproduce-chexnet"> Dr. John Zech\'s GitHub Repository </a></p>'
st.markdown(subtitle, unsafe_allow_html=True)

st.write("""""")

st.write("""
This application analyzes abnormal Chest X-rays and detects 14 common lung conditions using State-of-the-art AI technology. 
""")

st.write("""
Note that this is a Demo application and is not a replacement for a Radiologist.
""")

st.write("""""")
st.write("""""")

# Allow users to upload an image
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
     # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(uploaded_file)
    
    # Make predictions
    if st.button("👨‍⚕️  Report"):
        class_names =  [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia'
        ]
        
        predicted_class, class_probabilities = predict_image(model, input_tensor)
        
        report = f'<p style="font-size: 20px;">The Chest X-ray most likely shows: {class_names[predicted_class]}.'
        st.write(report, unsafe_allow_html=True)

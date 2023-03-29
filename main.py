import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader
import streamlit as st
import time
import zipfile
import base64

from custom_vit import ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = torch.load(model_path, map_location=device)
    return model

import os

def main():
    st.title("Image Classification App")

    # Set up sidebar
    model_path = "model/model.pt"
    uploaded_files = st.sidebar.file_uploader("Upload image files", accept_multiple_files=True)

    # Load model
    model = load_model(model_path)
    model.to(device)
    model.eval()

    # Process images
    if uploaded_files:
        if st.button("Run Prediction Model"):
            st.info(f"Running weld detection model on {len(uploaded_files)} images... :cd:")
            start_time = time.time()

            cw_count = 0
            non_cw_count = 0

            # Create directories if they don't exist
            os.makedirs("cw", exist_ok=True)
            os.makedirs("non-cw", exist_ok=True)

            for uploaded_file in uploaded_files:
                try:
                    # Load image
                    img = Image.open(uploaded_file).convert('RGB')
                    img_s = np.array(img)

                    # Apply transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])
                    img = transform(img)
                    img = img.unsqueeze(0).to(device)

                    # Make prediction
                    with torch.no_grad():
                        logits, _ = model(img)
                        predicted_class = torch.argmax(logits, dim=1)

                    # Save output images
                    predicted_label = predicted_class.item()
                    if predicted_label == 0:
                        cw_count += 1
                        cv2.imwrite(f"cw/cw_{cw_count}.jpg", img_s)
                    else:
                        non_cw_count += 1
                        cv2.imwrite(f"non-cw/non_cw_{non_cw_count}.jpg", img_s)

                except UnidentifiedImageError:
                    st.write(f"Error: Cannot identify image file '{filename}'")

            end_time = time.time()
            st.write("Finished processing.")

            # Show results
            st.write(f"Time elapsed: {end_time - start_time:.2f} seconds")
            st.write(f"Number of 'cw' images: {cw_count}")
            st.write(f"Number of 'non-cw' images: {non_cw_count}")

            # Create the zip files
            with zipfile.ZipFile('cw.zip', 'w') as zipf:
                for file in os.listdir('cw'):
                    zipf.write(os.path.join('cw', file))
            with zipfile.ZipFile('non-cw.zip', 'w') as zipf:
                for file in os.listdir('non-cw'):
                    zipf.write(os.path.join('non-cw', file))

            # Allow user to download folders separately

            with open("cw.zip", "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="cw.zip">Download cw folder</a>'
                st.markdown(href, unsafe_allow_html=True)

            with open("non-cw.zip", "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="non_cw.zip">Download non-cw folder</a>'
                st.markdown(href, unsafe_allow_html=True)



# Call the main function to start the app
if __name__ == "__main__":
    main()
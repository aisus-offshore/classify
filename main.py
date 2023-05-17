import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from PIL import UnidentifiedImageError
import streamlit as st
import time
import zipfile
import base64
import tensorflow as tf
import os
from custom_vit import ViTForImageClassification
from keras import layers
import tensorflow_addons as tfa

class RandomFlipWithTraining(layers.RandomFlip):
    def call(self, inputs, training=None):
        if training:
            return super().call(inputs)
        return inputs

#Load tensorflow models for Dark, Blur and Down View
def load_model_tf(model_path):
    with tf.keras.utils.custom_object_scope({'RandomFlipWithTraining': RandomFlipWithTraining, 'AdamW': tfa.optimizers.AdamW}):
        model = tf.keras.models.load_model(model_path)
    return model

def load_model_tw(model_path):
    with tf.keras.utils.custom_object_scope({'AdamW': tfa.optimizers.AdamW}):
        model = tf.keras.models.load_model(model_path)
    return model

#Activate cude if availiable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Weld Classifier model (pytorch)
def load_model_vit(model_path):
    model = torch.load(model_path, map_location=device)
    return model

#Zip folder for Blur, Dark and Down View Classifiers
def zip_folders(folder1):
    # Create the zip files for folder1
    with zipfile.ZipFile(f'{folder1}.zip', 'w') as zipf:
        for file in os.listdir(folder1):
            zipf.write(os.path.join(folder1, file))

    # Allow the user to download the folder separately
    with open(f'{folder1}.zip', "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{folder1}.zip">Download `{folder1}` folder</a>'
        st.markdown(href, unsafe_allow_html=True)

#Define ZipFolders for the last folders
def zip_folders_final(folder1, folder2):
    # Create the zip files for folder1
    with zipfile.ZipFile(f'{folder1}.zip', 'w') as zipf:
        for file in os.listdir(folder1):
            zipf.write(os.path.join(folder1, file))

    # Create the zip files for folder2
    with zipfile.ZipFile(f'{folder2}.zip', 'w') as zipf:
        for file in os.listdir(folder2):
            zipf.write(os.path.join(folder2, file))

    # Allow the user to download the folder separately
    with open(f'{folder1}.zip', "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{folder1}.zip">Download `{folder1}` folder</a>'
        st.markdown(href, unsafe_allow_html=True)

    with open(f'{folder2}.zip', "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{folder2}.zip">Download `{folder2}` folder</a>'
        st.markdown(href, unsafe_allow_html=True)


def disable_button(button: str):
    if button in st.session_state:
        st.session_state[button] = True


def main():
    st.set_page_config(page_title="Image Classification App", page_icon=":camera:")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Image Classification App")
    with col2:
        logo_image = "media/logo.png"
        st.image(logo_image, use_column_width=True)

    # Set up sidebar
    uploaded_files = st.sidebar.file_uploader("Upload image files", accept_multiple_files=True)

    if 'blur_run' not in st.session_state:
        st.session_state.blur_run = False
    if 'dark_run' not in st.session_state:
        st.session_state.dark_run = False
    if 'dv_run' not in st.session_state:
        st.session_state.dv_run = False
    if 'preprocessing_run' not in st.session_state:
        st.session_state.preprocessing_run = False

    #Define and load the model tensorflow
    model_path = "model/model.pt"
    blur_model = load_model_tf("model/effNet_blur.h5")
    dark_model = load_model_tf("model/effNet_dark.h5")
    views_model = load_model_tf("model/effNet_views.h5")
    tw_model = load_model_tw("model/effNet_tw.h5")

    # Load model pytorch
    model = load_model_vit(model_path)
    model.to(device)
    model.eval()

    # Process images
    if uploaded_files:
        non_count = 0
        # Preprocessing models for blur, dark, & dv
        st.header("Preprocessing")
        col1, col2, col3 = st.columns([1, 1, 1])

        # `blur` classifier
        with col1:
            # Define button
            if st.button("Run blur Classifier", disabled=st.session_state.blur_run, on_click=disable_button("blur")):
                st.info(f"Running image classification on {len(uploaded_files)} images... :cd:")
                start_time = time.time()
                # Add progress bar
                progress_bar = st.progress(0)
                blur_count = 0
                # Create directories if they don't exist
                os.makedirs("blur", exist_ok=True)
                os.makedirs("non", exist_ok=True)

                # for uploaded_file in uploaded_files:
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Extract the original filename
                        original_filename = uploaded_file.name

                        # Load image with PIL
                        img = Image.open(uploaded_file).convert('RGB')
                        img_s = np.array(img)
                        img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2BGR)

                        # Define the image transformation pipeline for tensorflow models
                        transform = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
                            tf.keras.layers.experimental.preprocessing.Normalization(),
                        ])

                        #Apply transforms
                        img = np.asarray(img)
                        img = transform(img)
                        img = np.expand_dims(img, axis=0)

                        #Theshold
                        threshold_blur = 0.5

                        # Make prediction
                        logits_blur = blur_model.predict(img) # 0:blur, 1:non

                        # Apply a sigmoid since our model returns logits
                        pred_blur = tf.nn.sigmoid(logits_blur)

                        # Apply custom threshold
                        pred_blur = tf.where(pred_blur < threshold_blur, 0, 1)

                        # Get the predicted class
                        predicted_class_blur = pred_blur.numpy()[0][0]

                        # Save output images
                        if predicted_class_blur <= threshold_blur:
                            blur_count += 1
                            cv2.imwrite(f"blur/blur_{blur_count}_{original_filename}", img_s)
                        else:
                            non_count += 1
                            cv2.imwrite(f"non/{original_filename}", img_s)


                    except UnidentifiedImageError:
                        st.write(f"Error: Cannot identify image file '{original_filename}'")

                    # Update progress bar
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                #Print processing time
                end_time = time.time()
                st.write(f"Time elapsed: {end_time - start_time:.2f} seconds")

                #Print info about the image classification
                st.info(f"Number of `blur` images: {len(os.listdir('blur'))}")
                
                #Zip Folders:
                if len(os.listdir('blur')) > 0:
                    st.write("Finished pre-processing. Zipping `blur` folder to download... :cd: ")
                    zip_folders("blur")

                # Update state
                st.session_state.blur_run = True
                if st.session_state.dark_run == True and st.session_state.dv_run == True:
                    st.session_state.preprocessing_run = True

        # `dark` classifier
        with col2:
            if st.button("Run dark Classifier", disabled=st.session_state.dark_run):
                st.info(f"Running image classification on {len(uploaded_files)} images... :cd:")
                start_time = time.time()
                # Add progress bar
                progress_bar = st.progress(0)
                dark_count = 0
                # Create directories if they don't exist
                os.makedirs("dark", exist_ok=True)
                os.makedirs("non", exist_ok=True)

                # for uploaded_file in uploaded_files:
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Extract the original filename
                        original_filename = uploaded_file.name

                        # Load image with PIL
                        img = Image.open(uploaded_file).convert('RGB')
                        img_s = np.array(img)
                        img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2BGR)

                        # Define the image transformation pipeline for tensorflow models
                        transform = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
                            tf.keras.layers.experimental.preprocessing.Normalization(),
                        ])

                        #Apply transforms
                        img = np.asarray(img)
                        img = transform(img)
                        img = np.expand_dims(img, axis=0)

                        #Theshold
                        threshold_dark = 0.5

                        # Make prediction
                        logits_dark = dark_model.predict(img) # 0:dark, 1:non

                        # Apply a sigmoid since our model returns logits
                        pred_dark = tf.nn.sigmoid(logits_dark)

                        # Apply custom threshold
                        pred_dark = tf.where(pred_dark < threshold_dark, 0, 1)

                        # Get the predicted class
                        predicted_class_dark = pred_dark.numpy()[0][0]

                        # Save output images
                        if predicted_class_dark <= threshold_dark:
                            dark_count += 1
                            cv2.imwrite(f"dark/dark_{dark_count}_{original_filename}", img_s)
                        else:
                            non_count += 1
                            cv2.imwrite(f"non/{original_filename}", img_s)


                    except UnidentifiedImageError:
                        st.write(f"Error: Cannot identify image file '{original_filename}'")

                    # Update progress bar
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                #Print processing time
                end_time = time.time()
                st.write(f"Time elapsed: {end_time - start_time:.2f} seconds")

                #Print info about the image classification
                st.info(f"Number of `dark` images: {len(os.listdir('dark'))}")
                
                #Zip Folders:
                if len(os.listdir('dark')) > 0:
                    st.write("Finished pre-processing. Zipping `dark` folder to download... :cd: ")
                    zip_folders("dark")

                # Update state
                st.session_state.dark_run = True
                if st.session_state.blur_run == True and st.session_state.dv_run == True:
                    st.session_state.preprocessing_run = True

        # `dv` classifier
        with col3:
            if st.button("Run dv Classifier", disabled=st.session_state.dv_run):
                st.info(f"Running image classification on {len(uploaded_files)} images... :cd:")
                start_time = time.time()
                # Add progress bar
                progress_bar = st.progress(0)
                dv_count = 0
                # Create directories if they don't exist
                os.makedirs("dv", exist_ok=True)
                os.makedirs("non", exist_ok=True)

                # for uploaded_file in uploaded_files:
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Extract the original filename
                        original_filename = uploaded_file.name

                        # Load image with PIL
                        img = Image.open(uploaded_file).convert('RGB')
                        img_s = np.array(img)
                        img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2BGR)

                        # Define the image transformation pipeline for tensorflow models
                        transform = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
                            tf.keras.layers.experimental.preprocessing.Normalization(),
                        ])

                        #Apply transforms
                        img = np.asarray(img)
                        img = transform(img)
                        img = np.expand_dims(img, axis=0)

                        #Theshold
                        threshold_dv = 0.5

                        # Make prediction
                        logits_dv = views_model.predict(img) # 0:dv, 1:non

                        # Apply a sigmoid since our model returns logits
                        pred_dv = tf.nn.sigmoid(logits_dv)

                        # Apply custom threshold
                        pred_dv = tf.where(pred_dv < threshold_dv, 0, 1)

                        # Get the predicted class
                        predicted_class_dv = pred_dv.numpy()[0][0]

                        # Save output images
                        if predicted_class_dv <= threshold_dv:
                            dv_count += 1
                            cv2.imwrite(f"dv/dv_{dv_count}_{original_filename}", img_s)
                        else:
                            non_count += 1
                            cv2.imwrite(f"non/{original_filename}", img_s)


                    except UnidentifiedImageError:
                        st.write(f"Error: Cannot identify image file '{original_filename}'")

                    # Update progress bar
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                #Print processing time
                end_time = time.time()
                st.write(f"Time elapsed: {end_time - start_time:.2f} seconds")

                #Print info about the image classification
                st.info(f"Number of `dv` images: {len(os.listdir('dv'))}")

                #Zip Folders:
                if len(os.listdir('dv')) > 0:
                    st.write("Finished pre-processing. Zipping `dv` folder to download... :cd: ")
                    zip_folders("dv")

                # Update state
                st.session_state.dv_run = True
                if st.session_state.blur_run == True and st.session_state.dark_run == True:
                    st.session_state.preprocessing_run = True

        # Classification models for anomaly and cw
        st.header("Classification")

        if st.button("Run anomaly Classifier", disabled=not st.session_state.preprocessing_run):
            start_time = time.time()

            st.info(f"Running weld classifier on {len(os.listdir('non'))} images... :cd:")

            #Folder path
            folder_path = 'non'

            # get a list of all the files in the folder
            file_list = os.listdir(folder_path)

            cw_count = 0
            non_cw_count = 0
            tw_count = 0

            # Create directories if they don't exist
            os.makedirs("cw", exist_ok=True)
            os.makedirs("non-cw", exist_ok=True)

            # Add second progress bar
            progress_bar_2 = st.progress(0)

            for idx, file_name in enumerate(file_list):
                # construct the full path to the image file
                full_path = os.path.join(folder_path, file_name)

                # get the original file name
                original_name = os.path.splitext(file_name)[0]

                try:
                    # Load image
                    img = Image.open(full_path).convert('RGB')
                    img_s = np.array(img) # For running into the pytorch model
                    img_tw = np.array(img) # For runing into the tensorflow (tw)
                    img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2BGR)

                    # Apply transforms for the pytorch
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])
                    img = transform(img)
                    img = img.unsqueeze(0).to(device)

                    # Make prediction for cw / non-cw #
                    with torch.no_grad():
                        logits, _ = model(img)
                        predicted_class = torch.argmax(logits, dim=1)

                    # Save output images
                    predicted_label = predicted_class.item()
                    if predicted_label == 0:
                        cw_count += 1
                        cv2.imwrite(f"cw/{original_name}_cw_{cw_count:04}.jpg", img_s)
                    else:
                        # Define the image transformation pipeline for the tw
                        transform = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
                            tf.keras.layers.experimental.preprocessing.Normalization(),
                        ])

                        # Apply transforms
                        imgt = np.asarray(img_tw)
                        imgt = transform(imgt)
                        imgt = np.expand_dims(imgt, axis=0)

                        # Theshold
                        threshold_tw = 0.59

                        # Make prediction
                        logits_tw = tw_model.predict(imgt)

                        # Apply a sigmoid since our model returns logits
                        pred_tw = tf.nn.sigmoid(logits_tw)

                        # Extract probability value as a scalar
                        prob_tw = pred_tw[0][0].numpy()

                        ##Here convert to have a correct prob scale
                        prob_tw = -1.5*prob_tw+1.45

                        # Convert probability to string
                        prob_str = "{:.2f}".format(prob_tw)

                        # Apply custom threshold
                        tw_classifier = tf.where(pred_tw < threshold_tw, 0, 1)

                        # Get the predicted class
                        predicted_class_tw = tw_classifier.numpy()[0][0]

                        if predicted_class_tw <= threshold_tw:
                            tw_count += 1
                            cv2.imwrite(f"non-cw/{original_name}_tw_{prob_str}.jpg", img_s)

                        else:
                            non_cw_count += 1
                            cv2.imwrite(f"non-cw/{original_name}.jpg", img_s)

                except UnidentifiedImageError:
                    st.write(f"Error: Cannot identify image file '{file_name}'")

                # Update the second progress bar
                progress_2 = (idx + 1) / len(file_list)
                progress_bar_2.progress(progress_2)

            end_time = time.time()
            st.write(f"Time elapsed: {end_time - start_time:.2f} seconds")

            # Show results

            st.info(f"Number of `cw` images: {cw_count}")
            st.info(f"Number of `non-cw` images: {non_cw_count}")
            st.write("Finished processing. Zipping folders... :cd:")

            # Create the zip files
            zip_folders_final("cw","non-cw")

# Call the main function to start the app
if __name__ == "__main__":
    main()

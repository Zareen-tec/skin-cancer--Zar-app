# -*- coding: utf-8 -*-
"""final.py"""

import os
import cv2
import numpy as np
import asyncio
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras import layers, models
import streamlit as st
from PIL import Image

# 🔧 Fix for Python 3.12 + Streamlit + Tornado (asyncio issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 1. Define Paths
dataset_path = "path/to/your/dataset"  # Replace with your dataset path
train_folder = "path/to/train/folder"  # Replace with your desired train folder path
test_folder = "path/to/test/folder"    # Replace with your desired test folder path

# Create folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 2. Load images and assign labels
def load_images_and_labels(dataset_path, limit=500):
    images, filenames, labels = [], [], []
    image_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))
            if len(image_files) >= limit:
                break
        if len(image_files) >= limit:
            break

    print(f"Found {len(image_files)} images. Loading...")

    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(file_path))
            label = 1 if "melanoma" in file_path.lower() else 0
            labels.append(label)

    if len(images) == 0:
        raise ValueError("No images found! Check dataset path.")

    return images, filenames, np.array(labels)

# 3. Preprocess images
def preprocess_images(images):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    return np.array(processed_images)

# 4. Split and save dataset
def split_and_save_dataset(X, y, filenames, test_size=0.2):
    X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(
        X, y, filenames, test_size=test_size, random_state=42
    )

    for img, name in zip(X_train, train_filenames):
        cv2.imwrite(os.path.join(train_folder, name), (img * 255).astype(np.uint8))
    for img, name in zip(X_test, test_filenames):
        cv2.imwrite(os.path.join(test_folder, name), (img * 255).astype(np.uint8))

    return X_train, X_test, y_train, y_test

# 5. Load and preprocess
images, filenames, labels = load_images_and_labels(dataset_path, limit=500)
processed_images = preprocess_images(images)
X_train, X_test, y_train, y_test = split_and_save_dataset(processed_images, labels, filenames)

# 6. Model: MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# 8. Save the trained model
model.save("skin_cancer_model.h5")
print("✅ Model saved successfully!")

# 9. Streamlit App Interface
def streamlit_interface():
    st.title("🧪 Skin Cancer Detection (Upload Image)")

    uploaded_image = st.file_uploader("Upload a skin image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_input)
        result = "melanoma" if prediction[0][0] > 0.5 else "Non-melanoma"

        st.subheader(f"Prediction: {result}")

# 10. Launch Streamlit app (this will work when running locally)
if __name__ == "__main__":
    streamlit_interface()

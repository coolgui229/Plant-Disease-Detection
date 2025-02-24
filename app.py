import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import pandas as pd


# Image preprocessing
def preprocess_image(image):
    img_size = (224, 224)  # Should match your training image size
    image = image.resize(img_size)
    image = np.array(image)
    if image.shape[-1] == 4:  # If image has an alpha channel
        image = image[..., :3]
    image = image / 255.0  # Normalize as per training
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Create the main Tkinter window
root = tk.Tk()
root.title("🌿 Plant Disease Classification")
root.geometry("500x500")

# Load your model and class dictionary
import keras

model_layer = keras.layers.TFSMLayer("/Users/ananyasakhalkar/Downloads/plant_ml/save_path", call_endpoint="serving_default")


def load_class_dict():
    class_dict_path = '/Users/ananyasakhalkar/Downloads/plant_ml/save_path'  # Update with your actual CSV path
    class_df = pd.read_csv(class_dict_path)
    class_mapping = {i: label for i, label in zip(class_df['class_index'], class_df['class'])}
    return class_mapping

# Load the model and class mapping
model = load_model()
class_mapping = load_class_dict()

# Function to open the file dialog and load an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Resize for display
            img = ImageTk.PhotoImage(image)
            panel.config(image=img)
            panel.image = img  # Keep reference to avoid garbage collection

            # Preprocess and predict
            processed_image = preprocess_image(image)
            preds = model.predict(processed_image)
            predicted_class = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_class] * 100

            # Display prediction result
            result_label.config(text=f"Prediction: {class_mapping[predicted_class]}")
            confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")

# Create the UI components
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
result_label.pack(pady=10)

confidence_label = tk.Label(root, text="Confidence: None", font=("Arial", 12))
confidence_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

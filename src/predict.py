# src/predict.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from precautions import precautions_dict

model = tf.keras.models.load_model('../models/model.h5')

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = list(precautions_dict.keys())[class_idx]

    print(f"🌿 Detected Disease: {class_label}")
    print(f"🩺 Precautions: {precautions_dict[class_label]}")

if __name__ == "__main__":
    predict_disease('PlantVillage/Potato___Early_blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG')


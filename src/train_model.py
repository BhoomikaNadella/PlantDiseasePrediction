# src/train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 🌿 Define dataset path (your PlantVillage folder should be inside project folder)
base_dir = os.path.join(os.getcwd(), 'PlantVillage')

# 🧪 Data generators (split 80/20 for training & validation)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 Build simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# ⚙️ Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# 💾 Create models folder safely (inside current project)
save_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(save_dir, exist_ok=True)

# 🧠 Save model
model_save_path = os.path.join(save_dir, 'model.h5')
model.save(model_save_path)

print(f"\n✅ Model training complete! Saved to: {model_save_path}")

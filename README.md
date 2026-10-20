# 🌿 Plant Disease Detection

A deep learning-based web application built using **Python, TensorFlow/Keras, and Streamlit** to identify plant diseases from leaf images. Users can upload an image or capture one using their webcam to receive a predicted disease, confidence score, and suggested precautions.

## ✨ Features

- 📸 Upload leaf images in JPG, JPEG, and PNG formats.
- 📷 Capture leaf images using a webcam.
- 🤖 Predict plant diseases using a trained deep learning model.
- 📊 Display the predicted disease and confidence score.
- 🌱 Provide disease-specific precautions and recommendations.
- 🖥️ Simple and interactive Streamlit user interface.

## 🛠️ Technologies Used

- **Programming Language:** Python 3.10
- **Deep Learning:** TensorFlow and Keras
- **Web Framework:** Streamlit
- **Image Processing:** OpenCV
- **Dataset:** PlantVillage

## 📂 Project Structure

```text
PlantDiseasePrediction/
├── app.py
├── requirements.txt
├── models/
│   └── model.h5
├── src/
│   ├── precautions.py
│   ├── predict.py
│   ├── train_model.py
│   └── utils.py
└── PlantVillage/
```

*Note: The structure above assumes these files and folders exist in your repository. Update it if your actual project structure differs.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/BhoomikaNadella/PlantDiseasePrediction.git
cd PlantDiseasePrediction
```

### 2. Create a Virtual Environment

Python 3.10 is recommended.

```powershell
py -3.10 -m venv venv
```

### 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install opencv-python
```

### 5. Add the Trained Model

Ensure the trained model is available at the path expected by your application:

```text
models/model.h5
```

If the model is not included in the repository, place your trained model in the correct location before running the application.

### 6. Run the Application

```powershell
streamlit run app.py
```

Open the local URL displayed in your terminal. The default address is:

http://localhost:8501

## 🚀 How to Use

1. Launch the application.
2. Upload a leaf image or capture one using the webcam.
3. Click the appropriate prediction button.
4. View the predicted plant disease and confidence score.
5. Read the suggested precautions for the predicted condition.

## 🧠 How It Works

1. **Image Input:** The user uploads or captures an image of a plant leaf.
2. **Preprocessing:** The image is prepared to match the input requirements of the trained model.
3. **Prediction:** The trained deep learning model analyzes the image and predicts a disease class.
4. **Confidence Score:** The application displays the model's prediction confidence.
5. **Precautions:** The application provides relevant precautionary suggestions based on the predicted class.

## 📊 Dataset

The project uses the **PlantVillage dataset**, which contains labeled images of healthy and diseased plant leaves. The dataset can be used to train and evaluate plant disease classification models.

The dataset is required for training if your training script uses it, but it is not necessary for running predictions when the trained model and other required application files are available.

## 🔮 Future Enhancements

- Improve model performance on real-world leaf images.
- Add support for more plant species and diseases.
- Provide multilingual disease descriptions and precautions.
- Deploy the application online for wider accessibility.
- Integrate additional agricultural guidance and recommendations.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions may be inaccurate, particularly for images captured under different lighting conditions or outside the training dataset. The results should not replace professional agricultural diagnosis or advice.

## 👩‍💻 Author

**Bhoomika Nadella**

GitHub: [BhoomikaNadella](https://github.com/BhoomikaNadella)

---

⭐ If you find this project interesting, consider giving the repository a star!

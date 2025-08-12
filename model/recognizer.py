import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("CUDA disabled manually")

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import requests

MODEL_PATH = "model/recogbio_classification_modnet50.keras"
MODEL_URL = os.getenv("MODEL_URL")

# Tự động tải nếu model chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print(f"⬇️ Downloading model from {MODEL_URL}")
    os.makedirs("model", exist_ok=True)
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")
else:
    print("✅ Model already exists.")

# Sau đó load model như bình thường
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    'ape', 'beans', 'bee', 'bird', 'buffalo', 'butterfly', 'cabbage', 'cauliflower', 'cat',
    'chicken', 'chipmunk', 'coffee', 'cow', 'cucumber', 'deer', 'dog', 'dragon_fruits', 'elephant',
    'fish', 'fox', 'ginger', 'goat', 'horse', 'jackfruit', 'jelly_fish', 'lion', 'litchi', 'longan',
    'lotus', 'maize', 'mouse', 'orchid', 'panther', 'papaya', 'peacock', 'pig', 'potato', 'rabbit',
    'rice_plant', 'seal', 'snake', 'spider', 'tiger', 'tomato', 'turtle', 'zebra']

def recognize_image(img_path):
    """
    Nhận diện ảnh và trả về tên lớp cùng với điểm tin cậy.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Thực hiện dự đoán
    preds = model.predict(x)
    
    # Lấy chỉ số và điểm tin cậy của lớp có xác suất cao nhất
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    
    # Trả về cả tên lớp và điểm tin cậy
    return CLASS_NAMES[class_idx], float(confidence)

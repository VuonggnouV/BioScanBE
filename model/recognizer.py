import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_PATH = "model/recogbio_classification_modnet50.keras"
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded")
else:
    print("⚠️ Model not found, skipping load")

CLASS_NAMES = [
    'ape', 'bat', 'bee', 'bird', 'buffalo', 'butterfly', 'carp', 'cat',
    'chicken', 'chipmunk', 'cow', 'dog', 'dove', 'duck', 'eagle',
    'elephant', 'fish', 'frog', 'horse', 'jelly_fish', 'lion', 'lobster',
    'mouse', 'panther', 'peacock', 'pig', 'rabbit', 'seal', 'snake',
    'spider', 'starfish', 'tiger', 'turtle', 'zebra'
]

# Tải model một lần khi ứng dụng khởi động
model = load_model(MODEL_PATH, compile=False)

def recognize_image(img_path):
    """
    Nhận diện ảnh và trả về tên lớp cùng với điểm tin cậy.
    """
    img = image.load_img(img_path, target_size=(256, 256))
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

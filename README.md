
# RecogBio Backend (Flask + Firebase + Gemini)

## 📁 Cấu trúc thư mục

```
recogbio_backend/
├── app.py                   # Flask App với API chính
├── model/
│   ├── recognizer.py        # Nhận diện ảnh bằng ResNet50
│   └── gemini.py            # Gọi API Gemini sinh mô tả
├── static/
│   ├── uploads/             # Ảnh gốc tải lên từ người dùng
│   └── outputs/             # File .txt chứa mô tả
├── requirements.txt
└── serviceAccountKey.json   # Firebase Admin SDK
```

## 🚀 Cách chạy

```bash
pip install -r requirements.txt
python app.py
```

## 🌐 API

### POST /predict
- Gửi: image, user_id, role (user | manager | guest | admin)
- Nhận: scan_id, class, textPath

### GET /history/<user_id>?role=user|guest|manager|admin
- Nhận: danh sách các lần quét + mô tả .txt từ Gemini

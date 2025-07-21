from flask import Flask, request, jsonify, send_from_directory
import os
from model.recognizer import recognize_image
from model.gemini import generate_description
import firebase_admin
from firebase_admin import credentials, firestore
import json

app = Flask(__name__)

# --- Khởi tạo Firebase ---
cred_json = os.getenv("FIREBASE_KEY_JSON")
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Biến cấu hình ---
CONFIDENCE_THRESHOLD = 0.7
BASE_URL = "https://bioscanbe-production.up.railway.app"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'user_id' not in request.form or 'role' not in request.form or 'scan_id' not in request.form:
        return jsonify({"error": "Missing required fields"}), 400

    image = request.files['image']
    user_id = request.form['user_id']
    role = request.form['role']
    scan_id = request.form['scan_id']

    # --- Lưu ảnh và text ---
    img_path_on_server = f"static/uploads/{scan_id}.jpg"
    txt_path_on_server = f"static/outputs/{scan_id}.txt"
    image.save(img_path_on_server)

    # --- Tạo URL công khai ---
    image_url = f"{BASE_URL}/uploads/{scan_id}.jpg"         # Đã đổi route
    text_url = f"{BASE_URL}/outputs/{scan_id}.txt"

    # --- Nhận dạng & mô tả ---
    predicted_class, confidence = recognize_image(img_path_on_server)
    final_class_name = predicted_class
    description = ""

    if confidence > CONFIDENCE_THRESHOLD:
        description = generate_description(final_class_name, img_path_on_server)
    else:
        final_class_name = "Không xác định"
        description = "Sinh vật này không nằm trong chương trình Sinh học Trung học Phổ thông, nên hệ thống không cung cấp thông tin chi tiết."

    with open(txt_path_on_server, "w", encoding="utf-8") as f:
        f.write(description)

    # --- Ghi kết quả vào Firestore ---
    collection_name = "archived_guests" if role == "guest" else "users"
    history_ref = db.collection(collection_name).document(user_id).collection("scanHistory").document(scan_id)

    history_ref.update({
        "infoFileUri": text_url,
        "imagePaths": [image_url],
        "class": final_class_name,
        "processingStatus": "completed"
    })

    return jsonify({"status": "success", "message": f"Processed scan_id {scan_id}"})


# --- ROUTE phục vụ ảnh ---
@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    try:
        return send_from_directory('static/uploads', filename)
    except FileNotFoundError:
        return "File not found", 404

# --- ROUTE phục vụ TXT ---
@app.route('/outputs/<filename>')
def serve_output_file(filename):
    try:
        return send_from_directory('static/outputs', filename, as_attachment=False, mimetype='text/plain; charset=utf-8')
    except FileNotFoundError:
        return "File not found", 404

# --- Trang chủ ---
@app.route("/")
def index():
    return "BioScan Backend is running!"

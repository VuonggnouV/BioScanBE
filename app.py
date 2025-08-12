# app.py
from flask import Flask, request, jsonify, send_from_directory
import os, json, mimetypes

# Model inference & LLM
from model.recognizer import recognize_image
from model.gemini import generate_description  # hàm này CHỈ gửi prompt cho Gemini

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Supabase Storage (lưu ảnh/txt bền vững)
from supabase import create_client

app = Flask(__name__)

# ---------------- Firebase ----------------
# FIREBASE_KEY_JSON: biến môi trường chứa toàn bộ JSON service account
cred_json = os.getenv("FIREBASE_KEY_JSON")
cred_dict = json.loads(cred_json)
firebase_admin.initialize_app(credentials.Certificate(cred_dict))
db = firestore.client()

# ---------------- Supabase Storage (lazy init) ----------------
_SB = None
SB_BUCKET = os.getenv("SB_BUCKET", "bioscan").strip()  # nhớ tạo bucket này và bật Public trên Supabase

def get_sb():
    """Khởi tạo Supabase client khi cần (tránh crash lúc import)."""
    global _SB
    if _SB is None:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        print("[DEBUG] SUPABASE_URL set?", bool(url), "| SUPABASE_KEY set?", bool(key), "| BUCKET:", SB_BUCKET, flush=True)
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        _SB = create_client(url, key)
    return _SB

def sb_upload(local_path: str, remote_path: str) -> str:
    """Upload file lên Supabase Storage (bucket public) và trả public URL."""
    with open(local_path, "rb") as f:
        # KHÔNG truyền options/headers để tránh lỗi bool header
        get_sb().storage.from_(SB_BUCKET).upload(remote_path, f)
    return get_sb().storage.from_(SB_BUCKET).get_public_url(remote_path)

# ---------------- App config ----------------
CONFIDENCE_THRESHOLD = 0.7
BASE_URL="https://bioscanbe-production.up.railway.app"

@app.route("/predict", methods=["POST"])
def predict():
    # Validate payload
    required = ('image', 'user_id', 'role', 'scan_id')
    if not all(k in (request.files if k == 'image' else request.form) for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    image_file = request.files['image']
    user_id = request.form['user_id']
    role = request.form['role']
    scan_id = request.form['scan_id']

    # Lưu tạm local để xử lý (như cũ)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/outputs", exist_ok=True)
    img_path_local = f"static/uploads/{scan_id}.jpg"
    txt_path_local = f"static/outputs/{scan_id}.txt"
    image_file.save(img_path_local)

    # Nhận diện
    predicted_class, confidence = recognize_image(img_path_local)
    final_class_name = predicted_class
    if confidence > CONFIDENCE_THRESHOLD:
        # generate_description hiện CHỈ gửi prompt (không gửi ảnh)
        description = generate_description(final_class_name, img_path_local)
    else:
        final_class_name = "Không xác định"
        description = (
            "Sinh vật này không nằm trong chương trình Sinh học Trung học Phổ thông, "
            "nên hệ thống không cung cấp thông tin chi tiết."
        )

    # Ghi mô tả ra file txt
    with open(txt_path_local, "w", encoding="utf-8") as f:
        f.write(description)

    # Upload ảnh & txt lên Supabase Storage -> nhận public URL bền
    img_url = sb_upload(img_path_local, f"uploads/{scan_id}.jpg")
    txt_url = sb_upload(txt_path_local, f"outputs/{scan_id}.txt")

    # Cập nhật Firestore history (giữ nguyên schema)
    collection_name = "archived_guests" if role == "guest" else "users"
    history_ref = (
        db.collection(collection_name)
          .document(user_id)
          .collection("scanHistory")
          .document(scan_id)
    )
    history_ref.update({
        "infoFileUri": txt_url,
        "imagePaths": [img_url],
        "class": final_class_name,
        "processingStatus": "completed"
    })

    return jsonify({"status": "success", "message": f"Processed scan_id {scan_id}"})

# --------- Các route cũ (không còn bắt buộc nhưng giữ cho tiện debug) ---------
@app.route('/static/uploads/<filename>')
def serve_uploaded_image(filename):
    try:
        return send_from_directory(os.path.join('static', 'uploads'), filename)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/outputs/<filename>')
def serve_output_file(filename):
    try:
        return send_from_directory('static/outputs', filename, as_attachment=False, mimetype='text/plain; charset=utf-8')
    except FileNotFoundError:
        return "File not found", 404

@app.route("/")
def index():
    return "BioScan Backend is running!"

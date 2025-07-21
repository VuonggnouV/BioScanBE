from flask import Flask, request, jsonify, send_from_directory
import os
from model.recognizer import recognize_image
from model.gemini import generate_description
import firebase_admin
from firebase_admin import credentials, firestore
import json

app = Flask(__name__)
cred_json = os.getenv("FIREBASE_KEY_JSON")
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

CONFIDENCE_THRESHOLD = 0.7

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'user_id' not in request.form or 'role' not in request.form or 'scan_id' not in request.form:
        return jsonify({"error": "Missing required fields"}), 400
        
    image = request.files['image']
    user_id = request.form['user_id']
    role = request.form['role']
    scan_id = request.form['scan_id']

    img_path_on_server = f"static/uploads/{scan_id}.jpg"
    txt_path_on_server = f"static/outputs/{scan_id}.txt"
    image.save(img_path_on_server)

    # Xử lý ảnh và gọi Gemini như cũ
    predicted_class, confidence = recognize_image(img_path_on_server)
    final_class_name = predicted_class
    description = ""

    if confidence > CONFIDENCE_THRESHOLD:
        description = generate_description(final_class_name)
    else:
        final_class_name = "Không xác định"
        description = "Sinh vật này không nằm trong chương trình Sinh học Trung học Phổ thông, nên hệ thống không cung cấp thông tin chi tiết."

    with open(txt_path_on_server, "w", encoding="utf-8") as f:
        f.write(description)

    collection_name = "archived_guests" if role == "guest" else "users"
    history_ref = db.collection(collection_name).document(user_id).collection("scanHistory").document(scan_id)
    
    # --- SỬA LỖI QUAN TRỌNG ---
    # Chỉ cập nhật các trường do backend tạo ra.
    # KHÔNG cập nhật lại trường 'imagePaths'.
    history_ref.update({
        "infoFileUri": txt_path_on_server,
        "class": final_class_name,
        "processingStatus": "completed"
    })
    # --- KẾT THÚC SỬA LỖI ---
    
    return jsonify({"status": "success", "message": f"Processed scan_id {scan_id}"})

# Các API phục vụ file không thay đổi
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import google.generativeai as genai

# Cấu hình API key của bạn
genai.configure(api_key="AIzaSyAMcJMdNsm9yBw2b7gaZ0b2Xz9qj1sqtAg")

# Khởi tạo model Gemini
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_description(label):
    prompt = f"""Đối tượng được nhận diện trong ảnh là: {label}.

Hãy cung cấp một đoạn mô tả hoàn chỉnh, rõ ràng và đầy đủ về đối tượng {label}, bao gồm:

Trong ảnh là {label}
1. Tên thông thường và tên khoa học (nếu có).
2. Phân loại sinh học.
3. Đặc điểm nhận diện dễ thấy.
4. Môi trường sống và khu vực phân bố phổ biến.
5. Vai trò sinh thái hoặc ý nghĩa đối với con người (như làm thực phẩm, thuốc, cảnh quan...).
6. Những lưu ý khi tiếp xúc hoặc sử dụng (nếu có).
7. Một số thông tin thú vị hoặc độc đáo liên quan đến loài này.

Trình bày dưới dạng văn bản hoàn chỉnh, hướng đến người dùng là học sinh trung học phổ thông, không cần định nghĩa lại yêu cầu. 

    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Lỗi khi gọi API Gemini: {e}")
        return f"Không thể tạo mô tả cho '{label}' vào lúc này."

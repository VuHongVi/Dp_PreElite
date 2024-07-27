import pandas as pd
import re
from collections import Counter
import openpyxl
from sklearn.metrics import accuracy_score

# Danh sách stop words
stop_words = set([
    "và", "có", "là", "của", "cho", "nên", "thể", "về", "ở", "với", "được",
    "trong", "rằng", "một", "này", "như", "nhưng", "đã", "không", "các", "để",
    "khi", "thì", "ra", "từ", "bởi", "vào", "lại", "những", "nhiều", "vẫn",
    "hơn", "cũng", "rất", "tôi", "anh", "em", "ông", "bà", "thầy", "cô",
    "học", "viên", "giảng", "dạy", "sinh", "ạ", "kiến", "thức", "hiểu"
])

# Từ điển chỉnh sửa chính tả
correct_mapping = {
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ",
    "h": "giờ",
    "chễ": "trễ"
}

# Hàm để chỉnh sửa chính tả
def correct_spelling(word):
    return correct_mapping.get(word, word)

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Loại bỏ dấu câu và chuyển về chữ thường
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    # Chỉnh sửa chính tả
    words = text.split()
    words = [correct_spelling(word) for word in words]
    # Loại bỏ stop words
    text = ' '.join([word for word in words if word not in stop_words])
    return text

# Đọc dữ liệu từ file Excel
file_path = r'C:\Users\Dell\Desktop\Dp_PreElite\Dataset.xlsx'
df = pd.read_excel(file_path)

# Tách phản hồi theo nhãn chú ý và không chú ý
attention_feedbacks = df[df['Chú ý'] == 1]['Nội dung comment']
non_attention_feedbacks = df[df['Chú ý'] == 0]['Nội dung comment']

# Tiền xử lý và đếm từ
attention_words = Counter()
non_attention_words = Counter()

for feedback in attention_feedbacks:
    processed_text = preprocess_text(feedback)
    attention_words.update(processed_text.split())

for feedback in non_attention_feedbacks:
    processed_text = preprocess_text(feedback)
    non_attention_words.update(processed_text.split())

# Lấy top 20 từ phổ biến nhất trong mỗi loại
top_attention_words = attention_words.most_common(20)
top_non_attention_words = non_attention_words.most_common(20)

# Bộ từ khóa phổ biến trong dataset
attention_keywords = [word for word, freq in top_attention_words]
non_attention_keywords = [word for word, freq in top_non_attention_words]

# Danh sách từ khóa tự thêm vào
custom_attention_keywords = ["tệ", "xấu", "thất vọng", "không tốt", "quá dở", "quá tệ", "không hài lòng", "ngu", "đần"]
custom_non_attention_keywords = ["tuyệt vời", "xuất sắc", "hoàn hảo", "tận tâm"]

# Kết hợp từ khóa từ dataset và từ khóa tự thêm vào
attention_keywords.extend(custom_attention_keywords)
non_attention_keywords.extend(custom_non_attention_keywords)

# Hàm kiểm tra từ khóa
def contains_keywords(text, keywords):
    for keyword in keywords:
        if keyword in text.lower():
            return True
    return False

# Hàm phân loại phản hồi
def classify_feedback(feedback):
    if contains_keywords(feedback, attention_keywords):
        return 1
    elif contains_keywords(feedback, non_attention_keywords):
        return 0
    else:
        return 0

# Đọc dữ liệu từ file input.xlsx
input_file_path = r'C:\Users\Dell\Desktop\Dp_PreElite\input.xlsx'
df_input = pd.read_excel(input_file_path)

# Tạo cột nhãn mới
df_input['label'] = df_input['Nội dung comment'].apply(classify_feedback)

# Xuất dữ liệu ra tệp output.csv
output_file = r'C:\Users\Dell\Desktop\Dp_PreElite\output.csv'
df_input.to_csv(output_file, index=False)

print("Phân loại hoàn tất và lưu vào tệp:", output_file)

# Áp dụng hàm phân loại lên dataset để dự đoán nhãn
df['predicted_label'] = df['Nội dung comment'].apply(classify_feedback)

# Tính toán độ chính xác
accuracy = accuracy_score(df['Chú ý'], df['predicted_label'])

print(f"Độ chính xác của mô hình: {accuracy:.2f}")

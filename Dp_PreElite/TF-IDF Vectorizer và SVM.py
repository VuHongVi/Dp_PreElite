import pandas as pd
import re
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Đọc dữ liệu từ file Excel
file_path = r'C:\Users\Dell\Desktop\Dp_PreElite\Dataset.xlsx'
df = pd.read_excel(file_path)

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
    "ship": "vận chuyển",
    "shop": "cửa hàng",
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
    "bgjo": "bao giờ"
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

# Áp dụng tiền xử lý cho dữ liệu
df['processed_comment'] = df['Nội dung comment'].apply(preprocess_text)

# Tạo các đặc trưng và huấn luyện mô hình sử dụng pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC(kernel='linear', probability=True))
])

# Nhãn
y = df['Chú ý']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df['processed_comment'], y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Dự đoán
y_pred = pipeline.predict(X_test)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình SVM với TF-IDF:", accuracy)

# Đọc dữ liệu từ file input.xlsx
input_file_path = r'C:\Users\Dell\Desktop\Dp_PreElite\input.xlsx'
df_input = pd.read_excel(input_file_path)

# Tiền xử lý văn bản cho dữ liệu input
df_input['processed_comment'] = df_input['Nội dung comment'].apply(preprocess_text)

# Dự đoán cho dữ liệu input
df_input['label'] = pipeline.predict(df_input['processed_comment'])

# Xuất dữ liệu ra tệp output.csv
output_file = r'C:\Users\Dell\Desktop\Dp_PreElite\output.csv'
df_input.to_csv(output_file, index=False)

print("Phân loại hoàn tất và lưu vào tệp:", output_file)

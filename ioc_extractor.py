import re
import logging
from pymongo import MongoClient

# Cấu hình logging để xem tiến trình chạy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IOCExtractor:
    def __init__(self):
        self.regex_patterns = {
            "ipv4": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            "hashes": r'\b([a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b',
            "cve": r'(?i)\bCVE-\d{4}-\d{4,7}\b',
            "domains": r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
        }

    def clean_text(self, text):
        """
        Làm sạch dữ liệu thô: Xóa khoảng trắng thừa, ký tự ẩn, HTML tags còn sót lại.
        Giúp thuật toán Chunking sau này cắt chữ chuẩn xác hơn.
        """
        if not text:
            return ""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.replace('\u200b', '').replace('\xa0', '')
        return text.strip()

    def extract_all_iocs(self, text):
        """
        Quét qua đoạn văn bản và nhặt ra toàn bộ IOCs.
        Trả về một Dictionary chứa các danh sách không trùng lặp.
        """
        extracted_iocs = {
            "ipv4": [],
            "hashes": [],
            "cve": [],
            "domains": []
        }
        
        if not text:
            return extracted_iocs

        for key, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, text)
            unique_matches = list(set([m for m in matches if m]))
            extracted_iocs[key] = unique_matches
            
        return extracted_iocs


def clean_mongo_collections(mongo_uri, db_name="threat_intel_db"):
    # 1. Kết nối DB và khởi tạo Extractor
    client = MongoClient(mongo_uri)
    db = client[db_name]
    extractor = IOCExtractor()

    # ==========================================
    # 2. DỌN DẸP BẢNG 'reports'
    # ==========================================
    reports_col = db["reports"]
    # Tìm tất cả các bài có trường "content"
    reports_cursor = reports_col.find({"content": {"$exists": True}})
    
    # Đếm số lượng document để log (sử dụng filter chính xác)
    total_reports = reports_col.count_documents({"content": {"$exists": True}})
    logging.info(f"Bat dau don dep bang 'reports' ({total_reports} documents)...")
    
    report_count = 0
    for doc in reports_cursor:
        raw_text = doc.get("content", "")
        if raw_text:
            # Clean text và trích xuất IOC
            cleaned_text = extractor.clean_text(raw_text)
            iocs = extractor.extract_all_iocs(cleaned_text)
            
            # Cập nhật lại vào MongoDB (Thêm trường mới, không xóa trường cũ)
            reports_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "cleaned_content": cleaned_text, 
                    "extracted_iocs": iocs
                }}
            )
            report_count += 1
            
    logging.info(f"Da don dep va trich xuat IOC cho {report_count} bai reports.")

    # ==========================================
    # 3. DỌN DẸP BẢNG 'mitre_attack'
    # ==========================================
    mitre_col = db["mitre_attack"]
    # Tìm tất cả các record có trường "description"
    mitre_cursor = mitre_col.find({"description": {"$exists": True}})
    
    total_mitre = mitre_col.count_documents({"description": {"$exists": True}})
    logging.info(f"Bat dau don dep bang 'mitre_attack' ({total_mitre} documents)...")
    
    mitre_count = 0
    for doc in mitre_cursor:
        raw_text = doc.get("description", "")
        if raw_text:
            # Dùng clean_text để xóa HTML tag hoặc khoảng trắng thừa trong description
            cleaned_text = extractor.clean_text(raw_text)
            iocs = extractor.extract_all_iocs(cleaned_text)
            
            mitre_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "cleaned_description": cleaned_text,
                    "extracted_iocs": iocs
                }}
            )
            mitre_count += 1

    logging.info(f"Da don dep xong {mitre_count} records trong mitre_attack.")


if __name__ == "__main__":
    # Thay bằng URI của bạn
    my_cloud_uri = "mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/"
    
    logging.info("KHOI DONG QUA TRINH DON DEP DATABASE...")
    clean_mongo_collections(mongo_uri=my_cloud_uri)
    logging.info("HOAN TAT QUÁ TRINH DON DEP!")
import pandas as pd
from pymongo import MongoClient
import certifi # Thư viện này giúp xác thực chứng chỉ tự động

# 1. Cấu hình kết nối
connection_string = 

try:
    # Sử dụng certifi.where() để xác thực kết nối an toàn (thay thế cho tlsCAFile=ca)
    client = MongoClient(connection_string, tlsCAFile=certifi.where())
    db = client['threat_intel_db']
    print("Kết nối thành công tới MongoDB Atlas!")
except Exception as e:
    print(f"Lỗi kết nối: {e}")
    exit()

# 2. Danh sách các collection cần xuất
collections = ['mitre_attack_data', 'processed_chunks', 'reports']

# 3. Vòng lặp xuất dữ liệu
for col_name in collections:
    print(f"--- Đang bắt đầu xuất: {col_name} ---")
    
    try:
        # Lấy toàn bộ dữ liệu
        cursor = list(db[col_name].find({}))
        
        if not cursor:
            print(f"Thông báo: Collection '{col_name}' trống, bỏ qua.")
            continue
            
        df = pd.DataFrame(cursor)
        
        # Xử lý cột '_id' (chuyển sang string để không bị lỗi định dạng)
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Lưu file
        df.to_csv(f'{col_name}.csv', index=False, encoding='utf-8')
        df.to_json(f'{col_name}.json', orient='records', indent=4)
        
        print(f"-> Đã xuất thành công: {col_name}.csv và {col_name}.json")
        
    except Exception as e:
        print(f"Lỗi khi xuất collection '{col_name}': {e}")

print("\nHoàn thành toàn bộ quy trình!")

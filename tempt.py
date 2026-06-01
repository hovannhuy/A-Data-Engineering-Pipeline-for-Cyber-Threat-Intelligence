import os
import json
from pymongo import MongoClient
from bson.json_util import dumps
import certifi
ca = certifi.where()
# 1. Cấu hình kết nối
# Lưu ý: Đảm bảo biến 'ca' đã được định nghĩa đúng như trong mã của bạn
uri = 'mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/?appName=Cluster0'
client = MongoClient(uri, tlsCAFile=ca)
db = client['threat_intel_db']

# 2. Tạo thư mục để lưu các file xuất ra
output_dir = "exported_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Lấy danh sách tất cả các collection và lưu chúng
collections = db.list_collection_names()

print(f"Bắt đầu xuất dữ liệu từ database: {db.name}")

for coll_name in collections:
    print(f"Đang xuất collection: {coll_name}...")
    
    collection = db[coll_name]
    
    # Truy vấn tất cả document trong collection
    cursor = collection.find({})
    
    # Chuyển đổi cursor thành danh sách và sau đó thành JSON
    # Sử dụng dumps của bson.json_util để xử lý các kiểu dữ liệu đặc biệt của Mongo (như ObjectId)
    data = list(cursor)
    json_data = dumps(data, indent=4)
    
    # Lưu vào file
    file_path = os.path.join(output_dir, f"{coll_name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
        
    print(f"Đã lưu {len(data)} document vào {file_path}")

print("Hoàn tất quá trình xuất dữ liệu.")
import requests
from pymongo import MongoClient

#Connect to MongoDB
client = MongoClient('mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/')
db = client['threat_intel_db']
collection = db['mitre_attack_data']

def fetch_mitre_attack_data():
  print('Đang tải dữ liệu MITRE ATT&CK...')
  url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
  response = requests.get(url)

  if response.status_code == 200:
    data = response.json()
    techniques = []

    for obj in data.get('objects', []):
      if obj.get('type') == 'attack-pattern':
        techniques_data = {
            "source": "MITRE ATT&CK",
            "name": obj.get('name'),
            "description": obj.get('description',""),
            "mitre_id": obj["external_references"][0]["external_id"] if "external_references" in obj else None       
           
        }
        techniques.append(techniques_data)
    if techniques:
        collection.insert_many(techniques)
        print(f"Đã lưu {len(techniques)} kỹ thuật MITRE vào MonoDB")
    else:
      print('Lỗi khi tải dữ liệu MITRE')
fetch_mitre_attack_data()
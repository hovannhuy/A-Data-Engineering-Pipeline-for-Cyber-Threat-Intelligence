import schedule
import time
from craw_data_MITRE import fetch_mitre_attack_data
from craw_data_Mandant_Kaspersky import crawl_threat_intel_blogs

def pipeline_job():
    print("--- Bắt đầu Pipeline tự động lấy dữ liệu TI ---")
    fetch_mitre_attack_data()
    crawl_threat_intel_blogs()
    print("--- Hoàn tất Pipeline ---")

pipeline_job()

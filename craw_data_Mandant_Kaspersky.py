import feedparser
import requests
from bs4 import BeautifulSoup
import os
from langchain_community.document_loaders import PyPDFLoader
from pymongo import MongoClient
import certifi
ca = certifi.where()
#Connect to MongoDB
client = MongoClient('mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/?appName=Cluster0', tlsCAFile=ca)
db = client['threat_intel_db']
db_reports = db['reports_data']
TI_SOURCES = {
    "Kaspersky": "https://securelist.com/feed/",
    "Mandiant": "https://www.mandiant.com/resources/blog/rss.xml", 
    "The Hacker News": "https://feeds.feedburner.com/TheHackersNews"
}
def scrape_arcticle_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers= headers, timeout = 10)
        soup = BeautifulSoup(res.content, 'html.parser')

        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])

        pdf_links = []
        for link in soup.find_all('a', href = True):
            if link['href'].endswith('.pdf'):
                pdf_links.append(link['href'])

        return text, pdf_links
    except Exception as e:
        print(f'Lỗi khi cào trang {url}: {e}')
        return '', []
    
def process_pdf_from_url(pdf_url, source_name):
    try:
        print(f'Đang tải PDF: {pdf_url}')
        temp_pdf_path = 'temp_report.pdf'
        response = requests.get(pdf_url, stream = True)
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)
    
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        full_text = "\n".join ([doc.page_content for doc in docs ])

        os.remove(temp_pdf_path)
        return full_text
    except Exception as e:
        print(f'Lỗi xử lý {pdf_url}: {e}')
        return ''
    
def crawl_threat_intel_blogs():
    for source_name, rss_url in TI_SOURCES.items():
        print(f'Đang quét nguồn: {source_name}')
        feed = feedparser.parse(rss_url)

        for entry in feed.entries[:5]:
            title = entry.title
            link = entry.link
            print(f'Xử lý bài: {title}')

            article_text, pdf_links = scrape_arcticle_text(link)
        
            if article_text.strip():
                db_reports.insert_one({
                        "title": title,
                        "url": link,
                        "content": article_text,
                        "type": "web_article",
                        "source": source_name
                    })
            for pdf_url in pdf_links:
                if pdf_url.startswith('/'):
                    from urllib.parse import urlparse
                    parsed_uri = urlparse(link)
                    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
                    pdf_url = base_url + pdf_url
                    
                pdf_text = process_pdf_from_url(pdf_url, source_name)
                
                if pdf_text.strip():
                    db_reports.insert_one({
                        "title": f"PDF Report for: {title}",
                        "url": pdf_url,
                        "content": pdf_text,
                        "type": "pdf_report",
                        "source": source_name
                    })
                    print("Đã lưu nội dung PDF vào MongoDB.")

crawl_threat_intel_blogs()    
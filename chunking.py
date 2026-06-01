import logging
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from ioc_extractor import IOCExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TIChunkingPipeline:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="threat_intel_db"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        # Định nghĩa danh sách các collection cần lấy dữ liệu
        self.target_collections = ["reports_data", "mitre_attack_data"]
        self.ioc_extractor = IOCExtractor()

        logging.info("Dang tai BAAI/bge-small-en-v1.5 embedding model...")
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def fetch_raw_documents(self):
        """Lấy dữ liệu từ tất cả các collections được chỉ định"""
        documents = []
        for col_name in self.target_collections:
            logging.info(f"Dang lay du lieu tu collection: {col_name}")
            collection = self.db[col_name]
            cursor = collection.find({})
            
            for doc in cursor:
                # Lay text linh hoat theo ten field
                raw_text = doc.get("content") or doc.get("description") or doc.get("text", "")
                
                if len(str(raw_text)) < 100:
                    continue
                    
                clean_text = self.ioc_extractor.clean_text(str(raw_text))  
                iocs = self.ioc_extractor.extract_all_iocs(clean_text)
                    
                lc_doc = Document(
                    page_content=clean_text, 
                    metadata={
                        "source_id": str(doc.get("_id")),
                        "source_collection": col_name,
                        "title": doc.get("title") or doc.get("name", "Unknown Title"),
                        "ioc_ipv4": ", ".join(iocs["ipv4"]),
                        "ioc_hashes": ", ".join(iocs["hashes"]),
                        "ioc_cve": ", ".join(iocs["cve"]),
                        "ioc_domains": ", ".join(iocs["domains"])
                    }
                )
                documents.append(lc_doc)
            
        logging.info(f"Tong cong da lay duoc {len(documents)} tai lieu tu tat ca cac nguon.")
        return documents

    def apply_chunking_strategies(self, documents):
        """Chay dong thoi 3 chien luoc chunking"""
        strategies = {}

        # 1. Fixed-size
        logging.info("Dang chay Fixed-size Chunking...")
        fixed_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=150)
        fixed_chunks = fixed_splitter.split_documents(documents)
        self._add_strategy_metadata(fixed_chunks, "fixed")
        strategies["fixed"] = fixed_chunks

        # 2. Recursive
        logging.info("Dang chay Recursive Chunking...")
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])
        recursive_chunks = recursive_splitter.split_documents(documents)
        self._add_strategy_metadata(recursive_chunks, "recursive")
        strategies["recursive"] = recursive_chunks

        # 3. Semantic
        logging.info("Dang chay Semantic Chunking...")
        semantic_splitter = SemanticChunker(self.embed_model, breakpoint_threshold_type="percentile")
        semantic_chunks = semantic_splitter.split_documents(documents)
        self._add_strategy_metadata(semantic_chunks, "semantic")
        strategies["semantic"] = semantic_chunks

        return strategies

    def _add_strategy_metadata(self, chunks, strategy_name):
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy_name

    def save_chunks_to_db(self, chunks_dict):
        for strategy, chunks in chunks_dict.items():
            target_col_name = f"processed_chunks_{strategy}"
            target_collection = self.db[target_col_name]
            
            # Xoa du lieu cu de tranh trung lap
            target_collection.delete_many({}) 
            
            all_records = []
            for chunk in chunks:
                all_records.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                })
            
            if all_records:
                target_collection.insert_many(all_records)
                logging.info(f"Da luu {len(all_records)} chunks vao {target_col_name}")

if __name__ == "__main__":
    my_cloud_uri = "mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/"
    pipeline = TIChunkingPipeline(mongo_uri=my_cloud_uri) 
    
    # Lay du lieu tu ca 2 bang
    docs = pipeline.fetch_raw_documents()
    
    if docs:
        # Ap dung 3 chien luoc
        processed_chunks = pipeline.apply_chunking_strategies(docs)
        # Luu 3 bang rieng biet
        pipeline.save_chunks_to_db(processed_chunks)
    else:
        logging.warning("Pipeline dung lai do khong co du lieu.")
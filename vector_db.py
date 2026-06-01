import logging
import time
from typing import List, Optional

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MONGO_URI = (
    "mongodb+srv://yhvn24_db_user:hovannhuy24"
    "@cluster0.4kaifw5.mongodb.net/?appName=Cluster0"
)
DB_NAME            = "threat_intel_db"
EMBEDDING_MODEL    = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM      = 384          # bge-small output dimension
VECTOR_INDEX_NAME  = "vector_index"
STRATEGIES         = ["fixed", "recursive", "semantic"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# HELPER: Atlas Search Index
# ──────────────────────────────────────────────
def ensure_collection_exists(db, col_name: str) -> None:
    """
    MongoDB Atlas chỉ tạo collection khi có document đầu tiên.
    Hàm này tạo collection tường minh để tránh lỗi 'does not exist'
    khi gọi create_search_index trước khi insert data.
    """
    if col_name not in db.list_collection_names():
        db.create_collection(col_name)
        log.info(f"  Đã tạo collection '{col_name}'.")


def create_vector_index_if_missing(collection, index_name: str, dim: int) -> None:
    """
    Tạo Atlas Vector Search index nếu chưa tồn tại.
    PHẢI gọi ensure_collection_exists() trước hàm này.
    """
    try:
        existing = list(collection.list_search_indexes())
        names    = [idx.get("name") for idx in existing]
    except Exception:
        names = []

    if index_name in names:
        log.info(f"  Index '{index_name}' đã tồn tại → bỏ qua.")
        return

    index_def = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": dim,
                        "similarity": "cosine",
                    }
                },
            }
        },
        name=index_name,
        type="vectorSearch",
    )

    try:
        collection.create_search_index(index_def)
        log.info(f"  Đã tạo index '{index_name}'. Đợi Atlas build (~30–60s)…")
        # Polling cho đến khi READY
        for _ in range(24):          # tối đa 2 phút
            time.sleep(5)
            states = [
                idx.get("status")
                for idx in collection.list_search_indexes()
                if idx.get("name") == index_name
            ]
            if states and states[0] == "READY":
                log.info(f"  Index '{index_name}' READY ✓")
                return
        log.warning(f"  Index '{index_name}' chưa READY sau 120s – tiếp tục bất đồng bộ.")
    except Exception as e:
        # M0 free-tier không hỗ trợ programmatic index creation
        # → nhắc người dùng tạo thủ công trên Atlas UI
        log.warning(
            f"  Không thể tạo index tự động ({e}).\n"
            f"  → Tạo thủ công trên Atlas UI:\n"
            f"     Collection : {collection.name}\n"
            f"     Index name : {index_name}\n"
            f"     Field      : embedding  (knnVector, dim={dim}, cosine)"
        )


# ──────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────
class TIVectorManager:
    """
    Đọc chunks từ MongoDB, embed bằng HuggingFace,
    rồi lưu vector store ngược lại MongoDB Atlas.
    """

    def __init__(self, mongo_uri: str = MONGO_URI):
        self.client = MongoClient(mongo_uri)
        self.db     = self.client[DB_NAME]

        log.info(f"Đang tải embedding model: {EMBEDDING_MODEL} …")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Embedding model sẵn sàng.")

    # ── 1. Load chunks ───────────────────────────
    def load_chunks(self, strategy: str) -> List[Document]:
            col_name   = f"processed_chunks_{strategy}"
            collection = self.db[col_name]
            documents  = []

            for row in collection.find({}):          # ← THIẾU DÒNG NÀY
                content = row.get("content", "").strip()

        # ── Lọc bỏ chunks rác ──────────────────────
                if len(content) < 100:               # quá ngắn
                    continue
                if content.count("Learn More") > 3:  # boilerplate nav
                    continue
                if len(set(content.split())) < 10:   # quá ít từ unique
                    continue
        # ────────────────────────────────────────────

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            **row.get("metadata", {}),
                            "strategy": strategy,
                            "source_id": str(row.get("_id", "")),
                        },
                    )
                )

            log.info(f"[{strategy}] Loaded {len(documents)} chunks (sau khi lọc rác).")
            return documents
    # ── 2. Build / rebuild vector store ──────────
    def build_vector_store(self, strategy: str) -> Optional[MongoDBAtlasVectorSearch]:
        """
        Embed chunks và lưu vào collection vector_store_<strategy>.
        Nếu collection đã có data → xóa sạch trước khi build lại.
        """
        documents = self.load_chunks(strategy)
        if not documents:
            log.warning(f"[{strategy}] Không có chunks → bỏ qua.")
            return None

        vs_col_name = f"vector_store_{strategy}"

        # ① Đảm bảo collection tồn tại (Atlas yêu cầu trước khi tạo Search Index)
        ensure_collection_exists(self.db, vs_col_name)
        vs_col = self.db[vs_col_name]

        # ② Xóa data cũ
        deleted = vs_col.delete_many({})
        log.info(f"[{strategy}] Đã xóa {deleted.deleted_count} vectors cũ.")

        # ③ Embed và insert documents (collection chắc chắn tồn tại)
        log.info(f"[{strategy}] Đang embed {len(documents)} chunks → '{vs_col_name}' …")
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection=vs_col,
            index_name=VECTOR_INDEX_NAME,
        )

        # ④ Tạo Vector Search Index SAU KHI đã có data
        create_vector_index_if_missing(vs_col, VECTOR_INDEX_NAME, EMBEDDING_DIM)
        log.info(f"[{strategy}] ✓ Build hoàn tất → '{vs_col_name}'.")
        return vector_store

    # ── 3. Load existing vector store ────────────
    def get_vector_store(self, strategy: str) -> MongoDBAtlasVectorSearch:
        """Kết nối tới vector store đã có trong MongoDB."""
        vs_col_name = f"vector_store_{strategy}"
        return MongoDBAtlasVectorSearch(
            collection=self.db[vs_col_name],
            embedding=self.embeddings,
            index_name=VECTOR_INDEX_NAME,
        )

    # ── 4. Similarity search ──────────────────────
    def search(self, query: str, strategy: str, k: int = 5) -> List[Document]:
        """Truy vấn vector store; trả về top-k Documents."""
        vs = self.get_vector_store(strategy)
        return vs.similarity_search(query, k=k)

    # ── 5. Search với score ───────────────────────
    def search_with_score(self, query: str, strategy: str, k: int = 5):
        """Trả về list[(Document, score)] – score là cosine similarity."""
        vs = self.get_vector_store(strategy)
        return vs.similarity_search_with_score(query, k=k)

    # ── 6. So sánh 3 strategies ───────────────────
    def compare_strategies(self, query: str, k: int = 3) -> None:
        """In kết quả truy vấn song song cho 3 strategies để so sánh."""
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        for strategy in STRATEGIES:
            print(f"\n▶ Strategy: {strategy.upper()}")
            print("-" * 50)
            try:
                results = self.search_with_score(query, strategy, k=k)
                for rank, (doc, score) in enumerate(results, 1):
                    src = doc.metadata.get("source", "N/A")
                    print(f"  [{rank}] Score={score:.4f} | Source: {src}")
                    print(f"       {doc.page_content[:200].strip()}…")
            except Exception as e:
                print(f"  ⚠ Lỗi: {e}")

        print("=" * 70 + "\n")


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    manager = TIVectorManager()

    # ── BƯỚC 1: Build 3 vector stores ────────────
    log.info("=" * 60)
    log.info("BƯỚC 1 – BUILD VECTOR STORES")
    log.info("=" * 60)
    for strat in STRATEGIES:
        manager.build_vector_store(strat)

    # ── BƯỚC 2: Test retrieval ────────────────────
    log.info("\n" + "=" * 60)
    log.info("BƯỚC 2 – TEST RETRIEVAL")
    log.info("=" * 60)

    queries = [
        "What techniques and tools are used for credential dumping and lateral movement?",
        "How do attackers persist after initial access?",
        "What are common indicators of compromise for ransomware?",
    ]

    for q in queries:
        manager.compare_strategies(q, k=3)
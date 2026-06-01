import logging
import time
import os
import shutil
import concurrent.futures
import multiprocessing as mp
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List
from pymongo import MongoClient

# ── Kiểm tra import (Chỉ để báo lỗi sớm nếu thiếu thư viện) ────────────
try:
    import chunking_evaluation
    import chromadb
except ImportError:
    raise SystemExit("\n[LỖI] Thiếu thư viện. Chạy: pip install chromadb sentence-transformers\n")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MONGO_URI = (
    "mongodb+srv://yhvn24_db_user:hovannhuy24"
    "@cluster0.4kaifw5.mongodb.net/?appName=Cluster0"
)
DB_NAME  = "threat_intel_db"

MODELS_TO_TEST = [
    "BAAI/bge-small-en-v1.5",                  # 384 dimensions
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    "sentence-transformers/all-mpnet-base-v2", # 768 dimensions
]

RETRIEVE_K = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# WORKER FUNCTION (CHẠY ĐỘC LẬP TRONG RAM)
# ──────────────────────────────────────────────
def worker_evaluate_model(model_name: str) -> dict:
    """
    Hàm này chạy hoàn toàn độc lập trong một Process riêng.
    Mọi class, import và cache sẽ bị xóa sạch sau khi hàm chạy xong.
    """
    from chunking_evaluation.chunking_evaluation import BaseChunker, GeneralEvaluation
    from chunking_evaluation.chunking_evaluation.chunking import RecursiveTokenChunker
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    # Khởi tạo Chunker ngay trong Worker để tránh lỗi Pickle
    class IsolatedRecursiveChunker(BaseChunker):
        def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
            self._inner = RecursiveTokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        def split_text(self, text: str) -> list[str]:
            return self._inner.split_text(text)

    best_chunker = IsolatedRecursiveChunker()
    safe_model_name = model_name.replace("/", "_")
    cache_dir = f"./chroma_eval_cache_emb_{safe_model_name}"
    
    # Dọn dẹp cache vật lý
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    try:
        evaluation = GeneralEvaluation(chroma_db_path=cache_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name=model_name)

        t0 = time.time()
        raw = evaluation.run(
            best_chunker,
            embedding_function=ef,
            retrieve=RETRIEVE_K,
            db_to_save_chunks=cache_dir,
        )
        elapsed = round(time.time() - t0, 1)

        p  = raw.get("precision_mean", 0)
        r  = raw.get("recall_mean", 0)
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        return {
            "model_name": model_name,
            "chunking_strategy": "recursive_500_50",
            "iou_mean": round(raw.get("iou_mean", 0), 4),
            "iou_std": round(raw.get("iou_std", 0), 4),
            "recall_mean": round(r, 4),
            "recall_std": round(raw.get("recall_std", 0), 4),
            "precision_mean": round(p, 4),
            "precision_std": round(raw.get("precision_std", 0), 4),
            "f1_mean": f1,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "model_name": model_name}

# ──────────────────────────────────────────────
# RESULT DATACLASS & EVALUATOR MANAGER
# ──────────────────────────────────────────────
@dataclass
class EmbeddingEvalResult:
    model_name:          str
    chunking_strategy:   str
    iou_mean:            float
    iou_std:             float
    recall_mean:         float
    recall_std:          float
    precision_mean:      float
    precision_std:       float
    f1_mean:             float
    elapsed_sec:         float
    timestamp:           str

class EmbeddingEvaluator:
    def __init__(self):
        log.info("Khởi tạo Benchmark cho Embedding Models (Chế độ Cách ly Tiến trình)…")
        self.mongo_client = MongoClient(MONGO_URI)
        self.db           = self.mongo_client[DB_NAME]

    def run_all(self) -> List[EmbeddingEvalResult]:
        results = []
        # Dùng context 'spawn' để đảm bảo RAM được làm sạch 100% trên cả Win/Mac/Linux
        ctx = mp.get_context('spawn')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            for i, model in enumerate(MODELS_TO_TEST, 1):
                log.info(f"\nPHASE {i}/{len(MODELS_TO_TEST)} — MODEL: {model}")
                future = executor.submit(worker_evaluate_model, model)
                
                res_dict = future.result()
                
                if "error" in res_dict:
                    log.error(f"Lỗi khi chạy {model}: {res_dict['error']}")
                    continue

                r = EmbeddingEvalResult(**res_dict)
                
                log.info(f"  IoU      = {r.iou_mean:.4f}")
                log.info(f"  Recall   = {r.recall_mean:.4f}")
                log.info(f"  Precision= {r.precision_mean:.4f}")
                log.info(f"  F1       = {r.f1_mean:.4f}  |  Time: {r.elapsed_sec}s")
                
                results.append(r)

        if results:
            self._save_to_mongo(results)
            self._print_summary(results)
            
        return results

    def _save_to_mongo(self, results: List[EmbeddingEvalResult]) -> None:
        col = self.db["evaluation_embedding"]
        col.insert_many([asdict(r) for r in results])
        log.info(f"\n✓ Đã lưu {len(results)} kết quả → collection 'evaluation_embedding'")

    @staticmethod
    def _print_summary(results: List[EmbeddingEvalResult]) -> None:
        ranked = sorted(results, key=lambda x: x.f1_mean, reverse=True)

        print("\n" + "=" * 90)
        print("  KẾT QUẢ — EVALUATING EMBEDDING MODELS")
        print("  Cố định Chunker : Recursive (Size: 500, Overlap: 50)")
        print("=" * 90)
        print(
            f"  {'Model Name':<42} "
            f"{'IoU ↑':>8} "
            f"{'Recall ↑':>9} {'Prec ↑':>8} {'F1 ↑':>7} {'Time(s)':>8}"
        )
        print("  " + "─" * 86)
        for r in ranked:
            best = " ◀ WINNER" if r == ranked[0] else ""
            print(
                f"  {r.model_name:<42} "
                f"{r.iou_mean:>8.4f} "
                f"{r.recall_mean:>9.4f} {r.precision_mean:>8.4f} "
                f"{r.f1_mean:>7.4f} {r.elapsed_sec:>8.1f}{best}"
            )
        print("=" * 90 + "\n")

# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Đóng băng quá trình chạy multiprocessing trên Windows
    mp.freeze_support()
    evaluator = EmbeddingEvaluator()
    results   = evaluator.run_all()


import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict

# ── Kiểm tra import trước khi chạy ────────────
try:
    from chunking_evaluation.chunking_evaluation import (
        BaseChunker,
        GeneralEvaluation,
    )

    from chunking_evaluation.chunking_evaluation.chunking import (
        RecursiveTokenChunker,
        ClusterSemanticChunker,
    )

except ImportError as e:
    raise SystemExit(
        f"\n[LỖI IMPORT] {e}\n"
        "Kiểm tra lại folder chunking_evaluation"
    )

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:
    raise SystemExit(
        "\n[LỖI] Chưa cài chromadb. Chạy:\n"
        "  pip install chromadb sentence-transformers\n"
    )

from pymongo import MongoClient

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MONGO_URI =
DB_NAME  = "threat_intel_db"

# SentenceTransformer — không cần API key
EVAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Số chunks retrieve mỗi query (Chroma paper dùng 5)
RETRIEVE_K = 5

# Cache ChromaDB (tái sử dụng embedding giữa các lần chạy)
CHROMA_CACHE_DIR = "./chroma_eval_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CHUNKER WRAPPERS
# ──────────────────────────────────────────────

class FixedChunker(BaseChunker):
    """
    Fixed-size chunking theo ký tự, có overlap.
    Baseline đơn giản nhất — tương đương strategy='fixed' trong pipeline.
    Ref: Chroma Technical Report (2024), Section 2.1
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        step   = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self._inner = RecursiveTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        return self._inner.split_text(text)


class SemanticChunker(BaseChunker):
    def __init__(self, embedding_function, max_chunk_size: int = 1000):
        self._inner = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=max_chunk_size,
        )

    def split_text(self, text: str) -> List[str]:
        return self._inner.split_text(text)


# ──────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────
@dataclass
class ChunkingEvalResult:
    strategy:            str
    chunk_size:          int
    chunk_overlap:       int
    iou_mean:            float
    iou_std:             float
    recall_mean:         float
    recall_std:          float
    precision_mean:      float
    precision_std:       float
    precision_omega:     float
    precision_omega_std: float
    f1_mean:             float
    elapsed_sec:         float
    corpora_scores:      Dict
    timestamp:           str


# ──────────────────────────────────────────────
# MAIN EVALUATOR
# ──────────────────────────────────────────────
class ChunkingEvaluator:

    def __init__(self):
        log.info("Khởi tạo GeneralEvaluation benchmark…")
        self.evaluation = GeneralEvaluation(chroma_db_path=CHROMA_CACHE_DIR)

        log.info(f"Tải embedding model: {EVAL_EMBEDDING_MODEL} (lần đầu ~1-2 phút)…")
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name=EVAL_EMBEDDING_MODEL
        )
        log.info("Embedding function sẵn sàng ✓")

        self.mongo_client = MongoClient(MONGO_URI)
        self.db           = self.mongo_client[DB_NAME]

    def _run_one(
        self,
        chunker: BaseChunker,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> ChunkingEvalResult:

        label = f"{strategy}(size={chunk_size}, overlap={chunk_overlap})"
        log.info(f"\n{'─'*60}")
        log.info(f"Đang đánh giá: {label}")
        log.info(f"{'─'*60}")

        t0 = time.time()
        try:
            raw = self.evaluation.run(
                chunker,
                embedding_function=self.ef,
                retrieve=RETRIEVE_K,
                db_to_save_chunks=CHROMA_CACHE_DIR,
            )
        except Exception as e:
            log.error(f"Lỗi khi chạy {label}: {e}")
            raise

        elapsed = round(time.time() - t0, 1)

        p  = raw.get("precision_mean", 0)
        r  = raw.get("recall_mean", 0)
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        result = ChunkingEvalResult(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            iou_mean=round(raw.get("iou_mean", 0), 4),
            iou_std=round(raw.get("iou_std", 0), 4),
            recall_mean=round(r, 4),
            recall_std=round(raw.get("recall_std", 0), 4),
            precision_mean=round(p, 4),
            precision_std=round(raw.get("precision_std", 0), 4),
            precision_omega=round(raw.get("precision_omega_mean", 0), 4),
            precision_omega_std=round(raw.get("precision_omega_std", 0), 4),
            f1_mean=f1,
            elapsed_sec=elapsed,
            corpora_scores=raw.get("corpora_scores", {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        log.info(f"  IoU      = {result.iou_mean:.4f} ± {result.iou_std:.4f}")
        log.info(f"  Recall   = {result.recall_mean:.4f} ± {result.recall_std:.4f}")
        log.info(f"  Precision= {result.precision_mean:.4f} ± {result.precision_std:.4f}")
        log.info(f"  Prec-Ω   = {result.precision_omega:.4f} ± {result.precision_omega_std:.4f}")
        log.info(f"  F1       = {result.f1_mean:.4f}  |  Time: {elapsed}s")

        return result

    def _save_to_mongo(self, results: List[ChunkingEvalResult]) -> None:
        col = self.db["evaluation_chunking"]
        col.insert_many([asdict(r) for r in results])
        log.info(f"\n✓ Đã lưu {len(results)} kết quả → collection 'evaluation_chunking'")

    def run_all(self) -> List[ChunkingEvalResult]:
        results = []

        # ── 1. FIXED ─────────────────────────────────
        log.info("\n" + "="*60)
        log.info("PHASE 1/3 — FIXED CHUNKING")
        log.info("="*60)
        for size, overlap in [(500, 50), (1000, 100), (1500, 150)]:
            r = self._run_one(FixedChunker(size, overlap), "fixed", size, overlap)
            results.append(r)

        # ── 2. RECURSIVE ─────────────────────────────
        log.info("\n" + "="*60)
        log.info("PHASE 2/3 — RECURSIVE CHUNKING")
        log.info("="*60)
        for size, overlap in [(500, 50), (1000, 100), (1500, 150)]:
            r = self._run_one(RecursiveChunker(size, overlap), "recursive", size, overlap)
            results.append(r)

        # ── 3. SEMANTIC ───────────────────────────────
        log.info("\n" + "="*60)
        log.info("PHASE 3/3 — SEMANTIC CHUNKING")
        log.info("="*60)
        for max_size in [500, 1000, 1500]:
            chunker = SemanticChunker(self.ef, max_size)
            r = self._run_one(chunker, "semantic", max_size, 0)
            results.append(r)

        self._save_to_mongo(results)
        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: List[ChunkingEvalResult]) -> None:
        ranked = sorted(results, key=lambda x: x.iou_mean, reverse=True)

        print("\n" + "=" * 82)
        print("  KẾT QUẢ — EVALUATING CHUNKING STRATEGIES")
        print("  Benchmark : GeneralEvaluation · 473 queries · 5 corpora")
        print("  Ref       : Smith & Troynikov, Chroma Technical Report, 2024")
        print("=" * 82)
        print(
            f"  {'Strategy':<12} {'Size':>5} {'Ovlp':>5} "
            f"{'IoU ↑':>8} {'±':>6} "
            f"{'Recall ↑':>9} {'Prec ↑':>8} {'Prec-Ω ↑':>9} {'F1 ↑':>7}"
        )
        print("  " + "─" * 76)
        for r in ranked:
            best = " ◀" if r == ranked[0] else ""
            print(
                f"  {r.strategy:<12} {r.chunk_size:>5} {r.chunk_overlap:>5} "
                f"{r.iou_mean:>8.4f} {r.iou_std:>6.4f} "
                f"{r.recall_mean:>9.4f} {r.precision_mean:>8.4f} "
                f"{r.precision_omega:>9.4f} {r.f1_mean:>7.4f}{best}"
            )
        print("=" * 82)
        print()
        print("  METRIC DEFINITIONS (Smith & Troynikov, 2024):")
        print("  IoU     = |retrieved ∩ ref| / |retrieved ∪ ref|  — metric chính")
        print("  Recall  = |retrieved ∩ ref| / |ref|              — bao phủ đủ không?")
        print("  Prec    = |retrieved ∩ ref| / |retrieved|        — trả về đúng không?")
        print("  Prec-Ω  = IoU không qua retrieval                — upper-bound chunking")
        print("  F1      = 2·P·R / (P+R)                          — harmonic mean")
        print("=" * 82 + "\n")


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = ChunkingEvaluator()
    results   = evaluator.run_all()

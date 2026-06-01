"""
embedding.py
============

Embedding Pipeline cho Cyber Threat Intelligence RAG System

Chức năng:
──────────────────────────────────────────────
1. Load nhiều embedding models
2. Generate embeddings cho chunks
3. Lưu embeddings vào MongoDB
4. Benchmark tốc độ embedding
5. So sánh dimensions / latency / throughput
6. Hỗ trợ RAG evaluation

Models:
──────────────────────────────────────────────
- BAAI/bge-small-en-v1.5
- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-base-en-v1.5

Collections:
──────────────────────────────────────────────
processed_chunks_fixed
processed_chunks_recursive
processed_chunks_semantic

Output:
──────────────────────────────────────────────
embeddings_fixed
embeddings_recursive
embeddings_semantic
embedding_benchmarks
"""

import logging
import time
import statistics
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================

MONGO_URI = (
    "mongodb+srv://yhvn24_db_user:hovannhuy24"
    "@cluster0.4kaifw5.mongodb.net/?appName=Cluster0"
)

DB_NAME = "threat_intel_db"

STRATEGIES = [
    "fixed",
    "recursive",
    "semantic",
]

EMBEDDING_MODELS = {
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
    },

    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    },

    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
    },
}

BATCH_SIZE = 32

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)

# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class EmbeddingBenchmark:

    strategy: str

    model_key: str

    model_name: str

    total_chunks: int

    dimension: int

    embedding_time_sec: float

    avg_time_per_chunk_ms: float

    throughput_chunks_per_sec: float

    timestamp: str

# =========================================================
# EMBEDDING PIPELINE
# =========================================================

class EmbeddingPipeline:

    def __init__(self):

        self.client = MongoClient(MONGO_URI)

        self.db = self.client[DB_NAME]

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self, model_key: str):

        cfg = EMBEDDING_MODELS[model_key]

        log.info(f"Loading model: {cfg['model_name']}")

        model = SentenceTransformer(
            cfg["model_name"]
        )

        return model

    # =====================================================
    # LOAD CHUNKS
    # =====================================================

    def load_chunks(
        self,
        strategy: str,
    ) -> List[Dict]:

        collection = self.db[
            f"processed_chunks_{strategy}"
        ]

        docs = list(
            collection.find(
                {},
                {
                    "content": 1,
                    "source": 1,
                    "chunk_id": 1,
                }
            )
        )

        return docs

    # =====================================================
    # GENERATE EMBEDDINGS
    # =====================================================

    def generate_embeddings(
        self,
        strategy: str,
        model_key: str,
    ) -> EmbeddingBenchmark:

        docs = self.load_chunks(strategy)

        if not docs:

            log.warning(
                f"[{strategy}] No chunks found."
            )

            return None

        model = self.load_model(model_key)

        texts = [
            d.get("content", "")
            for d in docs
        ]

        log.info(
            f"[{strategy}|{model_key}] "
            f"Generating embeddings "
            f"for {len(texts)} chunks..."
        )

        t0 = time.time()

        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        elapsed = round(
            time.time() - t0,
            2
        )

        # =================================================
        # SAVE TO MONGODB
        # =================================================

        out_collection = self.db[
            f"embeddings_{strategy}"
        ]

        out_collection.delete_many({
            "model_key": model_key
        })

        records = []

        for doc, vec in zip(docs, vectors):

            records.append({

                "chunk_id": doc.get("chunk_id"),

                "content": doc.get("content"),

                "source": doc.get("source"),

                "strategy": strategy,

                "model_key": model_key,

                "model_name": EMBEDDING_MODELS[
                    model_key
                ]["model_name"],

                "embedding_dimension": len(vec),

                "embedding": vec.tolist(),

                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

        if records:

            out_collection.insert_many(records)

        # =================================================
        # BENCHMARK
        # =================================================

        total_chunks = len(texts)

        avg_time_ms = round(
            (elapsed / total_chunks) * 1000,
            2
        )

        throughput = round(
            total_chunks / elapsed,
            2
        )

        benchmark = EmbeddingBenchmark(

            strategy=strategy,

            model_key=model_key,

            model_name=EMBEDDING_MODELS[
                model_key
            ]["model_name"],

            total_chunks=total_chunks,

            dimension=len(vectors[0]),

            embedding_time_sec=elapsed,

            avg_time_per_chunk_ms=avg_time_ms,

            throughput_chunks_per_sec=throughput,

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self.db["embedding_benchmarks"].insert_one(
            asdict(benchmark)
        )

        log.info(
            f"[{strategy}|{model_key}] DONE ✓"
        )

        log.info(
            f"Time: {elapsed}s | "
            f"Throughput: {throughput} chunks/s"
        )

        return benchmark

    # =====================================================
    # RUN ALL
    # =====================================================

    def run_all(self):

        benchmarks = []

        for strategy in STRATEGIES:

            log.info("=" * 60)

            log.info(
                f"PROCESSING STRATEGY: {strategy}"
            )

            log.info("=" * 60)

            for model_key in EMBEDDING_MODELS.keys():

                benchmark = self.generate_embeddings(
                    strategy,
                    model_key,
                )

                if benchmark:
                    benchmarks.append(benchmark)

        self.print_summary(benchmarks)

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    def print_summary(
        self,
        benchmarks: List[EmbeddingBenchmark]
    ):

        print("\n" + "=" * 90)

        print(
            "EMBEDDING MODEL BENCHMARK"
        )

        print("=" * 90)

        print(
            f"{'Strategy':<12}"
            f"{'Model':<15}"
            f"{'Dim':<8}"
            f"{'Chunks':<10}"
            f"{'Time(s)':<12}"
            f"{'ms/chunk':<12}"
            f"{'Chunks/s':<12}"
        )

        print("-" * 90)

        for b in benchmarks:

            print(
                f"{b.strategy:<12}"
                f"{b.model_key:<15}"
                f"{b.dimension:<8}"
                f"{b.total_chunks:<10}"
                f"{b.embedding_time_sec:<12}"
                f"{b.avg_time_per_chunk_ms:<12}"
                f"{b.throughput_chunks_per_sec:<12}"
            )

        print("=" * 90)

        # BEST THROUGHPUT

        fastest = max(
            benchmarks,
            key=lambda x: x.throughput_chunks_per_sec
        )

        print("\nFASTEST MODEL:")
        print(
            f"{fastest.model_key} "
            f"({fastest.strategy}) → "
            f"{fastest.throughput_chunks_per_sec} chunks/s"
        )

        print("=" * 90 + "\n")

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    pipeline = EmbeddingPipeline()

    pipeline.run_all()
import streamlit as st
from pymongo import MongoClient
import pandas as pd
from google import genai                    # google.genai (mới, không deprecated)
from pypdf import PdfReader
import sys, os, logging
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from typing import Any
from typing import cast
from chunking import TIChunkingPipeline
# ── Thêm thư mục project vào sys.path để import được các file cùng cấp ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Import từ các file thực tế của project ──────────────────────────────
from chunking import TIChunkingPipeline          # apply_chunking_strategies()
from ioc_extractor import IOCExtractor           # clean_text(), extract_all_iocs()
# vector_db.TIVectorManager KHÔNG dùng build_vector_store() (vì nó delete_many)
# → ta dùng MongoDBAtlasVectorSearch.from_documents() trực tiếp để APPEND

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
MONGO_URI          = "mongodb+srv://yhvn24_db_user:hovannhuy24@cluster0.4kaifw5.mongodb.net/?appName=Cluster0"
GEMINI_API_KEY     = "AIzaSyCm3yfWzYwIb_7HsQFgwtDOS1ZnvjmgmOc"
DB_NAME            = "threat_intel_db"
EMBEDDING_MODEL    = "BAAI/bge-small-en-v1.5"
VECTOR_INDEX_NAME  = "vector_index"

@st.cache_resource
def get_db_connection() -> Any:
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]
@st.cache_resource
def get_pipeline() -> Any:
    return TIChunkingPipeline(mongo_uri=MONGO_URI, db_name=DB_NAME)
db = get_db_connection()
pipeline = cast(TIChunkingPipeline, get_pipeline())
embedding=pipeline.embed_model,
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# 2. ĐỌC FILE (TXT + PDF)
# ─────────────────────────────────────────────
def read_file(file) -> str:
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────
# 3. GUARD: kiểm tra file đã xử lý chưa
# ─────────────────────────────────────────────
def file_already_processed(file_name: str) -> bool:
    """True nếu file đã có trong reports VÀ đã chạy pipeline đầy đủ."""
    doc = db["reports"].find_one({"title": file_name})
    return doc is not None and doc.get("pipeline_done", False)

# ─────────────────────────────────────────────
# 4. LƯU REPORT THÔ
# ─────────────────────────────────────────────
def save_raw_report(file_name: str, content: str) -> tuple[bool, str]:
    if db["reports"].find_one({"title": file_name}):
        return False, "⚠️ File đã tồn tại trong database!"
    db["reports"].insert_one({
        "title":         file_name,
        "content":       content,
        "created_at":    pd.Timestamp.now(),
        "pipeline_done": False,
        "extracted_iocs": {},
    })
    return True, "✅ Đã lưu báo cáo thô vào database!"

# ─────────────────────────────────────────────
# 5. PIPELINE — CHỈ CHẠY VỚI FILE MỚI, KHÔNG XÓA DATA CŨ
# ─────────────────────────────────────────────
def run_pipeline_for_new_file(file_name: str, raw_text: str) -> dict:
    """
    Chạy toàn bộ pipeline cho 1 file người dùng vừa upload:
      Bước 1 → IOCExtractor : clean_text + extract_all_iocs
      Bước 2 → TIChunkingPipeline.apply_chunking_strategies (3 strategies)
      Bước 3 → INSERT chunks vào processed_chunks_* (KHÔNG delete_many cũ)
      Bước 4 → Embed + APPEND vào vector_store_*    (KHÔNG delete_many cũ)
    """
    results = {}

    # ── BƯỚC 1: Clean text + Extract IOCs ────────────────────────────────
    ioc_extractor = IOCExtractor()
    cleaned_text  = ioc_extractor.clean_text(raw_text)
    iocs          = ioc_extractor.extract_all_iocs(cleaned_text)
    results["iocs"] = iocs

    # ── BƯỚC 2: Tạo LangChain Document để đưa vào chunking ───────────────
    lc_doc = Document(
        page_content=cleaned_text,
        metadata={
            "source_file":       file_name,
            "title":             file_name,
            "source_collection": "user_upload",          # phân biệt với reports_data
            "ioc_ipv4":          ", ".join(iocs["ipv4"]),
            "ioc_hashes":        ", ".join(iocs["hashes"]),
            "ioc_cve":           ", ".join(iocs["cve"]),
            "ioc_domains":       ", ".join(iocs["domains"]),
        }
    )

    # ── BƯỚC 3: Chunking bằng TIChunkingPipeline của project ─────────────
    # Khởi tạo pipeline với cloud URI — embed_model dùng lại ở bước 4
    chunking_pipeline = TIChunkingPipeline(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME
    )
    chunks_dict = chunking_pipeline.apply_chunking_strategies([lc_doc])
    # chunks_dict = {"fixed": [...], "recursive": [...], "semantic": [...]}

    # ── BƯỚC 4: Lưu chunks → INSERT ONLY (KHÔNG gọi save_chunks_to_db
    #            vì hàm đó có delete_many({}) — xóa hết data cũ!) ──────────
    for strategy, chunks in chunks_dict.items():
        col_name = f"processed_chunks_{strategy}"
        records  = [
            {
                "content":     c.page_content,
                "metadata":    c.metadata,
                "source_file": file_name,   # thêm field nhanh để query sau
            }
            for c in chunks
        ]
        if records:
            db[col_name].insert_many(records)
        results[f"chunks_{strategy}"] = len(records)

    # ── BƯỚC 5: Embed + APPEND vào Vector Store ───────────────────────────
    # Dùng embed_model đã load sẵn trong chunking_pipeline (tránh load lại)
    embed_model = chunking_pipeline.embed_model

    for strategy, chunks in chunks_dict.items():
        if not chunks:
            continue
        vs_col_name = f"vector_store_{strategy}"
        vs_col      = db[vs_col_name]

        # from_documents APPEND vào collection hiện có
        # (KHÔNG gọi TIVectorManager.build_vector_store vì nó delete_many cũ)
        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embed_model,
            collection=vs_col,
            index_name=VECTOR_INDEX_NAME,
        )
        results[f"vectors_{strategy}"] = len(chunks)

    # ── BƯỚC 6: Cập nhật trạng thái pipeline_done = True ──────────────────
    db["reports"].update_one(
        {"title": file_name},
        {"$set": {
            "pipeline_done":  True,
            "extracted_iocs": iocs,
        }}
    )

    return results

# ─────────────────────────────────────────────
# 6. GEMINI ANALYSIS (google.genai API mới)
# ─────────────────────────────────────────────
from google.genai import types
from google.genai.errors import ServerError
import time


def analyze_with_gemini(text: str) -> str:
    prompt = f"""
Bạn là một chuyên gia phân tích an ninh mạng cao cấp.
Hãy đọc báo cáo dưới đây và thực hiện các yêu cầu sau bằng Tiếng Việt:

1. Tóm tắt nội dung chính (Executive Summary).
2. Trích xuất các kỹ thuật tấn công theo khung MITRE ATT&CK.
3. Liệt kê các chỉ số thỏa hiệp (IOCs) nếu có (IP, Hash, Domains).
4. Đưa ra khuyến nghị phòng thủ.

Báo cáo:
{text[:12000]}
"""

    # Config an toàn + ổn định hơn
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config=config
            )

            # Kiểm tra response hợp lệ
            if response and hasattr(response, "text"):
                return response.text

            return "Không nhận được phản hồi từ Gemini."

        except ServerError as e:
            print(f"[Lần thử {attempt+1}] Gemini ServerError: {e}")

            # Retry nếu lỗi 503
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Đợi {wait_time}s rồi thử lại...")
                time.sleep(wait_time)
            else:
                return f"Lỗi Gemini API sau nhiều lần thử: {e}"

        except Exception as e:
            return f"Lỗi phân tích Gemini: {str(e)}"

# ─────────────────────────────────────────────
# 7. STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="CTI Analyst Platform", layout="wide")
st.sidebar.title("⚙️ Bảng điều khiển")
st.sidebar.info("Hệ thống phân tích mối đe dọa an ninh mạng")
st.title("🛡️ Threat Intelligence Platform")

tab1, tab2, tab3 = st.tabs(["📂 Quản lý Báo cáo", "🤖 Phân tích AI", "📊 Xuất dữ liệu"])

# ── TAB 1: Upload + Pipeline ──────────────────────────────────────────────
with tab1:
    st.header("Tải lên báo cáo mới")
    uploaded_file = st.file_uploader(
        "Kéo thả file báo cáo (TXT / PDF)",
        type=["txt", "pdf"]
    )

    if uploaded_file:
        # ── Chặn ngay nếu đã xử lý — không cho chạy lại ──────────────────
        if file_already_processed(uploaded_file.name):
            st.warning(
                f"⚠️ **{uploaded_file.name}** đã được xử lý đầy đủ.\n\n"
                "Dữ liệu gốc trong database được giữ nguyên để tránh ghi đè / trùng lặp."
            )
        else:
            if st.button("💾 Lưu & Chạy Pipeline"):
                # 1. Đọc nội dung file
                raw_text = read_file(uploaded_file)

                # 2. Lưu report thô
                ok, msg = save_raw_report(uploaded_file.name, raw_text)
                if not ok:
                    st.warning(msg)
                    st.stop()
                st.success(msg)

                # 3. Chạy pipeline — chỉ file này, không đụng data cũ
                with st.status("⚙️ Đang chạy pipeline...", expanded=True):
                    st.write("🔍 Bước 1/4 — Clean text + Extract IOCs...")
                    st.write("🔪 Bước 2-3/4 — Chunking (fixed / recursive / semantic)...")
                    st.write("🔢 Bước 4/4 — Embedding + Append vào Vector Store...")
                    results = run_pipeline_for_new_file(uploaded_file.name, raw_text)

                st.success("🎉 Pipeline hoàn tất — data cũ trong database không bị ảnh hưởng!")

                # Hiển thị kết quả
                col1, col2, col3 = st.columns(3)
                col1.metric("Fixed chunks",     results.get("chunks_fixed", 0))
                col2.metric("Recursive chunks", results.get("chunks_recursive", 0))
                col3.metric("Semantic chunks",  results.get("chunks_semantic", 0))

                iocs = results.get("iocs", {})
                if any(iocs.values()):
                    with st.expander("🔍 IOCs trích xuất được"):
                        st.json(iocs)

    st.divider()
    st.header("Danh sách báo cáo hiện có")
    reports = list(db["reports"].find({}, {"title": 1, "created_at": 1, "pipeline_done": 1}))
    if reports:
        df = pd.DataFrame(reports).drop(columns=["_id"], errors="ignore")
        df["pipeline_done"] = df["pipeline_done"].apply(
            lambda x: "✅ Đã xử lý" if x else "⏳ Chưa xử lý"
        )
        df.columns = ["Tên file", "Ngày tạo", "Trạng thái"]
        st.table(df)
    else:
        st.info("Chưa có báo cáo nào.")

# ── TAB 2: Gemini AI ──────────────────────────────────────────────────────
with tab2:
    st.header("🤖 Phân tích AI chuyên sâu (RAG)")
    
    # 1. Cấu hình tìm kiếm
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        strategy = st.selectbox(
            "Chọn chiến lược chunking để tham khảo:",
            ["fixed", "recursive", "semantic"]
        )
    with col_config2:
        k_chunks = st.slider("Số lượng ngữ cảnh (chunks) tham khảo:", 3, 10, 5)

    # 2. Câu hỏi người dùng
    user_query = st.text_area("Nhập câu hỏi hoặc yêu cầu phân tích:", 
                              "Hãy tóm tắt các kỹ thuật tấn công và IOCs từ các báo cáo liên quan.")

    if st.button("🚀 Thực thi truy vấn RAG"):
        if not user_query:
            st.warning("Vui lòng nhập câu hỏi!")
        else:
            with st.spinner("Đang tìm kiếm thông tin trong Vector DB..."):
                # Khởi tạo Vector Search
                vector_search = MongoDBAtlasVectorSearch(
                    collection=db[f"vector_store_{strategy}"],
                    embedding=pipeline.embed_model,  # pipeline thay vì chunking_pipeline
                    index_name=VECTOR_INDEX_NAME
                )
                
                # Tìm kiếm ngữ nghĩa
                docs = vector_search.similarity_search(user_query, k=k_chunks)
                context = "\n\n".join([d.page_content for d in docs])
                
                # Gửi lên Gemini với context đã tìm được
                st.write("🔍 Đang tổng hợp câu trả lời từ hệ thống...")
                final_prompt = f"""
                Bạn là một chuyên gia CTI (Cyber Threat Intelligence).
                Dựa vào thông tin ngữ cảnh được cung cấp bên dưới, hãy trả lời câu hỏi của người dùng.
                Nếu thông tin không đủ, hãy trả lời dựa trên kiến thức chuyên môn của bạn nhưng phải nêu rõ.

                Ngữ cảnh (Context):
                {context}

                Câu hỏi của người dùng:
                {user_query}
                """
                
                analysis = analyze_with_gemini(final_prompt)
                
                st.markdown("---")
                st.markdown("### 📝 Kết quả phân tích")
                st.write(analysis)
                
                # Hiển thị nguồn tham khảo
                with st.expander("🔗 Xem các nguồn dữ liệu đã sử dụng"):
                    sources = set([d.metadata.get("source_file") for d in docs])
                    st.write("Dữ liệu được trích xuất từ các file:", sources)

            for source_file in sources:
                db["reports"].update_one(
                    {"title": source_file},
                    {"$set": {"analysis": analysis}}
                )
            st.success("✅ Đã lưu kết quả phân tích vào database!")

# ── TAB 3: Export Chunks ──────────────────────────────────────────────────
with tab3:
    st.header("📊 Xuất dữ liệu Chunks")

    strategy = st.selectbox(
        "Chọn chiến lược chunking:",
        ["processed_chunks_fixed", "processed_chunks_recursive", "processed_chunks_semantic"]
    )

    # Lọc theo file nguồn (chỉ filter user_upload hay tất cả)
    source_files = db[strategy].distinct("source_file")
    filter_file  = st.selectbox(
        "Lọc theo file nguồn (tuỳ chọn):",
        ["-- Tất cả --"] + [f for f in source_files if f]
    )

    if st.button("Lấy dữ liệu để tải xuống"):
        query = {} if filter_file == "-- Tất cả --" else {"source_file": filter_file}
        data  = list(db[strategy].find(query, {"_id": 0}))

        if data:
            df = pd.DataFrame(data)
            st.write(f"Đã tìm thấy **{len(df)}** chunks.")
            st.dataframe(df.head(30), use_container_width=True)
            st.download_button(
                label="📥 Tải xuống CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"cti_data_{strategy}.csv",
                mime="text/csv"
            )
        else:
            st.error("Collection này không có dữ liệu hoặc chưa được xử lý.")
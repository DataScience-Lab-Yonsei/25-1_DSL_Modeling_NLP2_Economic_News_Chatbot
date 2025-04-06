import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from ner import extract_ner_from_text  

FAISS_INDEX_PATH = "news_index_final.faiss"
PICKLE_PATH = "news_data_final.pkl"
MODEL_NAME = "jhgan/ko-sroberta-multitask"

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(PICKLE_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(PICKLE_PATH, "rb") as f:
        news_df = pickle.load(f)
else:
    print("⚠️ FAISS 인덱스 또는 뉴스 데이터가 존재하지 않습니다.")
    faiss_index = None
    news_df = None

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def generate_weighted_query(text: str, top_companies: list):
    """
    GPT에서 받은 top 2 기업에 높은 가중치, person_ner 인물은 낮은 가중치로 쿼리 생성
    - top_companies: ["금융감독원", "금융투자협회"] 등 GPT 정제 결과
    """
    _, person_ner = extract_ner_from_text(text)


    weighted_company = []
    for comp in top_companies:
        weighted_company.extend([comp] * 5)


    weighted_person = []
    if person_ner:
        top_person = max(person_ner, key=person_ner.get)
        weighted_person = [top_person] * 2

    return " ".join(weighted_company + weighted_person)

def find_similar_news(query: str, top_k: int = 10, final_n: int = 4):
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    if faiss_index is None or news_df is None:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize_embeddings(query_embedding)

    distances, indices = faiss_index.search(query_embedding, top_k)
    indices = indices[0]

    candidate_df = news_df.iloc[indices].copy()
    final_news = candidate_df.iloc[1:final_n]

    # ✅ streamlit에서 바로 사용 가능한 딕셔너리 리스트 형태로 반환
    return final_news[["headline", "link"]].to_dict(orient="records")

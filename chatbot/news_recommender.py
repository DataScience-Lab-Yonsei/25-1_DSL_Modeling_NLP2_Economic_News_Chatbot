import numpy as np
import pandas as pd
import faiss
import os

news_embeddings = None
news_ids = None
faiss_index = None
df_news = None

def load_news_data(csv_path):
    global df_news
    df_news = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    print(f"뉴스 CSV 로드 완료! {len(df_news)}개 기사")

def load_embeddings_and_index(embedding_path, id_path, index_path):
    global news_embeddings, news_ids, faiss_index

    if not os.path.exists(embedding_path) or not os.path.exists(id_path):
        raise FileNotFoundError("❌ 벡터 파일이 없습니다.")
    if not os.path.exists(index_path):
        raise FileNotFoundError("❌ FAISS Index 파일이 없습니다.")

    news_embeddings = np.load(embedding_path).astype(np.float32)
    news_ids = np.load(id_path)
    faiss_index = faiss.read_index(index_path)

    print(f"벡터 및 인덱스 로드 완료! 총 {len(news_embeddings)}개 벡터")

def recommend_similar_news(news_title, news_content, top_n=20):
    global df_news, news_ids, news_embeddings, faiss_index

    # 제목과 본문에 해당하는 인덱스 찾기
    title_idx = df_news[df_news["headline"] == news_title].index
    content_idx = df_news[df_news["content"] == news_content].index

    if len(title_idx) == 0 or len(content_idx) == 0:
        raise ValueError("제목 또는 본문과 일치하는 뉴스가 데이터에 없습니다.")

    title_vector_idx = np.where(news_ids == title_idx[0])[0]
    content_vector_idx = np.where(news_ids == content_idx[0])[0]

    if len(title_vector_idx) == 0 or len(content_vector_idx) == 0:
        raise ValueError("벡터 인덱스 매핑 실패.")

    title_vec = news_embeddings[title_vector_idx[0]]
    content_vec = news_embeddings[content_vector_idx[0]]


    input_embedding = (2 * title_vec + content_vec) / 3
    input_embedding = input_embedding.reshape(1, -1).astype(np.float32)

    distances, indices = faiss_index.search(input_embedding, top_n)
    distances = distances[0]
    indices = indices[0]

    result_df = df_news.iloc[news_ids[indices]][["headline", "link"]].copy()
    result_df["l2_distance"] = np.round(distances, 4)

    result_df = result_df.iloc[1:].reset_index(drop=True)
    
    filtered = []
    for idx, row in result_df.iterrows():
        if not filtered:
            filtered.append(row)
        else:
            if all(abs(row["l2_distance"] - prev["l2_distance"]) >= 0.15 for prev in filtered):
                filtered.append(row)
        if len(filtered) == 3:
            break

    return pd.DataFrame(filtered)

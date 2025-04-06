import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re
from collections import Counter
import os

from dotenv import load_dotenv
import os
import openai

# ✅ 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# ✅ 금융 용어 추출 함수
def extract_financial_terms(news_article):
    prompt = f"""
    당신은 경제용어를 정확히 알고있는 경제전문가입니다. 아래 triple backticks로 구분된 경제기사에서 중요한 경제 용어나 개념을 추출해주세요.

    **추출 원칙** :
    1. 주어진 경제 기사에서 등장한 핵심 **전문 경제 용어** 5개를 선정하세요.
    2. 변화 상황이나 상태를 나타내는 표현(예: '금리 인하', '물가 상승')이 있더라도, **'금리', '물가'처럼 핵심 개념 단어(명사형)만 추출하세요.**
    3. **특정 기업명, 기관명, 고유명사는 제외**하고, 경제적 의미가 있는 개념 중심 용어만 추출하세요.
    4. 너무 일반적인 단어(예: 시장, 가격 등)는 제외하세요.
    5. **전문 용어 중심으로 작성**하고, 경제 기사 전반의 맥락에서 의미 있는 개념을 중심으로 선택하세요.
    6. 추출된 용어는 **세미콜론(;)으로 구분하여 한 줄로 나열**하세요.
    7. **반드시 용어와 세미콜론만 출력하고, 절대 마침표(.), 문장, 설명, 기타 텍스트를 포함하지 마세요.**
    ```
    {news_article}
    ```

    """

    # ✅ OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 경제 전문가로서 주어진 기사에서 경제 용어만 정확히 추출해야 합니다."},
            {"role": "user", "content": prompt}
        ]
    )

    # ✅ GPT 응답에서 용어 추출
    extracted_terms = response.choices[0].message.content.strip()

    # ✅ 세미콜론을 기준으로 리스트 변환
    terms_list = [term.strip() for term in extracted_terms.split(';') if term.strip()]

    # ✅ 다수의 GPT 호출을 통한 용어 빈도 분석
    all_terms = []
    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        response_terms = response.choices[0].message.content.strip()
        terms_list = [term.strip() for term in response_terms.split(';') if term.strip()]
        all_terms.extend(terms_list)

    # ✅ 빈도수 계산하여 상위 3개 용어 선택
    term_counts = Counter(all_terms)
    top_3_terms = [term for term, _ in term_counts.most_common(3)]

    return top_3_terms


# BGE-M3 모델 로드
model = SentenceTransformer("BAAI/bge-m3")

# 참조문서
file_path = "경제용어700선.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 용어 및 설명 분리
entries = re.split(r"\n\*\*(.+?)\*\*\n", text)
terms, docs, related_keywords = [], [], []
for i in range(1, len(entries), 2):
    term = entries[i].strip()
    desc_raw = entries[i+1].strip().replace('\n', ' ')
    desc_clean = re.sub(r'([다요임할]다\.|니다\.|습니까\?)', r'\1\n', desc_raw)
    terms.append(term)
    docs.append(desc_clean.strip())

    # 연관검색어 추출
    match = re.search(r'연관검색어\s*[:：]\s*(.+)', desc_raw)
    if match:
        keywords = [kw.strip() for kw in re.split(r'[ ,/]', match.group(1)) if kw.strip()]
    else:
        keywords = []
    related_keywords.append(keywords)

# 용어 매핑 사전 구성 (괄호 포함 용어 대응)
term_map = {}  # 예: {'유럽연합': idx, 'EU': idx, '유럽연합(EU)': idx}
for i, term in enumerate(terms):
    term_map[term] = i
    matches = re.findall(r"(.+?)\((.+?)\)", term)
    if matches:
        base, alias = matches[0]
        term_map[base.strip()] = i
        term_map[alias.strip()] = i

# 사전 임베딩, FAISS index 로드
doc_embeddings = np.load("doc_embeddings.npy")
index = faiss.read_index("faiss_index.index")
index.add(doc_embeddings)

# docs 설명 기반 유사 용어 추천 함수 (단어가 docs에 있지만 연관검색어가 없는 경우)
def suggest_similar_terms(query, top_k=5, threshold=0.5):
    query_idx = terms.index(query)
    query_description = docs[query_idx]
    query_emb = model.encode(["query: " + query_description], convert_to_numpy=True)

    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    sorted_idx = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_idx[:top_k]:
        if similarities[idx] >= threshold:
            results.append((terms[idx], round(float(similarities[idx]), 3)))
    return results


# GPT 설명 생성 함수 (사전에 없는 단어 설명)
def explain_query_with_gpt(query):
    prompt = f"""
    '{query}'는 경제지식이 거의 없는 초보자가 경제기사를 읽던 도중 궁금해하는 단어입니다. '{query}'에 대해 경제 초보자가 이해하기 쉽게 설명해 주세요.
    만약 {query}가 경제와 큰 관련이 없는 단어라면, "해당 단어는 경제 관련 용어가 아닙니다." 라는 설명을 추가하세요. 
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 경제 분야에 정통한 설명 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0
    )
    return response.choices[0].message.content

# GPT 설명 기반 유사 단어 추천 함수
def suggest_similar_by_gpt_description(query, top_k=5, threshold=0.5):
    query_explained = explain_query_with_gpt(query)
    query_emb = model.encode([query_explained], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    sorted_idx = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_idx[:top_k]:
        if similarities[idx] >= threshold:
            results.append((terms[idx], round(float(similarities[idx]), 3)))
    return results

# 유사 용어 추천함수 (모든 경우의 수)
def recommend_similar_words(query, top_k=5):
    if query in term_map:
        # print("이 용어는 경제사전에 있는 단어입니다.\n")
        idx = term_map[query]
        term_desc = docs[idx]
        if related_keywords[idx] != []:
            related = related_keywords[idx]
        else:
            related = [term for term, _ in suggest_similar_terms(query, top_k=5, threshold=0.5)[1:3]]
        return term_desc, related
    else:
        print("이 용어는 경제사전에 없는 단어입니다. GPT가 자체적으로 생산한 답변으로, 부정확할 수 있으니 참고해주세요. \n")
        gpt_desc = explain_query_with_gpt(query)
        related = [term for term, _ in suggest_similar_by_gpt_description(query, top_k=5, threshold=0.5)[:2]]
    return gpt_desc, related

# 답변 생성함수
def generate_answer(query, model="gpt-4o"):
    desc, related = recommend_similar_words(query)

    if query in term_map:
        prompt = f"""
        당신은 경제 분야에 정통한 경제 전문가입니다. triple backticks로 구분된 문서를 참고하여 {query}에 대해 자세히 설명하세요.
        1. '{query}'는 경제 신문에서 등장한 경제용어이다.
        2. 아래 문서의 내용을 바탕으로 경제를 잘 모르는 초보자도 알기 쉽게 설명해야 한다.
        3. 출력 형식은 다음과 같다.
        **{query}**
        {query}는 ~~

        **연관용어**
        {related}

        ```
        {desc}
        ```
        """
    else:
        prompt = f"""
        당신은 경제 분야에 정통한 경제 전문가입니다. 경제용어 {query}에 대해 자세히 설명하세요.
        1. '{query}'는 경제 신문에서 등장한 단어이다.
        2. 경제를 잘 모르는 초보자도 알기 쉽게 설명해야 한다.
        3. {query}가 경제 용어인지에 따라 출력 형식이 다르다. 각 경우에 따른 출력 형식은 다음과 같다.
            1) {query}가 경제용어가 아닌 경우(연관용어 출력하지 말 것):
            **{query}**
            이 용어는 본래 경제 용어가 아닙니다. ~~
            ---------------------------------------------------------
            2) {query}가 경제용어인 경우:
            **{query}**
            {query}에 대해 설명드리겠습니다. 
            **연관용어**            
            {related}
            """
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "system",
                "content":"당신은 경제 분야에 정통한 경제 전문가입니다. 경제용어를 초보자도 알기 쉽게 자세히 설명해주세요."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=1.0
    )

    return response.choices[0].message.content

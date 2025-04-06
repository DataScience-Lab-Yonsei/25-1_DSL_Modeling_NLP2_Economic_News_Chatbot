import sys
import streamlit as st
import pandas as pd
import random
from langchain_community.chat_models import ChatOpenAI
import subprocess

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

# ✅ 현재 디렉토리를 Python 모듈 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ 모듈 import
import news_summarizer
import economic_terms_RAG as term_explainer
import ner  
import ner_similarity_chatbot2 as ner_similarity
import company_info  
import news_recommender_final2 as news_recommender
import generate_quiz_module
import qa_prompt_module 

from news_summarizer import summarize_news
from ner import extract_ner_from_text 
from ner_similarity_chatbot2 import generate_weighted_query, find_similar_news 
from news_recommender_final2 import load_news_data, load_embeddings_and_index, recommend_similar_news
from generate_quiz_module import generate_quiz 

print("✅ chatbot.py 실행됨!")

# ✅ 데이터 로드
csv_file_path = "뉴스기사_최종.csv"
df_news = pd.read_csv(csv_file_path)

# ✅ FAISS 및 벡터 초기화 (사건 기반 추천용)
if "recommender_initialized" not in st.session_state:
    embedding_path = "news_embeddings.npy"
    id_path = "news_ids.npy"
    index_path = "news_faiss.index"

    news_recommender.load_news_data(csv_file_path)
    news_recommender.load_embeddings_and_index(embedding_path, id_path, index_path)

    st.session_state.recommender_initialized = True

# ✅ Streamlit UI 시작
st.title("📰 경제 뉴스 리딩메이트")

# 1️⃣ 사용자가 경제 카테고리 선택 (선택 없음 추가)
categories = ["선택 없음", "금융", "증권", "산업재계", "중기벤처", "부동산", "글로벌경제", "생활경제", "경제일반"]
selected_category = st.selectbox("📌 관심 있는 경제 카테고리를 선택하세요!", categories, index=0)

# ✅ 세션 초기화 (첫 실행 시)
if "random_articles" not in st.session_state:
    st.session_state.random_articles = None
if "selected_news" not in st.session_state:
    st.session_state.selected_news = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 대화 기록 저장

# ✅ 2️⃣ 올바른 카테고리를 선택해야 진행 가능
if selected_category == "선택 없음":
    st.warning("⚠️ 카테고리를 선택해주세요!")
    st.stop()

filtered_news = df_news[df_news["category"] == selected_category]

# ✅ 3️⃣ 헤드라인 랜덤 선택 
if "selected_category" not in st.session_state or st.session_state.selected_category != selected_category:
    st.session_state.selected_category = selected_category
    st.session_state.random_articles = filtered_news.sample(n=3) if not filtered_news.empty else None

    # 반드시 이전 상태 초기화
    for key in ["selected_news", "last_processed_news", "summary", "economic_terms", "ner_results", "chat_history", "quiz_text"]:
        st.session_state.pop(key, None)


# ✅ 4️⃣ 사용자가 헤드라인 선택
if st.session_state.random_articles is not None:
    st.write("🔎 관심 있는 뉴스를 선택하세요:")
    options = [f"{i+1}. {headline}" for i, headline in enumerate(st.session_state.random_articles["headline"])]
    selected_news = st.radio("뉴스 선택", options)

    if selected_news:
        selected_index = options.index(selected_news)
        new_selected = st.session_state.random_articles.iloc[selected_index].to_dict()

        # 새 뉴스가 기존과 다르면 상태 초기화
        if st.session_state.get("selected_news") != new_selected:
            st.session_state.selected_news = new_selected
            for key in ["summary", "economic_terms", "ner_results", "chat_history", "quiz_text"]:
                st.session_state.pop(key, None)
            st.session_state.last_processed_news = new_selected


# ✅ 5️⃣ 뉴스 정보 출력 (선택한 뉴스 유지)
if st.session_state.selected_news is not None:
    news_item = st.session_state.selected_news

    st.subheader("📌 선택한 뉴스")
    st.write(f"**📰 헤드라인:** {news_item['headline']}")
    st.write(f"🔗 [원문 보기]({news_item['link']})")

    # ✅ 6️⃣ 뉴스 본문 요약 (한 번만 실행)
    if "summary" not in st.session_state or st.session_state.selected_news["headline"] != news_item["headline"]:
        st.session_state.summary = summarize_news(news_item["content"])

    st.subheader("📌 뉴스 요약")
    st.write(st.session_state.summary)

    # ✅ 경제 용어 설명 (한 번만 실행)
    if "economic_terms" not in st.session_state or st.session_state.selected_news["headline"] != news_item["headline"]:
        st.session_state.economic_terms = term_explainer.extract_financial_terms(news_item["content"])

    st.subheader("📌 경제 용어 설명")
    for term in st.session_state.economic_terms:
        explanation, related_terms = term_explainer.recommend_similar_words(term)
        st.write(f"📖 **{term}**: {explanation}")

        if related_terms:
            related_str = ", ".join(related_terms)
            st.write(f"🔗 **연관 용어**: {related_str}")

    # # ✅ 8️⃣ Named Entity Recognition (NER) 적용
    # st.subheader("📌 Named Entity Recognition (NER) 결과")

    # ✅ NER 분석을 한 번만 실행하도록 session_state 활용
    if "ner_results" not in st.session_state or st.session_state.selected_news != news_item:
        selected_news_content = news_item["content"]
    
        # ✅ NER을 사용하여 기업명 추출
        company_ner, _ = extract_ner_from_text(selected_news_content)

        if company_ner:
            from collections import Counter

            # ✅ 기업별 등장 횟수 계산
            company_counts = Counter(company_ner)
            top_companies = company_counts.most_common(5)  # 상위 5개 기업

            # ✅ 상위 2개의 기업만 선택
            top_2_companies = top_companies[:2]

            # ✅ GPT에 전달할 기업 설명용 문자열 생성
            formatted_companies = "\n".join([f"- {company}: {count}회" for company, count in top_2_companies])

            # ✅ GPT를 활용하여 상위 2개 기업 설명 생성
            company_descriptions = company_info.get_company_info(formatted_companies)

            # ✅ 결과를 session_state에 저장 (한 번만 실행)
            st.session_state.ner_results = {"top_companies": top_2_companies, "descriptions": company_descriptions}
        else:
            st.session_state.ner_results = {"top_companies": [], "descriptions": "🚨 NER 결과가 없습니다. 뉴스 본문에서 기업명을 추출하지 못했습니다."}

    # ✅ 저장된 결과 불러오기
    ner_results = st.session_state.ner_results
    top_2_companies = ner_results["top_companies"]
    company_descriptions = ner_results["descriptions"]

    # # ✅ 결과 출력
    # if top_2_companies:
    #     for company, count in top_2_companies:
    #         st.write(f"🔹 **{company}** ({count}회 등장)")

    st.subheader("📌 기업 설명")
    st.write(company_descriptions)

    # # ✅ 9️⃣ 추가 질문 받기 (대화 히스토리 유지)    
    with st.expander("💬 추가 질문하기"):
        user_question = st.text_input("뉴스와 관련하여 궁금한 점을 질문하세요:")

        if st.button("질문하기") and user_question:
        # ✅ 경제 용어 dict 구성
            term_expl_dict = {
                term: term_explainer.recommend_similar_words(term)  # (설명, 연관용어)
                for term in st.session_state.economic_terms
            }

            # ✅ 프롬프트 구성
            prompt = qa_prompt_module.build_prompt(
                category=selected_category,
                news_content=news_item["content"],
                summary=st.session_state.summary,
                term_explanations=term_expl_dict,
                company_info=company_descriptions,
                user_question=user_question
            )

            # ✅ GPT 호출
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            response = completion.choices[0].message.content.strip()

            # ✅ 대화 기록 저장
            st.session_state.chat_history.append({"question": user_question, "answer": response})

    # ✅ 🔟 대화 히스토리 출력 (기존 질문+답변 유지)
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.subheader("🗂️ 대화 기록")
        for chat in st.session_state.chat_history:
            st.write(f"**❓ 질문:** {chat['question']}")
            st.write(f"**💡 답변:** {chat['answer']}")
            st.write("---")
    
    # ✅ 🔟 유사 뉴스 추천
    if st.button("질문 마치기"):
        st.subheader("🔎 유사한 뉴스 추천")

        # ✅ NER 기반 추천
        st.subheader("📌 NER 기반 추천") 
        top_2_company_names = [name for name, count in top_2_companies]
        query = generate_weighted_query(news_item["content"], top_2_company_names)
        print(query)
        recommended_news = find_similar_news(query)
        for idx, news in enumerate(recommended_news, 1):
            st.write(f"**{idx}. {news['headline']}**")
            st.write(f"🔗 [원문 보기]({news['link']})")

        # ✅ 사건 기반 추천
        st.subheader("📌 사건 기반 추천")

        news_title = news_item["headline"]
        news_content = news_item["content"]  # content 기준 버전이라면 content를 넣어야 함

        try:
            similar_news = recommend_similar_news(news_title, news_content, top_n=4)
            for idx, row in similar_news.iterrows():
                st.write(f"**{row['headline']}**")
                st.write(f"🔗 [원문 보기]({row['link']})")
        except Exception as e:
            st.error(f"❌ 유사 뉴스 추천 실패: {e}")
        
        # ✅ 퀴즈 생성
        st.markdown("---")
        st.subheader("🧠 뉴스 기반 경제 퀴즈 풀기")
        if "selected_news" in st.session_state:
            news_item = st.session_state.selected_news
            summary = st.session_state.summary
            company_info = st.session_state.ner_results["descriptions"]
            content = news_item["content"]

            # 퀴즈 자동 생성 및 저장
            with st.spinner("GPT가 퀴즈를 생성 중입니다..."):
                try:
                    quiz_text = generate_quiz(st.session_state, client)
                    st.session_state.quiz_text = quiz_text
                except Exception as e:
                    st.error(f"❌ 퀴즈 생성 중 오류 발생: {e}")


            # 퀴즈 출력
            if "quiz_text" in st.session_state:
                st.subheader("🧪 퀴즈 문제")

                quiz_blocks = st.session_state.quiz_text.strip().split("\n\n")
                for block in quiz_blocks:
                    lines = block.strip().split("\n")
                    if not lines or len(lines) < 3:
                        continue

                    question = lines[0]
                    options = [line for line in lines[1:] if line.startswith("①") or line.startswith("②") or line.startswith("③") or line.startswith("④")]
                    answer_line = next((line for line in lines if line.startswith("정답")), None)
                    explanation_line = next((line for line in lines if line.startswith("해설")), None)

                    with st.container():
                        st.markdown(f"**{question}**")
                        for opt in options:
                            st.markdown(f"- {opt}")
                        if answer_line:
                            st.success(answer_line)
                        if explanation_line:
                            st.info(explanation_line)
        else:
            st.warning("뉴스를 먼저 선택하고 요약, 기업 설명을 진행해주세요.")



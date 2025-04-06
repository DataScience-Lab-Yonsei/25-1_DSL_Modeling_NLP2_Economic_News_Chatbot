from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o")

summary_prompt = PromptTemplate(
    input_variables=["news_article"],
    template="""
    당신은 경제 지식이 부족한 사람들을 위해 경제 기사를 쉽게 풀어 설명해주는 요약 도우미입니다.
    아래 triple backticks으로 구분된 기사에 대한 요약을 제공하세요.
    
    요약은 다음과 같은 원칙을 따라야 합니다.
    1. 기사의 핵심내용을 포함하여야 한다.
    2. 어려운 경제 용어는 쉽게 풀어 설명해주어야 한다.
    3. 기사의 전체 내용을 3~5 문장 안팎으로 요약하여야 한다.
    4. 기사에서 어떤 사건이 발생하였고, 왜 발생하였는지 설명하여야 한다.

    ```
    {news_article}
    ```
    """
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# ✅ 요약 함수 정의
def summarize_news(news_article):
    """뉴스 기사를 요약하는 함수"""
    return summary_chain.run(news_article=news_article)

from dotenv import load_dotenv
import os
import openai

# ✅ 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_company_info(top_companies):
    """
    상위 2개 기업에 대한 설명을 GPT-4를 활용해 생성
    """
    if not top_companies:
        return []

    # 📌 GPT 프롬프트 구성
    prompt = f"""
    다음은 뉴스 기사에서 추출된 고유명사들과 그 등장 빈도입니다.

    이 중 실제 **기업, 금융 관련 기관, 정부 기관**만 골라 빈도수 순으로 정렬하고, 상위 2개의 기관만 선택해 주세요. 

    각 기관에 대한 설명은 다음 조건을 반드시 따르세요:

    1. 설명은 **300~400자 이내**로 작성합니다.
    2. 설명의 **문체는 '대학교 1학년이 이해할 수 있는 수준'**으로 쉽게 작성하며, 문장은 모두 **~습니다.** 형태로 끝맺습니다.
    3. **직책(예: 연구원, 회장, 대표, CEO, CTO 등), 대학, 지명, 일반 단어**는 제외하고, 기관만 선택합니다.
    4. 조사나 접미사가 붙은 경우(예: "삼성전자와")는 동일한 명칭으로 정리합니다.
    5. 동일한 기관이 여러 이름으로 등장하면 하나로 통합합니다 (예: '금감원' → '금융감독원').
    6. 기업의 브랜드도 기업으로 간주하고 설명해 주세요.
    7. 설명은 반드시 사실에 기반해야 하며, 추측은 금지입니다.

    📌 출력 형식 (반드시 이 형식을 따르세요):

    🏢 **[기관명]** : [300~400자 사이의 설명 문장, 문장 끝은 모두 ~습니다.]

    ---

    예시는 아래와 같으나, 실제 출력 시 포함하지 마세요. 참고용입니다.

    # 예시 시작 (출력 금지)
    🏢 **금융감독원** : 금융감독원은 은행, 보험사, 증권사 등 금융회사가 관련 법과 규정을 잘 지키는지 감독하는 국가 기관입니다. 국민들이 안심하고 금융서비스를 이용할 수 있도록 금융회사의 경영 상태를 점검하고, 문제가 있을 경우 조사와 제재를 진행합니다. 최근에는 가상자산, ESG(환경·사회·지배구조) 등 새로운 금융 환경에 대한 감독 기능도 강화하고 있습니다.
    # 예시 끝

    ---

    **고유명사 목록 (빈도 기준)**:  
    {top_companies}
    """

    # ✅ 최신 OpenAI API 사용 방식
    client = openai.OpenAI()  # ✅ 클라이언트 객체 생성 (V1 방식)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 금융 및 경제 기업 정보를 제공하는 AI입니다."},
            {"role": "user", "content": prompt}
        ]
    )

    # ✅ GPT 응답 반환
    company_info = response.choices[0].message.content.strip()
    return company_info

def generate_quiz(st_session_state, client):
    summarize = st_session_state.summary
    company_info = st_session_state.ner_results["descriptions"]
    content = st_session_state.selected_news["content"]

    use_company_info = bool(company_info.strip())

    prompt = f"""
    당신은 금융 지식이 부족한 초보자를 위해 뉴스 기반 퀴즈를 쉽고 재미있게 제공하는 AI 퀴즈 출제자입니다.

    아래 기사 요약, 기업 설명, 기사 본문을 참고하여 초보자도 풀 수 있는 경제 관련 퀴즈 3문제를 만들어주세요.

    퀴즈 구성 규칙:
    1. 기업 관련 퀴즈 1문제 → {'아래 기업 설명을 기반으로 하되, 내용이 없을 경우 기사 요약 또는 본문을 참고하여 출제하세요.' if not use_company_info else '반드시 아래 기업 설명을 바탕으로 출제하세요.'}
       - 기업 또는 기관의 성격, 역할, 소속 등 설명된 내용을 기반으로 해야 합니다.
    2. 경제 용어 퀴즈 1문제 → summarize 내용 속 "경제 개념 또는 경제 용어"만을 기반으로 출제하세요.
       - 반드시 경제 개념/용어를 다루어야 하며, 기술, 환경, 과학, 사회복지 등의 비경제 개념(예: 담수화, 스마트 농업 등)은 제외하세요.
    3. 기사 이해도 퀴즈 1문제 → content 기반
       - 숫자 문제는 피하고, 기관의 주요 정책 변화, 핵심 사건 중심으로 출제하세요.

    문제 형식:
    각 문제는 다음 포맷으로 작성합니다:

    [Q1] 문제 내용  
    ① 보기1  
    ② 보기2  
    ③ 보기3  
    ④ 보기4  
    정답: (정답 번호 또는 O/X)  
    해설: (초보자도 이해할 수 있는 한 문장 설명)

    ---

    📌 기업 설명:
    {company_info if use_company_info else '(해당 기사에 기업 설명이 없어 요약 또는 본문에서 대체 정보 사용)'}

    📌 기사 요약:
    {summarize}

    📌 기사 본문 (일부):
    {content[:800]} ...
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 금융 문맹도 재미있게 배울 수 있도록 도와주는 경제 퀴즈 마스터입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"퀴즈 생성 실패: {e}"

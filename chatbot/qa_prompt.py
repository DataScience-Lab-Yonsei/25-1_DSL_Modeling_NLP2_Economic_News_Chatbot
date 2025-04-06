def build_prompt(category: str, news_content: str, summary: str, term_explanations: dict, company_info: str,user_question: str) -> str:
    """
    사용자 질문과 뉴스 관련 정보를 바탕으로 GPT에게 전달할 프롬프트 생성
    """
    # 경제 용어 설명을 정리
    term_info = ""
    for term, (explanation, related_terms) in term_explanations.items():
        related_str = ", ".join(related_terms) if related_terms else "없음"
        term_info += f"\n- {term}: {explanation} (연관 용어: {related_str})"

    prompt = f"""
    너는 경제 뉴스를 쉽게 설명해주는 도우미 챗봇이야.
    사용자가 경제 개념를 잘 모르거나 뉴스 내용을 충분히 이해하지 못할 수 있기 때문에, 반드시 쉽게 설명해야 해.
    지금까지 뉴스에 대한 요약, 경제 용어 설명, 기업 관련 정보가 주어졌고,
    사용자가 추가 질문을 했어.

    답변 방식:
    1. 먼저, 질문한 대상이 경제 전반, 특히 사용자가 선택한 카테고리 안에서 어떤 의미를 갖는지 쉽게 설명해줘.
    2. 그 다음, 질문한 대상이 기사 본문의 내용과 직접적으로 관련이 있있다면 이 대상이 기사에서 어떤 의미로 사용되었는지 설명해줘. 만약 직접적인 관련이 없다면, 그 대상과 기사의 내용과의 관련성에 대해 설명해줘. 
    3. 복잡한 용어나 구조는 피하고, 경제를 잘 모르는 사람도 이해할 수 있게 쉽게 표현해줘.
    4. 사실에 기반해 설명하고, 추측이나 과장은 하지 마.

    아래는 지금까지의 뉴스 정보야:
    --------------------------
    [사용자가 선택한 카테고리]
    {category}

    [뉴스 본문]
    {news_content}

    [뉴스 요약]
    {summary}

    [경제 용어 설명]
    {term_info}

    [기업 설명 / NER 결과]
    {company_info}
    --------------------------

    아래는 사용자의 질문이야:
    "{user_question}"

    위 내용을 참고해, 흐름을 이어 자연스럽고 쉽게 설명해줘.

    """
    return prompt.format(
        category=category,
        summary=summary,
        term_info=term_info,
        company_info=company_info,
        user_question=user_question
    )


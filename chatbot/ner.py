# klue_roberta_ner.py
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import requests
from bs4 import BeautifulSoup
from collections import Counter

# 모델 및 토크나이저 초기화 (soddokayo/klue-roberta-base-ner 사용)
model_name = "soddokayo/klue-roberta-base-ner"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

def split_text_with_overlap(tokens, max_length=510, overlap=50):
    """
    긴 토큰 시퀀스를 슬라이딩 윈도우 기법으로 분할합니다.
    """
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length
        chunks.append(tokens[start:end])
        start += max_length - overlap  # 중복 영역 포함
    return chunks

def merge_wordpieces(entities):
    """
    WordPiece 토큰들을 단어 단위로 병합합니다.
    (특수 토큰(cls, sep)은 제외)
    """
    merged = []
    current_word, current_label = "", None
    for token, label in entities:
        if token in (tokenizer.cls_token, tokenizer.sep_token):
            continue
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word and current_label:
                merged.append((current_word, current_label))
            current_word = token
            current_label = label
    if current_word and current_label:
        merged.append((current_word, current_label))
    return merged

# BIO 스킴 기준 커스텀 라벨 매핑
custom_label_mapping = {
    "B-DT": "DT",  "I-DT": "DT",
    "B-LC": "LC",  "I-LC": "LC",
    "B-OG": "OG",  "I-OG": "OG",
    "B-PS": "PS",  "I-PS": "PS",
    "B-QT": "QT",  "I-QT": "QT",
    "B-TI": "TI",  "I-TI": "TI",
    "O": "O"
}

def normalize_entity(entity):
    """
    엔티티 문자열 끝에 붙은 한 글자 조사나 접미사를 제거합니다.
    (예: 'AI연구원이', 'AI연구원은' -> 'AI연구원')
    """
    particles = {'은', '는', '이', '가', '와', '과', '에', '의', '도', '만'}
    while entity and entity[-1] in particles:
        entity = entity[:-1]
    return entity

def extract_ner_from_text(article_text):
    """
    주어진 기사 전문에 대해 NER을 수행하고,
    회사(OG)와 인물(PS) 엔티티의 등장 횟수를 정규화하여 반환합니다.
    
    반환값: (company_ner, person_ner)
    - company_ner: dict, 예: {'LG': 7, 'AI연구원': 3, ...}
    - person_ner: dict, 예: {'이홍락': 3, '이상엽': 3, ...}
    """
    # 토큰화 (특수 토큰 제외)
    tokens = tokenizer.encode(article_text, add_special_tokens=False)
    token_chunks = split_text_with_overlap(tokens, max_length=510, overlap=50)
    
    extracted_entities = []
    for token_chunk in token_chunks:
        input_ids = [tokenizer.cls_token_id] + token_chunk + [tokenizer.sep_token_id]
        if len(input_ids) > 512:
            input_ids = input_ids[:512]
        inputs = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        tokens_decoded = tokenizer.convert_ids_to_tokens(inputs.squeeze().tolist())
        for token, pred in zip(tokens_decoded, predictions.squeeze().tolist()):
            label_str = model.config.id2label[pred]
            extracted_entities.append((token, label_str))
    
    merged_entities = merge_wordpieces(extracted_entities)
    
    final_entities = []
    for token, label in merged_entities:
        mapped_label = custom_label_mapping.get(label, label)
        # 인물(PS) 중 '기자'가 포함된 경우는 제외
        if mapped_label == "PS" and "기자" in token:
            continue
        if mapped_label != "O":
            final_entities.append((token, mapped_label))
    
    company_counter = Counter()
    person_counter = Counter()
    for token, label in final_entities:
        normalized = normalize_entity(token)
        if label == "OG":
            if len(normalized) > 1:
                company_counter[normalized] += 1
        elif label == "PS":
            if len(normalized) > 1:  # 한 글자인 인물 이름 제외
                person_counter[normalized] += 1
    
    company_ner = dict(company_counter)
    person_ner = dict(person_counter)
    return company_ner, person_ner



from openai import OpenAI
import requests
import json

# ✅ 1) OpenAI 임베딩 생성
client = OpenAI(api_key="api_key")  # ← 여기에 네 Key 입력

text = "도커로 OpenSearch를 배우는 강의"

embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
).data[0].embedding

print("생성된 벡터(앞부분):", embedding[:10])
print("벡터 길이:", len(embedding))

# ✅ 2) OpenSearch에 색인
OPENSEARCH_URL = "http://localhost:9200/my-ai-knn-index/_doc"

doc = {
    "title": text,
    "embedding": embedding
}

res = requests.post(
    OPENSEARCH_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(doc)
)

print("색인 결과:", res.json())

# ✅ 3) 검색용 벡터 생성
query_text = "OpenSearch 기초 강의"  # 검색하고 싶은 문장
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query_text
).data[0].embedding

print("검색용 벡터 길이:", len(query_embedding))
print("검색용 벡터 전체:", query_embedding)




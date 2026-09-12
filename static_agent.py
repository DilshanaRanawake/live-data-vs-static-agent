# static_agent.py
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
LLM_MODEL = "llama-3.2-3b-instruct"

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="../research-mcp-server/chroma_store")
collection = db.get_collection("research_papers")

def static_answer(question: str) -> str:
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer using only this context. If it's not sufficient, say so.

Context:
{context}

Question: {question}"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    q = "What accuracy did my thesis report for the ASL alphabet classification?"
    print(static_answer(q))
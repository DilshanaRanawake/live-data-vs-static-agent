# full_agent.py
import json
from openai import OpenAI
from live_search_tool import web_search
from sentence_transformers import SentenceTransformer
import chromadb

llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
LLM_MODEL = "llama-3.2-3b-instruct"

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="../research-mcp-server/chroma_store")
collection = db.get_collection("research_papers")

MAX_STEPS = 3

def search_papers(query: str) -> str:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    return "\n\n".join(results["documents"][0])

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def full_agent_answer(question: str) -> str:
    gathered = []   # list of (source_label, text) tuples

    for _ in range(MAX_STEPS):
        context_summary = "\n\n".join(f"[{src}]\n{text}" for src, text in gathered) or "(nothing gathered yet)"
        decide_prompt = f"""Question: {question}

Already gathered:
{context_summary}

You have two tools:
- search_papers: search the user's own static research papers (use this for anything about
  the papers' own content -- authors, methods, findings, numbers, specific claims made in them)
- web_search: search the live web (use this ONLY for questions about current, recent,
  2025/2026, or "latest" information that the static papers cannot possibly contain)

Default to search_papers first unless the question explicitly asks about something recent
or current that couldn't be in an older paper. Decide: do you have enough to answer
confidently, or do you need a tool?
Respond ONLY as JSON, no markdown fences:
{{"action": "search_papers" or "web_search" or "answer", "query": "search query if not answer, else empty"}}"""

        raw = call_llm(decide_prompt).strip().replace("```json", "").replace("```", "").strip()
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            decision = {"action": "answer", "query": ""}

        action = decision.get("action", "answer")
        if action == "answer":
            break
        elif action == "search_papers":
            result = search_papers(decision.get("query", question))
            # print(f"\n[DEBUG] search_papers retrieved:\n{result[:500]}\n")
            gathered.append(("FROM YOUR PAPERS", result[:800]))
        elif action == "web_search":
            result = web_search(decision.get("query", question))
            gathered.append(("FROM LIVE WEB SEARCH", result[:800]))
        else:
            break

    context = "\n\n".join(f"[{src}]\n{text}" for src, text in gathered) or "(no context gathered)"
    answer_prompt = f"""Answer the question below, directly and completely, using ONLY the
context provided. Each piece of context is labeled with where it came from. If the question
is about the papers' own content, trust [FROM YOUR PAPERS] over [FROM LIVE WEB SEARCH].

CRITICAL: If the context provides a specific, countable list of items (people, methods,
datasets, numbers), your answer must include exactly those items and no more -- never add
extra items from your own general knowledge, even if they seem plausible or well-known.
If the context does not contain enough information to answer the specific question asked,
say so honestly instead of guessing.

Context:
{context}

Question: {question}"""
    return call_llm(answer_prompt, temperature=0.1)

if __name__ == "__main__":
    for q in [
        "Who are the authors of the neural machine translation for Sinhala-Tamil thesis?",
        "According to the survey paper, why is Sinhala considered a resource-poor language for NLP?",
    ]:
        print(f"Q: {q}")
        print(full_agent_answer(q))
        print("---")
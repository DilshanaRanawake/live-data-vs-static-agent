# live_search_tool.py
from tavily import TavilyClient
import os
import time

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def web_search(query: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            results = tavily.search(query=query, max_results=3)
            return "\n\n".join(
                f"{r['title']}: {r['content'][:300]}" for r in results["results"]
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [web_search retry {attempt + 1}/{max_retries} after error: {e}]")
                time.sleep(2)
            else:
                return f"(web search failed after {max_retries} attempts: {e})"

if __name__ == "__main__":
    print(web_search("latest CVPR 2026 accepted papers gesture recognition"))
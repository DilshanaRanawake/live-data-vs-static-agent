# peek.py
import chromadb

c = chromadb.PersistentClient(path='../research-mcp-server/chroma_store')
col = c.get_collection('research_papers')

for paper_id in ['2', '3', '4', '6', '7', '8', '10', '59', '08818760 s-t']:
    results = col.get(where={'paper': paper_id}, limit=1)
    print(f"--- {paper_id} ---")
    if results['documents']:
        print(results['documents'][0][:400])
    else:
        print("(no chunks found)")
    print()
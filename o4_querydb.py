import os
import argparse
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL

from openai import OpenAI
from chromadb.config import Settings
from chromadb import PersistentClient

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

load_dotenv()

client_ai = OpenAI()

client = PersistentClient(path=CHROMA_DB_FULL_PATH)
print([c.name for c in client.list_collections()])
collection = client.get_collection("cpp_code")

def answer_query(query, k=5, metadata_filter=None):
    resp = client_ai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    q_emb = resp.data[0].embedding    

# build query args, only include `where` if a filter was passed
    query_args = {
        "query_embeddings": [q_emb],
        "n_results": k
    }
    if metadata_filter:
        query_args["where"] = metadata_filter

    # 2) Retrieve top-k code chunks
    results = collection.query(**query_args)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 3) Build context
    context = "\n\n".join(
       f"```cpp\n{doc}\n```\n— {m['file_path']}:{m['start_line']}-{m['end_line']}"
       for doc, m in zip(docs, metas)
    )

    # 4) Call ChatCompletion
    prompt = f"""
You are a C/C++ expert assistant. Use the following code context to answer the user’s question. Only reference code by its file path and line numbers.

Context:
{context}

Question:
{query}
"""
    chat = client_ai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":prompt}]
    )
    return chat.choices[0].message.content

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, nargs="?",
                       default="",
                       help="The query text.")
    args = parser.parse_args()
# "Explain nb3FixedConstrait class"  # "What is Convex Decomposition function?" 
# "In b3FixedConstraint constructor please explain frameInA argument"  
    if not args.query_text:
        query_text = "Show constructor of nb3FixedConstrait class"
    else:
        query_text = args.query_text

    answer = answer_query(query_text)
    print(answer)

if __name__ == "__main__":
    cli_main()

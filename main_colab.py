import argparse
from data_utils import load_and_chunk_csv
from embedder import embed_texts
from retriever import build_faiss_index
from reranker import load_reranker
from chat import run_chat_loop, run_single_query

def main():
    print("[System] Loading and chunking CSV...")
    chunk_df = load_and_chunk_csv("NZ_repository_scrapped_1.csv", chunk_size=500)

    print("[System] Generating embeddings...")
    chunk_texts = chunk_df['chunk'].iloc[:1000].tolist()
    embeddings = embed_texts(chunk_texts)

    print("[System] Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("[System] Loading reranker...")
    rerank_tokenizer, rerank_model = load_reranker()

    run_chat_loop(chunk_df, index)

if __name__ == "__main__":
    main()
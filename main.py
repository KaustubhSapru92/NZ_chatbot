import argparse
from data_utils import load_and_chunk_csv
from embedder import embed_texts
from retriever import build_faiss_index
from reranker import load_reranker
from chat import run_chat_loop, run_single_query

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help="Ask a single question")
    parser.add_argument('--interactive', nargs='?', const='1000', help="Start interactive chat. Optionally provide number of chunks or 'all'")

    args = parser.parse_args()

    print("[System] Loading and chunking CSV...")
    chunk_df = load_and_chunk_csv("NZ_repository_scrapped_1.csv", chunk_size=500)

   # Determine how many chunks to use
    if args.interactive:
        if args.interactive == 'all':
            chunk_texts = chunk_df['chunk'].tolist()
        else:
            try:
                n_chunks = int(args.interactive)
                chunk_texts = chunk_df['chunk'].iloc[:n_chunks].tolist()
            except ValueError:
                print("❗ Invalid value for --interactive. Use an integer or 'all'.")
                return
    elif args.query:
        chunk_texts = chunk_df['chunk'].iloc[:1000].tolist()  # Default for single query
    else:
        print("❗ Use --query or --interactive to run the chatbot.")
        return

    #print("[System] Generating embeddings...")
    #chunk_texts = chunk_df['chunk'].iloc[:1000].tolist()
    embeddings = embed_texts(chunk_texts)

    print("[System] Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("[System] Loading reranker...")
    rerank_tokenizer, rerank_model = load_reranker()

    if args.query:
         run_single_query(args.query, chunk_df, index, rerank_tokenizer, rerank_model)
    elif args.interactive:
        run_chat_loop(chunk_df, index)
    else:
        print("❗ Use --query or --interactive to run the chatbot.")

if __name__ == "__main__":
    main()

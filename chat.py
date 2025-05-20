from embedder import embed_texts
from retriever import retrieve_chunks
from reranker import rerank
from llm_client import generate_answer

def run_single_query(query, chunk_df, index):
    query_vec = embed_texts([query])  # Uses global tokenizer and model
    retrieved, _ = retrieve_chunks(index, query_vec, chunk_df)
    passages = retrieved['chunk'].tolist()
    top_passages = rerank(query, passages, top_n=1)  # Uses global reranker
    answer = generate_answer(top_passages, query)
    print(f"\nAnswer:\n{answer}\n")


def run_chat_loop(chunk_df, index):
    print("🟢 Chatbot is ready. Type your questions or 'exit'.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        run_single_query(user_input, chunk_df, index)


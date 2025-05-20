# NZ_chatbot

Steps to be flowed in the respective order for execution in a CLI:
1. Select a suitable directory to run the chatbot.
2. Clone this GitHub repository
3. python -m venv myenv  #you can give an appropriate name to your environment
4. Activate your environment. source myenv/bin/activate.
5. Time to install the requirements. pip install -r requirements.txt
6. Make we have all the information in our knowledge repository. Let's scrap the text fields if that are missing. Here is how to do it:
     i) Run the file scrap.py followed by the path of the knowledge repository. Like this :
    ii) python scrap.py /mnt/e/Downloads/NZ_AI_CHALLENGE_2025/NZ_AI_CHALLENGE_2025/wikipedia_sample_with_10_percent_batch_text_null.csv
   iii) This will save a new csv file with the scrapped details in the same directory with the name 'NZ_repository_scrapped_1.csv'.
    iv) Make sure this new file is in the same directory as the code base. I have already scrapped the missing text fields so I have already have a NZ_repository_scrapped_1.csv file in my directory.
        You can use the same file that I have created.
7. The rest of the code is modularized. All you need to run is : python main.py --interactive
8. As the knowledge repository is huge, there is additional optional parameter that can be passed after --interactive. It determines how many chunks of the data will be considered.
     i) if you want all the data to be considered, pass the argument 'all' like this : python main.py --interactive all
        This is time consuming and it takes about 3 hrs to convert the entire repository into embeddings.
    ii) To just test the chatbot we can start of with a small number like 50 or 100, like this:  python main.py --interactive 50
        This will take 1-2 mins to process.
   iii) If no argument is passed, the default value is set to a 1000, which take approximately 4 mins to process.


How semantic understanding and similarity matching are implemented
Semantic understanding is achieved using pre-trained transformer-based embedding models (likely from sentence-transformers) that convert text chunks into dense vector representations (embed_texts). These embeddings capture the meaning of the text beyond keyword matching.
Similarity matching is performed in two stages:

1. Retrieval:
FAISS is used to build a vector index from chunk embeddings (build_faiss_index). At query time, the input is embedded and nearest neighbor search retrieves the most semantically similar chunks (retrieve_chunks).

2. Re-ranking:
The retrieved chunks are further refined using a cross-encoder re-ranker (rerank function using BAAI/bge-reranker-base). This model scores query–passage pairs for relevance and reorders the top candidates based on deeper contextual alignment.

Challenges faced and how they were resolved
1. Slow Initialization (Latency)
Issue: Long startup times due to computing embeddings every time the script runs.

Resolution: Introduced optional arguments to control the number of chunks processed (e.g. --interactive 500 or --interactive all), reducing compute time during development and testing. Further optimization may involve caching or persisting embeddings for reuse.

2. Re-ranker Integration
Issue: The re-ranker model uses a cross-encoder architecture, which is more accurate but also computationally intensive.

Resolution: Applied re-ranking only on the top few retrieved results, significantly reducing overhead while preserving accuracy.

3. Dynamic Input Handling
Issue: Supporting multiple use cases such as single queries, full interactive chat, and variable dataset sizes required flexible command-line control.

Resolution: Used argparse to accept optional parameters and conditionally handle chunk sizes and input modes, enabling more adaptable workflows.

4. Hardware Limitations (GPU)
Issue: The local GPU was not functioning, which prevented full optimization for GPU acceleration.

Resolution: Despite the hardware constraint, the system was built to be modular and compatible with GPU support. Models and tensors were kept device-agnostic (e.g., with future to(device) calls in mind), ensuring that performance can be scaled when GPU support is restored or deployed to cloud environments.

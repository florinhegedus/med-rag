from langchain_openai.embeddings import OpenAIEmbeddings


def get_embedding_function():
    embeddings = OpenAIEmbeddings()
    return embeddings
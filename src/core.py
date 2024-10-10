from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_function(embedding_model: str = None) -> OllamaEmbeddings:
    if embedding_model is None:
        raise ValueError('No embedding model provided')
    return OllamaEmbeddings(model=embedding_model)

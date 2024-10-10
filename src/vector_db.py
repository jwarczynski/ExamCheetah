from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from docs_pipeline import create_chunk_ids

_CHROMA_PATH = 'chroma'


def add_to_chroma(chunks: List[Document], embedding_function: OllamaEmbeddings) -> None:
    db = Chroma(
        persist_directory=_CHROMA_PATH,
        embedding_function=embedding_function,
    )

    chunk_with_ids = create_chunk_ids(chunks)

    existing_items = db.get(include=[])
    exisitng_ids = set(existing_items["ids"])
    print(f'Number of existing items: {len(existing_items)}')

    new_chunks = []
    for chunk in chunk_with_ids:
        if chunk.metadata['id'] not in exisitng_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f'👉 Adding {len(new_chunks)} new chunks to Chroma')
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

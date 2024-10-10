from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document_from_pdf(pdf_dir_path: str) -> List[Document]:
    loader = PyPDFDirectoryLoader(pdf_dir_path)
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def create_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_idx = 0
    for chnk in chunks:
        source = chnk.metadata.get('source')
        page = chnk.metadata.get('page')
        page_id = f'{source}:{page}'
        if page_id != last_page_id:
            current_chunk_idx = 0
            last_page_id = page_id
        else:
            current_chunk_idx += 1
        chnk.metadata['id'] = f'{page_id}:{current_chunk_idx}'

    return chunks

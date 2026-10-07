import os
import uuid 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

#CHROMA_DIR = "vector_db"
#COLLECTION_NAME = "meeting_transcript"
COLLECTION_PREFIX = "transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def new_collection_name() -> str:
    return f"{COLLECTION_PREFIX}_{uuid.uuid4().hex}"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript: str, collection_name: str | None = None) -> Chroma:
    collection_name = collection_name or new_collection_name()
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    
    # Empty this collection first, so a new video replaces the old one
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    ).delete_collection()
    
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=collection_name,
        #persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


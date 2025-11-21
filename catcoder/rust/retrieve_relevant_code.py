import os
import torch
import warnings

import chromadb

from abc import ABC
from hashlib import sha1
from typing import overload, Callable, List, TypeVar, Optional, Any
from functools import lru_cache

from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.retrievers.ensemble import EnsembleRetriever

warnings.filterwarnings("ignore")

T = TypeVar('T')

class RustLoader(BaseLoader):
    def __init__(self, path: str, data) -> None:
        self.path = path
        self.data = data
        assert path.endswith('.rs'), f'{path} is not a Rust file'
    
    def load(self) -> List[Document]:
        with open(self.path, 'r') as f:
            lines = f.readlines()
        if self.path.endswith(self.data['path']):
            start, end = self.data['lines'][0], self.data['lines'][3]
            lines = lines[:start-1] + lines[end:]
        content = ''.join(lines)
        return [Document(page_content=content, metadata={'source': self.path})]

@lru_cache
def get_chroma_client(persist_directory):
    return chromadb.PersistentClient(persist_directory)


class CachedChroma(Chroma, ABC):
    @classmethod
    def from_documents_with_cache(
            cls,
            persist_directory: str,
            documents: List[Document],
            embedding: Optional[Embeddings] = None,
            ids: Optional[List[str]] = None,
            collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
            client_settings: Optional[chromadb.config.Settings] = None,
            **kwargs: Any,
    ) -> Chroma:
        client = get_chroma_client(persist_directory)
        collection_names = [c.name for c in client.list_collections()]

        if collection_name in collection_names:
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=persist_directory,
                client_settings=client_settings,
            )

        return Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            **kwargs
        )

@lru_cache
def get_embedding_model(model_path, embedding_cache_dir, namespace, multi_process=False):
    base_embedding = HuggingFaceEmbeddings(
        model_name=model_path, multi_process=multi_process,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    )
    embedding_cache_storage = LocalFileStore(embedding_cache_dir)
    embedding_model = CacheBackedEmbeddings.from_bytes_store(
        base_embedding, embedding_cache_storage, namespace=namespace, query_embedding_cache=True
    )
    return embedding_model

class RustProjectIndexer:
    def __init__(self, path: str, data, *,
                 chunk_size=1000,
                 chunk_overlap=0,
                 persist_directory='./.rag_cache',
                 embedding_model_path=None,
                 embedding_cache_dir='./.embedding_cache',
                 search_type='similarity',
                 **kwargs):
        self.path = path
        self.data = data
        self.embedding_model_path = embedding_model_path
        assert os.path.isdir(path), f'{path} is not a directory'
        self.embedding_model = get_embedding_model(embedding_model_path, embedding_cache_dir, self.namespace)
        self.loader = DirectoryLoader(self.path, glob='**/*.rs', loader_cls=RustLoader, 
                                      loader_kwargs={'data': data}, recursive=True, 
                                      exclude=['benches/*'], use_multithreading=True,
                                      max_concurrency=32)
        self.splitter = RecursiveCharacterTextSplitter.from_language(Language.RUST, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.docs = self.loader.load_and_split(self.splitter)
        vector_indices = CachedChroma.from_documents_with_cache(
            persist_directory, self.docs, self.embedding_model, collection_name=self.collection_name, **kwargs
        ).as_retriever(search_type=search_type, search_kwargs={'k': 8})
        bm25_indices = BM25Retriever.from_documents(self.docs)
        bm25_indices.k = 8
        self.indices = EnsembleRetriever(retrievers=[vector_indices, bm25_indices], weights=[0.7, 0.3])
    
    @property
    def namespace(self):
        return sha1(self.embedding_model_path.encode()).hexdigest()
    
    @property
    def collection_name(self):
        return sha1(self.path.encode()).hexdigest()
    
    @overload
    def search(self, query, *, 
               filter_fn: Callable[[Document], bool]=None, 
               map_fn: None=None) -> List[Document]:
        ...

    @overload
    def search(self, query, *,
               filter_fn: Callable[[Document], bool]=None, 
               map_fn: Callable[[Document], T]=None) -> List[T]:
        ...
    
    def search(self, query, *, filter_fn=None, map_fn=None):
        results = self.indices.invoke(query)
        if filter_fn is not None:
            results = list(filter(filter_fn, results))
        if map_fn is not None:
            results = list(map(map_fn, results))
        return results

def run_rag(data, embedding_model_path):
    path = f'crates/{data["package"]}'

    def drop_ground_truth(doc: Document):
        return f'fn {data["focal_fn_name"]}' not in doc.page_content

    def to_context(doc: Document):
        src = os.path.relpath(doc.metadata['source'], path)
        return f'/// {src}\n' + doc.page_content

    indexer = RustProjectIndexer(path, data, 
                                 embedding_model_path=embedding_model_path,
                                 persist_directory='./.rag_cache')
    results = indexer.search(data['hint'] + '\n' + data['focal_fn_signature'], filter_fn=drop_ground_truth, map_fn=to_context)
    return list(set(results))

def repocoder_rag(data, embedding_model_path, ref_code):
    path = f'crates/{data["package"]}'

    def drop_ground_truth(doc: Document):
        return f'fn {data["focal_fn_name"]}' not in doc.page_content

    def to_context(doc: Document):
        src = os.path.relpath(doc.metadata['source'], path)
        return f'/// {src}\n' + doc.page_content

    indexer = RustProjectIndexer(path, data, 
                                 embedding_model_path=embedding_model_path,
                                 persist_directory='./.rag_cache')
    results = indexer.search(ref_code, filter_fn=drop_ground_truth, map_fn=to_context)
    return list(set(results))

from .vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time
import psycopg

class pgVectorDocumentManager(DocumentManager):
    def __init__(self, client, index_name, embedding):
        pass
    def check_neo4j_version(self):
        pass
    def get_index_info(self):
        pass
    def _embed_doc(self, texts) -> List[float]:
        pass
    def search(self, query, k=10, **kwargs):
        pass
    def delete(self, ids=None, filters=None, **kwargs):
        pass
class pgVectorIndexManager:
    def __init__(self, client):
        pass
    def list_indexes(self):
        pass
    def delete_index(self, index_name):
        pass
    def get_index(self, index_name: str) -> Dict:
        pass
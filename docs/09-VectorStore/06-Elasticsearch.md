<style>
.custom {
    background-color: #008d8d;
    color: white;
    padding: 0.25em 0.5em 0.25em 0.5em;
    white-space: pre-wrap;       /* css-3 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;
}

pre {
    background-color: #027c7c;
    padding-left: 0.5em;
}

</style>

# Elasticsearch

- Author: [liniar](https://github.com/namyoungkim)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/06-Elasticsearch.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/06-Elasticsearch.ipynb)


## Overview  
- This tutorial is designed for beginners to get started with Elasticsearch and its integration with LangChain.
- You’ll learn how to set up the environment, prepare data, and explore advanced search features like hybrid and semantic search.
- By the end, you’ll be equipped to use Elasticsearch for powerful and intuitive search applications.

### Table of Contents  

- [Overview](#overview)  
- [Environment Setup](#environment-setup)  
- [Elasticsearch Setup](#elasticsearch-setup)  
- [Introduction to Elasticsearch](#introduction-to-elasticsearch)  
- [ElasticsearchManager](#elasticsearchmanager)  
- [Data Preparation for Tutorial](#data-preparation-for-tutorial)  
- [Initialization](#initialization)  
- [DB Handling](#db-handling)  
- [Advanced Search](#advanced-search)  

### References
- [LangChain VectorStore Documentation](https://python.langchain.com/docs/how_to/vectorstores/)
- [LangChain Elasticsearch Integration](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/)
- [Elasticsearch Official Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/index.html)  
- [Elasticsearch Vector Search Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
----

## Environment Setup  

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.  

**[Note]**  
- `langchain-opentutorial` is a package that provides a set of **easy-to-use environment setup,** **useful functions,** and **utilities for tutorials.**  
- You can check out the [`langchain-opentutorial` ](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.  


### 🛠️ **The following configurations will be set up**  

- **Jupyter Notebook Output Settings**
    - Display standard error ( `stderr` ) messages directly instead of capturing them.  
- **Install Required Packages** 
    - Ensure all necessary dependencies are installed.  
- **API Key Setup** 
    - Configure the API key for authentication.  
- **PyTorch Device Selection Setup** 
    - Automatically select the optimal computing device (CPU, CUDA, or MPS).
        - `{"device": "mps"}` : Perform embedding calculations using **MPS** instead of GPU. (For Mac users)
        - `{"device": "cuda"}` : Perform embedding calculations using **GPU.** (For Linux and Windows users, requires CUDA installation)
        - `{"device": "cpu"}` : Perform embedding calculations using **CPU.** (Available for all users)
- **Embedding Model Local Storage Path** 
    - Define a local path for storing embedding models.  

## Elasticsearch Setup
- In order to use the Elasticsearch vector search you must install the langchain-elasticsearch package.

### 🚀 Setting Up Elasticsearch with Elastic Cloud (Colab Compatible)
- Elastic Cloud allows you to manage Elasticsearch seamlessly in the cloud, eliminating the need for local installations.
- It integrates well with Google Colab, enabling efficient experimentation and prototyping.


### 📚 What is Elastic Cloud?  
- **Elastic Cloud** is a managed Elasticsearch service provided by Elastic.  
- Supports **custom cluster configurations** and **auto-scaling.** 
- Deployable on **AWS**, **GCP**, and **Azure.**  
- Compatible with **Google Colab,** allowing simplified cloud-based workflows.  

### 📌 Getting Started with Elastic Cloud  
1. **Sign up for Elastic Cloud’s Free Trial.**  
    - [Free Trial](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=documentation)
2. **Create an Elasticsearch Cluster.**  
3. **Retrieve your Elasticsearch URL** and **Elasticsearch API Key** from the Elastic Cloud Console.  
4. Add the following to your `.env` file
    > ```
    > ES_URL=https://my-elasticsearch-project-abd...:123
    > ES_API_KEY=bk9X...
    > ```
---

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain-core",
        "langchain_huggingface",
        "langchain_elasticsearch",
        "langchain_text_splitters",
        "elasticsearch",
        "python-dotenv",
        "uuid",
        "torch",
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "Elasticsearch",
            "HUGGINGFACEHUB_API_TOKEN": "",
            "ES_URL": "",
            "ES_API_KEY": "",
        }
    )
```

```python
# Automatically select the appropriate device
import torch
import platform


def get_device():
    if platform.system() == "Darwin":  # macOS specific
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Using MPS (Metal Performance Shaders) on macOS")
            return "mps"
    if torch.cuda.is_available():
        print("✅ Using CUDA (NVIDIA GPU)")
        return "cuda"
    else:
        print("✅ Using CPU")
        return "cpu"


# Set the device
device = get_device()
print("🖥️ Current device in use:", device)
```

<pre class="custom">✅ Using MPS (Metal Performance Shaders) on macOS
    🖥️ Current device in use: mps
</pre>

```python
# Embedding Model Local Storage Path
import os
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Set the download path to ./cache/
os.environ["HF_HOME"] = "./cache/"
```

## Introduction to Elasticsearch
- Elasticsearch is an open-source, distributed search and analytics engine designed to store, search, and analyze both structured and unstructured data in real-time.

### 📌 Key Features  
- **Real-Time Search:** Instantly searchable data upon ingestion  
- **Large-Scale Data Processing:** Efficient handling of vast datasets  
- **Scalability:** Flexible scaling through clustering and distributed architecture  
- **Versatile Search Support:** Keyword search, semantic search, and multimodal search  

### 📌 Use Cases  
- **Log Analytics:** Real-time monitoring of system and application logs  
- **Monitoring:** Server and network health tracking  
- **Product Recommendations:** Behavior-based recommendation systems  
- **Natural Language Processing (NLP):** Semantic text searches  
- **Multimodal Search:** Text-to-image and image-to-image searches  

### 🧠 Vector Database Functionality in Elasticsearch  
- Elasticsearch supports vector data storage and similarity search via **Dense Vector Fields.** As a vector database, it excels in applications like NLP, image search, and recommendation systems.

### 📌 Core Vector Database Features  
- **Dense Vector Field:** Store and query high-dimensional vectors  
- **KNN (k-Nearest Neighbors) Search:** Find vectors most similar to the input  
- **Semantic Search:** Perform meaning-based searches beyond keyword matching  
- **Multimodal Search:** Combine text and image data for advanced search capabilities  

### 📌 Vector Search Use Cases  
- **Semantic Search:** Understand user intent and deliver precise results  
- **Text-to-Image Search:** Retrieve relevant images from textual descriptions  
- **Image-to-Image Search:** Find visually similar images in a dataset  

### 🔗 Official Documentation Links  
- [Elasticsearch Official Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/index.html)  
- [Elasticsearch Vector Search Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)  

Elasticsearch goes beyond traditional text search engines, offering robust vector database capabilities essential for NLP and multimodal search applications. 🚀

---

## ElasticsearchManager
- `Purpose:` Simplifies interactions with Elasticsearch, allowing easy management of indices and documents through user-friendly methods.
- `Core Features` 
	- `Index management:` create, delete, and manage indices.
	- `Document operations:` upsert, retrieve, search, and delete documents.
	- `Bulk and parallel operations:` perform upserts in bulk or in parallel for high performance.

### Methods and Parameters

1. `__init__` 
	- Role: Initializes the ElasticsearchManager instance and connects to the Elasticsearch cluster.
	- Parameters
		- `es_url` (str): The URL of the Elasticsearch host (default: "http://localhost:9200").
		- `api_key` (Optional[str]): The API key for authentication (default: None).
	- Behavior
		- Establishes a connection to Elasticsearch.
		- Tests the connection using ping() and raises a ConnectionError if it fails.
	- Usage Example
		>```python
		>es_manager = ElasticsearchManager(es_url="http://localhost:9200")
		>```

2. `create_index` 
	- Role: Creates an Elasticsearch index with optional mappings and settings.
	- Parameters
		- `index_name` (str): The name of the index to create.
		- `mapping` (Optional[Dict]): A dictionary defining the index structure (field types, properties, etc.).
		- `settings` (Optional[Dict]): A dictionary defining index settings (e.g., number of shards, replicas).
	- Behavior
		- Checks if the index exists.
		- If the index does not exist, creates it using the provided mappings and settings.
	- Returns: A string message indicating success or failure.
	- Usage Example
		>```python
		>mapping = {"properties": {"name": {"type": "text"}}}
		>settings = {"number_of_shards": 1}
		>es_manager.create_index("my_index", mapping=mapping, settings=settings)
		>```

3. `delete_index` 
	- Role: Deletes an Elasticsearch index if it exists.
	- Parameters
		- `index_name` (str): The name of the index to delete.
	- Behavior
		- Checks if the index exists.
		- Deletes the index if it exists.
	- Returns: A string message indicating success or failure.
	- Usage Example
		>```python
		>es_manager.delete_index("my_index")
		```

4. `get_document` 
	- Role: Retrieves a single document by its ID.
	- Parameters
		- `index_name` (str): The name of the index to retrieve the document from.
		- `document_id` (str): The ID of the document to retrieve.
	- Behavior
		- Fetches the document using its ID.
		- Returns the _source field of the document (its contents).
	- Returns: The document contents (Dict) if found, otherwise None.
	- Usage Example
		>```python
		>document = es_manager.get_document("my_index", "1")
		>```

5. `search_documents` 
	- Role: Searches for documents in an index based on a query.
	- Parameters
		- `index_name` (str): The name of the index to search.
		- `query` (Dict): A query in Elasticsearch DSL format.
	- Behavior
		- Executes the query against the specified index.
		- Returns the _source field of all matching documents.
	- Returns: A list of matching documents (List[Dict]).
	- Usage Example
		>```python
		>query = {"match": {"name": "John"}}
		>results = es_manager.search_documents("my_index", query=query)
		>```
		
6. `upsert_document` 
	- Role: Inserts or updates a document by its ID.
	- Parameters
		- `index_name` (str): The index to perform the upsert on.
		- `document_id` (str): The ID of the document to upsert.
		- `document` (Dict): The content of the document.
	- Behavior
		- Updates the document if it exists or creates it if it does not.
		- Returns: The Elasticsearch response (Dict).
	- Usage Example
		>```python
		>document = {"name": "Alice", "age": 30}
		>es_manager.upsert_document("my_index", "1", document)
		>```

7. `bulk_upsert` 
	- Role: Performs a bulk upsert operation for multiple documents.
	- Parameters
		- `documents` (List[Dict]): A list of documents for the bulk operation.
			- Each document should specify _index, _id, _op_type, and doc_as_upsert.
	- Behavior
		- Uses Elasticsearch’s bulk API to upsert multiple documents in a single request.
	- Usage Example
		>```python
		>docs = [
		>	{"_index": "my_index", "_id": "1", "_op_type": "update", "doc": {"name": "Alice"}, "doc_as_upsert": True},
		>	{"_index": "my_index", "_id": "2", "_op_type": "update", "doc": {"name": "Bob"}, "doc_as_upsert": True}
		>]
		>es_manager.bulk_upsert(docs)
		>```

8. `parallel_bulk_upsert` 
	- Role: Performs a parallelized bulk upsert operation for large datasets.
	- Parameters
		- `documents` (List[Dict]): A list of documents for bulk upserts.
		- `batch_size` (int): Number of documents per batch (default: 100).
		- `max_workers` (int): Number of threads to use for parallel processing (default: 4).
	- Behavior
		- Splits the documents into batches and processes them in parallel using threads.
	- Usage Example
		>```python
		>es_manager.parallel_bulk_upsert(docs, batch_size=50, max_workers=4)
		>```

9. `delete_document` 
	- Role: Deletes a single document by its ID.
	- Parameters
		- `index_name` (str): The index containing the document.
		- `document_id` (str): The ID of the document to delete.
	- Behavior
		- Deletes the specified document using its ID.
	- Returns: The Elasticsearch response (Dict).
	- Usage Example
		>```python
		>es_manager.delete_document("my_index", "1")
		>```

10. `delete_by_query` 
	- Role: Deletes all documents that match a query.
	- Parameters
		- `index_name` (str): The index to delete documents from.
		- `query` (Dict): The query defining the documents to delete.
	- Behavior
		- Uses Elasticsearch’s delete_by_query API to remove documents matching the query.
	- Returns: The Elasticsearch response (Dict).
	- Usage Example
		>```python
		>delete_query = {"match": {"status": "inactive"}}
		>es_manager.delete_by_query("my_index", query=delete_query)
		>```

### Conclusion
- This class provides a robust and user-friendly interface to manage Elasticsearch operations.
- It encapsulates common tasks like creating indices, searching for documents, and performing upserts, making it ideal for use in data management pipelines or applications.

```python
from typing import Optional, Dict, List, Generator
from elasticsearch import Elasticsearch, helpers
from concurrent.futures import ThreadPoolExecutor


class ElasticsearchManager:
    def __init__(
        self, es_url: str = "http://localhost:9200", api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the ElasticsearchManager with a connection to the Elasticsearch instance.

        Parameters:
            es_url (str): URL of the Elasticsearch host.
            api_key (Optional[str]): API key for authentication (optional).
        """
        # Initialize the Elasticsearch client
        if api_key:
            self.es = Elasticsearch(es_url, api_key=api_key, timeout=120, retry_on_timeout=True)
        else:
            self.es = Elasticsearch(es_url, timeout=120, retry_on_timeout=True)

        # Test connection
        if self.es.ping():
            print("✅ Successfully connected to Elasticsearch!")
        else:
            raise ConnectionError("❌ Failed to connect to Elasticsearch.")

    def create_index(
        self,
        index_name: str,
        mapping: Optional[Dict] = None,
        settings: Optional[Dict] = None,
    ) -> str:
        """
        Create an Elasticsearch index with optional mapping and settings.

        Parameters:
            index_name (str): Name of the index to create.
            mapping (Optional[Dict]): Mapping definition for the index.
            settings (Optional[Dict]): Settings definition for the index.

        Returns:
            str: Success or warning message.
        """
        try:
            if not self.es.indices.exists(index=index_name):
                body = {}
                if mapping:
                    body["mappings"] = mapping
                if settings:
                    body["settings"] = settings
                self.es.indices.create(index=index_name, body=body)
                return f"✅ Index '{index_name}' created successfully."
            else:
                return f"⚠️ Index '{index_name}' already exists. Skipping creation."
        except Exception as e:
            return f"❌ Error creating index '{index_name}': {e}"

    def delete_index(self, index_name: str) -> str:
        """
        Delete an Elasticsearch index if it exists.

        Parameters:
            index_name (str): Name of the index to delete.

        Returns:
            str: Success or warning message.
        """
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                return f"✅ Index '{index_name}' deleted successfully."
            else:
                return f"⚠️ Index '{index_name}' does not exist."
        except Exception as e:
            return f"❌ Error deleting index '{index_name}': {e}"

    def get_document(self, index_name: str, document_id: str) -> Optional[Dict]:
        """
        Retrieve a single document by its ID.

        Parameters:
            index_name (str): The index to retrieve the document from.
            document_id (str): The ID of the document to retrieve.

        Returns:
            Optional[Dict]: The document's content if found, None otherwise.
        """
        try:
            response = self.es.get(index=index_name, id=document_id)
            return response["_source"]
        except Exception as e:
            print(f"❌ Error retrieving document: {e}")
            return None

    def search_documents(self, index_name: str, query: Dict) -> List[Dict]:
        """
        Search for documents based on a query.

        Parameters:
            index_name (str): The index to search.
            query (Dict): The query body for the search.

        Returns:
            List[Dict]: List of documents that match the query.
        """
        try:
            response = self.es.search(index=index_name, body={"query": query})
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

    def upsert_document(
        self, index_name: str, document_id: str, document: Dict
    ) -> Dict:
        """
        Perform an upsert operation on a single document.

        Parameters:
            index_name (str): The index to perform the upsert on.
            document_id (str): The ID of the document.
            document (Dict): The document content to upsert.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.update(
                index=index_name,
                id=document_id,
                body={"doc": document, "doc_as_upsert": True},
            )
            return response
        except Exception as e:
            print(f"❌ Error upserting document: {e}")
            return {}

    def bulk_upsert(
        self, index_name: str, documents: List[Dict], timeout: Optional[str] = None
    ) -> None:
        """
        Perform a bulk upsert operation.

        Parameters:
            index (str): Default index name for the documents.
            documents (List[Dict]): List of documents for bulk upsert.
            timeout (Optional[str]): Timeout duration (e.g., '60s', '2m'). If None, the default timeout is used.
        """
        try:
            # Ensure each document includes an `_index` field
            for doc in documents:
                if "_index" not in doc:
                    doc["_index"] = index_name

            # Perform the bulk operation
            helpers.bulk(self.es, documents, timeout=timeout)
            print("✅ Bulk upsert completed successfully.")
        except Exception as e:
            print(f"❌ Error in bulk upsert: {e}")

    def parallel_bulk_upsert(
        self,
        index_name: str,
        documents: List[Dict],
        batch_size: int = 100,
        max_workers: int = 4,
        timeout: Optional[str] = None,
    ) -> None:
        """
        Perform a parallel bulk upsert operation.

        Parameters:
            index_name (str): Default index name for documents.
            documents (List[Dict]): List of documents for bulk upsert.
            batch_size (int): Number of documents per batch.
            max_workers (int): Number of parallel threads.
            timeout (Optional[str]): Timeout duration (e.g., '60s', '2m'). If None, the default timeout is used.
        """

        def chunk_data(
            data: List[Dict], chunk_size: int
        ) -> Generator[List[Dict], None, None]:
            """Split data into chunks."""
            for i in range(0, len(data), chunk_size):
                yield data[i : i + chunk_size]

        # Ensure each document has an `_index` field
        for doc in documents:
            if "_index" not in doc:
                doc["_index"] = index_name

        batches = list(chunk_data(documents, batch_size))

        def bulk_upsert_batch(batch: List[Dict]):
            helpers.bulk(self.es, batch, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in batches:
                executor.submit(bulk_upsert_batch, batch)

    def delete_document(self, index_name: str, document_id: str) -> Dict:
        """
        Delete a single document by its ID.

        Parameters:
            index_name (str): The index to delete the document from.
            document_id (str): The ID of the document to delete.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.delete(index=index_name, id=document_id)
            return response
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return {}

    def delete_by_query(self, index_name: str, query: Dict) -> Dict:
        """
        Delete documents based on a query.

        Parameters:
            index_name (str): The index to delete documents from.
            query (Dict): The query body for the delete operation.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.delete_by_query(
                index=index_name, body={"query": query}, conflicts="proceed"
            )
            return response
        except Exception as e:
            print(f"❌ Error deleting documents by query: {e}")
            return {}
```

## Data Preparation for Tutorial
- Let’s process **The Little Prince** using the `RecursiveCharacterTextSplitter` to create document chunks.
- Then, we’ll generate embeddings for each text chunk and store the resulting data in a vector database to proceed with a vector database tutorial.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Function to read text from a file (Cross-Platform)
def read_text_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            # Normalize line endings (compatible with Windows, macOS, Linux)
            raw_text = f.read().replace("\r\n", "\n").replace("\r", "\n")
        return raw_text
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode the file with UTF-8 encoding: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified file was not found: {file_path}")

# Function to split the text into chunks
def split_text(raw_text, chunk_size=100, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Default string length function
        is_separator_regex=False,  # Default separator setting
    )
    split_docs = text_splitter.create_documents([raw_text])
    return [doc.page_content for doc in split_docs]

# Set file path and execute
file_path = "./data/the_little_prince.txt"
try:
    # Read the file
    raw_text = read_text_file(file_path)
    # Split the text
    docs = split_text(raw_text)
    
    # Verify output
    print(docs[:2])  # Print the first 5 chunks
    print(f"Total number of chunks: {len(docs)}")
except Exception as e:
    print(f"Error occurred: {e}")
```

<pre class="custom">['The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', '[ Antoine de Saiot-Exupery ]']
    Total number of chunks: 1359
</pre>

```python
%%time

## text embedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings_e5_instruct = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device},  # mps, cuda, cpu
    encode_kwargs={"normalize_embeddings": True},
)

embedded_documents = hf_embeddings_e5_instruct.embed_documents(docs)

print(len(embedded_documents))
print(len(embedded_documents[0]))
```

<pre class="custom">1359
    1024
    CPU times: user 9.33 s, sys: 3.24 s, total: 12.6 s
    Wall time: 23.3 s
</pre>

```python
from uuid import uuid4
from typing import List, Tuple, Dict


def prepare_documents_with_ids(
    docs: List[str], embedded_documents: List[List[float]]
) -> Tuple[List[Dict], List[str]]:
    """
    Prepare a list of documents with unique IDs and their corresponding embeddings.

    Parameters:
        docs (List[str]): List of document texts.
        embedded_documents (List[List[float]]): List of embedding vectors corresponding to the documents.

    Returns:
        Tuple[List[Dict], List[str]]: A tuple containing:
            - List of document dictionaries with `doc_id`, `text`, and `vector`.
            - List of unique document IDs (`doc_ids`).
    """
    # Generate unique IDs for each document
    doc_ids = [str(uuid4()) for _ in range(len(docs))]

    # Prepare the document list with IDs, texts, and embeddings
    documents = [
        {"doc_id": doc_id, "text": doc, "vector": embedding}
        for doc, doc_id, embedding in zip(docs, doc_ids, embedded_documents)
    ]

    return documents, doc_ids
```

```python
documents, doc_ids = prepare_documents_with_ids(docs, embedded_documents)
```

## Initialization
### Setting Up the Elasticsearch Client
- Begin by creating an Elasticsearch client.

```python
import os

# Load environment variables
ES_URL = os.environ["ES_URL"]  # Elasticsearch host URL
ES_API_KEY = os.environ["ES_API_KEY"]  # Elasticsearch API key

# Ensure required environment variables are set
if not ES_URL or not ES_API_KEY:
    raise ValueError("Both ES_URL and ES_API_KEY must be set in environment variables.")
```

```python
es_manager = ElasticsearchManager(es_url=ES_URL, api_key=ES_API_KEY)
```

<pre class="custom">✅ Successfully connected to Elasticsearch!
</pre>

## DB Handling
### Create index
- Use the index method to create a new document.

```python
# create index
index_name = "langchain_tutorial_es"

# vector dimension
dims = len(embedded_documents[0])


# 🛠️ Define the mapping for the new index
# This structure specifies the schema for documents stored in Elasticsearch
mapping = {
    "properties": {
        "metadata": {"properties": {"doc_id": {"type": "keyword"}}},
        "text": {"type": "text"},  # Field for storing textual content
        "vector": {  # Field for storing vector embeddings
            "type": "dense_vector",  # Specifies dense vector type
            "dims": dims,  # Number of dimensions in the vector
            "index": True,  # Enable indexing for vector search
            "similarity": "cosine",  # Use cosine similarity for vector comparisons
        },
    }
}
```

```python
es_manager.create_index(index_name, mapping=mapping)
```




<pre class="custom">"✅ Index 'langchain_tutorial_es' created successfully."</pre>



### Delete index
- You can delete an index as follows

```python
## delete index
es_manager.delete_index(index_name)
```




<pre class="custom">"✅ Index 'langchain_tutorial_es' deleted successfully."</pre>



### Upsert
- Let’s perform an upsert operation for **a single document.** 

```python
# Let’s upsert a single document.

es_manager.upsert_document(index_name, doc_ids[0], documents[0])
```




<pre class="custom">ObjectApiResponse({'_index': 'langchain_tutorial_es', '_id': 'fd9e7626-aac9-4c22-ae8f-2f09486be249', '_version': 1, 'result': 'created', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_seq_no': 0, '_primary_term': 1})</pre>



### Read
- Retrieve the upserted data using its `doc_id`  

```python
# get_document
result = es_manager.get_document(index_name, doc_ids[0])
print(result["doc_id"])
print(result["text"])
```

<pre class="custom">fd9e7626-aac9-4c22-ae8f-2f09486be249
    The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
</pre>

### Delete
- Delete using the `doc_id` 

```python
# delete_document
es_manager.delete_document(index_name, doc_ids[0])
```




<pre class="custom">ObjectApiResponse({'_index': 'langchain_tutorial_es', '_id': 'fd9e7626-aac9-4c22-ae8f-2f09486be249', '_version': 2, 'result': 'deleted', '_shards': {'total': 1, 'successful': 1, 'failed': 0}, '_seq_no': 1, '_primary_term': 1})</pre>



### Bulk Upsert
- Perform a bulk upsert of documents.
- In general, **“bulk”** refers to something large in quantity or volume, often handled or processed all at once.
- For example, “bulk operations” involve managing multiple items simultaneously.

```python
%%time

es_manager.bulk_upsert(index_name, documents)
```

<pre class="custom">✅ Bulk upsert completed successfully.
    CPU times: user 775 ms, sys: 136 ms, total: 912 ms
    Wall time: 37.4 s
</pre>

### Parallel Bulk Upsert
- Perform a bulk upsert of documents in parallel.
- **“parallel”** refers to tasks or processes happening at the same time or simultaneously, often independently of one another.

```python
%%time

# parallel_bulk_upsert
es_manager.parallel_bulk_upsert(index_name, documents, batch_size=100, max_workers=8)
```

<pre class="custom">CPU times: user 1.01 s, sys: 242 ms, total: 1.25 s
    Wall time: 26.1 s
</pre>

- It is evident that parallel_bulk_upsert is **faster.** 

### Read (Document Retrieval)
- Retrieve documents based on specific values.

```python
# search_documents
query = {"match": {"doc_id": doc_ids[0]}}
results = es_manager.search_documents(index_name, query=query)

print(len(results))
print(results[0]["doc_id"])
print(results[0]["text"])
```

<pre class="custom">2
    fd9e7626-aac9-4c22-ae8f-2f09486be249
    The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
</pre>

### Delete
- Delete documents based on specific values.

```python
# delete_by_query
delete_query = {"match": {"doc_id": doc_ids[0]}}
es_manager.delete_by_query(index_name, query=delete_query)
```




<pre class="custom">ObjectApiResponse({'took': 255, 'timed_out': False, 'total': 2, 'deleted': 2, 'batches': 1, 'version_conflicts': 0, 'noops': 0, 'retries': {'bulk': 0, 'search': 0}, 'throttled_millis': 0, 'requests_per_second': -1.0, 'throttled_until_millis': 0, 'failures': []})</pre>



- Delete all documents.

```python
# delete_by_query
delete_query = {"match_all": {}}
es_manager.delete_by_query(index_name, query=delete_query)
```




<pre class="custom">ObjectApiResponse({'took': 1385, 'timed_out': False, 'total': 2718, 'deleted': 2716, 'batches': 3, 'version_conflicts': 2, 'noops': 0, 'retries': {'bulk': 0, 'search': 0}, 'throttled_millis': 0, 'requests_per_second': -1.0, 'throttled_until_millis': 0, 'failures': []})</pre>



## Advanced Search
- **Keyword Search**  
    - This method matches documents that contain the exact keyword in their text field.
    - It performs a straightforward text-based search using Elasticsearch's `match` query.

- **Semantic Search**  
    - Semantic search leverages embeddings to find documents based on their contextual meaning rather than exact text matches.
    - It uses a pre-trained model (`hf_embeddings_e5_instruct`) to encode both the query and the documents into vector representations and retrieves the most similar results.

- **Hybrid Search**  
    - Hybrid search combines both keyword search and semantic search to provide more comprehensive results.
    - It uses a filtering mechanism to ensure documents meet specific keyword criteria while scoring and ranking results based on their semantic similarity to the query.  


```python
%%time

# parallel_bulk_upsert
es_manager.parallel_bulk_upsert(index_name, documents, batch_size=100, max_workers=8)
```

<pre class="custom">CPU times: user 863 ms, sys: 195 ms, total: 1.06 s
    Wall time: 21.9 s
</pre>

```python
# keyword search

keyword = "fox"

query = {"match": {"text": keyword}}
results = es_manager.search_documents(index_name, query=query)

for idx_, result in enumerate(results):
    if idx_ < 3:
        print(idx_, " :", result["text"])
```

<pre class="custom">0  : "I am a fox," said the fox.
    1  : "Good morning," said the fox.
    2  : "Ah," said the fox, "I shall cry."
</pre>

```python
from langchain_elasticsearch import ElasticsearchStore

# Initialize ElasticsearchStore
vector_store = ElasticsearchStore(
    index_name=index_name,  # Elasticsearch index name
    embedding=hf_embeddings_e5_instruct,  # Object responsible for text embeddings
    es_url=ES_URL,  # Elasticsearch host URL
    es_api_key=ES_API_KEY,  # Elasticsearch API key for authentication
)
```

```python
# Execute Semantic Search
search_query = "Who are the Little Prince’s friends?"
results = vector_store.similarity_search(search_query, k=3)

print("🔍 Question: ", search_query)
print("🤖 Semantic Search Results:")
for result in results:
    print(f"- {result.page_content}")
```

<pre class="custom">🔍 Question:  Who are the Little Prince’s friends?
    🤖 Semantic Search Results:
    - "Who are you?" said the little prince.
    - "Then what?" asked the little prince.
    - And the little prince asked himself:
</pre>

```python
# hybrid search with score
search_query = "Who are the Little Prince’s friends?"
keyword = "friend"


results = vector_store.similarity_search_with_score(
    query=search_query,
    k=1,
    filter=[{"term": {"text": keyword}}],
)

print("🔍 search_query: ", search_query)
print("🔍 keyword: ", keyword)

for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content}")
```

<pre class="custom">🔍 search_query:  Who are the Little Prince’s friends?
    🔍 keyword:  friend
    * [SIM=0.927641] "My friend the fox--" the little prince said to me.
</pre>

- **It is evident that conducting a Hybrid Search significantly enhances search performance.**  

- This approach ensures that the search results are both contextually meaningful and aligned with the specified keyword constraint, making it especially useful in scenarios where both precision and context matter.

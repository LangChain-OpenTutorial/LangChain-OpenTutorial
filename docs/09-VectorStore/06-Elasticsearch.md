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
- [Data Preparation for Tutorial](#data-preparation-for-tutorial)
- [Managing Elasticsearch Connections and Documents](#managing-elasticsearch-connections-and-documents)

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

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0
    [notice] To update, run: pip install --upgrade pip
</pre>

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
    CPU times: user 7.25 s, sys: 3.48 s, total: 10.7 s
    Wall time: 18.9 s
</pre>

## Managing Elasticsearch Connections and Documents
### ElasticsearchConnectionManager
- The `ElasticsearchConnectionManager` is a class designed to manage connections to an Elasticsearch instance.
- It facilitates connecting to the Elasticsearch server and provides functionalities for creating and deleting indices.

### Initialization
**Setting Up the Elasticsearch Client**
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
from utils.elasticsearch import ElasticsearchConnectionManager
```

```python
index_name = "langchain_tutorial_es"
```

```python
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

you'll learn how to generate text embeddings for documents using a Hugging Face model.
- First, we'll set up a multilingual model with the `HuggingFaceEmbeddings` class and choose the optimal device (mps, cuda, or cpu) for computation.
- Then, we'll generate embeddings for a list of documents and print the results to ensure everything is working correctly.

The `ElasticsearchConnectionManager` class manages the connection to an Elasticsearch server.
- This instance uses the server URL, API key, embedding model, and index name to connect to Elasticsearch and initialize the vector store.

```python
es_connection_manager = ElasticsearchConnectionManager(
    es_url=ES_URL,
    api_key=ES_API_KEY,
    embedding_model=hf_embeddings_e5_instruct,
    index_name=index_name,
)
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/ [status:200 duration:0.701s]
    INFO:utils.elasticsearch:✅ Successfully connected to Elasticsearch!
    INFO:elastic_transport.transport:GET https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/ [status:200 duration:0.555s]
    INFO:utils.elasticsearch:✅ Vector store initialized for index 'langchain_tutorial_es'.
</pre>

```python
## create index
es_connection_manager.create_index(index_name, mapping=mapping)
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:404 duration:0.183s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.259s]
</pre>




    "✅ Index 'langchain_tutorial_es' created successfully."



```python
## delete index
es_connection_manager.delete_index(index_name)
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.180s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.209s]
</pre>




    "✅ Index 'langchain_tutorial_es' deleted successfully."



### ElasticsearchDocumentManager
- The `ElasticsearchDocumentManager` leverages the `ElasticsearchConnectionManager` to handle document management tasks.
- This class performs operations such as inserting, deleting, and searching documents, with the capability to enhance performance through parallel processing.

```python
from utils.elasticsearch import ElasticsearchDocumentManager
```

```python
es_document_manager = ElasticsearchDocumentManager(
    connection_manager=es_connection_manager,
)
```

### Upsert
- The `upsert` method of the `es_document_manager` is used to insert or update documents in the specified Elasticsearch index.
- It takes the original texts, their corresponding embedded documents, and the index name to efficiently manage the document storage and retrieval process.

```python
%%time

es_document_manager.upsert(
    texts=docs,
    embedded_documents=embedded_documents,
    index_name=index_name,
)
```

<pre class="custom">INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:5.399s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:5.555s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:3.942s]
    INFO:utils.elasticsearch:✅ Bulk upsert completed successfully.
</pre>

    CPU times: user 591 ms, sys: 63 ms, total: 654 ms
    Wall time: 15.5 s
    

```python
es_document_manager.delete(index_name=index_name)
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_delete_by_query?conflicts=proceed [status:200 duration:0.354s]
</pre>

### Upsert_parallel
- The `upsert_parallel` method of the `es_document_manager` facilitates the parallel insertion or updating of documents in the specified Elasticsearch index.
- It processes the documents in batches of 100, utilizing up to 8 workers to enhance performance and efficiency in managing large datasets.

```python
%%time

es_document_manager.upsert_parallel(
    index_name=index_name,
    texts=docs,
    embedded_documents=embedded_documents,
    batch_size=100,
    max_workers=8,
)
```

<pre class="custom">INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.347s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:2.582s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:2.753s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:2.850s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.600s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.479s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.462s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.869s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:2.609s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.347s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.676s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:0.888s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.851s]
    INFO:elastic_transport.transport:PUT https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.626s]
</pre>

    CPU times: user 656 ms, sys: 45.4 ms, total: 702 ms
    Wall time: 7.21 s
    

- It is evident that parallel_upsert is **faster.** 

### Search
- The code performs a search query, "Who are the Little Prince’s friends?", using the `es_document_manager` to retrieve relevant documents from the specified Elasticsearch index.
- By default ( `use_similarity=False` ), it uses the **BM25** algorithm, which is a bag-of-words retrieval function that ranks documents based on the query terms' appearances, regardless of their semantic meaning.
- It fetches the top 10 results, then prints the query and each result in a formatted manner for easy review.

```python
search_query = "Who are the Little Prince’s friends?"

results = es_document_manager.search(index_name=index_name, query=search_query, k=10)

print("================================================")
print("🔍 Question: ", search_query)
print("================================================")
for idx_, result in enumerate(results):
    print(idx_, " :", result)
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.735s]
</pre>

    ================================================
    🔍 Question:  Who are the Little Prince’s friends?
    ================================================
    0  : "Who are you?" said the little prince.
    1  : "Who are you--Who are you--Who are you?" answered the echo.
    2  : people. For some, who are travelers, the stars are guides. For others they are no more than little
    3  : people. For some, who are travelers, the stars are guides. For others they are no more than little
    4  : (picture)
    "Who are you?" asked the little prince, and added, "You are very pretty to look at."
    5  : no more than little lights in the sky. For others, who are scholars, they are problems . For my
    6  : no more than little lights in the sky. For others, who are scholars, they are problems . For my
    7  : "Who are you?" he demanded, thunderstruck. 
    "We are roses," the roses said.
    8  : "No," said the little prince. "I am looking for friends. What does that mean-- ‘tame‘?"
    9  : "Just that," said the fox. "To me, you are still nothing more than a little boy who is just like a
    

Retrieves the top 10 relevant documents using similarity-based matching(cosine similarity).

```python
search_query = "Who are the Little Prince’s friends?"
results = es_document_manager.search(query=search_query, k=10, use_similarity=True)

print("================================================")
print("🔍 Question: ", search_query)
print("================================================")
for idx_, result in enumerate(results):
    print(idx_, " :", result)
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search?_source_includes=metadata,text [status:200 duration:0.377s]
    INFO:utils.elasticsearch:✅ Found 10 similar documents.
</pre>

    ================================================
    🔍 Question:  Who are the Little Prince’s friends?
    ================================================
    0  : "Who are you?" said the little prince.
    1  : "Then what?" asked the little prince.
    2  : And the little prince asked himself:
    3  : "Why is that?" asked the little prince.
    4  : "What do you do here?" the little prince asked.
    5  : [ Chapter 13 ]
    - the little prince visits the businessman
    6  : But the little prince was wondering... The planet was tiny. Over what could this king really rule?
    7  : "Where are the men?" the little prince asked, politely.
    8  : "No," said the little prince. "I am looking for friends. What does that mean-- ‘tame‘?"
    9  : But the little prince added:
    

This code performs a search for the query "Who are the Little Prince’s friends?" while also filtering results based on the **keyword "friend,"** retrieving the top 10 relevant documents and printing their content alongside additional information.

```python
search_query = "Who are the Little Prince’s friends?"
keyword = "friend"
results = es_document_manager.search(
    query=search_query, k=10, use_similarity=True, keyword=keyword
)

print("================================================")
print("🔍 Question: ", search_query)
print("================================================")
for idx_, contents in enumerate(results):
    print(idx_, " :", contents[0].page_content, contents[1])
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search?_source_includes=metadata,text [status:200 duration:0.248s]
    INFO:utils.elasticsearch:✅ Hybrid search completed. Found 10 results.
</pre>

    ================================================
    🔍 Question:  Who are the Little Prince’s friends?
    ================================================
    0  : "My friend the fox--" the little prince said to me. 0.9277072
    1  : any more. If you want a friend, tame me..." 0.91347504
    2  : a grown-up. I have a serious reason: he is the best friend I have in the world. I have another 0.905076
    3  : My friend broke into another peal of laughter: "But where do you think he would go?" 0.90468454
    4  : He was only a fox like a hundred thousand other foxes. But I have made him my friend, and now he is 0.9021255
    5  : that you have known me. You will always be my friend. You will want to laugh with me. And you will 0.89545083
    6  : a friend. And if I forget him, I may become like the grown-ups who are no longer interested in 0.8951793
    7  : that you have known me. You will always be my friend. You will want to laugh with me. And you will 0.8949666
    8  : "That man is the only one of them all whom I could have made my friend. But his planet is indeed 0.8948114
    9  : to seek, in other days, merely by pulling up his chair; and he wanted to help his friend. 0.8929472
    

- This approach ensures that the search results are both contextually meaningful and aligned with the specified keyword constraint, making it especially useful in scenarios where both precision and context matter.

### Read
- This code retrieves the IDs of all documents stored in the specified Elasticsearch index using the `get_documents_ids` method of the `es_document_manager`, and then prints the list of these document IDs for review.

```python
ids = es_document_manager.get_documents_ids(index_name)
print(ids[:10])
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.468s]
</pre>

    ['mfqx9ZQBuaU-CwHIDaXY', 'mvqx9ZQBuaU-CwHIDaXY', 'm_qx9ZQBuaU-CwHIDaXY', 'nPqx9ZQBuaU-CwHIDaXY', 'nfqx9ZQBuaU-CwHIDaXY', 'nvqx9ZQBuaU-CwHIDaXY', 'n_qx9ZQBuaU-CwHIDaXY', 'oPqx9ZQBuaU-CwHIDaXY', 'ofqx9ZQBuaU-CwHIDaXY', 'ovqx9ZQBuaU-CwHIDaXY']
    

This code fetches documents from the specified Elasticsearch index using a list of document IDs, specifically retrieving the first 10 IDs.

It then prints each document's ID along with its corresponding text for easy reference.

```python
responses = es_document_manager.get_documents_by_ids(index_name, ids[:10])

for response in responses:
    print(response["doc_id"], ": ", response["text"])
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.377s]
</pre>

    fb6a7033-465e-4a39-8577-5797fcc67c20 :  "What does this mean?" I demanded. "Why are you talking with snakes?"
    e549da15-6a9c-4589-9645-5263d9aa2615 :  I had loosened the golden muffler that he always wore. I had moistened his temples, and had given
    4c6a0aa2-a626-4a59-838d-989f07cff105 :  and had given him some water to drink. And now I did not dare ask him any more questions. He looked
    101c6f61-3bc8-4036-b0b8-e9a36119970f :  He looked at me very gravely, and put his arms around my neck. I felt his heart beating like the
    075b3e39-80c1-434e-b632-96b586c32f6b :  beating like the heart of a dying bird, shot with someone‘s rifle...
    d451662f-52dc-41cf-b8e9-2cc81a6f7138 :  "I am glad that you have found what was the matter with your engine," he said. "Now you can go back
    9ee4c5fa-68f1-4292-9d95-362834edc807 :  you can go back home--"
    dcdd7bc6-214e-454a-a8b3-a5e0acb19a1c :  "How do you know about that?"
    7468cc42-01aa-425d-acd0-0abf8fe50f0b :  I was just coming to tell him that my work had been successful, beyond anything that I had dared to
    7b3b9425-eca8-44b4-a6d5-d741d0100b0a :  that I had dared to hope. He made no answer to my question, but he added:
    

### Delete
- This code deletes documents from the specified Elasticsearch index using a list of document IDs, specifically retrieving the first 10 IDs. It then prints each document's ID along with its corresponding text for easy reference.

```python
es_document_manager.delete(index_name=index_name, ids=ids[:10])
```

<pre class="custom">INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/mfqx9ZQBuaU-CwHIDaXY [status:200 duration:0.190s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/mvqx9ZQBuaU-CwHIDaXY [status:200 duration:0.189s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/m_qx9ZQBuaU-CwHIDaXY [status:200 duration:0.194s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/nPqx9ZQBuaU-CwHIDaXY [status:200 duration:0.204s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/nfqx9ZQBuaU-CwHIDaXY [status:200 duration:0.188s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/nvqx9ZQBuaU-CwHIDaXY [status:200 duration:0.189s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/n_qx9ZQBuaU-CwHIDaXY [status:200 duration:0.185s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/oPqx9ZQBuaU-CwHIDaXY [status:200 duration:0.188s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/ofqx9ZQBuaU-CwHIDaXY [status:200 duration:0.187s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/ovqx9ZQBuaU-CwHIDaXY [status:200 duration:0.188s]
</pre>

```python
# Delete all documents
es_document_manager.delete(index_name=index_name)
```

<pre class="custom">INFO:elastic_transport.transport:POST https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_delete_by_query?conflicts=proceed [status:200 duration:0.374s]
</pre>

```python
## delete index
es_connection_manager.delete_index(index_name)
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.186s]
    INFO:elastic_transport.transport:DELETE https://e638d39188c94d828a30ae87af1733ce.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.215s]
</pre>




    "✅ Index 'langchain_tutorial_es' deleted successfully."



Remove a **Huggingface Cache**  , `embeddings` and `client` .

If you created a **vectordb** directory, please **remove** it at the end of this tutorial.

```python
from huggingface_hub import scan_cache_dir

del embedded_documents
del es_connection_manager
del es_document_manager
scan = scan_cache_dir()
scan.delete_revisions()
```




<pre class="custom">DeleteCacheStrategy(expected_freed_size=0, blobs=frozenset(), refs=frozenset(), repos=frozenset(), snapshots=frozenset())</pre>



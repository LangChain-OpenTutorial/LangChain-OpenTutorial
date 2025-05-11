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
- Peer Review: [Joseph](https://github.com/XaviereKU), [Sohyeon Yim](https://github.com/sohyunwriter), [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/your-notebook-file-name) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/your-notebook-file-name)

## Overview

This tutorial covers how to use **Elasticsearch** with **LangChain** .

This tutorial walks you through using **CRUD** operations with the **Elasticsearch** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Elasticsearch?](#what-is-Elasticsearch?)
- [Prepare Data](#Prepare-Data)
- [Setting up Elasticsearch](#Setting-up-Elasticsearch)
- [Document Manager](#document-manager)


### References
- [Elasticsearch Official Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/index.html)  
- [Elasticsearch Vector Search Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)  
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_openai",
        "elasticsearch",
        "python-dotenv",
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "Your OPENAI API KEY",
        "LANGCHAIN_API_KEY": "Your LangChain API KEY",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Elasticsearch",
        "ES_URL": "Your Elasticsearch URI",
        "ES_API_KEY": "Your Elasticsearch API KEY",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



### Setup Elasticsearch
In order to use the **Elasticsearch** vector search you must install the langchain-elasticsearch package.

 🚀 Setting Up Elasticsearch with Elastic Cloud (Colab Compatible)
- Elastic Cloud allows you to manage **Elasticsearch** seamlessly in the cloud, eliminating the need for local installations.
- It integrates well with Google Colab, enabling efficient experimentation and prototyping.


 📚 What is Elastic Cloud?  
- **Elastic Cloud** is a managed **Elasticsearch** service provided by Elastic.  
- Supports **custom cluster configurations** and **auto-scaling** . 
- Deployable on **AWS** , **GCP** , and **Azure** .  
- Compatible with **Google Colab,** allowing simplified cloud-based workflows.  

 📌 Getting Started with Elastic Cloud  
1. **Sign up for Elastic Cloud’s Free Trial** .  
    - [Free Trial](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=documentation)
2. Create an Elasticsearch **Cluster** .
3. Retrieve your **Elasticsearch URL** and **Elasticsearch API Key** from the Elastic Cloud Console.  
4. Add the following to your `.env` file
    ```
    ES_URL=https://my-elasticsearch-project-abd...:123
    ES_API_KEY=bk9X...
    ```


## What is Elasticsearch?
**Elasticsearch** is an open-source, distributed search and analytics engine designed to store, search, and analyze both structured and unstructured data in real-time.

### Key Features  
- **Real-Time Search:** Instantly searchable data upon ingestion  
- **Large-Scale Data Processing:** Efficient handling of vast datasets  
- **Scalability:** Flexible scaling through clustering and distributed architecture  
- **Versatile Search Support:** Keyword search, semantic search, and multimodal search  

### Use Cases  
- **Log Analytics:** Real-time monitoring of system and application logs  
- **Monitoring:** Server and network health tracking  
- **Product Recommendations:** Behavior-based recommendation systems  
- **Natural Language Processing (NLP):** Semantic text searches  
- **Multimodal Search:** Text-to-image and image-to-image searches  

### Vector Database Functionality in Elasticsearch  
- **Elasticsearch** supports vector data storage and similarity search via **Dense Vector Fields** . As a vector database, it excels in applications like NLP, image search, and recommendation systems.

### Core Vector Database Features  
- **Dense Vector Field:** Store and query high-dimensional vectors  
- **KNN (k-Nearest Neighbors) Search:** Find vectors most similar to the input  
- **Semantic Search:** Perform meaning-based searches beyond keyword matching  
- **Multimodal Search:** Combine text and image data for advanced search capabilities  

### Vector Search Use Cases  
- **Semantic Search:** Understand user intent and deliver precise results  
- **Text-to-Image Search:** Retrieve relevant images from textual descriptions  
- **Image-to-Image Search:** Find visually similar images in a dataset  

**Elasticsearch** goes beyond traditional text search engines, offering robust vector database capabilities essential for NLP and multimodal search applications.

---

## Prepare Data

This section guides you through the **data preparation process** .

This section includes the following components:

- Data Introduction

- Preprocess Data


### Data Introduction

In this tutorial, we will use the fairy tale **📗 The Little Prince** in PDF format as our data.

This material complies with the **Apache 2.0 license** .

The data is used in a text (.txt) format converted from the original PDF.

You can view the data at the link below.
- [Data Link](https://huggingface.co/datasets/sohyunwriter/the_little_prince)

### Preprocess Data

In this tutorial section, we will preprocess the text data from The Little Prince and convert it into a list of ```LangChain Document``` objects with metadata. 

Each document chunk will include a ```title``` field in the metadata, extracted from the first line of each section.

```python
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from typing import List

def preprocessing_data(content:str)->List[Document]:
    # 1. Split the text by double newlines to separate sections
    blocks = content.split("\n\n")

    # 2. Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,              # Maximum number of characters per chunk
        chunk_overlap=50,            # Overlap between chunks to preserve context
        separators=["\n\n", "\n", " "]  # Order of priority for splitting
    )

    documents = []

    # 3. Loop through each section
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Extract title from the first line using square brackets [ ]
        first_line = lines[0]
        title_match = re.search(r"\[(.*?)\]", first_line)
        title = title_match.group(1).strip() if title_match else ""

        # Remove the title line from content
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue

        # 4. Chunk the section using the text splitter
        chunks = text_splitter.split_text(body)

        # 5. Create a LangChain Document for each chunk with the same title metadata
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"title": title}))

    print(f"Generated {len(documents)} chunked documents.")

    return documents
```

```python
# Load the entire text file
with open("./data/the_little_prince.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Preprocess Data
docs = preprocessing_data(content=content)
```

<pre class="custom">Generated 262 chunked documents.
</pre>

## Setting up Elasticsearch

This part walks you through the initial setup of **Elasticsearch** .

This section includes the following components:

- Load Embedding Model

- Load Elasticsearch Client

### Load Embedding Model

In this section, you'll learn how to load an embedding model.

This tutorial uses **OpenAI's** **API-Key** for loading the model.

*💡 If you prefer to use another embedding model, see the instructions below.*
- [Embedding Models](https://python.langchain.com/docs/integrations/text_embedding/)

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Load Elasticsearch Client

In this section, we'll show you how to load the **database client object** using the **Python SDK** for **Elasticsearch** .
- [Elasticsearch Python SDK Docs](https://www.elastic.co/docs/reference/elasticsearch/clients/python)

```python
import os
import logging
from elasticsearch import Elasticsearch, exceptions as es_exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_client(
    es_url: str = None,
    api_key: str = None,
    timeout: int = 120,
    retry_on_timeout: bool = True
) -> Elasticsearch:
    """
    Initializes and returns an Elasticsearch client instance.
    
    This function loads configuration (e.g., API key, host) from environment
    variables or default values and creates a client object to interact
    with the Elasticsearch Python SDK.

    Args:
        es_url (str): Elasticsearch URL. If None, uses 'ES_URL' env var.
        api_key (str): API key. If None, uses 'ES_API_KEY' env var.
        timeout (int): Request timeout in seconds.
        retry_on_timeout (bool): Whether to retry on timeout.

    Returns:
        Elasticsearch: An instance of the Elasticsearch client.

    Raises:
        ValueError: If required configuration is missing.
        es_exceptions.ConnectionError: If connection fails.
    """
    es_url = es_url or os.getenv("ES_URL")
    api_key = api_key or os.getenv("ES_API_KEY")
    if not es_url or not api_key:
        raise ValueError("Elasticsearch URL and API key must be provided.")

    client = Elasticsearch(
        es_url, api_key=api_key, request_timeout=timeout, retry_on_timeout=retry_on_timeout
    )

    try:
        if client.ping():
            logger.info("✅ Successfully connected to Elasticsearch!")
        else:
            logger.error("❌ Failed to connect to Elasticsearch (ping returned False).")
            raise es_exceptions.ConnectionError("Failed to connect to Elasticsearch.")
    except Exception as e:
        logger.error(f"❌ Elasticsearch connection error: {e}")
        raise

    return client
```

```python
client = get_db_client()
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/ [status:200 duration:0.603s]
    INFO:__main__:✅ Successfully connected to Elasticsearch!
</pre>

### Create Index
If you are successfully connected to **Elasticsearch**, some basic indexes are already created.

But, in this tutorial we will create a new index with ```ElasticsearchIndexManager``` class.

```python
from utils.elasticsearch import ElasticsearchIndexManager

#  Create IndexManager Object
index_manger = ElasticsearchIndexManager(client)

# Create A New Index
index_name = "langchain_tutorial_es"

tutorial_index=index_manger.create_index(
    embedding, index_name=index_name, metric="cosine"
)

print(tutorial_index)
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:HEAD https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.194s]
</pre>

    ⚠️ Index 'langchain_tutorial_es' already exists. Skipping creation.
    {'status': 'exists', 'index_name': 'langchain_tutorial_es', 'embedding_dims': 3072, 'metric': 'cosine'}
    

### Delete Index
If you want to remove an existing index from **Elasticsearch**, you can use the `ElasticsearchIndexManager` class to delete it easily.

This is useful when you want to reset your data or clean up unused indexes during development or testing.

```python
# Delete A New Index
index_manger.delete_index(index_name)
```

<pre class="custom">INFO:elastic_transport.transport:HEAD https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.193s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.227s]
</pre>




    "✅ Index 'langchain_tutorial_es' deleted successfully."



To proceed with the tutorial, let’s create the index once again.

```python
tutorial_index=index_manger.create_index(
    embedding, index_name=index_name, metric="cosine"
)

print(tutorial_index)
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:HEAD https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:404 duration:0.195s]
    INFO:elastic_transport.transport:PUT https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es [status:200 duration:0.284s]
</pre>

    ✅ Index 'langchain_tutorial_es' created successfully.
    {'status': 'created', 'index_name': 'langchain_tutorial_es', 'embedding_dims': 3072, 'metric': 'cosine'}
    

## Document Manager

For the **LangChain-OpenTutorial**, we have implemented a custom set of **CRUD** functionalities for VectorDBs

The following operations are included:

- ```upsert``` : Update existing documents or insert if they don’t exist

- ```upsert_parallel``` : Perform upserts in parallel for large-scale data

- ```similarity_search``` : Search for similar documents based on embeddings

- ```delete``` : Remove documents based on filter conditions

Each of these features is implemented as class methods specific to each VectorDB.

In this tutorial, you'll learn how to use these methods to interact with your VectorDB.

*We plan to continuously expand the functionality by adding more common operations in the future.*

### Create Instance

First, create an instance of the Elasticsearch helper class to use its CRUD functionalities.

This class is initialized with the **Elasticsearch Python SDK client instance** and the **embedding model instance** , both of which were defined in the previous section.

```python
# import ElasticsearchDocumentManager
from utils.elasticsearch import ElasticsearchDocumentManager

# connect to tutorial_index
crud_manager = ElasticsearchDocumentManager(
    client=client, index_name=index_name, embedding=embedding
)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your **Elasticsearch** .

### Upsert Document

**Update** existing documents or **insert** if they don’t exist

**✅ Args**

- ```texts``` : Iterable[str] – List of text contents to be inserted/updated.

- ```metadatas``` : Optional[List[Dict]] – List of metadata dictionaries for each text (optional).

- ```ids``` : Optional[List[str]] – Custom IDs for the documents. If not provided, IDs will be auto-generated.

- ```**kwargs``` : Extra arguments for the underlying vector store.

**🔄 Return**

- None

```python
from uuid import uuid4

# Create ID for each document
ids = [str(uuid4()) for _ in docs]

args = {
    "texts": [doc.page_content for doc in docs[:2]],
    "metadatas": [doc.metadata for doc in docs[:2]],
    "ids": ids[:2],
    # Add additional parameters if you need
}

crud_manager.upsert(**args)
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:PUT https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:0.838s]
    INFO:utils.elasticsearch:✅ Bulk upsert completed successfully.
</pre>

### Upsert Parallel

Perform **upserts** in **parallel** for large-scale data

**✅ Args**

- ```texts``` : Iterable[str] – List of text contents to be inserted/updated.

- ```metadatas``` : Optional[List[Dict]] – List of metadata dictionaries for each text (optional).

- ```ids``` : Optional[List[str]] – Custom IDs for the documents. If not provided, IDs will be auto-generated.

- ```batch_size``` : int – Number of documents per batch (default: 32).

- ```workers``` : int – Number of parallel workers (default: 10).

- ```**kwargs``` : Extra arguments for the underlying vector store.

**🔄 Return**

- None

```python
from uuid import uuid4

args = {
    "texts": [doc.page_content for doc in docs],
    "metadatas": [doc.metadata for doc in docs],
    "ids": ids,
    # Add additional parameters if you need
}

crud_manager.upsert_parallel(**args)
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:PUT https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:1.858s]
    INFO:elastic_transport.transport:PUT https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:3.299s]
    INFO:elastic_transport.transport:PUT https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/_bulk [status:200 duration:3.570s]
</pre>

### Similarity Search

Search for **similar documents** based on **embeddings** .

This method uses **"cosine similarity"** .


**✅ Args**

- ```query``` : str – The text query for similarity search.

- ```k``` : int – Number of top results to return (default: 10).

- ```**kwargs``` : Additional search options (e.g., filters).

**🔄 Return**

- ```results``` : List[Document] – A list of LangChain Document objects ranked by similarity.

```python
# Search by Query
results = crud_manager.search(query="What is essential is invisible to the eye.", k=3)

for idx,doc in enumerate(results):
    print("="*100)
    print(f"Rank {idx+1} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print(f"Similarity Score : {doc.metadata['score']}")
    print()
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.954s]
</pre>

    ====================================================================================================
    Rank 1 | Title : Chapter 21
    Contents : And he went back to meet the fox. 
    "Goodbye," he said. 
    "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the heart that one can see rightly; what is essential is invisible to the eye." 
    "What is essential is invisible to the eye," the little prince repeated, so that he would be sure to remember.
    "It is the time you have wasted for your rose that makes your rose so important."
    Similarity Score : 0.7546974
    
    ====================================================================================================
    Rank 2 | Title : Chapter 24
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    Similarity Score : 0.7476631
    
    ====================================================================================================
    Rank 3 | Title : Chapter 25
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Similarity Score : 0.7111699
    
    

```python
# Search with filters
results = crud_manager.search(
    query="Which asteroid did the little prince come from?",
    k=3,
    filters={"title":"Chapter 4"}
    )

for idx,doc in enumerate(results):
    print("="*100)
    print(f"Rank {idx+1} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print(f"Similarity Score : {doc.metadata['score']}")
    print()
```

<pre class="custom">INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.584s]
</pre>

    ====================================================================================================
    Rank 1 | Title : Chapter 4
    Contents : I have serious reason to believe that the planet from which the little prince came is the asteroid known as B-612. This asteroid has only once been seen through the telescope. That was by a Turkish astronomer, in 1909. 
    (picture)
    On making his discovery, the astronomer had presented it to the International Astronomical Congress, in a great demonstration. But he was in Turkish costume, and so nobody would believe what he said.
    Grown-ups are like that...
    Similarity Score : 0.8311258
    
    ====================================================================================================
    Rank 2 | Title : Chapter 4
    Contents : - the narrator speculates as to which asteroid from which the little prince came　　
    I had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!
    Similarity Score : 0.81760435
    
    ====================================================================================================
    Rank 3 | Title : Chapter 9
    Contents : - the little prince leaves his planet
    Similarity Score : 0.8035729
    
    

### Delete Document

Delete documents based on filter conditions

**✅ Args**

- ```ids``` : Optional[List[str]] – List of document IDs to delete. If None, deletion is based on filter.

- ```filters``` : Optional[Dict] – Dictionary specifying filter conditions (e.g., metadata match).

- ```**kwargs``` : Any additional parameters.

**🔄 Return**

- None

```python
# Delete by ids
del_ids = ids[:5]  # The 'ids' value you want to delete
crud_manager.delete(ids=del_ids)
```

<pre class="custom">INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.196s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/egW1sJYBLV0ipYAYto5p [status:200 duration:0.194s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/HgW1sJYBLV0ipYAY1I_A [status:200 duration:0.194s]
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.197s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/ewW1sJYBLV0ipYAYto5p [status:200 duration:0.195s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/HwW1sJYBLV0ipYAY1I_A [status:200 duration:0.193s]
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.193s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/IAW1sJYBLV0ipYAY1I_A [status:200 duration:0.193s]
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.210s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/IQW1sJYBLV0ipYAY1I_A [status:200 duration:0.194s]
    INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_search [status:200 duration:0.194s]
    INFO:elastic_transport.transport:DELETE https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_doc/IgW1sJYBLV0ipYAY1I_A [status:200 duration:0.193s]
</pre>

```python
# Delete by ids with filters
filters = {"page": 6}
crud_manager.delete(filters={"title": "chapter 6"})
```

<pre class="custom">INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_delete_by_query?conflicts=proceed [status:200 duration:0.194s]
</pre>

```python
# Delete All
crud_manager.delete()
```

<pre class="custom">INFO:elastic_transport.transport:POST https://de7f5f4e2c8a452597e8e4db54c98b30.us-central1.gcp.cloud.es.io:443/langchain_tutorial_es/_delete_by_query?conflicts=proceed [status:200 duration:0.291s]
</pre>

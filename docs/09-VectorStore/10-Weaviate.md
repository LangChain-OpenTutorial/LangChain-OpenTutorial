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

# Weaviate

- Author: [Haseom Shin](https://github.com/IHAGI-c)
- Design: []()
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb)

## Overview

This comprehensive tutorial explores Weaviate, a powerful open-source vector database that enables efficient similarity search and semantic operations. Through hands-on examples, you'll learn:

- How to set up and configure Weaviate for production use
- Essential operations including document indexing, querying, and deletion
- Advanced features such as hybrid search, multi-tenancy, and batch processing
- Integration with LangChain for sophisticated applications like RAG and QA systems
- Best practices for managing and scaling your vector database

Whether you're building a semantic search engine, implementing RAG systems, or developing AI-powered applications, this tutorial provides the foundational knowledge and practical examples you need to leverage Weaviate effectively.

> [Weaviate](https://weaviate.io/) is an open-source vector database. It allows you to store data objects and vector embeddings from your favorite ML-models, and scale seamlessly into billions of data objects.

To use this integration, you need to have a running Weaviate database instance.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Credentials](#credentials)
  - [Setting up Weaviate Cloud Services](#setting-up-weaviate-cloud-services)
- [What is Weaviate?](#what-is-weaviate)
- [Why Use Weaviate?](#why-use-weaviate)
- [Initialization](#initialization)
  - [Creating Collections in Weaviate](#creating-collections-in-weaviate)
  - [Delete Collection](#delete-collection)
  - [List Collections](#list-collections)
  - [Data Preprocessing](#data-preprocessing)
  - [Document Preprocessing Function](#document-preprocessing-function)
- [Manage vector store](#manage-vector-store)
  - [Add items to vector store](#add-items-to-vector-store)
  - [Delete items from vector store](#delete-items-from-vector-store)
- [Finding Objects by Similarity](#finding-objects-by-similarity)
  - [Step 1: Preparing Your Data](#step-1-preparing-your-data)
  - [Step 2: Perform the search](#step-2-perform-the-search)
  - [Quantify Result Similarity](#quantify-result-similarity)
- [Search mechanism](#search-mechanism)
- [Persistence](#persistence)
- [Multi-tenancy](#multi-tenancy)
- [Retriever options](#retriever-options)
- [Use with LangChain](#use-with-langchain)
  - [Question Answering with Sources](#question-answering-with-sources)
  - [Retrieval-Augmented Generation](#retrieval-augmented-generation)


### References
- [Langchain-Weaviate](https://python.langchain.com/docs/integrations/providers/weaviate/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Introduction](https://weaviate.io/developers/weaviate/introduction)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "openai",
        "langsmith",
        "langchain",
        "tiktoken",
        "langchain-weaviate",
        "langchain-openai",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.2 -> 25.0
    [notice] To update, run: pip install --upgrade pip
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "WEAVIATE_API_KEY": "",
        "WEAVIATE_URL": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Weaviate",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Credentials

There are three main ways to connect to Weaviate:

1. **Local Connection**: Connect to a Weaviate instance running locally through Docker
2. **Weaviate Cloud(WCD)**: Use Weaviate's managed cloud service
3. **Custom Deployment**: Deploy Weaviate on Kubernetes or other custom configurations

For this notebook, we'll use Weaviate Cloud (WCD) as it provides the easiest way to get started without any local setup.

### Setting up Weaviate Cloud Services

1. First, sign up for a free account at [Weaviate Cloud Console](https://console.weaviate.cloud)
2. Create a new cluster
3. Get your API key
4. Set API key
5. Connect to your WCD cluster

#### 1. Weaviate Signup
![Weaviate Cloud Console](./img/10-weaviate-credentials-01.png)

#### 2. Create Cluster
![Weaviate Cloud Console](./img/10-weaviate-credentials-02.png)
![Weaviate Cloud Console](./img/10-weaviate-credentials-03.png)

#### 3. Get API Key
**If you using gRPC, please copy the gRPC URL**

![Weaviate Cloud Console](./img/10-weaviate-credentials-04.png)

#### 4. Set API Key
```
WEAVIATE_API_KEY="YOUR_WEAVIATE_API_KEY"
WEAVIATE_URL="YOUR_WEAVIATE_CLUSTER_URL"
```

#### 5. Connect to your WCD cluster

```python
import os
from langchain_openai import OpenAIEmbeddings
from utils.weaviate_vectordb import WeaviateDB

weaviate_url = os.environ.get("WEAVIATE_URL")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
weaviate_db = WeaviateDB(url=weaviate_url, api_key=weaviate_api_key, openai_api_key=openai_api_key, embeddings=embeddings)
client = weaviate_db.connect()

print(client.is_ready())
```

<pre class="custom">True
</pre>

## What is Weaviate?

Weaviate is a powerful open-source vector database that revolutionizes how we store and search data. It combines traditional database capabilities with advanced machine learning features, allowing you to:

- Weaviate is an open source [vector database](https://weaviate.io/blog/what-is-a-vector-database).
- Weaviate allows you to store and retrieve data objects based on their semantic properties by indexing them with [vectors](./concepts/vector-index.md).
- Weaviate can be used stand-alone (aka _bring your vectors_) or with a variety of [modules](./modules/index.md) that can do the vectorization for you and extend the core capabilities.
- Weaviate has a [GraphQL-API](./api/graphql/index.md) to access your data easily.
- Weaviate is fast (check our [open source benchmarks](./benchmarks/index.md)).

> 💡 **Key Feature** : Weaviate achieves millisecond-level query performance, making it suitable for production environments.

## Why Use Weaviate?

Weaviate stands out for several reasons:

1. **Versatility** : Supports multiple media types (text, images, etc.)
2. **Advanced Features** :
   - Semantic Search
   - Question-Answer Extraction
   - Classification
   - Custom ML Model Integration
3. **Production-Ready** : Built in Go for high performance and scalability
4. **Developer-Friendly** : Multiple access methods through GraphQL, REST, and various client libraries


## Initialization
Before initializing our vector store, let's connect to a Weaviate collection. If one named index_name doesn't exist, it will be created.

### Creating Collections in Weaviate

The `create_collection` function establishes a new collection in Weaviate, configuring it with specified properties and vector settings. This foundational operation requires six key parameters:

**Required Parameters:**
- `client` : Weaviate client instance for database connection
- `collection_name` : Unique identifier for your collection
- `description` : Detailed description of the collection's purpose
- `properties` : List of property definitions for data schema
- `vectorizer` : Configuration for vector embedding generation
- `metric` : Distance metric for similarity calculations

**Advanced Configuration Options:**
- For custom distance metrics: Utilize the `VectorDistances` class
- For alternative vectorization: Leverage the `Configure.Vectorizer` class

**Example Usage:**
```python
properties = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT)
]
vectorizer = Configure.Vectorizer.text2vec_openai()
create_collection(client, "Documents", "Document storage", properties, vectorizer)
```

> **Note:** Choose your distance metric and vectorizer carefully as they significantly impact search performance and accuracy.

Now let's use the `create_collection` function to create the collection we'll use in this tutorial.

```python
from weaviate.classes.config import Property, DataType, Configure

collection_name = "BookChunk"  # change if desired
description = "A chunk of a book's content"
vectorizer = Configure.Vectorizer.text2vec_openai(
    model="text-embedding-3-large"
)  # You can select other vectorizer
metric = "dot"  # You can select other distance metric
properties = [
    Property(
        name="text", data_type=DataType.TEXT, description="The content of the text"
    ),
    Property(
        name="order",
        data_type=DataType.INT,
        description="The order of the chunk in the book",
    ),
    Property(
        name="title", data_type=DataType.TEXT, description="The title of the book"
    ),
    Property(
        name="author", data_type=DataType.TEXT, description="The author of the book"
    ),
    Property(
        name="source", data_type=DataType.TEXT, description="The source of the book"
    ),
]

weaviate_db.create_collection(
    client, collection_name, description, properties, vectorizer, metric
)
```

<pre class="custom">Collection 'BookChunk' created successfully.
</pre>

### List Collections

Lists all collections in Weaviate, providing a comprehensive view of your database schema and configurations. The `list_collections` function helps you inspect and manage your Weaviate instance's structure.

**Key Information Returned:**
- Collection names
- Collection descriptions
- Property configurations
- Data types for each property

> **Note:** This operation is particularly useful for database maintenance, debugging, and documentation purposes.


```python
weaviate_db.list_collections(client)
```

<pre class="custom">Collections (indexes) in the Weaviate schema:
    - Collection name: BookChunk
      Description: A chunk of a book's content
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.INT
        - Name: title, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: source, Type: DataType.TEXT
    
</pre>

```python
print(weaviate_db.lookup_collection(collection_name))
```

<pre class="custom"><weaviate.Collection config={
      "name": "BookChunk",
      "description": "A chunk of a book's content",
      "generative_config": null,
      "inverted_index_config": {
        "bm25": {
          "b": 0.75,
          "k1": 1.2
        },
        "cleanup_interval_seconds": 60,
        "index_null_state": false,
        "index_property_length": false,
        "index_timestamps": false,
        "stopwords": {
          "preset": "en",
          "additions": null,
          "removals": null
        }
      },
      "multi_tenancy_config": {
        "enabled": false,
        "auto_tenant_creation": false,
        "auto_tenant_activation": false
      },
      "properties": [
        {
          "name": "text",
          "description": "The content of the text",
          "data_type": "text",
          "index_filterable": true,
          "index_range_filters": false,
          "index_searchable": true,
          "nested_properties": null,
          "tokenization": "word",
          "vectorizer_config": {
            "skip": false,
            "vectorize_property_name": true
          },
          "vectorizer": "text2vec-openai"
        },
        {
          "name": "order",
          "description": "The order of the chunk in the book",
          "data_type": "int",
          "index_filterable": true,
          "index_range_filters": false,
          "index_searchable": false,
          "nested_properties": null,
          "tokenization": null,
          "vectorizer_config": {
            "skip": false,
            "vectorize_property_name": true
          },
          "vectorizer": "text2vec-openai"
        },
        {
          "name": "title",
          "description": "The title of the book",
          "data_type": "text",
          "index_filterable": true,
          "index_range_filters": false,
          "index_searchable": true,
          "nested_properties": null,
          "tokenization": "word",
          "vectorizer_config": {
            "skip": false,
            "vectorize_property_name": true
          },
          "vectorizer": "text2vec-openai"
        },
        {
          "name": "author",
          "description": "The author of the book",
          "data_type": "text",
          "index_filterable": true,
          "index_range_filters": false,
          "index_searchable": true,
          "nested_properties": null,
          "tokenization": "word",
          "vectorizer_config": {
            "skip": false,
            "vectorize_property_name": true
          },
          "vectorizer": "text2vec-openai"
        },
        {
          "name": "source",
          "description": "The source of the book",
          "data_type": "text",
          "index_filterable": true,
          "index_range_filters": false,
          "index_searchable": true,
          "nested_properties": null,
          "tokenization": "word",
          "vectorizer_config": {
            "skip": false,
            "vectorize_property_name": true
          },
          "vectorizer": "text2vec-openai"
        }
      ],
      "references": [],
      "replication_config": {
        "factor": 1,
        "async_enabled": false,
        "deletion_strategy": "NoAutomatedResolution"
      },
      "reranker_config": null,
      "sharding_config": {
        "virtual_per_physical": 128,
        "desired_count": 1,
        "actual_count": 1,
        "desired_virtual_count": 128,
        "actual_virtual_count": 128,
        "key": "_id",
        "strategy": "hash",
        "function": "murmur3"
      },
      "vector_index_config": {
        "quantizer": null,
        "cleanup_interval_seconds": 300,
        "distance_metric": "dot",
        "dynamic_ef_min": 100,
        "dynamic_ef_max": 500,
        "dynamic_ef_factor": 8,
        "ef": -1,
        "ef_construction": 128,
        "filter_strategy": "sweeping",
        "flat_search_cutoff": 40000,
        "max_connections": 32,
        "skip": false,
        "vector_cache_max_objects": 1000000000000
      },
      "vector_index_type": "hnsw",
      "vectorizer_config": {
        "vectorizer": "text2vec-openai",
        "model": {
          "baseURL": "https://api.openai.com",
          "model": "text-embedding-3-large"
        },
        "vectorize_collection_name": true
      },
      "vectorizer": "text2vec-openai",
      "vector_config": null
    }>
</pre>

### Data Preprocessing

Before storing documents in Weaviate, it's essential to preprocess them into manageable chunks. This section demonstrates how to effectively prepare your documents using the `RecursiveCharacterTextSplitter` for optimal vector storage and retrieval.

**Key Preprocessing Steps:**
- Text chunking for better semantic representation
- Metadata assignment for enhanced searchability
- Document structure optimization
- Batch preparation for efficient storage

> **Note:** While this example uses `RecursiveCharacterTextSplitter`, choose your text splitter based on your specific content type and requirements. The chunk size and overlap parameters significantly impact search quality and performance.

```python
# This is a long document we can split up.
with open("./data/the_little_prince.txt",encoding='utf-8') as f:
    raw_text = f.read()
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)

split_docs = text_splitter.create_documents([raw_text])

print(split_docs[:5])
```

<pre class="custom">[Document(metadata={}, page_content='The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)'), Document(metadata={}, page_content='[ Antoine de Saiot-Exupery ]'), Document(metadata={}, page_content='Over the past century, the thrill of flying has inspired some to perform remarkable feats of daring. For others, their desire to soar into the skies led to dramatic leaps in technology. For Antoine de Saint-Exupéry, his love of aviation inspired stories, which have touched the hearts of millions'), Document(metadata={}, page_content='have touched the hearts of millions around the world.'), Document(metadata={}, page_content='Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the')]
</pre>

### Document Preprocessing Function

The `preprocess_documents` function transforms pre-split documents into a format suitable for Weaviate storage. This utility function handles both document content and metadata, ensuring proper organization of your data.

**Function Parameters:**
- `split_docs` : List of LangChain Document objects containing page content and metadata
- `metadata` : Optional dictionary of additional metadata to include with each chunk

**Processing Steps:**
- Iterates through Document objects
- Assigns sequential order numbers
- Combines document metadata with additional metadata
- Formats data for Weaviate ingestion

> **Best Practice:** When preprocessing documents, always maintain consistent metadata structure across your collection. This ensures efficient querying and filtering capabilities later.

```python
from typing import List, Dict
from langchain_core.documents import Document


def preprocess_documents(
    split_docs: List[Document], metadata: Dict[str, str] = None
) -> List[Dict[str, Dict[str, object]]]:
    """
    Processes a list of pre-split documents into a format suitable for storing in Weaviate.

    :param split_docs: List of LangChain Document objects (each containing page_content and metadata).
    :param metadata: Additional metadata to include in each chunk (e.g., title, source).
    :return: A list of dictionaries, each representing a chunk in the format:
             {'properties': {'text': ..., 'order': ..., ...metadata}}
    """
    processed_chunks = []
    texts = []
    metadatas = []
    # Iterate over Document objects
    for idx, doc in enumerate(split_docs, start=1):
        # Extract text from page_content and include metadata
        chunk_data = {"text": doc.page_content, "order": idx}
        # Combine with metadata from Document and additional metadata if provided
        if metadata:
            chunk_data.update(metadata)
        if doc.metadata:
            chunk_data.update(doc.metadata)

        # Format for Weaviate
        processed_chunks.append(chunk_data)
        texts.append(doc.page_content)
        metadatas.append(metadata)

    return processed_chunks, texts, metadatas


metadata = {
    "title": "The Little Prince",
    "author": "Antoine de Saint-Exupéry",
    "source": "Original Text",
}

processed_chunks, texts, metadatas = preprocess_documents(split_docs, metadata=metadata)

processed_chunks[:5]
```




<pre class="custom">[{'text': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)',
      'order': 1,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': '[ Antoine de Saiot-Exupery ]',
      'order': 2,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of daring. For others, their desire to soar into the skies led to dramatic leaps in technology. For Antoine de Saint-Exupéry, his love of aviation inspired stories, which have touched the hearts of millions',
      'order': 3,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'have touched the hearts of millions around the world.',
      'order': 4,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the',
      'order': 5,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'}]</pre>



## Manage vector store
Once you have created your vector store, we can interact with it by adding and deleting different items.

### Add Items to Vector Store

Weaviate provides flexible methods for adding documents to your vector store. This section explores two efficient approaches: standard insertion and parallel batch processing, each optimized for different use cases.

**Standard Insertion**

Best for smaller datasets or when processing order is important:
- Sequential document processing
- Automatic UUID generation
- Built-in duplicate handling
- Real-time progress tracking

**Parallel Batch Processing**

Optimized for large-scale document ingestion:
- Multi-threaded processing
- Configurable batch sizes
- Concurrent execution
- Enhanced throughput

**Configuration Options:**

- `batch_size` : Control memory usage and processing chunks
- `max_workers` : Adjust concurrent processing threads
- `unique_key` : Define document identification field
- `show_progress` : Monitor ingestion progress

**Performance Tips:**

- For datasets < 1000 documents: Use standard insertion
- For datasets > 1000 documents: Consider parallel processing
- Monitor memory usage when increasing batch size
- Adjust worker count based on available CPU cores

> **Best Practice:** Choose your ingestion method based on dataset size and system resources. Start with conservative batch sizes and gradually optimize based on performance metrics.

```python
from weaviate.util import generate_uuid5

def generate_ids(collection_name: str, unique_values: List[str]):
  ids = []

  for unique_value in unique_values:
    ids.append(generate_uuid5(collection_name, unique_value))
  return ids

ids = generate_ids(collection_name, [str(processed_chunk["order"]) for processed_chunk in processed_chunks])
```

```python
import time

start_time = time.time()
# Example usage
results = weaviate_db.upsert(
    texts=texts,
    metadatas=metadatas,
    ids=ids,
    collection_name=collection_name,
    batch_size=100,
    show_progress=True,
)

end_time = time.time()
print(f"\nProcessing complete")
print(f"Number of successfully processed documents: {len(results)}")
print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
```

<pre class="custom">문서 처리 중 오류 발생 (ID: 8e5d8d25-c745-5628-91cc-72f035859618): Object was not added! Unexpected status code: 503, with response body: None.
    Processed batch 1/5
    Processed batch 2/5
    Processed batch 3/5
    Processed batch 4/5
    Processed batch 5/5
    
    Processing complete
    Number of successfully processed documents: 457
    Total elapsed time: 214.33 seconds
</pre>

```python
import time

start_time = time.time()

results = weaviate_db.upsert_parallel(
    texts=texts,
    metadatas=metadatas,
    ids=ids,
    collection_name=collection_name,
    text_key="text",
)

end_time = time.time()
print(f"\nProcessing complete")
print(f"Number of successfully processed documents: {len(results)}")
print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
```

<pre class="custom">
    Processing complete
    Number of successfully processed documents: 458
    Total elapsed time: 8.07 seconds
</pre>

### Search items from Weaviate

You can search items from `weaviate` by filter

```python
weaviate_db.search(
    query="What is the little prince about?",
    filters={"author": "Antoine de Saint-Exupéry"},
    k=2,
    collection_name=collection_name,
    show_progress=True,
)
```




<pre class="custom">[Document(metadata={'title': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry', 'source': 'Original Text', 'order': 9, 'uuid': 'c78af9d2-00b1-5637-9904-f925cb8e2107'}, page_content='To console himself, he drew upon his experiences over the Saharan desert to write and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is stranded in the'),
     Document(metadata={'title': 'The Little Prince', 'order': 10, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': '00d8fa75-c17d-5d21-8820-0175c0d461d1'}, page_content='In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince discovers the true meaning of life. At the end of his conversation with the Little Prince, the aviator')]</pre>



### Delete items from Weaviate

You can delete items from `weaviate` by filter

First, let's search for documents that contain the text `Hum! Hum!` in the `text` property.

```python
weaviate_db.keyword_search(
    query="Hum! Hum!",
    filters={"author": "Antoine de Saint-Exupéry"},
    k=2,
    collection_name=collection_name,
    show_progress=True,
)
```




<pre class="custom">[Document(metadata={'title': 'The Little Prince', 'order': 199, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': 'bef162c8-9707-5016-b1b4-3fe66a35f32b'}, page_content='"Hum! Hum!" replied the king; and before saying anything else he consulted a bulky almanac. "Hum! Hum! That will be about-- about-- that will be this evening about twenty minutes to eight. And you will see how well I am obeyed."'),
     Document(metadata={'title': 'The Little Prince', 'order': 185, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': 'dd0f094c-35e4-5fbd-b24c-8a638b06cb77'}, page_content='"Hum! Hum!" replied the king. "Then I-- I order you sometimes to yawn and sometimes to--"\nHe sputtered a little, and seemed vexed.')]</pre>



Now let's delete the document with the filter applied.

```python
weaviate_db.delete(collection_name=collection_name, ids=None, filters={"author": "Antoine de Saint-Exupéry"})
```




<pre class="custom">True</pre>



Let's verify that the document was deleted properly.

```python
weaviate_db.keyword_search(
    query="Hum! Hum!",
    filters={"author": "Antoine de Saint-Exupéry"},
    k=2,
    collection_name=collection_name,
    show_progress=True,
)
```




<pre class="custom">[]</pre>



Great job, now let's dive into Similarity Search with Langchain Vector Store.

----

## Finding Objects by Similarity

Weaviate allows you to find objects that are semantically similar to your query. Let's walk through a complete example, from importing data to executing similarity searches.

### Step 1: Preparing Your Data

Before we can perform similarity searches, we need to populate our Weaviate instance with data. We'll start by loading and chunking a text file into manageable pieces.

> 💡 **Tip** : Breaking down large texts into smaller chunks helps optimize vector search performance and relevance.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = WeaviateVectorStore(
    client=client, index_name=collection_name, embedding=embeddings, text_key="text"
)

vector_store.add_documents(split_docs[:5])
```




<pre class="custom">['6b892c6d-7f7c-4687-a6de-b27029724070',
     '45107ac6-4dfe-4cd5-a020-9ccc208aa012',
     '25503fea-c128-49d5-9e1d-a0ef6c5529f9',
     '64410acb-3f7e-4762-a656-1e0f661f9f7d',
     '28abbe7e-56d0-48a0-962b-ad06b0c9b14f']</pre>



### Step 2: Perform the search

We can now perform a similarity search. This will return the most similar documents to the query text, based on the embeddings stored in Weaviate and an equivalent embedding generated from the query text.

```python
from utils.weaviate_vectordb import WeaviateSearch

query = "What is the little prince about?"
searcher = WeaviateSearch(vector_store)
docs = searcher.similarity_search(query, k=1)

for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)
```

<pre class="custom">
    Document 1:
    The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
</pre>

You can also add filters, which will either include or exclude results based on the filter conditions. (See [more filter examples](https://weaviate.io/developers/weaviate/search/filters).)

It is also possible to provide `k` , which is the upper limit of the number of results to return.

```python
from weaviate.classes.query import Filter

filter_query = Filter.by_property("text").equal("In the book, a pilot is")

searcher.similarity_search(
    query=query,
    filter_query=filter_query,
    k=1,
)
```




<pre class="custom">[]</pre>



### Quantify Result Similarity

When performing similarity searches, you might want to know not just which documents are similar, but how similar they are. Weaviate provides this information through a relevance score.
> 💡 Tip: The relevance score helps you understand the relative similarity between search results.

```python
docs = searcher.similarity_search_with_score(query, k=5)

for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content)
```

<pre class="custom">1.000 : The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    0.391 : [ Antoine de Saiot-Exupery ]
    0.333 : Over the past century, the thrill of flying has inspired some to perform remarkable feats of daring. For others, their desire to soar into the skies led to dramatic leaps in technology. For Antoine de Saint-Exupéry, his love of aviation inspired stories, which have touched the hearts of millions
    0.164 : Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the
    0.000 : have touched the hearts of millions around the world.
</pre>

## Search mechanism

`similarity_search` uses Weaviate's [hybrid search](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid).

A hybrid search combines a vector and a keyword search, with `alpha` as the weight of the vector search. The `similarity_search` function allows you to pass additional arguments as kwargs. See this [reference doc](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid) for the available arguments.

So, you can perform a pure keyword search by adding `alpha=0` as shown below:

```python
docs = searcher.similarity_search(query, alpha=0)
docs[0]
```




<pre class="custom">Document(metadata={'title': None, 'author': None, 'source': None, 'order': None}, page_content='The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)')</pre>



## Persistence

Any data added through `langchain-weaviate` will persist in Weaviate according to its configuration. 

WCS instances, for example, are configured to persist data indefinitely, and Docker instances can be set up to persist data in a volume. Read more about [Weaviate's persistence](https://weaviate.io/developers/weaviate/configuration/persistence).

## Multi-tenancy

[Multi-tenancy](https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy) allows you to have a high number of isolated collections of data, with the same collection configuration, in a single Weaviate instance. This is great for multi-user environments such as building a SaaS app, where each end user will have their own isolated data collection.

To use multi-tenancy, the vector store need to be aware of the `tenant` parameter. 

So when adding any data, provide the `tenant` parameter as shown below.

```python
# 2. Create a vector store with a specific tenant
vector_store_with_tenant = WeaviateVectorStore.from_documents(
    docs, embeddings, client=client, tenant="tenant1"  # specify the tenant name
)
```

<pre class="custom">2025-Feb-08 12:03 AM - langchain_weaviate.vectorstores - INFO - Tenant tenant1 does not exist in index LangChain_faa4f5a05fab42fba487b3487000b232. Creating tenant.
</pre>

```python
results = vector_store_with_tenant.similarity_search(
    query, tenant="tenant1"  # use the same tenant name
)

for doc in results:
    print(doc.page_content)
```

<pre class="custom">The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
</pre>

```python
vector_store_with_tenant = WeaviateVectorStore.from_documents(
    docs, embeddings, client=client, tenant="tenant1", mt=True
)
```

<pre class="custom">2025-Feb-08 12:03 AM - langchain_weaviate.vectorstores - INFO - Tenant tenant1 does not exist in index LangChain_c255d8854e9146c28d3698df6bb51d46. Creating tenant.
</pre>

And when performing queries, provide the `tenant` parameter also.

```python
vector_store_with_tenant.similarity_search(query, tenant="tenant1")
```




<pre class="custom">[Document(metadata={'title': None, 'author': None, 'source': None, 'order': None}, page_content='The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)')]</pre>



## Retriever options

Weaviate can also be used as a retriever

### Maximal marginal relevance search (MMR)

In addition to using similaritysearch  in the retriever object, you can also use `mmr`

```python
retriever = vector_store.as_retriever(search_type="mmr")
retriever.invoke(query)[0]
```

<pre class="custom">Failed to multipart ingest runs: langsmith.utils.LangSmithRateLimitError: Rate limit exceeded for https://api.smith.langchain.com/runs/multipart. HTTPError('429 Client Error: Too Many Requests for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Monthly unique traces usage limit exceeded"}')trace=10200441-130b-4dc2-94d8-0c74fcfa107c,id=10200441-130b-4dc2-94d8-0c74fcfa107c
</pre>




    Document(metadata={'title': None, 'author': None, 'source': None, 'order': None}, page_content='The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)')



    Failed to multipart ingest runs: langsmith.utils.LangSmithRateLimitError: Rate limit exceeded for https://api.smith.langchain.com/runs/multipart. HTTPError('429 Client Error: Too Many Requests for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Monthly unique traces usage limit exceeded"}')trace=10200441-130b-4dc2-94d8-0c74fcfa107c,id=10200441-130b-4dc2-94d8-0c74fcfa107c
    

### Delete Collection

Managing collections in Weaviate includes the ability to remove them when they're no longer needed. The `delete_collection` function provides a straightforward way to remove collections from your Weaviate instance.

**Function Signature:**
- `client` : Weaviate client instance for database connection
- `collection_name` : Name of the collection to be deleted

**Advanced Operations:**
For batch operations or managing multiple collections, you can use the `delete_all_collections()` function, which removes all collections from your Weaviate instance.

> **Important:** Collection deletion is permanent and cannot be undone. Always ensure you have appropriate backups before deleting collections in production environments.

```python
# weaviate_db.delete_all_collections(client)    # if you want to delete all collections, uncomment this line
weaviate_db.delete_collection(client, collection_name)
```

<pre class="custom">Deleted index: BookChunk
</pre>

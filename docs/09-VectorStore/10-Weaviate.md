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
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
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
import weaviate
from weaviate.classes.init import Auth

weaviate_url = os.environ.get("WEAVIATE_URL")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers={"X-Openai-Api-Key": os.environ.get("OPENAI_API_KEY")},
)

print(client.is_ready())
```

<pre class="custom">True
</pre>

```python
## api key Lookup
def get_api_key():
    return weaviate_api_key


print(get_api_key())
```

## What is Weaviate?

Weaviate is a powerful open-source vector database that revolutionizes how we store and search data. It combines traditional database capabilities with advanced machine learning features, allowing you to:

- Weaviate is an open source [vector database](https://weaviate.io/blog/what-is-a-vector-database).
- Weaviate allows you to store and retrieve data objects based on their semantic properties by indexing them with [vectors](./concepts/vector-index.md).
- Weaviate can be used stand-alone (aka _bring your vectors_) or with a variety of [modules](./modules/index.md) that can do the vectorization for you and extend the core capabilities.
- Weaviate has a [GraphQL-API](./api/graphql/index.md) to access your data easily.
- Weaviate is fast (check our [open source benchmarks](./benchmarks/index.md)).

> 💡 **Key Feature**: Weaviate achieves millisecond-level query performance, making it suitable for production environments.

## Why Use Weaviate?

Weaviate stands out for several reasons:

1. **Versatility**: Supports multiple media types (text, images, etc.)
2. **Advanced Features**:
   - Semantic Search
   - Question-Answer Extraction
   - Classification
   - Custom ML Model Integration
3. **Production-Ready**: Built in Go for high performance and scalability
4. **Developer-Friendly**: Multiple access methods through GraphQL, REST, and various client libraries


## Initialization
Before initializing our vector store, let's connect to a Weaviate collection. If one named index_name doesn't exist, it will be created.

### Creating Collections in Weaviate

The `create_collection` function establishes a new collection in Weaviate, configuring it with specified properties and vector settings. This foundational operation requires six key parameters:

**Required Parameters:**
- `client`: Weaviate client instance for database connection
- `collection_name`: Unique identifier for your collection
- `description`: Detailed description of the collection's purpose
- `properties`: List of property definitions for data schema
- `vectorizer`: Configuration for vector embedding generation
- `metric`: Distance metric for similarity calculations

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

```python
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from typing import List


def create_collection(
    client: weaviate.Client,
    collection_name: str,
    description: str,
    properties: List[Property],
    vectorizer: Configure.Vectorizer,
    metric: str = "cosine",
) -> None:
    """
    Creates a new index (collection) in Weaviate with the specified properties.

    :param client: Weaviate client instance
    :param collection_name: Name of the index (collection) (e.g., "BookChunk")
    :param description: Description of the index (e.g., "A collection for storing book chunks")
    :param properties: List of properties, where each property is a dictionary with keys:
        - name (str): Name of the property
        - dataType (list[str]): Data types for the property (e.g., ["text"], ["int"])
        - description (str): Description of the property
    :param vectorizer: Vectorizer configuration created using Configure.Vectorizer
                       (e.g., Configure.Vectorizer.text2vec_openai())
    :return: None
    """
    distance_metric = getattr(VectorDistances, metric.upper(), None)

    # Set vector_index_config to hnsw
    vector_index_config = Configure.VectorIndex.hnsw(distance_metric=distance_metric)

    # Create the collection in Weaviate
    try:
        client.collections.create(
            name=collection_name,
            description=description,
            properties=properties,
            vectorizer_config=vectorizer,
            vector_index_config=vector_index_config,
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create collection '{collection_name}': {e}")
```

Now let's use the `create_collection` function to create the collection we'll use in this tutorial.

```python
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

create_collection(client, collection_name, description, properties, vectorizer, metric)
```

<pre class="custom">Collection 'BookChunk' created successfully.
</pre>

### Delete Collection

Managing collections in Weaviate includes the ability to remove them when they're no longer needed. The `delete_collection` function provides a straightforward way to remove collections from your Weaviate instance.

**Function Signature:**
- `client`: Weaviate client instance for database connection
- `collection_name`: Name of the collection to be deleted

**Advanced Operations:**
For batch operations or managing multiple collections, you can use the `delete_all_collections()` function, which removes all collections from your Weaviate instance.

> **Important:** Collection deletion is permanent and cannot be undone. Always ensure you have appropriate backups before deleting collections in production environments.

```python
def delete_collection(client, collection_name):
    client.collections.delete(collection_name)
    print(f"Deleted index: {collection_name}")


def delete_all_collections():
    client.collections.delete_all()
    print("Deleted all collections")


# delete_all_collections()    # if you want to delete all collections, uncomment this line
delete_collection(client, collection_name)
```

<pre class="custom">Deleted index: BookChunk
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
def list_collections():
    """
    Lists all collections (indexes) in the Weaviate database, including their properties.
    """
    # Retrieve all collection configurations
    collections = client.collections.list_all()

    # Check if there are any collections
    if collections:
        print("Collections (indexes) in the Weaviate schema:")
        for name, config in collections.items():
            print(f"- Collection name: {name}")
            print(
                f"  Description: {config.description if config.description else 'No description available'}"
            )
            print(f"  Properties:")
            for prop in config.properties:
                print(f"    - Name: {prop.name}, Type: {prop.data_type}")
            print()
    else:
        print("No collections found in the schema.")


list_collections()
```

<pre class="custom">Collections (indexes) in the Weaviate schema:
    - Collection name: LangChain_4c510d6dc12d46069d5b6a74a742c4ff
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.NUMBER
        - Name: source, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: title, Type: DataType.TEXT
    
    - Collection name: LangChain_25ab58a0f16d476a8d261bd4a11245be
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
    
    - Collection name: BookChunk
      Description: A chunk of a book's content
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.INT
        - Name: title, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: source, Type: DataType.TEXT
    
    - Collection name: LangChain_e63c8e8a49cc4915995dae2fcdf1aef1
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.NUMBER
        - Name: source, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: title, Type: DataType.TEXT
    
    - Collection name: LangChain_a6190f02a2f64ff4aca85e3c24f8e8cb
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
    
    - Collection name: LangChain_be71f63889d74d09b2ade15d384ec210
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: source, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: title, Type: DataType.TEXT
        - Name: order, Type: DataType.NUMBER
    
    - Collection name: LangChain_bd62d989508f479a8ab02fcc3190010e
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.NUMBER
        - Name: source, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: title, Type: DataType.TEXT
    
    - Collection name: LangChain_0a18b4c9d03f4f3d8ab2e7a6258d9a2c
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
        - Name: order, Type: DataType.NUMBER
        - Name: source, Type: DataType.TEXT
        - Name: author, Type: DataType.TEXT
        - Name: title, Type: DataType.TEXT
    
    - Collection name: LangChain_7ead0866ef9f4e3eb559142c74f79446
      Description: No description available
      Properties:
        - Name: text, Type: DataType.TEXT
    
</pre>

```python
def lookup_collection(collection_name: str):
    return client.collections.get(collection_name)


print(lookup_collection(collection_name))
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
with open("./data/the_little_prince.txt") as f:
    raw_text = f.read()
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False,
)

split_docs = text_splitter.create_documents([raw_text])

print(split_docs[:20])
```

<pre class="custom">[Document(metadata={}, page_content='The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)'), Document(metadata={}, page_content='[ Antoine de Saiot-Exupery ]'), Document(metadata={}, page_content='Over the past century, the thrill of flying has inspired some to perform remarkable feats of daring. For others, their desire to soar into the skies led to dramatic leaps in technology. For Antoine'), Document(metadata={}, page_content='in technology. For Antoine de Saint-Exupéry, his love of aviation inspired stories, which have touched the hearts of millions around the world.'), Document(metadata={}, page_content='Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French'), Document(metadata={}, page_content='hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the military in order to begin flying air mail between remote settlements in'), Document(metadata={}, page_content='between remote settlements in the Sahara desert.'), Document(metadata={}, page_content="For Saint-Exupéry, it was a grand adventure - one with dangers lurking at every corner. Flying his open cockpit biplane, Saint-Exupéry had to fight the desert's swirling sandstorms. Worse, still, he"), Document(metadata={}, page_content="sandstorms. Worse, still, he ran the risk of being shot at by unfriendly tribesmen below. Saint-Exupéry couldn't have been more thrilled. Soaring across the Sahara inspired him to spend his nights"), Document(metadata={}, page_content='him to spend his nights writing about his love affair with flying.'), Document(metadata={}, page_content='When World War II broke out, Saint-Exupéry rejoined the French Air Force. After Nazi troops overtook France in 1940, Saint-Exupéry fled to the United States. He had hoped to join the U. S. war effort'), Document(metadata={}, page_content='to join the U. S. war effort as a fighter pilot, but was dismissed because of his age. To console himself, he drew upon his experiences over the Saharan desert to write and illustrate what would'), Document(metadata={}, page_content='and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is'), Document(metadata={}, page_content='In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince'), Document(metadata={}, page_content='the book, the little prince discovers the true meaning of life. At the end of his conversation with the Little Prince, the aviator manages to fix his plane and both he and the little prince continue'), Document(metadata={}, page_content='the little prince continue on their journeys'), Document(metadata={}, page_content='Shortly after completing the book, Saint-Exupéry finally got his wish. He returned to North Africa to fly a warplane for his country. On July 31, 1944, Saint-Exupéry took off on a mission. Sadly, he'), Document(metadata={}, page_content='off on a mission. Sadly, he was never heard from again.'), Document(metadata={}, page_content='[ TO LEON WERTH ]'), Document(metadata={}, page_content='I ask the indulgence of the children who may read this book for dedicating it to a grown-up. I have a serious reason: he is the best friend I have in the world. I have another reason: this grown-up')]
</pre>

### Document Preprocessing Function

The `preprocess_documents` function transforms pre-split documents into a format suitable for Weaviate storage. This utility function handles both document content and metadata, ensuring proper organization of your data.

**Function Parameters:**
- `split_docs`: List of LangChain Document objects containing page content and metadata
- `metadata`: Optional dictionary of additional metadata to include with each chunk

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

    return processed_chunks


metadata = {
    "title": "The Little Prince",
    "author": "Antoine de Saint-Exupéry",
    "source": "Original Text",
}

processed_chunks = preprocess_documents(split_docs, metadata=metadata)

processed_chunks[:10]
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
     {'text': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of daring. For others, their desire to soar into the skies led to dramatic leaps in technology. For Antoine',
      'order': 3,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'in technology. For Antoine de Saint-Exupéry, his love of aviation inspired stories, which have touched the hearts of millions around the world.',
      'order': 4,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French',
      'order': 5,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the military in order to begin flying air mail between remote settlements in',
      'order': 6,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'between remote settlements in the Sahara desert.',
      'order': 7,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': "For Saint-Exupéry, it was a grand adventure - one with dangers lurking at every corner. Flying his open cockpit biplane, Saint-Exupéry had to fight the desert's swirling sandstorms. Worse, still, he",
      'order': 8,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': "sandstorms. Worse, still, he ran the risk of being shot at by unfriendly tribesmen below. Saint-Exupéry couldn't have been more thrilled. Soaring across the Sahara inspired him to spend his nights",
      'order': 9,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'},
     {'text': 'him to spend his nights writing about his love affair with flying.',
      'order': 10,
      'title': 'The Little Prince',
      'author': 'Antoine de Saint-Exupéry',
      'source': 'Original Text'}]</pre>



## Manage vector store
Once you have created your vector store, we can interact with it by adding and deleting different items.

### Add Items to Vector Store

Weaviate provides flexible methods for adding documents to your vector store. This section explores two efficient approaches: standard insertion and parallel batch processing, each optimized for different use cases.

#### Standard Insertion
Best for smaller datasets or when processing order is important:
- Sequential document processing
- Automatic UUID generation
- Built-in duplicate handling
- Real-time progress tracking

#### Parallel Batch Processing
Optimized for large-scale document ingestion:
- Multi-threaded processing
- Configurable batch sizes
- Concurrent execution
- Enhanced throughput

**Configuration Options:**
- `batch_size`: Control memory usage and processing chunks
- `max_workers`: Adjust concurrent processing threads
- `unique_key`: Define document identification field
- `show_progress`: Monitor ingestion progress

**Performance Tips:**
- For datasets < 1000 documents: Use standard insertion
- For datasets > 1000 documents: Consider parallel processing
- Monitor memory usage when increasing batch size
- Adjust worker count based on available CPU cores

> **Best Practice:** Choose your ingestion method based on dataset size and system resources. Start with conservative batch sizes and gradually optimize based on performance metrics.

```python
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = WeaviateVectorStore(
    client=client, index_name=collection_name, embedding=embeddings, text_key="text"
)
```

```python
from weaviate.util import generate_uuid5
import time


def upsert_documents(
    vector_store: WeaviateVectorStore,
    docs: List[Dict],
    unique_key: str = "order",
    batch_size: int = 100,
    show_progress: bool = True,
) -> List[str]:
    """
    Upserts documents into the WeaviateVectorStore.
    """
    # Prepare Document objects and IDs
    documents = []
    ids = []

    for doc in docs:
        unique_value = str(doc[unique_key])
        doc_id = generate_uuid5(vector_store._index_name, unique_value)

        documents.append(
            Document(
                page_content=doc["text"],
                metadata={k: v for k, v in doc.items() if k != "text"},
            )
        )
        ids.append(doc_id)

    # Generate embeddings
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    embeddings = vector_store.embeddings.embed_documents(texts)

    # Get the collection
    collection = vector_store._client.collections.get(vector_store._index_name)
    successful_ids = []

    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size] if metadatas else None

            for j, text in enumerate(batch_texts):
                properties = {"text": text}
                if batch_metadatas:
                    properties.update(batch_metadatas[j])

                try:
                    # First, check if the object exists
                    exists = collection.data.exists(uuid=batch_ids[j])

                    if exists:
                        # If the object exists, update it
                        collection.data.replace(
                            uuid=batch_ids[j],
                            properties=properties,
                            vector=batch_embeddings[j],
                        )
                    else:
                        # If the object does not exist, insert it
                        collection.data.insert(
                            uuid=batch_ids[j],
                            properties=properties,
                            vector=batch_embeddings[j],
                        )
                    successful_ids.append(batch_ids[j])

                except Exception as e:
                    print(f"Error processing document (ID: {batch_ids[j]}): {e}")
                    continue

            if show_progress:
                print(
                    f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
                )

    except Exception as e:
        print(f"Error during batch processing: {e}")

    return successful_ids


start_time = time.time()

# Example usage
results = upsert_documents(
    vector_store=vector_store,
    docs=processed_chunks,
    unique_key="order",
    batch_size=100,
    show_progress=True,
)

end_time = time.time()
print(f"\nProcessing complete")
print(f"Number of successfully processed documents: {len(results)}")
print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
```

<pre class="custom">Processed batch 1/7
    Processed batch 2/7
    Processed batch 3/7
    Processed batch 4/7
    Processed batch 5/7
    Processed batch 6/7
    Processed batch 7/7
    
    Processing complete
    Number of successfully processed documents: 698
    Total elapsed time: 316.36 seconds
</pre>

```python
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


def upsert_documents_parallel(
    vector_store: WeaviateVectorStore,
    docs: List[Dict],
    unique_key: str = "order",
    batch_size: int = 100,
    max_workers: Optional[int] = 4,
    show_progress: bool = True,
) -> List[str]:
    """
    Upserts documents in parallel to WeaviateVectorStore.

    Args:
        vector_store: WeaviateVectorStore instance
        docs: List of documents to upsert
        unique_key: Key to use as the unique identifier
        batch_size: Size of each batch
        max_workers: Maximum number of workers
        show_progress: Whether to show progress
    Returns:
        List[str]: List of IDs of successfully processed documents
    """

    # Divide data into batches
    def create_batches(data: List, size: int) -> List[List]:
        return [data[i : i + size] for i in range(0, len(data), size)]

    batched_docs = create_batches(docs, batch_size)

    def process_batch(batch: List[Dict]) -> List[str]:
        try:
            return upsert_documents(
                vector_store=vector_store,
                docs=batch,
                unique_key=unique_key,
                batch_size=len(batch),
                show_progress=False,  # Do not show progress for individual batches
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            return []

    successful_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch): i
            for i, batch in enumerate(batched_docs)
        }

        if show_progress:
            with tqdm(total=len(batched_docs), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    batch_result = future.result()
                    successful_ids.extend(batch_result)
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                batch_result = future.result()
                successful_ids.extend(batch_result)

    return successful_ids


# Example usage
start_time = time.time()

results = upsert_documents_parallel(
    vector_store=vector_store,
    docs=processed_chunks,
    unique_key="order",
    batch_size=100,  # Set batch size
    max_workers=4,  # Set maximum number of workers
    show_progress=True,
)

end_time = time.time()
print(f"\nProcessing complete")
print(f"Number of successfully processed documents: {len(results)}")
print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
```

<pre class="custom">Processing batches: 100%|██████████| 7/7 [01:31<00:00, 13.02s/it]</pre>

    
    Processing complete
    Number of successfully processed documents: 698
    Total elapsed time: 94.17 seconds
    

    
    

```python
from langchain_weaviate import WeaviateVectorStore
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from weaviate.collections.classes.filters import Filter
from typing import Any, List, Dict, Optional, Union, Tuple
from langchain_core.documents import Document
from weaviate.collections.classes.filters import Filter


class WeaviateSearch:
    def __init__(self, vector_store: WeaviateVectorStore):
        """
        Initialize Weaviate search class
        """
        self.vector_store = vector_store
        self.collection = vector_store._client.collections.get(vector_store._index_name)
        self.text_key = vector_store._text_key

    def _format_filter(self, filter_query: Filter) -> str:
        """
        Converts a Filter object to a readable string.

        Args:
            filter_query: Weaviate Filter object

        Returns:
            str: Filter description string
        """
        if not filter_query:
            return "No filter"

        try:
            # Converts the internal structure of the Filter object to a string
            if hasattr(filter_query, "filters"):  # Composite filter (AND/OR)
                operator = "AND" if filter_query.operator == "And" else "OR"
                filter_strs = []
                for f in filter_query.filters:
                    if hasattr(f, "value"):  # Single filter
                        filter_strs.append(
                            f"({f.target} {f.operator.lower()} {f.value})"
                        )
                return f" {operator} ".join(filter_strs)
            elif hasattr(filter_query, "value"):  # Single filter
                return f"{filter_query.target} {filter_query.operator.lower()} {filter_query.value}"
            else:
                return str(filter_query)
        except Exception:
            return "Complex filter"

    def similarity_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        **kwargs: Any,
    ):
        """
        Perform basic similarity search
        """
        documents = self.vector_store.similarity_search(
            query, k=k, filters=filter_query, **kwargs
        )
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        **kwargs: Any,
    ):
        """
        Perform similarity search with score
        """
        documents_and_scores = self.vector_store.similarity_search_with_score(
            query, k=k, filters=filter_query, **kwargs
        )
        return documents_and_scores

    def mmr_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        fetch_k: int = 10,
        **kwargs: Any,
    ):
        """
        Perform MMR algorithm-based diverse search
        """
        documents = self.vector_store.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, filters=filter_query, **kwargs
        )
        return documents

    def hybrid_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        alpha: float = 0.5,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Hybrid search (keyword + vector search)

        Args:
            query: Text to search
            filter_dict: Filter condition dictionary
            alpha: Weight for keyword and vector search (0: keyword only, 1: vector only)
            limit: Number of documents to return
            return_score: Whether to return similarity score

        Returns:
            List of Documents hybrid search results
        """
        embedding_vector = self.vector_store.embeddings.embed_query(query)
        results = self.collection.query.hybrid(
            query=query,
            vector=embedding_vector,
            alpha=alpha,
            limit=limit,
            filters=filter_query,
            **kwargs,
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)

            if hasattr(obj.metadata, "score"):
                metadata["score"] = obj.metadata.score

            doc = Document(
                page_content=obj.properties.get(self.text_key, str(obj.properties)),
                metadata=metadata,
            )

            documents.append(doc)

        return documents

    def semantic_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Semantic search (vector-based)
        """
        results = self.collection.query.near_text(
            query=query, limit=limit, filters=filter_query, **kwargs
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)
            documents.append(
                Document(
                    page_content=obj.properties.get(self.text_key, str(obj.properties)),
                    metadata=metadata,
                )
            )

        return documents

    def keyword_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Keyword-based search (BM25)
        """
        results = self.collection.query.bm25(
            query=query, limit=limit, filters=filter_query, **kwargs
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)
            documents.append(
                Document(
                    page_content=obj.properties.get(self.text_key, str(obj.properties)),
                    metadata=metadata,
                )
            )

        return documents

    def create_qa_chain(
        self,
        llm: BaseChatModel = None,
        chain_type: str = "stuff",
        retriever: BaseRetriever = None,
        **kwargs: Any,
    ):
        """
        Create search-QA chain
        """
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            **kwargs,
        )
        return qa_chain

    def print_results(
        self,
        results: Union[List[Document], List[Tuple[Document, float]]],
        search_type: str,
        filter_query: Optional[Filter] = None,
    ) -> None:
        """
        Print search results in a readable format

        Args:
            results: List of Document or (Document, score) tuples
            search_type: Search type (e.g., "Hybrid", "Semantic" etc.)
            filter_dict: Applied filter information
        """
        print(f"\n=== {search_type.upper()} SEARCH RESULTS ===")
        if filter_query:
            print(f"Filter: {self._format_filter(filter_query)}")

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")

            # Separate Document object and score
            if isinstance(result, tuple):
                doc, score = result
                print(f"Score: {score:.4f}")
            else:
                doc = result

            # Print content
            print(f"Content: {doc.page_content}")

            # Print metadata
            if doc.metadata:
                print("\nMetadata:")
                for key, value in doc.metadata.items():
                    if (
                        key != "score" and key != "uuid"
                    ):  # Exclude already printed information
                        print(f"  {key}: {value}")

            print("-" * 50)

    def print_search_comparison(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 5,
        alpha: float = 0.5,
        fetch_k: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Print comparison of all search methods' results

        Args:
            query: Search query
            filter_dict: Filter condition
            limit: Number of results
            alpha: Weight for hybrid search (0: keyword only, 1: vector only)
            fetch_k: Number of candidate documents for MMR search
            **kwargs: Additional search parameters
        """
        search_methods = [
            # 1. Basic similarity search
            {
                "name": "Similarity Search",
                "method": self.similarity_search,
                "params": {"k": limit},
            },
            # 2. Similarity search with score
            {
                "name": "Similarity Search with Score",
                "method": self.similarity_search_with_score,
                "params": {"k": limit},
            },
            # 3. MMR search
            {
                "name": "MMR Search",
                "method": self.mmr_search,
                "params": {"k": limit, "fetch_k": fetch_k},
            },
            # 4. Hybrid search
            {
                "name": "Hybrid Search",
                "method": self.hybrid_search,
                "params": {"limit": limit, "alpha": alpha},
            },
            # 5. Semantic search
            {
                "name": "Semantic Search",
                "method": self.semantic_search,
                "params": {"limit": limit},
            },
            # 6. Keyword search
            {
                "name": "Keyword Search",
                "method": self.keyword_search,
                "params": {"limit": limit},
            },
        ]

        print("\n=== SEARCH METHODS COMPARISON ===")
        print(f"Query: {query}")
        if filter_query:
            print(f"Filter: {self._format_filter(filter_query)}")
        print("=" * 50)

        for search_config in search_methods:
            try:
                method_params = {
                    **search_config["params"],
                    "query": query,
                    "filter_query": filter_query,
                    **kwargs,
                }

                results = search_config["method"](**method_params)

                print(f"\n>>> {search_config['name'].upper()} <<<")
                self.print_results(results, search_config["name"], filter_query)

            except Exception as e:
                print(f"\nError in {search_config['name']}: {str(e)}")

            print("\n" + "=" * 50)
```

```python
searcher = WeaviateSearch(vector_store)

filter_query = Filter.by_property("author").equal("Antoine de Saint-Exupéry")

searcher.print_search_comparison(
    query="What is the little prince about?",
    filter_query=filter_query,
    limit=3,
    alpha=0.5,  # keyword/vector weight for hybrid search
    fetch_k=10,  # number of candidate documents for MMR search
)
```

<pre class="custom">
    === SEARCH METHODS COMPARISON ===
    Query: What is the little prince about?
    Filter: author equal Antoine de Saint-Exupéry
    ==================================================
    
    >>> SIMILARITY SEARCH <<<
    
    === SIMILARITY SEARCH SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Content: In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    
    Metadata:
      title: The Little Prince
      author: Antoine de Saint-Exupéry
      source: Original Text
      order: 14
    --------------------------------------------------
    
    Result 2:
    Content: and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is
    
    Metadata:
      title: The Little Prince
      order: 13
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 3:
    Content: The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    
    Metadata:
      title: The Little Prince
      author: Antoine de Saint-Exupéry
      source: Original Text
      order: 1
    --------------------------------------------------
    
    ==================================================
    
    >>> SIMILARITY SEARCH WITH SCORE <<<
    
    === SIMILARITY SEARCH WITH SCORE SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Score: 0.7000
    Content: In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    
    Metadata:
      title: The Little Prince
      order: 14
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 2:
    Score: 0.6264
    Content: and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is
    
    Metadata:
      title: The Little Prince
      order: 13
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 3:
    Score: 0.6003
    Content: The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    
    Metadata:
      title: The Little Prince
      author: Antoine de Saint-Exupéry
      source: Original Text
      order: 1
    --------------------------------------------------
    
    ==================================================
    
    >>> MMR SEARCH <<<
    
    === MMR SEARCH SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Content: In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    
    Metadata:
      title: The Little Prince
      order: 14
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 2:
    Content: The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    
    Metadata:
      title: The Little Prince
      author: Antoine de Saint-Exupéry
      source: Original Text
      order: 1
    --------------------------------------------------
    
    Result 3:
    Content: And that is how I made the acquaintance of the little prince.
    
    Metadata:
      title: The Little Prince
      author: Antoine de Saint-Exupéry
      source: Original Text
      order: 78
    --------------------------------------------------
    
    ==================================================
    
    >>> HYBRID SEARCH <<<
    
    === HYBRID SEARCH SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Content: [ Chapter 7 ]
    - the narrator learns about the secret of the little prince‘s life
    
    Metadata:
      title: The Little Prince
      order: 174
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 2:
    Content: [ Chapter 3 ]
    - the narrator learns more about from where the little prince came
    
    Metadata:
      title: The Little Prince
      order: 79
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 3:
    Content: In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    
    Metadata:
      title: The Little Prince
      order: 14
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    ==================================================
    
    >>> SEMANTIC SEARCH <<<
    
    === SEMANTIC SEARCH SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Content: In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    
    Metadata:
      title: The Little Prince
      order: 14
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 2:
    Content: and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is
    
    Metadata:
      title: The Little Prince
      order: 13
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 3:
    Content: The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    
    Metadata:
      title: The Little Prince
      order: 1
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    ==================================================
    
    >>> KEYWORD SEARCH <<<
    
    === KEYWORD SEARCH SEARCH RESULTS ===
    Filter: author equal Antoine de Saint-Exupéry
    
    Result 1:
    Content: "Hum! Hum!" replied the king; and before saying anything else he consulted a bulky almanac. "Hum! Hum! That will be about-- about-- that will be this evening about twenty minutes to eight. And you
    
    Metadata:
      title: The Little Prince
      order: 291
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 2:
    Content: have made a new friend, they never ask you any questions about essential matters. They never say to you, "What does his voice sound like? What games does he love best? Does he collect butterflies?"
    
    Metadata:
      title: The Little Prince
      order: 110
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    Result 3:
    Content: figures do they think they have learned anything about him.
    
    Metadata:
      title: The Little Prince
      order: 112
      source: Original Text
      author: Antoine de Saint-Exupéry
    --------------------------------------------------
    
    ==================================================
</pre>

### Delete items from vector store

You can delete items from vector store by filter

First, let's search for documents that contain the text `Hum! Hum!` in the `text` property.

```python
filter_query = Filter.by_property("text").equal("Hum! Hum!")

searcher.keyword_search(
    query="Hum! Hum!",
    filter_query=filter_query,
    limit=3,
)
```




<pre class="custom">[Document(metadata={'title': 'The Little Prince', 'order': 291, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': '16ddf535-a610-510c-b597-1fd3ce13360f'}, page_content='"Hum! Hum!" replied the king; and before saying anything else he consulted a bulky almanac. "Hum! Hum! That will be about-- about-- that will be this evening about twenty minutes to eight. And you'),
     Document(metadata={'title': 'The Little Prince', 'order': 269, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': 'a4c46e83-a491-5c1a-be06-e6635dfa58e5'}, page_content='"That frightens me... I cannot, any more..." murmured the little prince, now completely abashed.\n"Hum! Hum!" replied the king. "Then I-- I order you sometimes to yawn and sometimes to--"'),
     Document(metadata={'title': 'The Little Prince', 'order': 301, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry', 'uuid': 'a8ff68c1-db62-51f6-a03b-5e12aceda12f'}, page_content='"Hum! Hum!" said the king. "I have good reason to believe that somewhere on my planet there is an old rat. I hear him at night. You can judge this old rat. From time to time you will condemn him to')]</pre>



Now let's delete the document with the filter applied.

```python
from weaviate.collections.classes.filters import Filter


def delete_by_filter(collection_name: str, filter_query: Filter) -> int:
    try:
        # Retrieve the collection
        collection = client.collections.get(collection_name)

        # Check the number of documents that match the filter before deletion
        query_result = collection.query.fetch_objects(
            filters=filter_query,
        )
        initial_count = len(query_result.objects)

        # Delete documents that match the filter condition
        collection.data.delete_many(where=filter_query)

        print(f"Number of documents deleted: {initial_count}")
        return initial_count

    except Exception as e:
        print(f"Error occurred during deletion: {e}")
        raise


delete_by_filter(collection_name=collection_name, filter_query=filter_query)
```

<pre class="custom">Number of documents deleted: 3
</pre>




    3



Let's verify that the document was deleted properly.

```python
searcher.keyword_search(
    query="Hum! Hum!",
    filter_query=filter_query,
    limit=3,
)
```




<pre class="custom">[]</pre>



Great job, now let's dive into Similarity Search with a simple example.

----

## Finding Objects by Similarity

Weaviate allows you to find objects that are semantically similar to your query. Let's walk through a complete example, from importing data to executing similarity searches.

### Step 1: Preparing Your Data

Before we can perform similarity searches, we need to populate our Weaviate instance with data. We'll start by loading and chunking a text file into manageable pieces.

> 💡 **Tip**: Breaking down large texts into smaller chunks helps optimize vector search performance and relevance.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This is a long document we can split up.
with open("./data/the_little_prince.txt") as f:
    raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False,
)

split_docs = text_splitter.create_documents([raw_text])
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = WeaviateVectorStore(
    client=client, index_name=collection_name, embedding=embeddings, text_key="text"
)
```

### Step 2: Perform the search

We can now perform a similarity search. This will return the most similar documents to the query text, based on the embeddings stored in Weaviate and an equivalent embedding generated from the query text.

```python
query = "What is the little prince about?"
searcher = WeaviateSearch(vector_store)
docs = searcher.similarity_search(query, k=1)

for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)
```

<pre class="custom">
    Document 1:
    In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
</pre>

You can also add filters, which will either include or exclude results based on the filter conditions. (See [more filter examples](https://weaviate.io/developers/weaviate/search/filters).)

It is also possible to provide `k`, which is the upper limit of the number of results to return.

```python
from weaviate.classes.query import Filter

filter_query = Filter.by_property("text").equal("In the book, a pilot is")

searcher.similarity_search(
    query=query,
    filter_query=filter_query,
    k=1,
)
```




<pre class="custom">[Document(metadata={'title': 'The Little Prince', 'order': 14, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry'}, page_content='In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince')]</pre>



### Quantify Result Similarity

When performing similarity searches, you might want to know not just which documents are similar, but how similar they are. Weaviate provides this information through a relevance score.
> 💡 Tip: The relevance score helps you understand the relative similarity between search results.

```python
docs = searcher.similarity_search_with_score(query, k=5)

for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content)
```

<pre class="custom">0.700 : In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince
    0.627 : and illustrate what would become his most famous book, The Little Prince (1943). Mystical and enchanting, this small book has fascinated both children and adults for decades. In the book, a pilot is
    0.600 : The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    0.525 : [ Chapter 7 ]
    - the narrator learns about the secret of the little prince‘s life
    0.519 : [ Chapter 3 ]
    - the narrator learns more about from where the little prince came
</pre>

## Search mechanism

`similarity_search` uses Weaviate's [hybrid search](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid).

A hybrid search combines a vector and a keyword search, with `alpha` as the weight of the vector search. The `similarity_search` function allows you to pass additional arguments as kwargs. See this [reference doc](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid) for the available arguments.

So, you can perform a pure keyword search by adding `alpha=0` as shown below:

```python
docs = searcher.similarity_search(query, alpha=0)
docs[0]
```




<pre class="custom">Document(metadata={'title': 'The Little Prince', 'order': 110, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry'}, page_content='have made a new friend, they never ask you any questions about essential matters. They never say to you, "What does his voice sound like? What games does he love best? Does he collect butterflies?"')</pre>



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

<pre class="custom">2025-Jan-19 09:14 PM - langchain_weaviate.vectorstores - INFO - Tenant tenant1 does not exist in index LangChain_866945876dc24c83bb0247ce4324bdbd. Creating tenant.
</pre>

```python
results = vector_store_with_tenant.similarity_search(
    query, tenant="tenant1"  # use the same tenant name
)

for doc in results:
    print(doc.page_content)
```

<pre class="custom">"Yes?" said the little prince, who did not understand what the conceited man was talking about. 
    "Clap your hands, one against the other," the conceited man now directed him.
    have made a new friend, they never ask you any questions about essential matters. They never say to you, "What does his voice sound like? What games does he love best? Does he collect butterflies?"
    figures do they think they have learned anything about him.
</pre>

```python
vector_store_with_tenant = WeaviateVectorStore.from_documents(
    docs, embeddings, client=client, tenant="tenant1", mt=True
)
```

<pre class="custom">2025-Jan-19 09:14 PM - langchain_weaviate.vectorstores - INFO - Tenant tenant1 does not exist in index LangChain_c07a19db3f994319935be1ccdeb957c0. Creating tenant.
</pre>

And when performing queries, provide the `tenant` parameter also.

```python
vector_store_with_tenant.similarity_search(query, tenant="tenant1")
```




<pre class="custom">[Document(metadata={'title': 'The Little Prince', 'order': 313.0, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry'}, page_content='"Yes?" said the little prince, who did not understand what the conceited man was talking about. \n"Clap your hands, one against the other," the conceited man now directed him.'),
     Document(metadata={'title': 'The Little Prince', 'order': 110.0, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry'}, page_content='have made a new friend, they never ask you any questions about essential matters. They never say to you, "What does his voice sound like? What games does he love best? Does he collect butterflies?"'),
     Document(metadata={'title': 'The Little Prince', 'order': 112.0, 'source': 'Original Text', 'author': 'Antoine de Saint-Exupéry'}, page_content='figures do they think they have learned anything about him.')]</pre>



## Retriever options

Weaviate can also be used as a retriever

### Maximal marginal relevance search (MMR)

In addition to using similaritysearch  in the retriever object, you can also use `mmr`.

```python
retriever = vector_store.as_retriever(search_type="mmr")
retriever.invoke(query)[0]
```




<pre class="custom">Document(metadata={'title': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry', 'source': 'Original Text', 'order': 14}, page_content='In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little prince')</pre>



## Use with LangChain

A known limitation of large language models (LLMs) is that their training data can be outdated, or not include the specific domain knowledge that you require.

Take a look at the example below:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
result = llm.invoke(query)
print(result.content)
```

<pre class="custom">"The Little Prince" is a novella written by Antoine de Saint-Exupéry, first published in 1943. The story is narrated by a pilot who crashes in the Sahara Desert and meets a young boy who appears to be a prince. The little prince hails from a small asteroid called B-612 and shares his adventures and experiences as he travels from one planet to another.
    
    Throughout the story, the little prince encounters various inhabitants of different planets, each representing different aspects of human nature and society, such as a king, a vain man, a drunkard, a businessman, a geographer, and a fox. These encounters serve as allegories for adult behaviors and societal norms, often highlighting themes of loneliness, love, friendship, and the loss of innocence.
    
    One of the central messages of the book is the importance of seeing with the heart rather than just the eyes, emphasizing that true understanding and connection come from emotional and spiritual insight rather than superficial appearances. The story also explores themes of childhood, imagination, and the essence of what it means to be human.
    
    Ultimately, "The Little Prince" is a poignant reflection on the nature of relationships, the value of love, and the wisdom that can be found in simplicity and innocence. It has resonated with readers of all ages and is considered a classic of world literature.
</pre>

Vector stores complement LLMs by providing a way to store and retrieve relevant information. This allow you to combine the strengths of LLMs and vector stores, by using LLM's reasoning and linguistic capabilities with vector stores' ability to retrieve relevant information.

Two well-known applications for combining LLMs and vector stores are:
- Question answering
- Retrieval-augmented generation (RAG)

### Question Answering with Sources

Question answering in langchain can be enhanced by the use of vector stores. Let's see how this can be done.

This section uses the `RetrievalQAWithSourcesChain`, which does the lookup of the documents from an Index. 

We can construct the chain, with the retriever specified:

```python
searcher = WeaviateSearch(vector_store)

chain = searcher.create_qa_chain(
    llm=llm, retriever=vector_store.as_retriever(), chain_type="stuff"
)
```

```python
chain.invoke(
    {"question": query},
    return_only_outputs=True,
)
```




<pre class="custom">{'answer': 'The Little Prince is about a pilot who is stranded in the Sahara Desert and encounters a tiny prince from another world. The prince is traveling the universe to understand life. The story is mystical and enchanting, captivating both children and adults for decades.\n\n',
     'sources': 'Original Text'}</pre>



### Retrieval-Augmented Generation

Another very popular application of combining LLMs and vector stores is retrieval-augmented generation (RAG). This is a technique that uses a retriever to find relevant information from a vector store, and then uses an LLM to provide an output based on the retrieved data and a prompt.

We begin with a similar setup:

We need to construct a template for the RAG model so that the retrieved information will be populated in the template.

```python
from langchain_core.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
```

<pre class="custom">input_variables=['context', 'question'] input_types={} partial_variables={} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:\n"), additional_kwargs={})]
</pre>

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke(query)
```




<pre class="custom">'"The Little Prince" is about a pilot who, while stranded in the Sahara, meets a young prince from another world who is exploring the universe to understand life. The story contrasts the prince\'s innocent perspective with the often misguided views of adults. It explores themes of love, loss, and the importance of seeing beyond the surface.'</pre>



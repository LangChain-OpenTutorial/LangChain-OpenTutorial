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

# Faiss

- Author: [Ilgyun Jeong](https://github.com/johnny9210)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers how to use ```Faiss``` with **LangChain** .

```Faiss``` (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also includes supporting code for evaluation and parameter tuning.

This tutorial walks you through using **CRUD** operations with the ```Faiss``` **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Faiss?](#what-is-Faiss?)
- [Data](#data)
- [Initial Setting Faiss](#initial-setting-Faiss)
- [Document Manager](#document-manager)


### References

- [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/Faiss-a-library-for-efficient-similarity-search/)
- [Faiss Library paper](https://arxiv.org/pdf/2401.08281)
- [Faiss documentation](https://Faiss.ai/)
- [Langchain Faiss document](https://python.langchain.com/docs/integrations/vectorstores/Faiss/)
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
# Install necessary package
%pip install -qU langchain-community Faiss-cpu

# Note that you can also install Faiss-gpu if you want to use the GPU enabled version
# Install necessary package
# %pip install -qU langchain-community Faiss-gpu
```

<pre class="custom">ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    langchain-weaviate 0.0.3 requires simsimd<5.0.0,>=3.6.1, but you have simsimd 6.2.1 which is incompatible.
    
    [notice] A new release of pip is available: 24.3.1 -> 25.1.1
    [notice] To update, run: pip install --upgrade pip
    Note: you may need to restart the kernel to use updated packages.
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "faiss-cpu",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.1.1
    [notice] To update, run: pip install --upgrade pip
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "Your OPENAI API KEY",
        "LANGCHAIN_API_KEY": "Your LangChain API KEY",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Faiss",
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



## What is Faiss?

```Faiss``` (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.

* Core Concepts
  * ```Similarity search```: Finding vectors that are closest to a query vector
  * ```Scaling```: Handles vector sets of any size, including those exceeding RAM
  * ```Efficiency```: Optimized for memory usage and search speed

* Vector Operations
  * ```Nearest neighbor```: Finding k vectors closest to a query vector
  * ```Maximum inner product```: Finding vectors with highest dot product
  * ```Clustering```: Grouping similar vectors together

* Index Types
  * ```Flat```: Exact search with exhaustive comparison
  * ```IVF```: Inverted file structure for faster approximate search
  * ```HNSW```: Hierarchical navigable small world graphs for high-quality search
  * ```PQ```: Product quantization for memory compression
  * ```OPQ```: Optimized product quantization for better accuracy

* Performance Metrics
  * ```Speed```: Query time for finding similar vectors
  * ```Memory```: RAM requirements for index storage
  * ```Accuracy```: How well results match exhaustive search (recall)

* Technical Features
  * ```GPU support```: State-of-the-art GPU implementations with 5-20x speedup
  * ```Multi-threading```: Parallel processing across CPU cores
  * ```SIMD optimization```: Vectorized operations for faster computation
  * ```Half-precision```: Float16 support for better performance

* Applications
  * ```Image similarity```: Finding visually similar images
  * ```Text embeddings```: Semantic search in document collections
  * ```Recommendation systems```: Finding similar items for users
  * ```Classification```: Computing maximum inner-products for classification

## Data

This part walks you through the **data preparation process** .

This section includes the following components:

- Introduce Data

- Preprocessing Data


### Introduce Data

In this tutorial, we will use the fairy tale **📗 The Little Prince** in PDF format as our data.

This material complies with the **Apache 2.0 license** .

The data is used in a text (.txt) format converted from the original PDF.

You can view the data at the link below.
- [Data Link](https://huggingface.co/datasets/sohyunwriter/the_little_prince)

### Preprocessing Data

In this tutorial section, we will preprocess the text data from The Little Prince and convert it into a list of ```LangChain Document``` objects with metadata. 

Each document chunk will include a ```title``` field in the metadata, extracted from the first line of each section.

```python
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from typing import List


def preprocessing_data(content: str) -> List[Document]:
    # 1. Split the text by double newlines to separate sections
    blocks = content.split("\n\n")

    # 2. Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Maximum number of characters per chunk
        chunk_overlap=50,  # Overlap between chunks to preserve context
        separators=["\n\n", "\n", " "],  # Order of priority for splitting
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
        title = title_match.group(1).strip() if title_match else None

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

# Preprocessing Data

docs = preprocessing_data(content=content)
```

<pre class="custom">Generated 262 chunked documents.
</pre>

## Initial Setting Faiss

This part walks you through the initial setup of ```Faiss``` .

This section includes the following components:

- Load Embedding Model

- Load ```Faiss``` Client

### Load Embedding Model

In the **Load Embedding Model** section, you'll learn how to load an embedding model.

This tutorial uses **OpenAI's** **API-Key** for loading the model.

*💡 If you prefer to use another embedding model, see the instructions below.*
- [Embedding Models](https://python.langchain.com/docs/integrations/text_embedding/)

```python
import os
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Load Faiss Client

In the **Load ```Faiss``` Client** section, we cover how to load the **database client object** using the **Python SDK** for ```Faiss``` .
- [Faiss Python SDK Docs](https://github.com/facebookresearch/faiss/wiki/getting-started?utm_source=chatgpt.com)

```python
# Create Database Client Object Function
import faiss
import numpy as np


def get_db_client(dim: int = 128):
    """

    Initializes and returns a VectorStore client instance.


    This function loads configuration (e.g., API key, host) from environment

    variables or default values and creates a client object to interact

    with the faiss Python SDK.


    Returns:

        client:ClientType - An instance of the faiss client.


    Raises:

        ValueError: If required configuration is missing.

    """

    base_index = faiss.IndexFlatL2(dim)  # L2 거리 기반 인덱스 생성
    client = faiss.IndexIDMap(base_index)  # ID 매핑 지원 추가

    return client
```

```python
# Get DB Client Object
client = get_db_client()
```

## Document Manager

To support the **Langchain-Opentutorial** , we implemented a custom set of **CRUD** functionalities for VectorDBs. 

The following operations are included:

- ```upsert``` : Update existing documents or insert if they don’t exist

- ```upsert_parallel``` : Perform upserts in parallel for large-scale data

- ```similarity_search``` : Search for similar documents based on embeddings

- ```delete``` : Remove documents based on filter conditions

Each of these features is implemented as class methods specific to each VectorDB.

In this tutorial, you can easily utilize these methods to interact with your VectorDB.

*We plan to continuously expand the functionality by adding more common operations in the future.*

### Create Instance

First, we create an instance of the ```faiss``` helper class to use its CRUD functionalities.

This class is initialized with the **```faiss``` Python SDK client instance**, **index name** and the **embedding model instance** , both of which were defined in the previous section.

```python
# import FaissCRUDManager
from utils.faiss import FaissCRUDManager

# connect to tutorial_index
crud_manager = FaissCRUDManager(dim=3072, embedding=embedding)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your ```faiss``` .

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
    # if you want args, add params.
}

crud_manager.upsert(**args)
```

### Upsert Parallel Document

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
    # if you want args, add params.
}

crud_manager.upsert_parallel(**args)
```

### Similarity Search

Search for **similar documents** based on **embeddings** .

This method uses **"cosine similarity"** .


**✅ Args**

- ```query``` : str – The text query for similarity search.

- ```k``` : int – Number of top results to return (default: 10).

```**kwargs``` : Additional search options (e.g., filters).

**🔄 Return**

- ```results``` : List[Document] – A list of LangChain Document objects ranked by similarity.

```python
# Search by Query

results = crud_manager.search(query="What is essential is invisible to the eye.", k=3)
for idx, result in enumerate(results):
    print(f"Rank {idx+1}")
    print(f"Contents : {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Similarity Score: {result['score']}")
    print()
```

<pre class="custom">Rank 1
    Contents : And he went back to meet the fox. 
    "Goodbye," he said. 
    "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the heart that one can see rightly; what is essential is invisible to the eye." 
    "What is essential is invisible to the eye," the little prince repeated, so that he would be sure to remember.
    "It is the time you have wasted for your rose that makes your rose so important."
    Metadata: {'id': 'e9bcc211-223c-464b-9141-362e177d46cc', 'title': 'Chapter 21'}
    Similarity Score: 0.504
    
    Rank 2
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    Metadata: {'id': 'ceeaae40-51e4-4ad9-842d-f96d6590dd0c', 'title': 'Chapter 24'}
    Similarity Score: 0.498
    
    Rank 3
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'id': 'c75865c7-a928-43d0-a109-ef32bceebbae', 'title': 'Chapter 25'}
    Similarity Score: 0.464
    
</pre>

### Delete Document

Remove documents based on filter conditions

**✅ Args**

- ```ids``` : Optional[List[str]] – List of document IDs to delete. If None, deletion is based on filter.

- ```filters``` : Optional[Dict] – Dictionary specifying filter conditions (e.g., metadata match).

- ```**kwargs``` : Any additional parameters.

**🔄 Return**

- Boolean

```python
# Delete by ids

ids = ids[:5]  # The 'ids' value you want to delete
crud_manager.delete(ids=ids)
```




<pre class="custom">True</pre>



```python
# Delete by ids with filters

filters = {"page": 6}
crud_manager.delete(filters={"title": "chapter 6"})
```




<pre class="custom">True</pre>



```python
# Delete All

crud_manager.delete()
```




<pre class="custom">True</pre>



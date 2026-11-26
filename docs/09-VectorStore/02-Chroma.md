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

# Chroma

- Author: [Pupba](https://github.com/pupba)
- Peer Review: [liniar](https://github.com/namyoungkim), [Youngin Kim](https://github.com/Normalist-K), [BokyungisaGod](https://github.com/BokyungisaGod), [Sohyeon Yim](https://github.com/sohyunwriter)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb)

## Overview

This tutorial covers how to use **Chroma** with **LangChain** .

**Chroma** is an open-source vector database optimized for semantic search and RAG applications. It offers **fast similarity search**, **metadata filtering**, and supports both **in-memory** and **persistent storage**. With built-in or **custom embedding functions** and a **simple Python API**, it's easy to integrate into ML pipelines.

This tutorial walks you through using **CRUD** operations with the **Chroma** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Chroma?](#what-is-chroma?)
- [Prepare Data](#Prepare-Data)
- [Setting up Chroma](#Setting-up-Chroma)
- [Document Manager](#document-manager)


### References
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
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Chroma",
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



## What is Chroma?

![chroma](./img/02-chroma-with-langchain-chroma-logo.png)

```Chroma``` is an open-source embedding database built for enabling semantic search in AI applications.  
It is commonly used in Retrieval-Augmented Generation(RAG) pipelines to manage and search document embeddings efficiently.

Unlike traditional databases or pure vector stores, ```Chroma``` combines vector similarity with structured metadata filtering.  

This allows developers to build hybrid search systems that consider both the meaning of the text and metadata constraints.

### Key Features

- **Easy-to-use API** : Simplifies vector management and querying through a clean Python interface.

- **Persistent storage** : Supports both in-memory and on-disk storage for scalable deployment.

- **Metadata filtering** : Enables precise search using custom fields stored alongside vectors.

- **Built-in similarity search** : Provides fast approximate nearest-neighbor (ANN) retrieval using cosine distance.

- **Local-first and open-source** : No cloud lock-in; can run entirely on local or edge environments.

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

## Setting up Chroma

This part walks you through the initial setup of ```Chroma``` .

This section includes the following components:

- Load Embedding Model

- Load Qdrant Client

### Load Embedding Model

In this section, you'll learn how to load an embedding model.

This tutorial uses **OpenAI's** **API-Key** for loading the model.

*💡 If you prefer to use another embedding model, see the instructions below.*
- [Embedding Models](https://python.langchain.com/docs/integrations/text_embedding/)

```python
import os
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Load Chroma Client
In this section, we'll show you how to load the **database client object** using the **Python SDK** for ```Chroma``` .
- [Python SDK Docs](https://docs.trychroma.com/docs/overview/introduction)

```python
# Create Database Client Object Function
import chromadb


def get_db_client():
    """
    Initializes and returns a VectorStore client instance.

    This function loads configuration (e.g., API key, host) from environment
    variables or default values and creates a client object to interact
    with the Chroma Python SDK.

    Returns:
        client:ClientType - An instance of the Chroma client.

    Raises:
        ValueError: If required configuration is missing.
    """
    client = chromadb.Client()  # in-memory
    return client
```

```python
# Get DB Client Object
client = get_db_client()
```

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

First, create an instance of the **Qdrant** helper class to use its CRUD functionalities.

This class is initialized with the **Qdrant Python SDK client instance** and the **embedding model instance** , both of which were defined in the previous section.

```python
from utils.chroma import ChromaDocumentMangager

crud_manager = ChromaDocumentMangager(client=client, embedding=embedding)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your ```Chroma``` .

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

args = {
    "texts": [doc.page_content for doc in docs[:2]],
    "metadatas": [doc.metadata for doc in docs[:2]],
    "ids": [str(uuid4()) for _ in docs[:2]],
    # Add additional parameters if you need
}
crud_manager.upsert(**args)
```

### Upsert Parallel

Perform **upsert** in **parallel** for large-scale data

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
    "ids": [str(uuid4()) for _ in docs],
    # Add additional parameters if you need
}

crud_manager.upsert_parallel(**args)
```

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
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print()
```

<pre class="custom">Rank 0 | Title : Chapter 21
    Contents : And he went back to meet the fox. 
    "Goodbye," he said. 
    "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the heart that one can see rightly; what is essential is invisible to the eye." 
    "What is essential is invisible to the eye," the little prince repeated, so that he would be sure to remember.
    "It is the time you have wasted for your rose that makes your rose so important."
    
    Rank 1 | Title : Chapter 24
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    
    Rank 2 | Title : Chapter 25
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    
</pre>

```python
# Filter Search
results = crud_manager.search(
    query="Which asteroid did the little prince come from?",
    k=3,
    where={"title": "Chapter 4"},
)
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print()
```

<pre class="custom">Rank 0 | Title : Chapter 4
    Contents : I have serious reason to believe that the planet from which the little prince came is the asteroid known as B-612. This asteroid has only once been seen through the telescope. That was by a Turkish astronomer, in 1909. 
    (picture)
    On making his discovery, the astronomer had presented it to the International Astronomical Congress, in a great demonstration. But he was in Turkish costume, and so nobody would believe what he said.
    Grown-ups are like that...
    
    Rank 1 | Title : Chapter 4
    Contents : - the narrator speculates as to which asteroid from which the little prince came　　
    I had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!
    
    Rank 2 | Title : Chapter 4
    Contents : Just so, you might say to them: "The proof that the little prince existed is that he was charming, that he laughed, and that he was looking for a sheep. If anybody wants a sheep, that is a proof that he exists." And what good would it do to tell them that? They would shrug their shoulders, and treat you like a child. But if you said to them: "The planet he came from is Asteroid B-612," then they would be convinced, and leave you in peace from their questions.
    
</pre>

### as_retriever

The ```as_retriever()``` method creates a LangChain-compatible retriever wrapper.

This function allows a ```DocumentManager``` class to return a retriever object by wrapping the internal ```search()``` method, while staying lightweight and independent from full LangChain ```VectorStore``` dependencies.

The retriever obtained through this function is compatible with existing LangChain retrievers and can be used in LangChain Pipelines (e.g., RetrievalQA, ConversationalRetrievalChain, Tool, etc.)

**✅ Args**

- ```search_fn``` : Callable - The function used to retrieve relevant documents. Typically this is ```self.search``` from a ```DocumentManager``` instance.

- ```search_kwargs``` : Optional[Dict] - A dictionary of keyword arguments passed to ```search_fn```, such as ```k``` for top-K results or metadata filters.

**🔄 Return**

- ```LightCustomRetriever``` :BaseRetriever - A lightweight LangChain-compatible retriever that internally uses the given ```search_fn``` and ```search_kwargs```.

```python
# Search without filters
ret = crud_manager.as_retriever(
    search_fn=crud_manager.search, search_kwargs={"k": 1}
)
```

```python
ret.invoke("Which asteroid did the little prince come from?")
```




<pre class="custom">[Document(metadata={'id': '2b08e739-220f-4297-a287-d93f51780dd2', 'score': 0.66, 'title': 'Chapter 4'}, page_content='I have serious reason to believe that the planet from which the little prince came is the asteroid known as B-612. This asteroid has only once been seen through the telescope. That was by a Turkish astronomer, in 1909. \n(picture)\nOn making his discovery, the astronomer had presented it to the International Astronomical Congress, in a great demonstration. But he was in Turkish costume, and so nobody would believe what he said.\nGrown-ups are like that...')]</pre>



```python
# Search with filters
ret = crud_manager.as_retriever(
    search_fn=crud_manager.search,
    search_kwargs={
        "k": 2,
        "where": {"title": "Chapter 4"}  # Filter to only search in Chapter 4
    }
)
```

```python
print("Example 2: Search with title filter (Chapter 4)")
print(ret.invoke("Which asteroid did the little prince come from?"))
```

<pre class="custom">Example 2: Search with title filter (Chapter 4)
    [Document(metadata={'id': '2b08e739-220f-4297-a287-d93f51780dd2', 'score': 0.66, 'title': 'Chapter 4'}, page_content='I have serious reason to believe that the planet from which the little prince came is the asteroid known as B-612. This asteroid has only once been seen through the telescope. That was by a Turkish astronomer, in 1909. \n(picture)\nOn making his discovery, the astronomer had presented it to the International Astronomical Congress, in a great demonstration. But he was in Turkish costume, and so nobody would believe what he said.\nGrown-ups are like that...'), Document(metadata={'id': 'd3f87cbe-6dae-4007-bb1a-4f119721a24a', 'score': 0.64, 'title': 'Chapter 4'}, page_content='- the narrator speculates as to which asteroid from which the little prince came\u3000\u3000\nI had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!')]
</pre>

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
ids = args["ids"][:3]  # The 'ids' value you want to delete
crud_manager.delete(ids=ids)
```

<pre class="custom">3 data deleted
</pre>

```python
# Delete by ids with filters
ids = args["ids"][3:]  # The `ids` value corresponding to chapter 6
crud_manager.delete(ids=ids, filters={"where": {"title": "Chapter 6"}})
```

<pre class="custom">4 data deleted
</pre>

```python
# Delete All
crud_manager.delete()
```

<pre class="custom">257 data deleted
</pre>

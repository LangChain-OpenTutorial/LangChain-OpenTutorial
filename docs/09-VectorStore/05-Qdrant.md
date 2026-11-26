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

# Qdrant

- Author: [HyeonJong Moon](https://github.com/hj0302), [Pupba](#https://github.com/pupba)
- Peer Review: [liniar](https://github.com/namyoungkim), [hellohotkey](https://github.com/hellohotkey), [Sohyeon Yim](https://github.com/sohyunwriter)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/05-Qdrant.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/05-Qdrant.ipynb)

## Overview

This tutorial covers how to use **Qdrant****Qdrant** with **LangChain** .

**Qdrant** is a high-performance, open-source vector database that stands out with advanced filtering, payload indexing, and native support for hybrid (vector + keyword) search.

This tutorial walks you through using **CRUD** operations with the **Qdrant** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Qdrant?](#what-is-qdrant?)
- [Prepare Data](#Prepare-Data)
- [Setting up Qdrant](#Setting-up-Qdrant)
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
        "qdrant-client",
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
        "LANGCHAIN_PROJECT": "Qdrant",
        "QDRANT_API_KEY": "",
        "QDRANT_URL": "",
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



## What is Qdrant?

![qdrant-logo](./img/05-qdrant-logo.png)

Qdrant is an open-source vector database and similarity search engine built in Rust, designed to handle high-dimensional vector data efficiently.

It provides a production-ready service with a user-friendly API for storing, searching, and managing vectors along with additional payload data.

### Key Features

- **High Performance** : Built in Rust for speed and reliability, handling billions of vectors with low latency.  

- **Advanced Filtering** : Supports complex filtering with JSON payloads, enabling precise searches based on metadata.  

- **Hybrid Search** : Combines vector similarity with keyword-based filtering for enhanced search capabilities.  

- **Scalable Deployment** : Offers cloud-native scalability with options for on-premise, cloud, and hybrid deployments.  

- **Multi-language Support** : Provides client libraries for Python, JavaScript/TypeScript, Go, and more.  

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

## Setting up Qdrant

This part walks you through the initial setup of **Qdrant** .

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

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
```

### Load Qdrant Client

In this section, we'll show you how to load the **database client object** using the **Python SDK** for **Qdrant** .
- [Python SDK Docs](https://python-client.qdrant.tech/)

```python
# Create Database Client Object Function
from qdrant_client import QdrantClient
import os


def get_db_client():
    """
    Initializes and returns a VectorStore client instance.

    This function loads configuration (e.g., API key, host) from environment
    variables or default values and creates a client object to interact
    with the Qdrant Python SDK.

    Returns:
        client:ClientType - An instance of the Qdrant client.

    Raises:
        ValueError: If required configuration is missing.
    """

    # In this tutorial, use Qdrant Cloud.
    # Get your personal Qdrant Cloud URL and API_Key on the official website.
    # https://qdrant.tech/documentation/cloud-intro/
    # If you want to use on-premise, please refer to -> https://qdrant.tech/documentation/quickstart/

    host = os.environ.get("QDRANT_URL", None)
    api_key = os.environ.get("QDRANT_API_KEY", None)
    try:
        client = QdrantClient(url=host, api_key=api_key, timeout=30)
    except Exception as e:
        print("Error")
        print(f"{e}")
        return None
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
from utils.qdrant import QdrantDocumentManager

# ❗ Qdrant vectorizes using the embedding function. Transfer the "Embedding Function" as a parameter.
crud_manager = QdrantDocumentManager(client=client, embedding=embedding.embed_documents)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your **Qdrant** .

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
    "result_view": True,
    # Add additional parameters if you need
}

crud_manager.upsert(**args)
```

<pre class="custom">Operation_id : 70 | Status : completed
</pre>

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
# Search by query
results = crud_manager.search(query="What is essential is invisible to the eye.", k=3)
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print()
```

<pre class="custom">Rank 0 | Title : TO LEON WERTH
    Contents : TO LEON WERTH WHEN HE WAS A LITTLE BOY
    
    Rank 1 | Title : Chapter 21
    Contents : you see the grain-fields down yonder? I do not ea t bread. Wheat is of no use to me. The wheat fields have nothing to say to me. And that is sad. But you have hair that is the colour of gold. Think how wonderful that will be when you have tamed me! The grain, which is also golden, will bring me bac k the thought of you. And I shall love to listen to the wind in the wheat..."
    
    Rank 2 | Title : Chapter 27
    Contents : Look up at the sky. Ask yourselves: is it yes or no? Has the sheep eaten the flower? And you will see how everything changes... 
    And no grown-up will ever understand that this is a matter of so much importance! 
    (picture)
    This is, to me, the loveliest and saddest landscape in the world. It is the same as that on the preceding page, but I have drawn it again to impress it on your memory. It is here that the little prince appeared on Earth, and disappeared.
    
</pre>

```python
# Search by query with filters
results = crud_manager.search(
    query="Which asteroid did the little prince come from?",
    k=3,
    filters=[{"title": "Chapter 4"}],
)
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print()
```

<pre class="custom">Rank 0 | Title : Chapter 4
    Contents : But that did not really surprise me much. I knew very well that in addition to the great planets-- such as the Earth, Jupiter, Mars, Venus-- to which we have given names, there are also hundreds of others, some of which are so small that one has a hard time seeing them through the telescope. When an astronomer discovers one of these he does not give it a name, but only a number. He might call it, for example, "Asteroid 325."
    
    Rank 1 | Title : Chapter 4
    Contents : - the narrator speculates as to which asteroid from which the little prince came　　
    I had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!
    
    Rank 2 | Title : Chapter 4
    Contents : weigh? How much money does his father make?" Only from these figures do they think they have learned anything about him.
    
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




<pre class="custom">[Document(metadata={'title': 'TO LEON WERTH', 'score': 0.2232914, 'id': 'cfa0c496-ab0b-4a31-8f13-90f97ed713da'}, page_content='TO LEON WERTH WHEN HE WAS A LITTLE BOY')]</pre>



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
    [Document(metadata={'title': 'TO LEON WERTH', 'score': 0.2232914, 'id': 'cfa0c496-ab0b-4a31-8f13-90f97ed713da'}, page_content='TO LEON WERTH WHEN HE WAS A LITTLE BOY'), Document(metadata={'title': 'Chapter 21', 'score': 0.17259452, 'id': 'b72da2e6-8793-4344-b864-7b833b615ea4'}, page_content='you see the grain-fields down yonder? I do not ea t bread. Wheat is of no use to me. The wheat fields have nothing to say to me. And that is sad. But you have hair that is the colour of gold. Think how wonderful that will be when you have tamed me! The grain, which is also golden, will bring me bac k the thought of you. And I shall love to listen to the wind in the wheat..."')]
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

<pre class="custom">3 data delete...
</pre>

```python
# Delete by ids with filters
ids = args["ids"][3:]  # The `ids` value corresponding to chapter 6
crud_manager.delete(ids=ids, filters=[{"title": "Chapter 6"}])
```

<pre class="custom">4 data delete...
    Delete All Finished
</pre>

```python
# Delete All
crud_manager.delete()
```

<pre class="custom">256 data delete...
    1 data delete...
    Delete All Finished
</pre>

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

# {VectorStore Name}

- Author: [Author Name](#Author's-Profile-Link)
- Design: [Designer](#Designer's-Profile-Link)
- Peer Review: [Reviewer Name](#Reviewer-Profile-Link)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/your-notebook-file-name) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/your-notebook-file-name)

## Overview

This tutorial covers how to use **{Vector Store Name}** with **LangChain** .

{A short introduction to vectordb}

This tutorial walks you through using **CRUD** operations with the **{VectorDB}** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is {vectordb}?](#what-is-{vectordb}?)
- [Data](#data)
- [Initial Setting {vectordb}](#initial-setting-{vectordb})
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
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "{Project Name}",
    }
)
```

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```

## What is {vectordb}?

Please write down what you need to set up the Vectorstore here.

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

# Preprocessing Data
docs = preprocessing_data(content=content)
```

## Initial Setting {vectordb}

This part walks you through the initial setup of **{vectordb}** .

This section includes the following components:

- Load Embedding Model

- Load {vectordb} Client

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

### Load {vectordb} Client

In the **Load {vectordb} Client** section, we cover how to load the **database client object** using the **Python SDK** for **{vectordb}** .
- [Python SDK Docs]()

```python
# Create Database Client Object Function
def get_db_client():
    """
    Initializes and returns a VectorStore client instance.

    This function loads configuration (e.g., API key, host) from environment
    variables or default values and creates a client object to interact
    with the {vectordb} Python SDK.

    Returns:
        client:ClientType - An instance of the {vectordb} client.

    Raises:
        ValueError: If required configuration is missing.
    """
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

First, we create an instance of the **{vectordb}** helper class to use its CRUD functionalities.

This class is initialized with the **{vectordb} Python SDK client instance** and the **embedding model instance** , both of which were defined in the previous section.

```python
# crud_manager = <Your Vectordb Class>(client=client, embedding=embedding)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your **{vectordb}** .

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
    "ids": [str(uuid4()) for _ in docs[:2]]
    # if you want args, add params.
}

# crud_manager.upsert(**args)
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
    "ids": [str(uuid4()) for _ in docs]
    # if you want args, add params.
}

# crud_manager.upsert_parallel(**args)
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
# results = crud_manager.search(query="What is essential is invisible to the eye.",k=3)
# for idx,doc in enumerate(results):
#     print(f"Rank {idx} | Title : {doc.metadata['title']}")
#     print(f"Contents : {doc.page_content}")
#     print()
```

```python
# Filter Search
# results = crud_manager.search(query="Which asteroid did the little prince come from?",k=3,<filters>={"title":"Chapter 4"})
# for idx,doc in enumerate(results):
#     print(f"Rank {idx} | Title : {doc.metadata['title']}")
#     print(f"Contents : {doc.page_content}")
#     print()
```

### as_retriever

The ```as_retriever()``` method creates a LangChain-compatible retriever wrapper.

This function allows a ```DocumentManager``` class to return a retriever object by wrapping the internal ```search()``` method, while staying lightweight and independent from full LangChain ```VectorStore``` dependencies.

The retriever obtained through this function can be used the same as the existing LangChain retriever and is **compatible with LangChain Pipeline(e.g. RetrievalQA,ConversationalRetrievalChain,Tool,...)**.

**✅ Args**

- ```search_fn``` : Callable - The function used to retrieve relevant documents. Typically this is ```self.search``` from a ```DocumentManager``` instance.

- ```search_kwargs``` : Optional[Dict] - A dictionary of keyword arguments passed to ```search_fn```, such as ```k``` for top-K results or metadata filters.

**🔄 Return**

- ```LightCustomRetriever``` :BaseRetriever - A lightweight LangChain-compatible retriever that internally uses the given ```search_fn``` and ```search_kwargs```.

```python
# Basic Search without filters
ret = crud_manager.as_retriever(
    search_fn=crud_manager.search, search_kwargs={"k": 1}
)
```

```python
ret.invoke("Which asteroid did the little prince come from?")
```

```python
# Search with title filter
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

### Delete Document

Remove documents based on filter conditions

**✅ Args**

- ```ids``` : Optional[List[str]] – List of document IDs to delete. If None, deletion is based on filter.

- ```filters``` : Optional[Dict] – Dictionary specifying filter conditions (e.g., metadata match).

- ```**kwargs``` : Any additional parameters.

**🔄 Return**

- None

```python
# Delete by ids
# ids = [] # The 'ids' value you want to delete
# crud_manager.delete(ids=ids)
```

```python
# Delete by ids with filters
# ids = [] # The `ids` value corresponding to chapter 6
# crud_manager.delete(ids=ids,filters={"title":"chapter 6"}) 
```

```python
# Delete All
# crud_manager.delete()
```

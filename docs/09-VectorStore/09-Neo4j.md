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

# Neo4j

- Author: [Jongho Lee](https://github.com/XaviereKU)
- Peer Review: [HyeonJong Moon](https://github.com/hj0302), [Haseom Shin](https://github.com/IHAGI-c), [Sohyeon Yim](https://github.com/sohyunwriter)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers how to use ```Neo4j``` with **LangChain** .

```Neo4j``` is a graph database backed by vector store and can be deployed locally or on cloud.

To fully utilize ```Neo4j```, you need to learn about ```Cypher```, declarative query language.

This tutorial walks you through using **CRUD** operations with the ```Neo4j``` **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Neo4j?](#what-is-neo4j?)
- [Prepare Data](#prepare-data)
- [Setting up Neo4j](#setting-up-neo4j)
- [Document Manager](#document-manager)


### References

- [Cypher](https://neo4j.com/docs/cypher-manual/current/introduction/)
- [Neo4j Docker Installation](https://hub.docker.com/_/neo4j)
- [Neo4j Official Installation guide](https://neo4j.com/docs/operations-manual/current/installation/)
- [Neo4j Python SDK document](https://neo4j.com/docs/api/python-driver/current/index.html)
- [Neo4j document](https://neo4j.com/docs/)
- [Langchain Neo4j document](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector/)
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
%pip install -qU neo4j
```

<pre class="custom">Note: you may need to restart the kernel to use updated packages.
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
        "langchain_openai",
        "neo4j",
        "nltk",
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
        "LANGCHAIN_PROJECT": "Neo4j",
        "NEO4J_URI": "Your Neo4j Aura URI",
        "NEO4J_USERNAME": "Your Neo4j Aura Username",
        "NEO4J_PASSWORD": "Your Neo4j Aura Password",
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



### Setup Neo4j
We have two options to start with: cloud or local deployment.

In this tutorial, we will use the cloud service called ```Aura```, provided by ```Neo4j```.

We will also describe how to deploy ```Neo4j``` using ```Docker```.

* **Getting started with Aura**
  
  You can create a new **Neo4j Aura** account on the [Neo4j](https://neo4j.com/) official website.

  Visit the website and click "Get Started" Free at the top right.

  Once you have signed in, you will see a button, **Create instance**, and after that, you will see your username and password.

  To get your API key, click **Download and continue** to download a .txt file that contains the API key to connect your **NEO4j Aura** .

* **Getting started with Docker**

  Here is the description for how to run ```Neo4j``` using ```Docker```.

  To run **Neo4j container** , use the following command.
  ```
  docker run \
      -itd \
      --publish=7474:7474 --publish=7687:7687 \
      --volume=$HOME/neo4j/data:/data \
      --env=NEO4J_AUTH=none \
      --name neo4j \
      neo4j
  ```

  You can visit **Neo4j Docker installation** reference to check more detailed information.

**[NOTE]**
* ```Neo4j``` also supports native deployment on macOS, Windows and Linux. Visit the **Neo4j Official Installation guide** reference for more details.
* The ```Neo4j community edition``` only supports one database.

## What is Neo4j?

```Neo4j``` is a native graph database, which means it represents data as nodes and edges.

* Nodes
  * ```label```: tag to represent node role in a domain.
  * ```property```: key-value pairs, e.g. name-John.

* Edges
  * Represents relationship between two nodes.
  * Directional, which means it has start and end node.
  * ```property```: like nodes, edge can have properties.

* NoSQL
  * Neo4j does not require predefined schema allowing flexible data modeling.
  
* ```Cypher```
  * ```Neo4j``` uses ```Cypher```, a declarative query language, to interact with the database.
  * ```Cypher``` expression resembles how humans think about relationships.

## Prepare Data

This section guides you through the **data preparation process** .

This section includes the following components:

- Data Introduction

- Preprocess Data


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

## Setting up Neo4j

This part walks you through the initial setup of ```Neo4j``` .

This section includes the following components:

- Load Embedding Model

- Load ```Neo4j``` Client

- Create Index

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

### Load Neo4j Client

In the **Load ```Neo4j``` Client** section, we cover how to load the **database client object** using the **Python SDK** for ```Neo4j``` .
- [Neo4j Python SDK Docs](https://neo4j.com/docs/api/python-driver/current/index.html)

```python
import time
import neo4j


# Create Database Client Object Function


def get_db_client(uri, username, password):
    """

    Initializes and returns a VectorStore client instance.


    This function loads configuration (e.g., API key, host) from environment

    variables or default values and creates a client object to interact

    with the Neo4j Python SDK.


    Returns:

        client:ClientType - An instance of the Neo4j client.


    Raises:

        ValueError: If required configuration is missing.

    """

    client = neo4j.GraphDatabase.driver(uri=uri, auth=(username, password))

    return client
```

```python
# Get DB Client Object
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

client = get_db_client(uri, username, password)
```

### Create Index
If you are successfully connected to **Neo4j Aura**, some basic indexes are already created.

But, in this tutorial we will create a new index with ```Neo4jIndexManager``` class.

```python
from utils.neo4j import Neo4jIndexManager

#  Create IndexManager Object
indexManger = Neo4jIndexManager(client)

# Create A New Index
index_name = "tutorial_index"
node_label = "tutorial_node"

# create a new index
try:
    tutorial_index = indexManger.create_index(
        embedding, index_name=index_name, metric="cosine", node_label=node_label
    )
except Exception as e:
    print("Index creation failed due to")
    print(type(e))
    print(str(e))
```

<pre class="custom">Index with name tutorial_index already exists.
    Returning Neo4jDBManager object.
    Created index information
    ('Index name: tutorial_index', 'Node label: tutorial_node', 'Similarity metric: COSINE', 'Embedding dimension: 3072', 'Embedding node property: embedding', 'Text node property: text')
    Index creation successful. Return Neo4jDBManager object.
    Index creation failed due to
    <class 'NameError'>
    name 'Neo4jCRUDManager' is not defined
</pre>

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

First, we create an instance of the ```Neo4j``` helper class to use its CRUD functionalities.

This class is initialized with the **```Neo4j``` Python SDK client instance**, **index name** and the **embedding model instance** , both of which were defined in the previous section.

```python
from utils.neo4j import Neo4jDocumentManager

crud_manager = Neo4jDocumentManager(
    client=client, index_name="tutorial_index", embedding=embedding
)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your ```Neo4j``` .

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

```**kwargs``` : Additional search options (e.g., filters).

**🔄 Return**

- ```results``` : List[Document] – A list of LangChain Document objects ranked by similarity.

```python
# Search by query
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
    Metadata: {'id': '148b9b3f-2231-4ebd-86d8-6aa841c4ac1b', 'title': 'Chapter 21', 'embedding': None}
    Similarity Score: 0.755
    
    Rank 2
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    Metadata: {'id': '62df5e3c-2668-4f5c-96ea-23c5b7a38351', 'title': 'Chapter 24', 'embedding': None}
    Similarity Score: 0.748
    
    Rank 3
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'id': 'ff93762d-6bde-44f4-b3d2-c6dc466b46a8', 'title': 'Chapter 25', 'embedding': None}
    Similarity Score: 0.711
    
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



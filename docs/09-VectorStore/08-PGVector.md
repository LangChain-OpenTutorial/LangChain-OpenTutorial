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

# PGVector

- Author: [Min-su Jung](https://github.com/effort-type), [Joonha Jeon](https://github.com/realjoonha), [Jongho Lee](https://github.com/XaviereKU)
- Peer Review : [Joonha Jeon](https://github.com/realjoonha), [Changwon Jeon](https://github.com/changwonjeon), [Sohyeon Yim](https://github.com/sohyunwriter), [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/07-PGVector.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/07-PGVector.ipynb)

## Overview

This tutorial covers how to use **PGVector** with **LangChain** .

[**PGVector**](https://github.com/pgvector/pgvector) is an open-source extension for PostgreSQL that allows you to store and search vector data alongside your regular database information.

This tutorial walks you through using **CRUD** operations with the **PGVector** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is PGVector?](#what-is-pgvector)
- [Prepare Data](#Prepare-Data)
- [Setting up PGVector](#Setting-up-PGVector)
- [Document Manager](#document-manager)


### References

- [langchain-postgres](https://github.com/langchain-ai/langchain-postgres/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Install pgvector on Windows](https://dev.to/mehmetakar/install-pgvector-on-windows-6gl)
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
        "langchain",
        "langchain_core",
        "langchain_openai",
        "pgvector",
        "psycopg",
        "psycopg-binary",
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
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "PGVector",
        "OPENAI_API_KEY": "",
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



### Set up PGVector

If you are using Windows and have installed postgresql for Windows, you are required to install **vector** extension for postgresql. The following may help [Install pgvector on Windows](https://dev.to/mehmetakar/install-pgvector-on-windows-6gl).

But in this tutorial, we will use **Docker** container. If you are using Mac or Windows, check [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install).

If you are using **Docker** desktop, you can easily set up **PGVector** by running the following command that spins up a **Docker** container:

```bash
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6088:5432 -d pgvector/pgvector:pg16
```

For more detailed instructions, please refer to [the official documentation](https://github.com/pgvector/pgvector) 

**[ NOTE ]**
* If you want to maintain the stored data even after container being deleted, you must mount volume like below:
```bash
docker run --name pgvector-container -v {/mount/path}:/var/lib/postgresql/data -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6088:5432 -d pgvector/pgvector:pg16
```


## What is PGVector?

**PGVector** is a **PostgreSQL** extension that enables vector similarity search directly within your **PostgreSQL** database, making it ideal for AI applications, semantic search, and recommendation systems.

This is particularly valuable for who already use **PostgreSQL** who want to add vector search capabilities without managing separate infrastructure or learning new query languages.

**Features** :
1. Native **PostgreSQL** integration with standard SQL queries
2. Multiple similarity search methods including L2, Inner Product, Cosine
3. Several indexing options including HNSW and IVFFlat
4. Support for up to 2,000 dimensions per vector
5. ACID compliance inherited from **PostgreSQL**

**Advantages** :

1. Free and open-source
2. Easy integration with existing **PostgreSQL** databases
3. Full SQL functionality and transactional support
4. No additional infrastructure needed
5. Supports hybrid searches combining vector and traditional SQL queries

**Disadvantages** :
1. Performance limitations with very large datasets (billions of vectors)
2. Limited to single-node deployment
3. Memory-intensive for large vector dimensions
4. Requires manual optimization for best performance
5. Less specialized features compared to dedicated vector databases

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

# Preprocess Data
docs = preprocessing_data(content=content)
```

<pre class="custom">Generated 262 chunked documents.
</pre>

## Setting up PGVector

This part walks you through the initial setup of **PGVector** .

This section includes the following components:

- Load Embedding Model

- Load PGVector Client

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

### Load PGVector Client

In this section, we'll show you how to load the **database client object** using the **Python SDK** for ```PGVector``` .
- [PGVector Python SDK Docs](https://github.com/pgvector/pgvector)

```python
# Create Database Client Object Function
from sqlalchemy import create_engine


def get_db_client(conn_str):
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

    try:
        client = create_engine(url=conn_str, **({}))
    except Exception as e:
        raise e
    else:
        return client
```

```python
# Get DB Client Object
conn_str = "postgresql+psycopg://langchain:langchain@localhost:6088/langchain"
client = get_db_client(conn_str)
```

If you are successfully running the ```PGVector``` container and get client objecct, you can use ```PGVectorIndexManager``` from ```pgvector.py``` in utils directory to handle collections.

You can also initialize ```pGVectorIndexManager``` by passing full connection string or each parameter separately instead of passing client.

```python
from utils.pgvector import PGVectorIndexManager

# Initialize pgVectorIndexManaer
index_manager = PGVectorIndexManager(client=client)
```

When you initialize ```PGVectorIndexManager```, the procedure will automatically create two tables
**langchain_pg_collection** and **langchain_pg_embedding.**

* langchain_pg_collection
    * Stores **names** of the collections.
    * Distinguish collection by uuid and name.
* langchain_pg_embedding
    * Stores actual data.
    
So, when you create a new collection and insert data to the collection, the data will be stored in **langchain_pg_embedding** table.

As you can see below, the uuid column in langchain_pg_collection table matched with collection_id column in langchain_pg_embedding table.

![pgVector Entity Relation](./img/08-pgvector-entityRelation.png)


![pgVector Collection](./img/08-pgvector-collection.png)


![pgVector Data](./img/08-pgvector-data.png)

### Create collection
Now we can create collection with ```index_manager```.

To create collection, you need to pass **embedding** model and **collection_name** when calling the ```create_index``` method.

In this tutorial we will use ```text-embedding-3-large``` model of OpenAI.

If creation is successful, the method will return ```PGVectorDocumentManager``` class that can handle actual data.

In this tutorial we will create an collection with name **langchain_opentutorial.**

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

```python
# create new collection
_ = index_manager.create_index(
    collection_name="tutorial_collection", embedding=embeddings
)
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

### Filtering
The PGVector support following filtering operations.

| Operator | Meaning/Category        |
|----------|-------------------------|
| \$eq      | Equality (==)           |
| \$ne      | Inequality (!=)         |
| \$lt      | Less than (&lt;)           |
| \$lte     | Less than or equal (&lt;=) |
| \$gt      | Greater than (>)        |
| \$gte     | Greater than or equal (>=) |
| \$in      | Special Cased (in)      |
| \$nin     | Special Cased (not in)  |
| \$between | Special Cased (between) |
| \$like    | Text (like)             |
| \$ilike   | Text (case-insensitive like) |
| \$and     | Logical (and)           |
| \$or      | Logical (or)            |

Filter can be used with ```delete```, and ```search``` methods.

To apply filter, we create a dictionary and pass it to ```filter``` parameter like the following
```python
{"page": {"$between": [10,20]}}
```

### Create Instance

First, we create an instance of the PGVector helper class to use its CRUD functionalities.

This class is initialized with the **PGVector Python SDK client instance** and the **embedding model instance** , both of which were defined in the previous section.

```python
from utils.pgvector import PGVectorDocumentManager

crud_manager = PGVectorDocumentManager(
    client=client, embedding=embedding, collection_name="tutorial_collection"
)
```

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your ```PGVector```.

### Upsert Document

**Update** existing documents or **insert** if they don’t exist

**✅ Args**

- ```texts``` : Iterable[str] – List of text contents to be inserted/updated.

- ```metadatas``` : Optional[List[Dict]] – List of metadata dictionaries for each text (optional).

- ```ids``` : Optional[List[str]] – Custom IDs for the documents. If not provided, IDs will be auto-generated.

- ```**kwargs``` : Extra arguments for the underlying vector store.

**🔄 Return**

- ```ids``` : IDs of the upserted documents.

```python
from uuid import uuid4

ids = [str(uuid4()) for _ in docs]


args = {
    "texts": [doc.page_content for doc in docs[:2]],
    "metadatas": [doc.metadata for doc in docs[:2]],
    "ids": ids[:2],
}


upsert_result = crud_manager.upsert(**args)
```

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

- ```ids``` : IDs of the upserted documents.

```python
args = {
    "texts": [doc.page_content for doc in docs],
    "metadatas": [doc.metadata for doc in docs],
    "ids": ids,
    "batch_size": 32,
    "max_workers": 8,
}

upsert_parallel_result = crud_manager.upsert_parallel(**args)
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
    print(f"Contents : {result['content']}")
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
    Metadata: {'title': 'Chapter 21'}
    Similarity Score: 0.5081315211410451
    
    Rank 2
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    Metadata: {'title': 'Chapter 24'}
    Similarity Score: 0.49512916658197925
    
    Rank 3
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'title': 'Chapter 25'}
    Similarity Score: 0.4223734643904644
    
</pre>

```python
# Search by query with filters
results = crud_manager.search(
    query="Which asteroid did the little prince come from?",
    k=3,
    filter={"title": "Chapter 4"},
)
for idx, doc in enumerate(results):
    print(f"Rank {idx+1}")
    print(f"Contents : {result['content']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Similarity Score: {result['score']}")
    print()
```

<pre class="custom">Rank 1
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'title': 'Chapter 25'}
    Similarity Score: 0.4223734643904644
    
    Rank 2
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'title': 'Chapter 25'}
    Similarity Score: 0.4223734643904644
    
    Rank 3
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'title': 'Chapter 25'}
    Similarity Score: 0.4223734643904644
    
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
crud_manager.delete(ids=ids[:10])
```

<pre class="custom">Delete done successfully
</pre>




    True



```python
# Delete by filters
crud_manager.delete(filters={"title": {"$eq": "chapter 4"}})
```

<pre class="custom">Delete done successfully
</pre>




    True



```python
# Delete All
crud_manager.delete()
```

<pre class="custom">Delete done successfully
</pre>




    True



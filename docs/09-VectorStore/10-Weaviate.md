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
- Design: [Haseom Shin](https://github.com/IHAGI-c)
- Peer Review: [Joonha Jeon](https://github.com/realjoonha), [Musang Kim](https://github.com/musangk), [Sohyeon Yim](https://github.com/sohyunwriter), [BokyungisaGod](https://github.com/BokyungisaGod), [Pupba](https://github.com/pupba)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb)

## Overview

This tutorial covers how to use **Weaviate** with **LangChain** .

[Weaviate](https://weaviate.io/) is an open-source vector database. It allows you to store data objects and vector embeddings from your favorite ML-models, and scale seamlessly into billions of data objects.

This tutorial walks you through using **CRUD** operations with the **Weaviate** **storing** , **updating** , **deleting** documents, and performing **similarity-based retrieval** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Weaviate?](#what-is-weaviate?)
- [Prepare Data](#Prepare-Data)
- [Setting up Weaviate](#setting-up-weaviate)
- [Document Manager](#document-manager)


### References
- [Langchain-Weaviate](https://python.langchain.com/docs/integrations/providers/weaviate/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Introduction](https://weaviate.io/developers/weaviate/introduction)
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
        "weaviate-client",
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

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



Please write down what you need to set up the Vectorstore here.

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

<pre class="custom">Generated 262 chunked documents.
</pre>

## Setting up Weaviate

This part walks you through the initial setup of **Weaviate** .

This section includes the following components:

- Load Embedding Model

- Load Weaviate Client

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

### Load Weaviate Client

In the **Load Weaviate Client** section, we cover how to load the **database client object** using the **Python SDK** for **Weaviate** .
- [Python SDK Docs](https://weaviate.io/developers/weaviate/client-libraries/python)

```python
import weaviate
from weaviate.classes.init import Auth

# Create Database Client Object Function
def get_db_client():
    """
    Initializes and returns a VectorStore client instance.

    This function loads configuration (e.g., API key, host) from environment
    variables or default values and creates a client object to interact
    with the Weaviate Python SDK.

    Returns:
        client:ClientType - An instance of the Weaviate client.

    Raises:
        ValueError: If required configuration is missing.
    """
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
    weaviate_url = os.environ.get("WEAVIATE_URL")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers={"X-OpenAI-Api-key": openai_api_key},
    )

    return client
```

```python
# Get DB Client Object
client = get_db_client()
```

<pre class="custom">/Users/sohyun/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-FtaFqYLT-py3.11/lib/python3.11/site-packages/weaviate/warnings.py:133: DeprecationWarning: Dep005: You are using weaviate-client version 4.10.4. The latest version is 4.14.1.
                Consider upgrading to the latest version. See https://weaviate.io/developers/weaviate/client-libraries/python for details.
      warnings.warn(
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

First, we create an instance of the **Weaviate** helper class to use its CRUD functionalities.

This class is initialized with the **Weaviate Python SDK client instance** and the **embedding model instance** , both of which were defined in the previous section.

```python
from utils.weaviate import WeaviateDocumentManager

crud_manager = WeaviateDocumentManager(
    client, collection_name="tutorial_collection", embeddings=embedding
)
```

<pre class="custom">[Weaviate] Collection 'tutorial_collection' exists
</pre>

    /Users/sohyun/project/LangChain-OpenTutorial/09-VectorStore/utils/vectordbinterface.py:76: DeprecationWarning: Retrievers must implement abstract `_get_relevant_documents` method instead of `get_relevant_documents`
      class LightCustomRetriever(BaseRetriever):
    

Now you can use the following **CRUD** operations with the ```crud_manager``` instance.

These instance allow you to easily manage documents in your **Weaviate** .

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
    "show_progress": True,
    "text_key": "text",
}

crud_manager.upsert(**args);
```

<pre class="custom">Upserted 1/2
    Upserted 2/2
</pre>

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
    "ids": [str(uuid4()) for _ in docs],
    "batch_size": 30,
    "workers": 10,
    "show_progress": True,
    "text_key": "text",
}

crud_manager.upsert_parallel(**args);
```

<pre class="custom">Upserted 30/262 documents
    Upserted 60/262 documents
    Upserted 90/262 documents
    Upserted 120/262 documents
    Upserted 150/262 documents
    Upserted 180/262 documents
    Upserted 210/262 documents
    Upserted 240/262 documents
    [Weaviate] 262 Documents Insert Complete
</pre>

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
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity Distance: {doc.metadata['distance']}")
    print()
```

<pre class="custom">Rank 0 | Title : Chapter 21
    Contents : And he went back to meet the fox. 
    "Goodbye," he said. 
    "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the heart that one can see rightly; what is essential is invisible to the eye." 
    "What is essential is invisible to the eye," the little prince repeated, so that he would be sure to remember.
    "It is the time you have wasted for your rose that makes your rose so important."
    Metadata: {'title': 'Chapter 21', 'uuid': '4476d736-4090-4da1-abf3-216fb62f03b9', 'distance': 0.4915975332260132}
    Similarity Distance: 0.4915975332260132
    
    Rank 1 | Title : Chapter 24
    Contents : "Yes," I said to the little prince. "The house, the stars, the desert-- what gives them their beauty is something that is invisible!" 
    "I am glad," he said, "that you agree with my fox."
    Metadata: {'title': 'Chapter 24', 'uuid': '329c3f94-f5f5-48bb-9712-fda3ce36ffe3', 'distance': 0.5046807527542114}
    Similarity Distance: 0.5046807527542114
    
    Rank 2 | Title : Chapter 25
    Contents : "The men where you live," said the little prince, "raise five thousand roses in the same garden-- and they do not find in it what they are looking for." 
    "They do not find it," I replied. 
    "And yet what they are looking for could be found in one single rose, or in a little water." 
    "Yes, that is true," I said. 
    And the little prince added: 
    "But the eyes are blind. One must look with the heart..."
    Metadata: {'title': 'Chapter 25', 'uuid': '45b8fd3e-f78b-4e27-99a4-cfd4b867b059', 'distance': 0.5776596665382385}
    Similarity Distance: 0.5776596665382385
    
</pre>

```python
from weaviate.classes.query import Filter

# Filter Search
results = crud_manager.search(
    query="Which asteroid did the little prince come from?",
    k=3,
    filters=Filter.by_property("title").equal("Chapter 4"),
)
for idx, doc in enumerate(results):
    print(f"Rank {idx} | Title : {doc.metadata['title']}")
    print(f"Contents : {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity Distance: {doc.metadata['distance']}")
    print()
```

<pre class="custom">Rank 0 | Title : Chapter 4
    Contents : I have serious reason to believe that the planet from which the little prince came is the asteroid known as B-612. This asteroid has only once been seen through the telescope. That was by a Turkish astronomer, in 1909. 
    (picture)
    On making his discovery, the astronomer had presented it to the International Astronomical Congress, in a great demonstration. But he was in Turkish costume, and so nobody would believe what he said.
    Grown-ups are like that...
    Metadata: {'title': 'Chapter 4', 'uuid': 'f06392b8-f2d2-4edd-b256-4198c17b1335', 'distance': 0.33773601055145264}
    Similarity Distance: 0.33773601055145264
    
    Rank 1 | Title : Chapter 4
    Contents : - the narrator speculates as to which asteroid from which the little prince came　　
    I had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!
    Metadata: {'title': 'Chapter 4', 'uuid': 'fa672b51-ccd0-488b-8a9d-ec16a2398136', 'distance': 0.3647916316986084}
    Similarity Distance: 0.3647916316986084
    
    Rank 2 | Title : Chapter 4
    Contents : Just so, you might say to them: "The proof that the little prince existed is that he was charming, that he laughed, and that he was looking for a sheep. If anybody wants a sheep, that is a proof that he exists." And what good would it do to tell them that? They would shrug their shoulders, and treat you like a child. But if you said to them: "The planet he came from is Asteroid B-612," then they would be convinced, and leave you in peace from their questions.
    Metadata: {'title': 'Chapter 4', 'uuid': 'd27c848d-97d4-4230-8a3c-de4f2c78d507', 'distance': 0.5127066373825073}
    Similarity Distance: 0.5127066373825073
    
</pre>

### As_retrever

The ```as_retriever()``` method creates a LangChain-compatible retriever wrapper.

This function allows a ```DocumentManager``` class to return a retriever object by wrapping the internal ```search()``` method, while staying lightweight and independent from full LangChain ```VectorStore``` dependencies.

The retriever obtained through this function can be used the same as the existing LangChain retriever and is **compatible with LangChain Pipeline(e.g. RetrievalQA,ConversationalRetrievalChain,Tool,...)**.

**✅ Args**

- ```search_fn``` : Callable - The function used to retrieve relevant documents. Typically this is ```self.search``` from a ```DocumentManager``` instance.

- ```search_kwargs``` : Optional[Dict] - A dictionary of keyword arguments passed to ```search_fn```, such as ```k``` for top-K results or metadata filters.

**🔄 Return**

- ```LightCustomRetriever``` :BaseRetriever - A lightweight LangChain-compatible retriever that internally uses the given ```search_fn``` and ```search_kwargs```.

```python
ret = crud_manager.as_retriever(
    search_fn=crud_manager.search,
    search_kwargs={
        "k": 3,
        "filters": Filter.by_property("title").equal("Chapter 5"),
    },
)
```

```python
ret.invoke("Which asteroid did the little prince come from?")
```




<pre class="custom">[Document(metadata={'title': 'Chapter 5', 'uuid': '518b958b-a645-4ed2-8f16-3fcccb005329', 'distance': 0.5090157389640808}, page_content='So, as the little prince described it to me, I have made a drawing of that planet. I do not much like to take the tone of a moralist. But the danger of the baobabs is so little understood, and such considerable risks would be run by anyone who might get lost on an asteroid, that for once I am breaking through my reserve. "Children," I say plainly, "watch out for the baobabs!"'),
     Document(metadata={'title': 'Chapter 5', 'uuid': '76618ef2-c13b-4ab0-b8d1-f1d641c729d8', 'distance': 0.5463582873344421}, page_content='Indeed, as I learned, there were on the planet where the little prince lived-- as on all planets-- good plants and bad plants. In consequence, there were good seeds from good plants, and bad seeds from bad plants. But seeds are invisible. They sleep deep in the heart of the earth‘s darkness, until some one among them is seized with the desire to awaken. Then this little seed will stretch itself and begin-- timidly at first-- to push a charming little sprig inoffensively upward toward the sun.'),
     Document(metadata={'title': 'Chapter 5', 'uuid': 'f8098593-28c8-45d4-9215-fc9e2062fda6', 'distance': 0.5707676410675049}, page_content='"It is a question of discipline," the little prince said to me later on. "When you‘ve finished your own toilet in the morning, then it is time to attend to the toilet of your planet, just so, with the greatest care. You must see to it that you pull up regularly all the baobabs, at the very first moment when they can be distinguished from the rosebushes which they resemble so closely in their earliest youth. It is very tedious work," the little prince added, "but very easy."')]</pre>



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
ids = [
    "fe6bfb27-4785-4105-b24a-7a69a45252ee",
    "4ca818c7-e95b-4af0-8f7c-3f1638f72a39",
    "b3364a5a-51a8-4076-80df-a9f28d38be61",
    "33193a48-dde9-4aa3-9818-c0a3f7381a9a",
]  # The 'ids' value you want to delete
crud_manager.delete(ids=ids);
```

<pre class="custom">[Weaviate] 0 document(s) deleted by ID
</pre>

```python
# Delete by ids with filters
ids = [
    "b0e85aca-ab8a-4a59-a56b-5ef437ff0e53",
    "29d9e56e-b4e5-4898-9be9-badc2c294295",
    "1554250f-04ed-45b0-b4ac-b639f858d3ad",
    "b539aade-848a-4a54-bc8f-bc5387db9d46",
]  # The `ids` value corresponding to chapter 6
crud_manager.delete(ids=ids, filters={"title": "chapter 6"});
```

<pre class="custom">[Weaviate] 0 document(s) deleted by ID+filter
</pre>

```python
# Delete All
crud_manager.delete()
```

<pre class="custom">[Weaviate] Deleted collection: tutorial_collection
</pre>




    True



```python
client.close()
```

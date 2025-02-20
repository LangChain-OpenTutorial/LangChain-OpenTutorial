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

- Author: [Gwangwon Jung](https://github.com/pupba)
- Design: []()
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb)

## Overview

This tutorial covers how to use **Chroma Vector Store** with **LangChain** .

`Chroma` is an **open-source AI application database** .

In this tutorial, after learning how to use `langchain-chroma` , we will implement examples of a simple **Text Search** engine using `Chroma` .

![search-example](./img/02-chroma-with-langchain-flow-search-example.png)

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Chroma?](#what-is-chroma?)
- [LangChain Chroma Basic](#langchain-chroma-basic)
- [Manage Store](#manage-store)
- [Query Vector Store](#query-vector-store)
- [Document Manager](#document-manager)


### References

- [Chroma Docs](https://docs.trychroma.com/docs/overview/introduction)
- [Langchain-Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [List of VectorStore supported by Langchain](https://python.langchain.com/docs/integrations/vectorstores/)
----

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
        "langsmith",
        "langchain-core",
        "langchain-chroma",
        "chromadb",
        "langchain-text-splitters",
        "langchain-huggingface",
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
        "LANGCHAIN_PROJECT": "Chroma",
        "HUGGINGFACEHUB_API_TOKEN": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is Chroma?

![logo](./img/02-chroma-with-langchain-chroma-logo.png)

`Chroma` is the **open-source vector database** designed for AI application. 

It specializes in storing high-dimensional vectors and performing fast similariy search, making it ideal for tasks like **semantic search** , **recommendation systems** and **multimodal search** .

With its **developer-friendly APIs** and seamless integration with frameworks like **LangChain** , `Chroma` is powerful tool for building scalable, AI-driven solutions.

The biggest feature of `Chroma` is that it internally **Indexing ([HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world))** and **Embedding ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))** are used when storing data.

## LangChain Chroma Basic

### Select Embedding Model

We load the **Embedding Model** with `langchain_huggingface` .

If you want to use a different model, use a different model.

```python
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "Alibaba-NLP/gte-base-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs={"trust_remote_code": True}
)
```

### Create VectorDB

The **library** supported by **LangChain** has no `upsert` function and lacks interface uniformity with other **Vector DBs**, so we have implemented a new **Python** class.

First, Load a **Python** class from **utils/chroma/basic.py** .

```python
from utils.chroma.basic import ChromaDB

vector_store = ChromaDB(embeddings=embeddings)
```

Create `ChromaDB` object.

- **Mode** : `persistent`

- **Persistent Path** : `data/chroma.sqlite` (Used `SQLite` DB)

- **collection** : `test`

- **hnsw:space** : `cosine`

```python
configs = {
    "mode": "persistent",
    "persistent_path": "data/chroma_text",
    "collection": "test",
    "hnsw:space": "cosine",
}

vector_store.connect(**configs)
```

### Load Text Documents Data

In this tutorial, we will use the **A Little Prince** fairy tale document.

To put this data in **Chroma** ,we will do data preprocessing first.

First of all, we will load the `data/the_little_prince.txt` file that extracted only the text of the fairy tale document.


```python
# If your "OS" is "Windows", add 'encoding=utf-8' to the open function
with open("./data/the_little_prince.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
```

Second, chunking the text imported into the `RecursiveCharacterTextSplitter` .

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

split_docs = text_splitter.create_documents([raw_text])

for docs in split_docs[:2]:
    print(f"Content: {docs.page_content}\nMetadata: {docs.metadata}", end="\n\n")
```

<pre class="custom">Content: The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)
    Metadata: {}
    
    Content: [ Antoine de Saiot-Exupery ]
    Metadata: {}
    
</pre>

Preprocessing document for **Chroma** .

```python
pre_dosc = vector_store.preprocess_documents(
    documents=split_docs,
    source="The Little Prince",
    author="Antoine de Saint-Exupéry",
    chapter=True,
)
```

```python
pre_dosc[:2]
```




<pre class="custom">[Document(metadata={'source': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'id': 'd5fcd392-cf17-475e-9bd6-2aecfa481ffe'}, page_content='- we are introduced to the narrator, a pilot, and his ideas about grown-ups'),
     Document(metadata={'source': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'id': '3b2062be-649a-4b11-adf7-2eb5c2634e6e'}, page_content='Once when I was six years old I saw a magnificent picture in a book, called True Stories from')]</pre>



## Manage Store

This section introduces four basic functions.

- `add`

- `upsert(parallel)`

- `query`

- `delete`

### Add

Add the new **Documents** .

An error occurs if you have the same **ID** .

```python
vector_store.add(pre_documents=pre_dosc[:2])
```

```python
uids = list(vector_store.unique_ids)
uids
```




<pre class="custom">['3b2062be-649a-4b11-adf7-2eb5c2634e6e',
     'd5fcd392-cf17-475e-9bd6-2aecfa481ffe']</pre>



```python
vector_store.chroma.get(ids=uids[0])
```




<pre class="custom">{'ids': ['3b2062be-649a-4b11-adf7-2eb5c2634e6e'],
     'embeddings': None,
     'documents': ['Once when I was six years old I saw a magnificent picture in a book, called True Stories from'],
     'uris': None,
     'data': None,
     'metadatas': [{'author': 'Antoine de Saint-Exupéry',
       'chapter': 1,
       'source': 'The Little Prince'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



Error occurs when trying to `add` duplicate `ids` .

```python
vector_store.add(pre_documents=pre_dosc[:2])
```

<pre class="custom">Insert of existing embedding ID: d5fcd392-cf17-475e-9bd6-2aecfa481ffe
    Insert of existing embedding ID: 3b2062be-649a-4b11-adf7-2eb5c2634e6e
    Add of existing embedding ID: d5fcd392-cf17-475e-9bd6-2aecfa481ffe
    Add of existing embedding ID: 3b2062be-649a-4b11-adf7-2eb5c2634e6e
</pre>

### Upsert(parallel)

`Upsert` will `Update` a document or `Add` a new document if the same `ID` exists.

```python
tmp_ids = [docs.metadata["id"] for docs in pre_dosc[:2]]
vector_store.chroma.get(ids=tmp_ids)
```




<pre class="custom">{'ids': ['d5fcd392-cf17-475e-9bd6-2aecfa481ffe',
      '3b2062be-649a-4b11-adf7-2eb5c2634e6e'],
     'embeddings': None,
     'documents': ['- we are introduced to the narrator, a pilot, and his ideas about grown-ups',
      'Once when I was six years old I saw a magnificent picture in a book, called True Stories from'],
     'uris': None,
     'data': None,
     'metadatas': [{'author': 'Antoine de Saint-Exupéry',
       'chapter': 1,
       'source': 'The Little Prince'},
      {'author': 'Antoine de Saint-Exupéry',
       'chapter': 1,
       'source': 'The Little Prince'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



```python
pre_dosc[0].page_content = "Changed Content"
pre_dosc[0]
```




<pre class="custom">Document(metadata={'source': 'The Little Prince', 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'id': 'd5fcd392-cf17-475e-9bd6-2aecfa481ffe'}, page_content='Changed Content')</pre>



```python
vector_store.upsert_documents(
    documents=pre_dosc[:2],
)
tmp_ids = [docs.metadata["id"] for docs in pre_dosc[:2]]
vector_store.chroma.get(ids=tmp_ids)
```




<pre class="custom">{'ids': ['d5fcd392-cf17-475e-9bd6-2aecfa481ffe',
      '3b2062be-649a-4b11-adf7-2eb5c2634e6e'],
     'embeddings': None,
     'documents': ['Changed Content',
      'Once when I was six years old I saw a magnificent picture in a book, called True Stories from'],
     'uris': None,
     'data': None,
     'metadatas': [{'author': 'Antoine de Saint-Exupéry',
       'chapter': 1,
       'id': 'd5fcd392-cf17-475e-9bd6-2aecfa481ffe',
       'source': 'The Little Prince'},
      {'author': 'Antoine de Saint-Exupéry',
       'chapter': 1,
       'id': '3b2062be-649a-4b11-adf7-2eb5c2634e6e',
       'source': 'The Little Prince'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



```python
# parallel upsert
vector_store.upsert_documents_parallel(
    documents=pre_dosc,
    batch_size=32,
    max_workers=10,
)
```

## Query Vector Store

There are two ways to **Query** the **LangChain Chroma Vector Store** .

- **Directly** : Query the vector store directly using methods like `similarity_search` or `similarity_search_with_score` .

- **Turning into retriever** : Convert the vector store into a **retriever** object, which can be used in **LangChain** pipelines or chains.

### Query

This method is created by wrapping the methods of the `langchain-chroma` .

**Parameters**

- `query:str` - Query text to search for.

- `k:int = DEFAULT_K` - Number of results to return. Defaults to 4.

- `filter: Dict[str, str] | None = None` - Filter by metadata. Defaults to None.

- `where_document: Dict[str, str] | None = None` - dict used to filter by the documents. E.g. {$contains: {"text": "hello"}}.

- `**kwargs:Any` : Additional keyword arguments to pass to Chroma collection query.


**Returns**
- `List[Document]` - List of documents most similar to the query text and distance in float for each. Lower score represents more similarity.

**Simple Search**

```python
docs = vector_store.query(query="Prince", top_k=2)

for doc in docs:
    print("ID:", doc.metadata["id"])
    print("Chapter:", doc.metadata["chapter"])
    print("Page Content:", doc.page_content)
    print()
```

<pre class="custom">ID: 8a3ea7ca-0bda-4da5-aca1-3adbac2f07a1
    Chapter: 7
    Page Content: prince disturbed my thoughts.
    
    ID: 2a2c0daf-8c25-418a-821f-20f8f7141cd3
    Chapter: 6
    Page Content: Oh, little prince! Bit by bit I came to understand the secrets of your sad little life... For a
    
</pre>

**Filtering Search**

```python
docs = vector_store.query(query="Prince", top_k=2, filters={"chapter": 20})

for doc in docs:
    print("ID:", doc.metadata["id"])
    print("Chapter:", doc.metadata["chapter"])
    print("Page Content:", doc.page_content)
    print()
```

<pre class="custom">ID: 543ed504-aef4-4eeb-9946-616c448a4ad8
    Chapter: 20
    Page Content: snow, the little prince at last came upon a road. And all roads lead to the abodes of men.
    
    ID: b02ae0a4-b881-49aa-838f-5e25377f6724
    Chapter: 20
    Page Content: extinct forever... that doesn‘t make me a very great prince..."
    
</pre>

**Cosine Similarity Search**

```python
# Cosine Similarity
results = vector_store.query(query="Prince", top_k=2, cs=True, filters={"chapter": 20})

for doc, score in results:
    print("ID:", doc.metadata["id"])
    print("Chapter:", doc.metadata["chapter"])
    print("Page Content:", doc.page_content)
    print(f"Similarity Score: {round(score,2)*100:.1f}%")
    print()
```

<pre class="custom">ID: b38c0471-f74b-4fa6-a9c9-872b01fd87bb
    Chapter: 20
    Page Content: snow, the little prince at last came upon a road. And all roads lead to the abodes of men.
    Similarity Score: 60.0%
    
    ID: 02092b04-eeb3-496e-885c-174e0f864a80
    Chapter: 20
    Page Content: extinct forever... that doesn‘t make me a very great prince..."
    Similarity Score: 54.0%
    
</pre>

### as_retriever()

The `as_retriever()` method converts a `VectorStore` object into a `Retriever` object.

A `Retriever` is an interface used in `LangChain` to query a vector store and retrieve relevant documents.

**Parameters**

- `search_type:Optional[str]` - Defines the type of search that the Retriever should perform. Can be `similarity` (default), `mmr` , or `similarity_score_threshold`

- `search_kwargs:Optional[Dict]` - Keyword arguments to pass to the search function. 

    Can include things like:

    `k` : Amount of documents to return (Default: 4)

    `score_threshold` : Minimum relevance threshold for similarity_score_threshold

    `fetch_k` : Amount of documents to pass to `MMR` algorithm(Default: 20)
        
    `lambda_mult` : Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)

    `filter` : Filter by document metadata


**Returns**

- `VectorStoreRetriever` - Retriever class for VectorStore.


### invoke()

Invoke the retriever to get relevant documents.

Main entry point for synchronous retriever invocations.

**Parameters**

- `input:str` - The query string.

- `config:RunnableConfig | None = None` - Configuration for the retriever. Defaults to None.

- `**kwargs:Any` - Additional arguments to pass to the retriever.


**Returns**

- `List[Document]` : List of relevant documents.

```python
from langchain_chroma import Chroma

client = Chroma(
    collection_name="test",
    persist_directory="data/chroma_text",
    collection_metadata={"hnsw:space": "cosine"},
    embedding_function=embeddings,
)
```

```python
retriever = client.as_retriever(search_type="similarity", search_kwargs={"k": 2})
docs = retriever.invoke("Prince", filter={"chapter": 5})

for doc in docs:
    print("ID:", doc.id)
    print("Chapter:", doc.metadata["chapter"])
    print("Page Content:", doc.page_content)
    print()
```

<pre class="custom">ID: 63c0f702-49d4-411e-beab-e18dbf2543ff
    Chapter: 5
    Page Content: Indeed, as I learned, there were on the planet where the little prince lived-- as on all planets--
    
    ID: 2d4abb40-f615-4f22-8183-066e8a43f317
    Chapter: 5
    Page Content: Now there were some terrible seeds on the planet that was the home of the little prince; and these
    
</pre>

### Delete

`Delete` the Documents.

You can use with `filter` .

```python
len(vector_store.unique_ids)
```




<pre class="custom">1317</pre>



```python
len([docs for docs in pre_dosc if docs.metadata["chapter"] == 1])
```




<pre class="custom">43</pre>



```python
vector_store.delete_by_filter(
    unique_ids=list(vector_store.unique_ids), filters={"chapter": 1}
)
```

<pre class="custom">Success Delete 43 Documents
</pre>

```python
len(vector_store.unique_ids)
```




<pre class="custom">1274</pre>



```python
vector_store.delete_by_filter(unique_ids=list(vector_store.unique_ids))
```

<pre class="custom">Success Delete 1274 Documents
</pre>

```python
len(vector_store.unique_ids)
```




<pre class="custom">0</pre>



Remove a **Huggingface Cache** , `vector_store` , `embeddings` and `client` .

If you created a **vectordb** directory, please **remove** it at the end of this tutorial.

```python
from huggingface_hub import scan_cache_dir

del embeddings
del vector_store
del client
scan = scan_cache_dir()
scan.delete_revisions()
```




<pre class="custom">DeleteCacheStrategy(expected_freed_size=0, blobs=frozenset(), refs=frozenset(), repos=frozenset(), snapshots=frozenset())</pre>



## Document Manager

We have developed an interface that makes **CRUD** of **VectorDB** easy to use in tutorials.

Features are as follows

- `upsert` : Inserts or updates documents in the vector database with optional metadata and embeddings.

- `upsert_parellel` : Processes batch insertions or updates in parallel for improved performance.

- `search` : Searches for the top k most similar documents using **cosine similarity** (In this tutorial, we fix the similarity score as cosine similarity) .

- `delete` : Deletes documents by IDs or filters based on metadata or content.

Each function was inherited and developed for each vector DB.

In this tutorial, it was developed for **Chroma** .

Load **Chroma Client** and **Embedding** .

```python
import chromadb

client = chromadb.Client()  # in-memory
```

```python
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "Alibaba-NLP/gte-base-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs={"trust_remote_code": True}
)
```

Load `ChromaDocument Manager` .

```python
from utils.chroma.crud import ChromaDocumentMangager

cdm = ChromaDocumentMangager(
    client=client,
    embedding=embeddings,
    name="chroma",
    metadata={"created_by": "pupba"},
)
```

Preprocessing for `Upsert` .

```python
test_docs = pre_dosc[:50]

ids = [doc.metadata["id"] for doc in test_docs]
texts = [doc.page_content for doc in test_docs]
metadatas = [{k: v for k, v in doc.metadata.items() if k != "id"} for doc in test_docs]
```

### Upsert

The upsert method is designed to **insert** or **update** documents in a vector database. 

It takes the following parameters:

- **texts** : A collection of document texts to be inserted or updated.

- **metadatas** : Optional metadata associated with each document.

- **ids** : Optional unique identifiers for each document.

- ****kwargs** : Additional keyword arguments for flexibility.

```python
cdm.upsert(texts=texts[:5], metadatas=metadatas[:5], ids=ids[:5])
```

```python
cdm.collection.get()["ids"]
```




<pre class="custom">['7c84ae6c-fcda-495a-9f83-4014e17cde17',
     '68222d17-0405-4627-861a-24f74234f600',
     '35b4b8ac-ec66-4303-baf0-00396235ee50',
     'adfbaa1e-3f24-4303-a074-73ef9d7e434d',
     '096e29ba-fd45-411d-87a9-3b3b62938c4a']</pre>



### Upsert-Parellel

The `upsert_parallel` method is an optimized version of `upsert` that processes documents in parallel.

The following parameters are added.

- **batch_size** : The number of documents to process in each batch (default: 32).

- **workers** : The number of parallel workers to use (default: 10).

```python
cdm.upsert_parallel(
    texts=texts,
    metadatas=metadatas,
    ids=ids,
)
```

```python
len(cdm.collection.get()["ids"])
```




<pre class="custom">50</pre>



### Search

The `search` method returns a list of Document objects, which are the top k most similar documents to the query. 

- **query** : A string representing the search query.

- **k** : An integer specifying the number of top results to return (default is 10).

- ****kwargs** : Additional keyword arguments for flexibility in search options. This can include metadata filters( `where` , `where_document` ).

Default search

```python
results = cdm.search("prince", k=2)
results
```




<pre class="custom">[Document(metadata={'id': 'a33ebef8-e878-4a8b-aae2-39a5a0c1fb04', 'score': 0.52, 'author': 'Antoine de Saint-Exupéry', 'chapter': 2, 'source': 'The Little Prince'}, page_content='- the narrator crashes in the desert and makes the acquaintance of the little prince'),
     Document(metadata={'id': '9719e551-884f-4726-85b4-78b6ec09e136', 'score': 0.45, 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'source': 'The Little Prince'}, page_content='I pondered deeply, then, over the adventures of the jungle. And after some work with a colored')]</pre>



```python
cdm.collection.get()["metadatas"][:3]
```




<pre class="custom">[{'author': 'Antoine de Saint-Exupéry',
      'chapter': 1,
      'source': 'The Little Prince'},
     {'author': 'Antoine de Saint-Exupéry',
      'chapter': 1,
      'source': 'The Little Prince'},
     {'author': 'Antoine de Saint-Exupéry',
      'chapter': 1,
      'source': 'The Little Prince'}]</pre>



Search with filters

```python
results = cdm.search("prince", k=2, where={"chapter": 1})
results
```




<pre class="custom">[Document(metadata={'id': '9719e551-884f-4726-85b4-78b6ec09e136', 'score': 0.45, 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'source': 'The Little Prince'}, page_content='I pondered deeply, then, over the adventures of the jungle. And after some work with a colored'),
     Document(metadata={'id': '313ef6e6-146d-4722-a3d7-0e7861fe0f7e', 'score': 0.41, 'author': 'Antoine de Saint-Exupéry', 'chapter': 1, 'source': 'The Little Prince'}, page_content='to be always and forever explaining things to them.')]</pre>



### Delete

The `delete` method removes documents from the vector database based on specified criteria.

- `ids` : A list of document IDs to be deleted. If None, all documents delete.

- `filters` : A dictionary specifying filtering criteria for deletion. This can include metadata filters( `where` , `where_document` ).

- `**kwargs` : Additional keyword arguments for custom deletion options.


Delete with ids

```python
len(cdm.collection.get()["ids"])
```




<pre class="custom">50</pre>



```python
ids = cdm.collection.get()["ids"][:20]
cdm.delete(ids=ids)
len(cdm.collection.get()["ids"])
```




<pre class="custom">30</pre>



Delete with filters

```python
ids = cdm.collection.get(where={"chapter": 1})["ids"]
print("Chapter 1 documents counts:", len(ids))
```

<pre class="custom">Chapter 1 documents counts: 27
</pre>

```python
cdm.delete(ids=cdm.collection.get()["ids"], filters={"where": {"chapter": 1}})
```

```python
ids = cdm.collection.get(where={"chapter": 1})["ids"]
print("Chapter 1 documents counts:", len(ids))
```

<pre class="custom">Chapter 1 documents counts: 0
</pre>

Delete all

```python
len(cdm.collection.get()["ids"])
```




<pre class="custom">3</pre>



```python
cdm.delete()
```

```python
len(cdm.collection.get()["ids"])
```




<pre class="custom">0</pre>



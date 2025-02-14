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

# Chroma With Langchain

- Author: [Gwangwon Jung](https://github.com/pupba)
- Design: []()
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/02-Chroma.ipynb)

## Overview

This tutorial covers how to use `Chroma Vector Store` with `LangChain` .

`Chroma` is an `open-source AI application database` .

In this tutorial, after learning how to use `langchain-chroma` , we will implement examples of a simple **Text Search** engine and **Multimodal Search** engine using `Chroma` .

![search-example](./img/02-chroma-with-langchain-flow-search-example.png)

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [What is Chroma?](#what-is-chroma?)
- [LangChain Chroma Basic](#langchain-chroma-basic)
- [Text Search](#text-search)
- [Multimodal Search](#multimodal-search)


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
        "langchain-huggingface",
        "langchain-experimental",
        "pillow",
        "open_clip_torch",
        "scikit-learn",
        "numpy",
        "requests",
        "python-dotenv",
        "datasets >= 3.2.0",  # Requirements >= 3.2.0
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
        "LANGCHAIN_PROJECT": "Chroma With Langchain",  # title 과 동일하게 설정해 주세요
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

`Chroma` is the `open-source vector database` designed for AI application. 

It specializes in storing high-dimensional vectors and performing fast similariy search, making it ideal for tasks like `semantic search` , `recommendation systems` and `multimodal search` .

With its **developer-friendly APIs** and seamless integration with frameworks like `LangChain` , `Chroma` is powerful tool for building scalable, AI-driven solutions.


## LangChain Chroma Basic

### Load Text Documents Data(Temporary)

The following code demonstrates how to load text documents into a structured format using the `Document` class from `langchain-core` .

Each document contains `page_content` (the text) and `metadata` (additional information about the soruce).

Unique **IDs** are also generated for each document using `uuid4` .

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
```

### Create Vector Store with Embedding

First, load the **Embedding Model**. 

We use the `sentence-transformers/all-mpnet-base-v2` embedding model, which is loaded using the `HuggingFaceEmbeddings` class from `langchain-huggingface` integration.

This model is a powerful choice for generating high-quality embeddings for text data.

```python
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)
```

<pre class="custom">/Users/hi/anaconda3/envs/langchain-opentutorial/lib/python3.11/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
</pre>

Create a `Chroma DB` instance using the `Chroma` class from `langchain-chroma` .

**Parameters**

- `collection_name:str` – Name of the collection to create.

- `embedding_function:Optional[Embeddings]` – Embedding class object. Used to embed texts.

- `persist_directory:Optional[str]` – Directory to persist the collection.

- `client_settings:Optional[chromadb.config.Settings]` – Chroma client settings

- `collection_metadata:Optional[Dict]` – Collection configurations.

- `client:Optional[chromadb.ClientAPI]` – Chroma client. Documentation: https://docs.trychroma.com/reference/python/client

- `relevance_score_fn:Optional[Callable[[float], float]]` – Function to calculate relevance score from distance. Used only in similarity_search_with_relevance_scores

- `create_collection_if_not_exists:Optional[bool]` – Whether to create collection if it doesn’t exist. Defaults to True.

**Returns**

- `Chroma:langchain_chroma.vectorstores.Chroma` - Chroma instance

```python
# Create Vector DB
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./data/test_chroma_db",
    # Where to save data locally, remove if not necessary
)
print(vector_store)
```

<pre class="custom"><langchain_chroma.vectorstores.Chroma object at 0x10a8f77d0>
</pre>

### Manage Vector Store

Add `documents` into the vector store with `UUIDs` as identifiers.

```python
# Add documents
vector_store.add_documents(documents=documents, ids=uuids)
```




<pre class="custom">['aa7c6807-d7bd-44ba-8c8a-98e2b32be65b',
     'ab8bf686-e814-4ffb-919b-49046d3c8a4f',
     '75c06423-90b3-42fe-bac9-1ee941357bbf',
     '03a556b1-10e3-4c06-9de5-a477c028374c',
     '1847b67f-5bf5-4bb9-952f-4b69ca57d93b',
     'e450db27-331c-4f83-80f9-986d13a6ee60',
     'ba08da44-6cdf-4421-a35b-060dedbb42dd',
     '31999f88-488f-4477-a3d1-6b3532492787',
     '9e4783e4-461b-4472-b772-10842da6bd30',
     '5fbd949b-025a-4dba-89ad-173881f15a8a']</pre>



```python
# Verifying saved data
vector_store.get()
```




<pre class="custom">{'ids': ['b085ad68-acf8-4bc8-b727-821d4dbb2d74',
      'e15ecaf0-1b5f-4f76-976c-7e6ec99ce597',
      '1e8b68aa-fcef-407c-82de-d630286c78f2',
      'd91101ae-43ec-4cae-81e2-4a98d1c9db59',
      'c0e24517-3fa8-4759-9d5d-e8e860ff3022',
      'db2bccbf-e83d-4e5a-96c9-15e3180b152a',
      'fe9f36e9-87d5-4238-9b42-ba40ed908907',
      '93b0e75a-7dd0-44df-997a-d43add67f0cd',
      '85f1cfd7-c2c8-463d-8b24-e2dc051c963b',
      'aa7c6807-d7bd-44ba-8c8a-98e2b32be65b',
      'ab8bf686-e814-4ffb-919b-49046d3c8a4f',
      '75c06423-90b3-42fe-bac9-1ee941357bbf',
      '03a556b1-10e3-4c06-9de5-a477c028374c',
      '1847b67f-5bf5-4bb9-952f-4b69ca57d93b',
      'e450db27-331c-4f83-80f9-986d13a6ee60',
      'ba08da44-6cdf-4421-a35b-060dedbb42dd',
      '31999f88-488f-4477-a3d1-6b3532492787',
      '9e4783e4-461b-4472-b772-10842da6bd30',
      '5fbd949b-025a-4dba-89ad-173881f15a8a'],
     'embeddings': None,
     'documents': ['I had fish&chip and fried eggs for breakfast this morning.',
      'The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.',
      'I had chocolate chip pancakes and scrambled eggs for breakfast this morning.',
      'The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.',
      'I have a bad feeling I am going to get deleted :('],
     'uris': None,
     'data': None,
     'metadatas': [{'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



In addition to adding items this way, you can freely `update` or `delete` items in the vector store.

```python
updated_document_1 = Document(
    page_content="I had fish&chip and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)
```

```python
# Update exmpale

vector_store.update_document(
    document_id=uuids[0], document=updated_document_1
)  # document update

vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)  # documents update
```

```python
vector_store.get()
```




<pre class="custom">{'ids': ['b085ad68-acf8-4bc8-b727-821d4dbb2d74',
      'e15ecaf0-1b5f-4f76-976c-7e6ec99ce597',
      '1e8b68aa-fcef-407c-82de-d630286c78f2',
      'd91101ae-43ec-4cae-81e2-4a98d1c9db59',
      'c0e24517-3fa8-4759-9d5d-e8e860ff3022',
      'db2bccbf-e83d-4e5a-96c9-15e3180b152a',
      'fe9f36e9-87d5-4238-9b42-ba40ed908907',
      '93b0e75a-7dd0-44df-997a-d43add67f0cd',
      '85f1cfd7-c2c8-463d-8b24-e2dc051c963b',
      'aa7c6807-d7bd-44ba-8c8a-98e2b32be65b',
      'ab8bf686-e814-4ffb-919b-49046d3c8a4f',
      '75c06423-90b3-42fe-bac9-1ee941357bbf',
      '03a556b1-10e3-4c06-9de5-a477c028374c',
      '1847b67f-5bf5-4bb9-952f-4b69ca57d93b',
      'e450db27-331c-4f83-80f9-986d13a6ee60',
      'ba08da44-6cdf-4421-a35b-060dedbb42dd',
      '31999f88-488f-4477-a3d1-6b3532492787',
      '9e4783e4-461b-4472-b772-10842da6bd30',
      '5fbd949b-025a-4dba-89ad-173881f15a8a'],
     'embeddings': None,
     'documents': ['I had fish&chip and fried eggs for breakfast this morning.',
      'The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.',
      'I had fish&chip and fried eggs for breakfast this morning.',
      'The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.',
      'I have a bad feeling I am going to get deleted :('],
     'uris': None,
     'data': None,
     'metadatas': [{'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



```python
# Delete exmple
vector_store.delete(ids=uuids[-1])
```

```python
vector_store.get()
```




<pre class="custom">{'ids': ['b085ad68-acf8-4bc8-b727-821d4dbb2d74',
      'e15ecaf0-1b5f-4f76-976c-7e6ec99ce597',
      '1e8b68aa-fcef-407c-82de-d630286c78f2',
      'd91101ae-43ec-4cae-81e2-4a98d1c9db59',
      'c0e24517-3fa8-4759-9d5d-e8e860ff3022',
      'db2bccbf-e83d-4e5a-96c9-15e3180b152a',
      'fe9f36e9-87d5-4238-9b42-ba40ed908907',
      '93b0e75a-7dd0-44df-997a-d43add67f0cd',
      '85f1cfd7-c2c8-463d-8b24-e2dc051c963b',
      'aa7c6807-d7bd-44ba-8c8a-98e2b32be65b',
      'ab8bf686-e814-4ffb-919b-49046d3c8a4f',
      '75c06423-90b3-42fe-bac9-1ee941357bbf',
      '03a556b1-10e3-4c06-9de5-a477c028374c',
      '1847b67f-5bf5-4bb9-952f-4b69ca57d93b',
      'e450db27-331c-4f83-80f9-986d13a6ee60',
      'ba08da44-6cdf-4421-a35b-060dedbb42dd',
      '31999f88-488f-4477-a3d1-6b3532492787',
      '9e4783e4-461b-4472-b772-10842da6bd30'],
     'embeddings': None,
     'documents': ['I had fish&chip and fried eggs for breakfast this morning.',
      'The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.',
      'I had fish&chip and fried eggs for breakfast this morning.',
      'The weather forecast for tomorrow is windy and cold, with a high of 82 degrees.',
      'Building an exciting new project with LangChain - come check it out!',
      'Robbers broke into the city bank and stole $1 million in cash.',
      "Wow! That was an amazing movie. I can't wait to see it again.",
      'Is the new iPhone worth the price? Read this review to find out.',
      'The top 10 soccer players in the world right now.',
      'LangGraph is the best framework for building stateful, agentic applications!',
      'The stock market is down 500 points today due to fears of a recession.'],
     'uris': None,
     'data': None,
     'metadatas': [{'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'news'},
      {'source': 'tweet'},
      {'source': 'website'},
      {'source': 'website'},
      {'source': 'tweet'},
      {'source': 'news'}],
     'included': [<IncludeEnum.documents: 'documents'>,
      <IncludeEnum.metadatas: 'metadatas'>]}</pre>



### Query Vector Store

There are two ways to `Query` the `Vector Store` .

- **Directly** : Query the vector store directly using methods like `similarity_search` or `similarity_search_with_score` .

- **Turning into retriever** : Convert the vector store into a `retriever` object, which can be used in `LangChain` pipelines or chains.

## Text Search

With the `Directly` way, you can simply search for `Text` through `Similarity` without much implementation.

The `Directly` way includes `similarity_search` and `similarity_search_with_score` .

### similarity_search()

`similarity_search()` is run similarity search with Chroma.

**Parameters**

- `query:str` - Query text to search for.

- `k: int = DEFAULT_K` - Number of results to return. Defaults to 4.    

- `filter: Dict[str, str] | None = None` - Filter by metadata. Defaults to None.

- `**kwargs:Any` - Additional keyword arguments to pass to Chroma collection query.


**Returns**
- `List[Documents]` - List of documents most similar to the query text.



### similarity_search_with_score()

`similarity_search_with_score()` is run similarity search with Chroma with distance.

**Parameters**

- `query:str` - Query text to search for.

- `k:int = DEFAULT_K` - Number of results to return. Defaults to 4.

- `filter: Dict[str, str] | None = None` - Filter by metadata. Defaults to None.

- `where_document: Dict[str, str] | None = None` - dict used to filter by the documents. E.g. {$contains: {"text": "hello"}}.

- `**kwargs:Any` : Additional keyword arguments to pass to Chroma collection query.


**Returns**
- `List[Tuple[Document, float]]` - List of documents most similar to the query text and distance in float for each. Lower score represents more similarity.

```python
# Directly - similarity_search

results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)

for idx, res in enumerate(results):
    print(f"{idx}: {res.page_content} [{res.metadata}]")
```

<pre class="custom">0: Building an exciting new project with LangChain - come check it out! [{'source': 'tweet'}]
    1: Building an exciting new project with LangChain - come check it out! [{'source': 'tweet'}]
</pre>

```python
# Directly - similarity_search_with_score

results = vector_store.similarity_search_with_score(
    query="Will it be cold tomorrow?",
    k=1,
    filter={"source": "news"},
)

for idx, (res, score) in enumerate(results):
    print(
        f"{idx}: [Similarity Score: {round(score,3)*100}%] {res.page_content} [{res.metadata}]"
    )
```

<pre class="custom">0: [Similarity Score: 78.9%] The weather forecast for tomorrow is windy and cold, with a high of 82 degrees. [{'source': 'news'}]
</pre>

You can also use `Turning into retrievals` way to search for text.

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
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
```

<pre class="custom">huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
</pre>




    [Document(id='d91101ae-43ec-4cae-81e2-4a98d1c9db59', metadata={'source': 'news'}, page_content='Robbers broke into the city bank and stole $1 million in cash.'),
     Document(id='03a556b1-10e3-4c06-9de5-a477c028374c', metadata={'source': 'news'}, page_content='Robbers broke into the city bank and stole $1 million in cash.')]



## Multimodal Search

`Chorma` supports `Multimodal Collections` , which means it can handle and store embeddings from different types of data, such as `text` , `images` , `audio` , or even `video` .

We can search for `images` using `Chroma` .

### Setting `image` and `image_info` data

This dataset is made by `SDXL` . 

**Dataset: Animal-180**

- [animal-180](https://huggingface.co/datasets/Pupba/animal-180)

This dataset, named `animal-180` , is a collection of 180 realistic animal images generated using `Stable-Diffusion XL(SDXL)` .

It includes images of `lions` , `rabbits` , `cats` , `dogs` , `elephants` and `tigers` , with 30 images per animal category.

All images are free to use for any purpose, as they are synthetically generated and not subject to copyright restrictions.

```python
import tempfile
from PIL import Image


def save_temp_gen_url(image: Image) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file, format="PNG")
    temp_file.close()
    return temp_file.name
```

```python
from datasets import load_dataset

dataset = load_dataset("Pupba/animal-180", split="train")

# slice 50 set
images = dataset[:50]["png"]
image_paths = [save_temp_gen_url(img) for img in images]
metas = dataset[:50]["json"]
prompts = [data["prompt"] for data in metas]
categories = [data["category"] for data in metas]
```

```python
print("Image Path:", image_paths[0])
print("Prompt:", prompts[0])
print("Category:", categories[0])
images[0]
```

<pre class="custom">Image Path: /var/folders/fv/qst03yf9073b83fw3cm8y50h0000gn/T/tmpqme5j8lv.png
    Prompt: a fluffy white rabbit sitting in a grassy meadow, soft sunlight illuminating its fur, highly detailed, 8k resolution.
    Category: rabbit
</pre>




    
![png](./img/output_33_1.png)
    



Load `OpenCLIP` for `Multimodal Embedding` .

- [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main)

```python
from langchain_experimental.open_clip import OpenCLIPEmbeddings

MODEL = "ViT-H-14-quickgelu"
CHECKPOINT = "dfn5b"

image_embedding = OpenCLIPEmbeddings(model_name=MODEL, checkpoint=CHECKPOINT)
```

## Create a Multimodal Vector Store

Create a `Multimodal Vector Store` and add the `Image uri` and `Metadata(file_path, category, prompt)`

```python
image_vector_db = Chroma(
    collection_name="multimodal",
    embedding_function=image_embedding,
)

image_vector_db.add_images(
    uris=image_paths,
    metadatas=[
        {"file_path": file_path, "category": category, "prompt": prompt}
        for file_path, category, prompt in zip(image_paths, categories, prompts)
    ],
)
```




<pre class="custom">['a6400e26-d398-4489-8b10-3793e97b4ee6',
     '4695add5-4fec-419f-86f6-1570d36f75a2',
     '7cc82cd0-4d05-4fc8-8243-6c084e88cc1a',
     '69dc736a-0717-4198-a1dc-46e9907c8b44',
     '0454356a-1693-462d-ac9c-d012a0e5627a',
     '89e353b8-b68d-4d7a-bbb7-4a446278693c',
     '33b890d4-da1f-4997-bcb0-83c3340f49ec',
     '85154a7b-4847-4ccb-b15d-ab5a78f8eb62',
     'b49f2148-9495-4480-83da-1a33b59d4659',
     '2ad4e5db-2557-40d9-9a7e-d575138d2200',
     'ec1c6b44-55b5-42ec-88e5-02b446b4e5c0',
     'ea716316-b70a-443d-9856-d00c55229fa6',
     '6544ef09-55d3-49f3-a67e-b41bc19002a3',
     '02c92a0f-4287-4003-96ab-49723832c59e',
     'bd298c95-1b42-432b-b803-07d8aa71c74c',
     '3990928b-f23e-4416-bf79-1e5e2e40caa3',
     '86cd5aaa-aade-4f5c-904a-36441fc10b58',
     '1634c252-1d39-4327-95fb-519d9429a66a',
     '256a9146-cb06-4730-a0d0-ba9986ea6e0e',
     '47ca868c-0a52-40a1-8b15-98fe7e5af767',
     '783d5dab-673f-4b8b-b21c-5241ee2745a4',
     '0498df78-b06a-4d6b-b622-2d6b1b0a77fd',
     'ea6d54e1-a387-4cb9-ba2a-292ddf7e4343',
     '221cd854-30f8-4939-b5d8-fd89ef5c1de6',
     '7153f87a-5410-402b-af67-70158a770b36',
     'c93687d4-277f-4372-86b4-6cadf94b2772',
     'd1efe60b-4f81-449f-b228-7ef5424f4cf4',
     '06be3378-8aa1-4351-bed1-55ba23d96659',
     'e77d27fd-7115-4848-a631-d8db789ae4dc',
     '8a107858-6ed3-4f14-a7a4-f90e25934bac',
     '19a2b454-a045-4a7b-943c-0fb91dc59852',
     'e4a226bc-65f6-4892-8c77-2c0c596db061',
     '890dfa8c-c4d7-4880-be78-c8ba86c77835',
     'd0e33f46-f515-43b9-a07d-cb28a59471db',
     '81f1942c-90dc-4b05-a49a-dcf5d8257f4c',
     '5b769e26-eeab-45e4-aba3-c4db903d8c25',
     'e4e424f2-9c58-4b6a-8965-2c8c7deb0324',
     '43793395-9256-47b9-974b-64edca5278e8',
     '46608f34-3953-45d2-a6ea-5f7520145326',
     'd56d8a92-406c-4523-ade6-c8217dfe0f19',
     '153adebe-528d-4dfc-9fa2-a6f8f0d77635',
     'fc840a7f-7309-483c-971a-ea8e9f035fa9',
     '4a8174ca-52b8-46d5-bd9d-29889685691b',
     'a18e35b7-9b31-41bb-9957-489119577cb0',
     'd5484bcd-dea7-44ee-86d9-fae098482116',
     'd41c461d-be67-48b8-ba42-2431da7be51e',
     '6029d2a4-6673-406e-acc6-35c8d37bf6a6',
     '3650ab8b-9238-4c39-b249-2e84bf036cfb',
     '0f9298bb-c31e-4b6b-a32e-0ce087716824',
     '521eed1d-eb83-4547-b006-44c4f5360fb2']</pre>



```python
image_vector_db.get()
```


Make `ImageSearcher`

```python
from typing import Optional, Dict


class ImageSearcher:
    def __init__(self, image_vector_store: Chroma):
        self.__vector_store = image_vector_store

    def searching_text_query(self, query: str) -> Optional[Dict]:
        docs = self.__vector_store.as_retriever().invoke(query)

        if docs and isinstance(docs[0], Document):
            metadata = docs[0].metadata
            return {
                "category": metadata["category"],
                "image": Image.open(metadata["file_path"]),
                "prompt": metadata["prompt"],
            }
        else:
            return None

    def searching_image_query(self, image_uri: str) -> Optional[Dict]:
        docs = self.__vector_store.similarity_search_by_image(uri=image_uri, k=1)
        if docs and isinstance(docs[0], Document):
            metadata = docs[0].metadata
            return {
                "category": metadata["category"],
                "image": Image.open(metadata["file_path"]),
                "prompt": metadata["prompt"],
            }
        else:
            return None


image_search = ImageSearcher(image_vector_store=image_vector_db)
```

Text Query Search

```python
result = image_search.searching_text_query(query="a elephant run")

print(f"Category: {result['category']}")
print(f"Prompt: {result['prompt']}")
result["image"]
```

<pre class="custom">Category: elephant
    Prompt: a majestic elephant walking through the savanna, golden sunlight illuminating its wrinkled skin, highly detailed, 8k resolution.
</pre>




    
![png](./img/output_42_1.png)
    



Image Query Search

```python
# query image url
import requests
from io import BytesIO


def load_image_from_url(url: str, resolution: int = 512) -> Image.Image:
    """
    Load an image from a URL and return it as a PIL Image object.

    Args:
        url (str): The URL of the image.

    Returns:
        Image.Image: The loaded PIL Image object.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for failed requests
    image = Image.open(BytesIO(response.content))
    image = image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    return image


def save_image_to_tempfile(url: str) -> str:
    """
    Download an image from a URL and save it to a temporary file.

    Args:
        url (str): The URL of the image.

    Returns:
        str: The file path to the saved image.
    """
    response = requests.get(url)

    # Raise an error for failed requests
    response.raise_for_status()

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(response.content)

    # Close the file to allow other processes to access it
    temp_file.close()
    return temp_file.name
```

```python
# rabbit image
img_url = "https://i.pinimg.com/736x/b2/e9/f4/b2e9f449c1c5f8a29e31cafb8671c8b2.jpg"

image_query = load_image_from_url(img_url)
image_query_url = save_image_to_tempfile(img_url)

image_query
```




    
![png](./img/output_45_0.png)
    



```python
result = image_search.searching_image_query(image_uri=image_query_url)

print(f"Category: {result['category']}")
print(f"Prompt: {result['prompt']}")
result["image"]
```

<pre class="custom">Category: rabbit
    Prompt: a rabbit standing on its hind legs, looking at the camera, soft golden lighting, highly detailed, 8k resolution.
</pre>




    
![png](./img/output_46_1.png)
    



Remove a `Huggingface Cache`

```python
dataset.cleanup_cache_files()
```




<pre class="custom">0</pre>



Disconnect `Chroma` DB and Remove Local DB file

```python
del vector_store
del image_vector_db
```

```python
import os
import shutil

shutil.rmtree(os.path.join(os.getcwd(), "data", "test_chroma_db"))
```

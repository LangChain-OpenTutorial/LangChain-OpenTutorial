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

# CacheBackedEmbeddings

- Author: [byoon](https://github.com/acho98)
- Peer Review: [ro__o_jun](https://github.com/ro-jun)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/02-CacheBackedEmbeddings.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/02-CacheBackedEmbeddings.ipynb)
## Overview

Embeddings can be stored or temporarily cached to avoid recalculation.

Caching embeddings can be done using ```CacheBackedEmbeddings```. A cache-backed embedder is a wrapper around an embedder that caches embeddings in a key-value store. The text is hashed, and the hash is used as a key in the cache.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Using Embeddings with LocalFileStore (Persistent Storage)](#using-embeddings-with-localfilestore-persistent-storage)
- [Using InMemoryByteStore (Non-Persistent)](#using-inmemorybytestore-non-persistent)


### References

- [CacheBackedEmbeddings](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html)
- [InMemoryByteStore](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html)
----

## Environment-setup

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
        "langchain",
        "langchain_community",
        "langchain_openai",
        "faiss-cpu"
    ],
    verbose=False,
    upgrade=False,
)
```

Configuration file for managing API keys as environment variables.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "CacheBackedEmbeddings",
    },
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">False</pre>



Check and create the ./cache/ directory for persistent storage.

```python
import os

os.makedirs("./cache/", exist_ok=True)
print(os.path.exists("./cache/"))  # Check if the directory exists
print(os.access("./cache/", os.W_OK))  # Check if the directory is writable
```

<pre class="custom">True
    True
</pre>

## Using Embeddings with LocalFileStore (Persistent Storage)

The primary supported method for initializing ```CacheBackedEmbeddings``` is ```from_bytes_store```.  

It accepts the following parameters:

- ```underlying_embeddings```: The embedder is used for generating embeddings.
- ```document_embedding_cache```: One of the ```ByteStore``` implementations for caching document embeddings.
- ```namespace```: (Optional, default is ```""```) A namespace is used for the document cache. This is utilized to avoid conflicts with other caches. For example, set it to the name of the embedding model being used.

**Note**: It is important to set the ```namespace``` parameter to avoid conflicts when the same text is embedded using different embedding models.

First, let's look at an example of storing embeddings using the local file system and retrieving them with the FAISS vector store.

```python
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Configure basic embeddings using OpenAI embeddings
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Set up a local file storage
store = LocalFileStore("./cache/")

# Create embeddings with caching support
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings, 
    document_embedding_cache=store, 
    namespace=underlying_embeddings.model, # Create a cache-backed embedder using the base embedding and storage
)
```

The cache is empty prior to embedding

```python
list(store.yield_keys())
```




<pre class="custom">[]</pre>



Load the document, split it into chunks, embed each chunk and load it into the vector store.


```python
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

raw_documents = TextLoader("./data/state_of_the_union.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
```

Create FAISS database from documents.

```python
%time db = FAISS.from_documents(documents, cached_embedder)
```

<pre class="custom">CPU times: user 105 ms, sys: 14.3 ms, total: 119 ms
    Wall time: 1.49 s
</pre>

If we try to create the vector store again, it'll be much faster since it does not need to re-compute any embeddings.


```python
# Create FAISS database using cached embeddings
%time db2 = FAISS.from_documents(documents, cached_embedder)
```

<pre class="custom">CPU times: user 21.8 ms, sys: 3.23 ms, total: 25 ms
    Wall time: 23.8 ms
</pre>

Here are some of the embeddings that got created.

```python
list(store.yield_keys())[:5]
```




<pre class="custom">['text-embedding-3-small464862c8-03d2-5854-b32c-65a075e612a2',
     'text-embedding-3-small6d6cb8fc-721a-5a4c-bfe9-c83d2920c2bb',
     'text-embedding-3-small5990258b-0781-5651-8444-c69cb5b6da3d',
     'text-embedding-3-small01dbc21f-5e4c-5fb5-8d13-517dbe7a32d4',
     'text-embedding-3-small704c76af-3696-5383-9858-6585616669ef']</pre>



## Using ```InMemoryByteStore``` (Non-Persistent)

To use a different ```ByteStore```, simply specify the desired ```ByteStore``` when creating the ```CacheBackedEmbeddings```.

Below is an example of creating the same cached embedding object using the non-persistent ```InMemoryByteStore```.



```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

# Create an in-memory byte store
store = InMemoryByteStore()

underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
```

```python
%time db = FAISS.from_documents(documents, cached_embedder)  
list(store.yield_keys())[:5]
```

<pre class="custom">CPU times: user 87.4 ms, sys: 8.58 ms, total: 96 ms
    Wall time: 1.14 s
</pre>




    ['text-embedding-3-small305efb5c-3f01-5657-bcf2-2b92fb1747ca',
     'text-embedding-3-small01dbc21f-5e4c-5fb5-8d13-517dbe7a32d4',
     'text-embedding-3-smalla5ef11e4-0474-5725-8d80-81c91943b37f',
     'text-embedding-3-small6d6cb8fc-721a-5a4c-bfe9-c83d2920c2bb',
     'text-embedding-3-small81426526-23fe-58be-9e84-6c7c72c8ca9a']



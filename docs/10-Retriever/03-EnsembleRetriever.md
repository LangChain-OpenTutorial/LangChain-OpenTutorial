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

# Ensemble Retriever

- Author: [3dkids](https://github.com/3dkids)
- Peer Review: [r14minji](https://github.com/r14minji), [jeongkpa](https://github.com/jeongkpa)
- Proofread : [jishin86](https://github.com/jishin86)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/03-EnsembleRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/03-EnsembleRetriever.ipynb)
## Overview

This notebook explores the creation and use of an EnsembleRetriever in LangChain to improve information retrieval by combining multiple retrieval methods.<br> 
The EnsembleRetriever integrates the strengths of sparse and dense retrieval algorithms, using weights and runtime configurations for tailored performance.<br>

**Key Features**
1. integrate multiple searchers: take different types of searchers as input and combine results.
2. result re-ranking: uses the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm to re-rank results.
3. hybrid search: mainly uses a combination of ```sparse retriever``` (e.g. BM25) and ```dense retriever``` (e.g. embedding similarity).

**Advantages**
- Sparse retriever: effective for keyword-based searches
- Dense retriever: effective for semantic similarity-based searches

Due to these complementary characteristics, ```EnsembleRetriever``` can provide improved performance in a variety of search scenarios.

For more information, please refer to the [LangChain official documentation](https://python.langchain.com/api_reference/langchain/retrievers.html)



### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Creating and Configuring Ensemble Retrievers](#creating-and-configuring-ensemble-retrievers)
- [Query Execution](#query-execution)
- [Change runtime config](#change-runtime-config)


### References

- [LangChain: EnsembleRetriever](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html#ensembleretriever)
- [LangChain: BM25Retriever](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.bm25.BM25Retriever.html)
- [LangChain: ConfigurableField](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_core",  # Core functionality of LangChain
        "langchain_community",  # Community-supported integrations
        "langchain_openai",  # OpenAI integration for embeddings and models
        "rank_bm25",  # BM25 ranking algorithm for information retrieval
    ],
    verbose=False,  # Suppress detailed installation logs
    upgrade=False,  # Do not upgrade packages if already installed
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
        "LANGCHAIN_PROJECT": "Conversation-With-History",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Configuration file to manage API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">False</pre>



## Creating and Configuring Ensemble Retrievers
Initializing an ensemble retriever
Ensemble retrievers combine two discovery mechanisms

- Sparse search: Uses BM25Retriever for keyword-based matching.
- Dense search: Uses FAISS with OpenAI embedding for semantic similarity.

- Initialize ```EnsembleRetriever``` to combine the ```BM25Retriever``` and ```FAISS``` searchers. Set the weights for each searcher.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# list sample documents
doc_list = [
    "I like apples",
    "I like apple company",
    "I like apple's iphone",
    "Apple is my favorite company",
    "I like apple's ipad",
    "I like apple's macbook",
]

# Initialize the bm25 retriever and faiss retriever.
bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)
bm25_retriever.k = 1  # Set the number of search results for BM25Retriever to 1.

embedding = OpenAIEmbeddings()  # Enable OpenAI embedding.

faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# Initialize the ensemble retriever.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)
```

## Query Execution
Perform retrieval for a given query using ensemble_retriever and compare results across retrievers.
- Call the ```get_relevant_documents()``` method of the ```ensemble_retriever``` object to retrieve relevant documents.


```python
# Get the search results document.
query = "my favorite fruit is apple"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# Output the fetched documents.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()
```

<pre class="custom">[Ensemble Retriever]
    Content: Apple is my favorite company
    
    Content: I like apples
    
    [BM25 Retriever]
    Content: Apple is my favorite company
    
    [FAISS Retriever]
    Content: I like apples
    
</pre>

```python
# Get the search results document.
query = "Apple company makes my favorite iphone"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# Output the fetched documents.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()
```

<pre class="custom">[Ensemble Retriever]
    Content: Apple is my favorite company
    
    Content: I like apple's iphone
    
    [BM25 Retriever]
    Content: Apple is my favorite company
    
    [FAISS Retriever]
    Content: I like apple's iphone
    
</pre>

## Change runtime config

You can also change the properties of a retriever at runtime. This is possible using the ```ConfigurableField``` class.

- Define the ```weights``` parameter as a ```ConfigurableField``` object.
  - Set the field's ID to “ensemble_weights”.


```python
from langchain_core.runnables import ConfigurableField

ensemble_retriever = EnsembleRetriever(
    # Set the list of retrievers. Here we use bm25_retriever and faiss_retriever.
    retrievers=[bm25_retriever, faiss_retriever],
).configurable_fields(
    weights=ConfigurableField(
        # Set a unique identifier for the search parameter.
        id="ensemble_weights",
        # Set a name for the search parameter.
        name="Ensemble Weights",
        # Write a description of the search parameters.
        description="Ensemble Weights",
    )
)
```

- Specify the search settings via the ```config``` parameter when searching.
  - Set the weight of the ```ensemble_weights``` option to [1, 0] so that **all search results are weighted more heavily toward BM25 retriever**.

```python
config = {"configurable": {"ensemble_weights": [1, 0]}}

# Use the config parameter to specify search settings.
docs = ensemble_retriever.invoke("my favorite fruit is apple", config=config)
docs  # Print the search result, docs.
```




<pre class="custom">[Document(metadata={}, page_content='Apple is my favorite company'),
     Document(id='6280c2a3-b58f-474e-aeb6-d480bb44d49e', metadata={}, page_content='I like apples')]</pre>



This time, we want all search results to be weighted **more heavily in favor of the FAISS retriever**.

```python
config = {"configurable": {"ensemble_weights": [0, 1]}}

# Use the config parameter to specify search settings.
docs = ensemble_retriever.invoke("my favorite fruit is apple", config=config)
docs  # Print the search result, docs.
```




<pre class="custom">[Document(id='6280c2a3-b58f-474e-aeb6-d480bb44d49e', metadata={}, page_content='I like apples'),
     Document(metadata={}, page_content='Apple is my favorite company')]</pre>



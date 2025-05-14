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

# Ensemble Retriever with Convex Combination (CC)

- Author: [Harheem Kim](https://github.com/harheem)
- Peer Review:
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/11-Convex-Combination-Ensemble-Retriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/11-Convex-Combination-Ensemble-Retriever.ipynb)
## Overview

This tutorial focuses on implementing and comparing different ensemble retrieval methods in LangChain. While LangChain's built-in ```EnsembleRetriever``` uses the **Reciprocal Rank Fusion (```RRF```)** method, we'll explore an additional approach by implementing the **Convex Combination (```CC```)** method.<br>
The tutorial guides you through creating custom implementations of both **```RRF``` and ```CC``` methods** , allowing for a direct performance comparison between these ensemble techniques.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Process Document](#process-document)
- [Initialize Retrievers](#initialize-retrievers)
- [Implement Ensemble Retrievers](#implement-ensemble-retrievers)
- [Compare and Test](#compare-and-test)

### References

- [EnsembleRetriever](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html)
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
        "langchain_community",
        "langchain_openai",
        "langchain_core",
        "faiss-cpu",
        "pdfplumber",
        "rank_bm25",
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
        "LANGCHAIN_PROJECT": "Conversation-With-History",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in **.env** file and load it.

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



##  Process Document

This section outlines the preparation process for processing PDF documents before storing them in a vector store. 

We use ```PDFPlumberLoader``` to load the PDF file and leverage ```RecursiveCharacterTextSplitter``` to break down the document into smaller, manageable chunks. 

The chunk size is set to 200 characters with no overlap, allowing for efficient processing while maintaining the document's semantic integrity.

```python
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF document
loader = PDFPlumberLoader("data/Introduction_LangChain.pdf")
# Split the document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
split_documents = loader.load_and_split(text_splitter)
```

## Initialize Retrievers

This section initializes retrievers to implement two different search approaches. We create embeddings using OpenAI's ```text-embedding-3-small``` model and set up ```FAISS``` vector search based on these embeddings. 

Additionally, we configure a ```BM25``` retriever for keyword-based search, with both retrievers set to return the top 5 most relevant results.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize FAISS retriever with vector embeddings
faiss = FAISS.from_documents(
    documents=split_documents, embedding=embeddings
).as_retriever(search_kwargs={"k": 5})

# Initialize BM25 retriever for keyword-based search
bm25 = BM25Retriever.from_documents(documents=split_documents)
bm25.k = 5
```

## Implement Ensemble Retrievers

This section introduces a custom retriever implementing two ensemble search methods, designed to compare performance against LangChain's built-in ```EnsembleRetriever``` . 

We implement both **Reciprocal Rank Fusion (```RRF```)** , which combines results based on document rankings, and **Convex Combination (```CC```)** , which utilizes normalized scores. 

Both methods integrate results from ```FAISS``` and ```BM25``` retrievers to provide more accurate and diverse search results, allowing users to select the most suitable ensemble approach for their needs.

```python
from enum import Enum
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, model_validator


class EnsembleMethod(str, Enum):
    RRF = "rrf"  # Reciprocal Rank Fusion
    CC = "cc"  # Convex Combination


class EnsembleRetriever(BaseRetriever, BaseModel):
    retrievers: List[BaseRetriever]
    weights: Optional[List[float]] = None
    method: EnsembleMethod = EnsembleMethod.RRF
    c: int = 60

    @model_validator(mode="before")
    def validate_weights(cls, values):
        weights = values.get("weights")
        method = values.get("method", EnsembleMethod.RRF)

        if not weights:
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        elif method == EnsembleMethod.CC and abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("CC method의 경우 weights의 합이 1이어야 합니다")

        return values

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_list = [
            retriever.get_relevant_documents(query) for retriever in self.retrievers
        ]

        if self.method == EnsembleMethod.RRF:
            return self._rrf_fusion(docs_list)
        else:
            return self._cc_fusion(docs_list)

    def _rrf_fusion(self, docs_list: List[List[Document]]) -> List[Document]:
        """
        Implements Reciprocal Rank Fusion algorithm
        - Combines results based on document rankings
        - Uses a constant 'c' to prevent high ranks from dominating
        - Applies weights to different retrievers' contributions
        """
        from collections import defaultdict

        scores = defaultdict(float)
        for docs, weight in zip(docs_list, self.weights):
            for rank, doc in enumerate(docs, 1):
                scores[doc.page_content] += weight / (rank + self.c)

        all_docs = []
        seen = set()
        for docs in docs_list:
            for doc in docs:
                if doc.page_content not in seen:
                    all_docs.append(doc)
                    seen.add(doc.page_content)

        return sorted(all_docs, key=lambda x: scores[x.page_content], reverse=True)

    def _cc_fusion(self, docs_list: List[List[Document]]) -> List[Document]:
        """
        Implements Convex Combination fusion
        - Combines normalized scores from different retrievers
        - Requires weights to sum to 1.0
        - Handles cases with missing or zero scores
        """
        from collections import defaultdict

        scores = defaultdict(float)
        for docs, weight in zip(docs_list, self.weights):
            max_score = max(
                (doc.metadata.get("score", 1.0) for doc in docs), default=1.0
            )
            if max_score == 0:
                max_score = 1.0

            for doc in docs:
                norm_score = doc.metadata.get("score", 1.0) / max_score
                scores[doc.page_content] += weight * norm_score

        all_docs = []
        seen = set()
        for docs in docs_list:
            for doc in docs:
                if doc.page_content not in seen:
                    all_docs.append(doc)
                    seen.add(doc.page_content)

        return sorted(all_docs, key=lambda x: scores[x.page_content], reverse=True)
```

```python
from langchain.retrievers import EnsembleRetriever as OriginalEnsembleRetriever

# Initialize the original LangChain EnsembleRetriever
original_ensemble_retriever = OriginalEnsembleRetriever(retrievers=[faiss, bm25])

# Initialize Ensemble Retriever with RRF (Reciprocal Rank Fusion) method
rrf_ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss, bm25], method=EnsembleMethod.RRF
)

# Initialize Ensemble Retriever with CC (Convex Combination) method
cc_ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss, bm25],
    method=EnsembleMethod.CC,
    weights=[0.5, 0.5],  # Equal weights for both retrievers
)
```

## Compare and Test

This section presents a test function for comparing ensemble retrieval results. 

While the **```RRF``` method** , which follows LangChain's default implementation, produces identical results to **Original** , the **```CC``` method** utilizing normalized scores and weights offers different search patterns. 

By testing with real queries and comparing these approaches, we can identify which ensemble method better suits our project requirements.

```python
def pretty_print(query):
    for i, (original_doc, cc_doc, rrf_doc) in enumerate(
        zip(
            original_ensemble_retriever.invoke(query),
            cc_ensemble_retriever.invoke(query),
            rrf_ensemble_retriever.invoke(query),
        )
    ):
        print(f"[{i}] [Original] Q: {query}", end="\n\n")
        print(original_doc.page_content)
        print("-" * 100)
        print(f"[{i}] [RRF] Q: {query}", end="\n\n")
        print(rrf_doc.page_content)
        print("-" * 100)
        print(f"[{i}] [CC] Q: {query}", end="\n\n")
        print(cc_doc.page_content)
        print("=" * 100, end="\n\n")
```

```python
pretty_print("What are the advantages of LangChain?")
```

<pre class="custom">[0] [Original] Q: What are the advantages of LangChain?
    
    Introductions to all the key parts of LangChain you’ll need to know! Here you'll find high level
    explanations of all LangChain concepts.
    ----------------------------------------------------------------------------------------------------
    [0] [RRF] Q: What are the advantages of LangChain?
    
    Introductions to all the key parts of LangChain you’ll need to know! Here you'll find high level
    explanations of all LangChain concepts.
    ----------------------------------------------------------------------------------------------------
    [0] [CC] Q: What are the advantages of LangChain?
    
    Introductions to all the key parts of LangChain you’ll need to know! Here you'll find high level
    explanations of all LangChain concepts.
    ====================================================================================================
    
    [1] [Original] Q: What are the advantages of LangChain?
    
    For a deeper dive into LangGraph concepts, check out this page.
    Integrations
    LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it. If
    ----------------------------------------------------------------------------------------------------
    [1] [RRF] Q: What are the advantages of LangChain?
    
    For a deeper dive into LangGraph concepts, check out this page.
    Integrations
    LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it. If
    ----------------------------------------------------------------------------------------------------
    [1] [CC] Q: What are the advantages of LangChain?
    
    For a deeper dive into LangGraph concepts, check out this page.
    Integrations
    LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it. If
    ====================================================================================================
    
    [2] [Original] Q: What are the advantages of LangChain?
    
    If you're looking to build something specific or are more of a hands-on learner, check out our tutorials
    section. This is the best place to get started.
    These are the best ones to get started with:
    ----------------------------------------------------------------------------------------------------
    [2] [RRF] Q: What are the advantages of LangChain?
    
    If you're looking to build something specific or are more of a hands-on learner, check out our tutorials
    section. This is the best place to get started.
    These are the best ones to get started with:
    ----------------------------------------------------------------------------------------------------
    [2] [CC] Q: What are the advantages of LangChain?
    
    LangChain simplifies every stage of the LLM application lifecycle:
    Development: Build your applications using LangChain's open-source components and third-party
    ====================================================================================================
    
    [3] [Original] Q: What are the advantages of LangChain?
    
    Integration packages (e.g. , , etc.): Important
    langchain-openai langchain-anthropic
    integrations have been split into lightweight packages that are co-maintained by the LangChain
    ----------------------------------------------------------------------------------------------------
    [3] [RRF] Q: What are the advantages of LangChain?
    
    Integration packages (e.g. , , etc.): Important
    langchain-openai langchain-anthropic
    integrations have been split into lightweight packages that are co-maintained by the LangChain
    ----------------------------------------------------------------------------------------------------
    [3] [CC] Q: What are the advantages of LangChain?
    
    The LangChain framework consists of multiple open-source libraries. Read more in the Architecture
    page.
    : Base abstractions for chat models and other components.
    langchain-core
    ====================================================================================================
    
    [4] [Original] Q: What are the advantages of LangChain?
    
    LangChain simplifies every stage of the LLM application lifecycle:
    Development: Build your applications using LangChain's open-source components and third-party
    ----------------------------------------------------------------------------------------------------
    [4] [RRF] Q: What are the advantages of LangChain?
    
    LangChain simplifies every stage of the LLM application lifecycle:
    Development: Build your applications using LangChain's open-source components and third-party
    ----------------------------------------------------------------------------------------------------
    [4] [CC] Q: What are the advantages of LangChain?
    
    Read up on security best practices to make sure you're developing safely with LangChain.
    Contributing
    ====================================================================================================
    
    [5] [Original] Q: What are the advantages of LangChain?
    
    : Third-party integrations that are community maintained.
    langchain-community
    : Orchestration framework for combining LangChain components into production-ready
    langgraph
    ----------------------------------------------------------------------------------------------------
    [5] [RRF] Q: What are the advantages of LangChain?
    
    : Third-party integrations that are community maintained.
    langchain-community
    : Orchestration framework for combining LangChain components into production-ready
    langgraph
    ----------------------------------------------------------------------------------------------------
    [5] [CC] Q: What are the advantages of LangChain?
    
    If you're looking to build something specific or are more of a hands-on learner, check out our tutorials
    section. This is the best place to get started.
    These are the best ones to get started with:
    ====================================================================================================
    
    [6] [Original] Q: What are the advantages of LangChain?
    
    The LangChain framework consists of multiple open-source libraries. Read more in the Architecture
    page.
    : Base abstractions for chat models and other components.
    langchain-core
    ----------------------------------------------------------------------------------------------------
    [6] [RRF] Q: What are the advantages of LangChain?
    
    The LangChain framework consists of multiple open-source libraries. Read more in the Architecture
    page.
    : Base abstractions for chat models and other components.
    langchain-core
    ----------------------------------------------------------------------------------------------------
    [6] [CC] Q: What are the advantages of LangChain?
    
    Integration packages (e.g. , , etc.): Important
    langchain-openai langchain-anthropic
    integrations have been split into lightweight packages that are co-maintained by the LangChain
    ====================================================================================================
    
    [7] [Original] Q: What are the advantages of LangChain?
    
    Read up on security best practices to make sure you're developing safely with LangChain.
    Contributing
    ----------------------------------------------------------------------------------------------------
    [7] [RRF] Q: What are the advantages of LangChain?
    
    Read up on security best practices to make sure you're developing safely with LangChain.
    Contributing
    ----------------------------------------------------------------------------------------------------
    [7] [CC] Q: What are the advantages of LangChain?
    
    : Third-party integrations that are community maintained.
    langchain-community
    : Orchestration framework for combining LangChain components into production-ready
    langgraph
    ====================================================================================================
    
    [8] [Original] Q: What are the advantages of LangChain?
    
    Head to the reference section for full documentation of all classes and methods in the LangChain
    Python packages.
    Ecosystem
    🦜🛠 LangSmith
    ----------------------------------------------------------------------------------------------------
    [8] [RRF] Q: What are the advantages of LangChain?
    
    Head to the reference section for full documentation of all classes and methods in the LangChain
    Python packages.
    Ecosystem
    🦜🛠 LangSmith
    ----------------------------------------------------------------------------------------------------
    [8] [CC] Q: What are the advantages of LangChain?
    
    Head to the reference section for full documentation of all classes and methods in the LangChain
    Python packages.
    Ecosystem
    🦜🛠 LangSmith
    ====================================================================================================
    
</pre>

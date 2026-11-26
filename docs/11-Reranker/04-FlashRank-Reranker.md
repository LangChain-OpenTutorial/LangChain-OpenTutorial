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

# FlashRank Reranker

- Author: [Hwayoung Cha](https://github.com/forwardyoung)
- Peer Review: []()
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/04-FlashRank-Reranker.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/04-FlashRank-Reranker.ipynb)
## Overview

> [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) is an ultra-lightweight and ultra-fast Python library designed to add reranking to existing search and **retrieval** pipelines. It is based on state-of-the-art (**SoTA**) **cross-encoders**.

This notebook introduces the use of ```FlashRank-Reranker``` within the LangChain framework, showcasing how to apply reranking techniques to improve the quality of search or **retrieval** results. It provides practical code examples and explanations for integrating ```FlashRank``` into a LangChain pipeline, highlighting its efficiency and effectiveness. The focus is on leveraging ```FlashRank```'s capabilities to enhance the ranking of outputs in a streamlined and scalable way.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [FlashRankRerank](#flashrankrerank)

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
    }
)
```

You can alternatively set OPENAI_API_KEY in ```.env``` file and load it.

[Note] This is not necessary if you've already set OPENAI_API_KEY in previous steps.

```python
# Configuration file to manage API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "flashrank"
    ],
    verbose=False,
    upgrade=False,
)
```

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )
```

## FlashrankRerank

Load data for a simple example and create a **retriever**.

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load the documents
documents = TextLoader("./data/appendix-keywords.txt").load()

# Initialized the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split the documents
texts = text_splitter.split_documents(documents)

# Add a unique ID to each text
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    
# Initialize the retriever
retriever = FAISS.from_documents(
    texts, OpenAIEmbeddings()
).as_retriever(search_kwargs={"k": 10})

# query
query = "Tell me about Word2Vec"

# Search for documents
docs = retriever.invoke(query)

# Print the document
pretty_print_docs(docs)
```

Now, let's wrap the base **retriever** with a ```ContextualCompressionRetriever``` and use ```FlashrankRerank``` as the compressor.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(temperature=0)

# Initialize the FlshrankRerank
compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")

# Initialize the ContextualCompressioinRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Search for compressed documents
compressed_docs = compression_retriever.invoke(
    "Tell me about Word2Vec."
)

# Print the document ID
print([doc.metadata["id"] for doc in compressed_docs])
```

Compare the results after reranker is applied.

```python
# Print the results of document compressions
pretty_print_docs(compressed_docs)
```

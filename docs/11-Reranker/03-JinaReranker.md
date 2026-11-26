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

# JinaReranker

- Author: [hyeyeoon](https://github.com/hyeyeoon)
- Peer Review: 
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/03-JinaReranker.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/03-JinaReranker.ipynb)
## Overview

```Jina Reranker``` is a document re-ranking and compression tool that reorders retrieved documents or results to prioritize the most relevant items. It is primarily used in information retrieval and natural language processing (NLP) tasks, designed to extract critical information more quickly and accurately from large datasets.

---

**Key Features**

- Relevance-based Re-ranking

    Jina Reranker analyzes search results and reorders documents based on relevance scores. This ensures that users can access more relevant information first.

- Multilingual Support

    Jina Reranker supports multilingual models, such as ```jina-reranker-v2-base-multilingual```, enabling the processing of data in various languages.

- Document Compression

    It selects only the top N most relevant documents (```top_n```), compressing the search results to reduce noise and optimize performance.

- Integration with LangChain

    Jina Reranker integrates seamlessly with workflow tools like LangChain, making it easy to connect to natural language processing pipelines.

---

**How It Works**

- Document Retrieval

    The base retriever is used to fetch initial search results.

- Relevance Score Calculation

    Jina Reranker utilizes pre-trained models (e.g., ```jina-reranker-v2-base-multilingual```) to calculate relevance scores for each document.

- Document Re-ranking and Compression

    Based on the relevance scores, it selects the top N documents and provides reordered results.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Jina Reranker](#Jina-Reranker)
- [Performing Re-ranking with JinaRerank](#Performing-re-ranking-with-JinaRerank)

### References

- [LangChain Documentation](https://python.langchain.com/docs/how_to/lcel_cheatsheet/)
- [Jina Reranker](https://jina.ai/reranker/)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.
**Issuing an API Key for JinaReranker**
- Add the following to your .env file
    >JINA_API_KEY="YOUR_JINA_API_KEY"

```python
%%capture --no-stderr
!pip install langchain-opentutorial
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
    ],
    verbose=False,
    upgrade=False,
)
```

You can also load the ```OPEN_API_KEY``` from the ```.env``` file.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Set local environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "03-JinaReranker",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Jina Reranker

- Load data for a simple example and create a retriever.

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

- A text document is loaded into the system.

- The document is split into smaller chunks for better processing.

- ```FAISS``` is used with ```OpenAI embeddings``` to create a retriever.

- The retriever processes a query to find and display the most relevant documents.


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load the document
documents = TextLoader("./data/appendix-keywords.txt").load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split the document into chunks
texts = text_splitter.split_documents(documents)

# Initialize the retriever
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever(
    search_kwargs={"k": 10}
)

# Define the query
query = "Tell me about Word2Vec."

# Retrieve relevant documents
docs = retriever.invoke(query)

# Print the retrieved documents
pretty_print_docs(docs)
```

<pre class="custom">Document 1:
    
    Word2Vec
    Definition: Word2Vec is a technique in NLP that maps words to a vector space, representing their semantic relationships based on context.
    Example: In a Word2Vec model, "king" and "queen" are represented by vectors located close to each other.
    Related Keywords: Natural Language Processing (NLP), Embedding, Semantic Similarity
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    Embedding
    Definition: Embedding is the process of converting textual data, such as words or sentences, into low-dimensional continuous vectors that computers can process and understand.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing (NLP), Vectorization, Deep Learning
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    VectorStore
    Definition: A VectorStore is a system designed to store data in vector format, enabling efficient retrieval, classification, and analysis tasks.
    Example: Storing word embedding vectors in a database for quick access during semantic search.
    Related Keywords: Embedding, Database, Vectorization
    ----------------------------------------------------------------------------------------------------
    Document 4:
    
    TF-IDF (Term Frequency-Inverse Document Frequency)
    Definition: TF-IDF is a statistical measure used to evaluate the importance of a word within a document by considering its frequency and rarity across a corpus.
    Example: Words with high TF-IDF values are often unique and critical for understanding the document.
    Related Keywords: Natural Language Processing (NLP), Information Retrieval, Data Mining
    ----------------------------------------------------------------------------------------------------
    Document 5:
    
    GPT (Generative Pretrained Transformer)
    Definition: GPT is a generative language model pre-trained on vast datasets, capable of performing various text-based tasks. It generates natural and coherent text based on input.
    Example: A chatbot generating detailed answers to user queries is powered by GPT models.
    Related Keywords: Natural Language Processing (NLP), Text Generation, Deep Learning
    ----------------------------------------------------------------------------------------------------
    Document 6:
    
    Tokenizer
    Definition: A tokenizer is a tool that splits text data into tokens, often used for preprocessing in natural language processing tasks.
    Example: The sentence "I love programming." is tokenized into ["I", "love", "programming", "."].
    Related Keywords: Tokenization, Natural Language Processing (NLP), Syntax Analysis.
    ----------------------------------------------------------------------------------------------------
    Document 7:
    
    LLM (Large Language Model)
    Definition: LLMs are massive language models trained on large-scale text data, used for various natural language understanding and generation tasks.
    Example: OpenAI's GPT series is a prominent example of LLMs.
    Related Keywords: Natural Language Processing (NLP), Deep Learning, Text Generation
    ----------------------------------------------------------------------------------------------------
    Document 8:
    
    Transformer
    Definition: A Transformer is a type of deep learning model widely used in natural language processing tasks like translation, summarization, and text generation. It is based on the Attention mechanism.
    Example: Google Translate utilizes a Transformer model for multilingual translation.
    Related Keywords: Deep Learning, Natural Language Processing (NLP), Attention mechanism
    ----------------------------------------------------------------------------------------------------
    Document 9:
    
    Semantic Search
    Definition: Semantic search is a search technique that understands the meaning of a user's query beyond simple keyword matching, returning results that are contextually relevant.
    Example: If a user searches for "planets in the solar system," the system provides information about planets like Jupiter and Mars.
    Related Keywords: Natural Language Processing (NLP), Search Algorithms, Data Mining
    ----------------------------------------------------------------------------------------------------
    Document 10:
    
    HuggingFace
    Definition: HuggingFace is a library offering pre-trained models and tools for natural language processing, making NLP tasks accessible to researchers and developers.
    Example: HuggingFace's Transformers library can be used for sentiment analysis and text generation.
    Related Keywords: Natural Language Processing (NLP), Deep Learning, Library.
</pre>

## Performing Re-ranking with JinaRerank

- A document compression system is initialized using JinaRerank to prioritize the most relevant documents.

- Retrieved documents are compressed by selecting the top 3 (top_n=3) based on relevance.

- A ```ContextualCompressionRetriever``` is created with the JinaRerank compressor and an existing retriever.

- The system processes a query to retrieve and compress relevant documents.

```python
from ast import mod
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

# Initialize the JinaRerank compressor
compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=3)

# Initialize the document compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Retrieve and compress relevant documents
compressed_docs = compression_retriever.invoke("Explain Word2Vec.")

```

```python
# Display the compressed documents in a readable format
pretty_print_docs(compressed_docs)
```

<pre class="custom">Document 1:
    
    Word2Vec
    Definition: Word2Vec is a technique in NLP that maps words to a vector space, representing their semantic relationships based on context.
    Example: In a Word2Vec model, "king" and "queen" are represented by vectors located close to each other.
    Related Keywords: Natural Language Processing (NLP), Embedding, Semantic Similarity
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    Embedding
    Definition: Embedding is the process of converting textual data, such as words or sentences, into low-dimensional continuous vectors that computers can process and understand.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing (NLP), Vectorization, Deep Learning
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    VectorStore
    Definition: A VectorStore is a system designed to store data in vector format, enabling efficient retrieval, classification, and analysis tasks.
    Example: Storing word embedding vectors in a database for quick access during semantic search.
    Related Keywords: Embedding, Database, Vectorization
</pre>

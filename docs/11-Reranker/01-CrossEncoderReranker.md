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

# Cross Encoder Reranker

- Author: [JeongHo Shin](https://github.com/ThePurpleCollar)
- Peer Review:
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/01-CrossEncoderReranker.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/11-Reranker/01-CrossEncoderReranker.ipynb)
## Overview

The **Cross Encoder Reranker** is a technique designed to enhance the performance of **Retrieval-Augmented Generation (RAG)** systems.<br>This guide explains how to implement a reranker using Hugging Face's Cross Encoders to refine the ranking of retrieved documents, promoting those most relevant to a query.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Key Features and Mechanism](#key-features-and-mechanism)
- [Practical Applications](#practical-applications)
- [Implementation](#implementation)
- [Key Advantages of Reranker](#key-advantages-of-reranker)
- [Document Count Settings for Reranker](#document-count-settings-for-reranker)
- [Trade-offs When Using a Reranker](#trade-offs-when-using-a-reranker)

### References

[Hugging Face Cross Encoders](https://huggingface.co/cross-encoder)

----

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

You can alternatively set OPENAI_API_KEY in .env file and load it.

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

## Key Features and Mechanism

### Purpose
- Re-rank retrieved documents to refine their ranking, prioritizing the most relevant results for the query.

### Structure
- Accepts both the **query** and **document** as a single input pair, enabling joint processing.

### Mechanism
- **Single Input Pair** :  
  Processes the **query** and **document** as a combined input to output a relevance score directly.
- **Self-Attention Mechanism** :  
  Uses self-attention to jointly analyze the **query** and **document** , effectively capturing their semantic relationship.

### Advantages
- **Higher Accuracy** :  
  Provides more precise similarity scores.
- **Deep Contextual Analysis** :  
  Explores semantic nuances between **query** and **document** .

### Limitations
- **High Computational Costs** :  
  Processing can be time-intensive.
- **Scalability Issues** :  
  Not suitable for large-scale document collections without optimization.

## Practical Applications
- A **Bi-Encoder** quickly retrieves candidate **documents** by computing lightweight similarity scores.  
- A **Cross Encoder** refines these results by deeply analyzing the semantic relationship between the **query** and the retrieved **documents** .

## Implementation
- Use Hugging Face cross encoders or ```BAAI/bge-reranker``` models.
- Easily integrate with frameworks like **LangChain** through the ```CrossEncoderReranker``` component.

## Key Advantages of Reranker
- **Precise Similarity Scoring** :  
  Delivers highly accurate measurements of relevance between the **query** and **documents**.
- **Semantic Depth** :  
  Analyzes deeper semantic relationships, uncovering nuances in **query** - **document** interactions.
- **Refined Search Quality** :  
  Improves the relevance and quality of the retrieved **documents** .
- **RAG System Boost** :  
  Enhances the performance of **Retrieval-Augmented Generation (RAG)** systems by refining input relevance.
- **Seamless Integration** :  
  Easily adaptable to various workflows and compatible with multiple frameworks.
- **Model Versatility** :  
  Offers flexibility with a wide range of pre-trained models for tailored use cases.

## Document Count Settings for Reranker
- Reranking is generally performed on the top **5–10** **documents** retrieved during the initial search.
- The ideal number of **documents** for reranking should be determined through experimentation and evaluation, as it depends on the dataset characteristics and computational resources available.

## Trade-offs When Using a Reranker
- **Accuracy vs Processing Time** :  
  Striking a balance between achieving higher accuracy and minimizing processing time.
- **Performance Improvement vs Computational Cost** :  
  Weighing the benefits of improved performance against the additional computational resources required.
- **Search Speed vs Relevance Accuracy** :  
  Managing the trade-off between faster retrieval and maintaining high relevance in results.
- **System Requirements** :  
  Ensuring the system meets the necessary hardware and software requirements to support reranking.
- **Dataset Characteristics** :  
  Considering the scale, diversity, and specific attributes of the dataset to optimize reranker performance.

Explaining the Implementation of **Cross Encoder Reranker** with a Simple Example

```python
# Helper function to format and print document content
def pretty_print_docs(docs):
    # Print each document in the list with a separator between them
    print(
        f"\n{'-' * 100}\n".join(  # Separator line for better readability
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]  # Format: Document number + content
        )
    )
```

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
documents = TextLoader("./data/appendix-keywords.txt").load()

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split documents into chunks
texts = text_splitter.split_documents(documents)

# # Set up the embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5",
    model_kwargs = {"tokenizer_kwargs": {"clean_up_tokenization_spaces": False}}
)

# Create FAISS index from documents and set up retriever
retriever = FAISS.from_documents(texts, embeddings_model).as_retriever(
    search_kwargs={"k": 10}
)

# Define the query
query = "Can you tell me about Word2Vec?"

# Execute the query and retrieve results
docs = retriever.invoke(query)

# Display the retrieved documents
pretty_print_docs(docs)
```

<pre class="custom">Document 1:
    
    Word2Vec
    Definition: Word2Vec is a technique in NLP that maps words to a vector space, representing their semantic relationships based on context.
    Example: In a Word2Vec model, "king" and "queen" are represented by vectors located close to each other.
    Related Keywords: Natural Language Processing (NLP), Embedding, Semantic Similarity
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    Token
    Definition: A token refers to a smaller unit of text obtained by splitting a larger piece of text. It can be a word, phrase, or sentence.
    Example: The sentence "I go to school" can be tokenized into "I," "go," "to," and "school."
    Related Keywords: Tokenization, Natural Language Processing (NLP), Syntax Analysis
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    Example: A customer information table in a relational database is an example of structured data.
    Related Keywords: Database, Data Analysis, Data Modeling
    ----------------------------------------------------------------------------------------------------
    Document 4:
    
    Schema
    Definition: A schema defines the structure of a database or file, detailing how data is organized and stored.
    Example: A relational database schema specifies column names, data types, and key constraints.
    Related Keywords: Database, Data Modeling, Data Management
    ----------------------------------------------------------------------------------------------------
    Document 5:
    
    Keyword Search
    Definition: Keyword search involves finding information based on user-inputted keywords, commonly used in search engines and database systems.
    Example: Searching 
    When a user searches for "coffee shops in Seoul," the system returns a list of relevant coffee shops.
    Related Keywords: Search Engine, Data Search, Information Retrieval
    ----------------------------------------------------------------------------------------------------
    Document 6:
    
    TF-IDF (Term Frequency-Inverse Document Frequency)
    Definition: TF-IDF is a statistical measure used to evaluate the importance of a word within a document by considering its frequency and rarity across a corpus.
    Example: Words with high TF-IDF values are often unique and critical for understanding the document.
    Related Keywords: Natural Language Processing (NLP), Information Retrieval, Data Mining
    ----------------------------------------------------------------------------------------------------
    Document 7:
    
    SQL
    Definition: SQL (Structured Query Language) is a programming language for managing data in databases. 
    It allows you to perform various operations such as querying, updating, inserting, and deleting data.
    Example: SELECT * FROM users WHERE age > 18; retrieves information about users aged above 18.
    Related Keywords: Database, Query, Data Management
    ----------------------------------------------------------------------------------------------------
    Document 8:
    
    Open Source
    Definition: Open source software allows its source code to be freely used, modified, and distributed, fostering collaboration and innovation.
    Example: The Linux operating system is a well-known open source project.
    Related Keywords: Software Development, Community, Technical Collaboration
    Structured Data
    Definition: Structured data is organized according to a specific format or schema, making it easy to search and analyze.
    ----------------------------------------------------------------------------------------------------
    Document 9:
    
    Semantic Search
    Definition: Semantic search is a search technique that understands the meaning of a user's query beyond simple keyword matching, returning results that are contextually relevant.
    Example: If a user searches for "planets in the solar system," the system provides information about planets like Jupiter and Mars.
    Related Keywords: Natural Language Processing (NLP), Search Algorithms, Data Mining
    ----------------------------------------------------------------------------------------------------
    Document 10:
    
    GPT (Generative Pretrained Transformer)
    Definition: GPT is a generative language model pre-trained on vast datasets, capable of performing various text-based tasks. It generates natural and coherent text based on input.
    Example: A chatbot generating detailed answers to user queries is powered by GPT models.
    Related Keywords: Natural Language Processing (NLP), Text Generation, Deep Learning
</pre>

Now, let's wrap the ```base_retriever``` with a ```ContextualCompressionRetriever``` . The ```CrossEncoderReranker``` leverages ```HuggingFaceCrossEncoder``` to re-rank the retrieved results.


Multilingual Support BGE Reranker: [```bge-reranker-v2-m3```](https://huggingface.co/BAAI/bge-reranker-v2-m3)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize the model
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# Select the top 3 documents
compressor = CrossEncoderReranker(model=model, top_n=3)

# Initialize the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Retrieve compressed documents
compressed_docs = compression_retriever.invoke("Can you tell me about Word2Vec?")

# Display the documents
pretty_print_docs(compressed_docs)
```

<pre class="custom">Document 1:
    
    Word2Vec
    Definition: Word2Vec is a technique in NLP that maps words to a vector space, representing their semantic relationships based on context.
    Example: In a Word2Vec model, "king" and "queen" are represented by vectors located close to each other.
    Related Keywords: Natural Language Processing (NLP), Embedding, Semantic Similarity
    ----------------------------------------------------------------------------------------------------
    Document 2:
    
    Open Source
    Definition: Open source software allows its source code to be freely used, modified, and distributed, fostering collaboration and innovation.
    Example: The Linux operating system is a well-known open source project.
    Related Keywords: Software Development, Community, Technical Collaboration
    Structured Data
    Definition: Structured data is organized according to a specific format or schema, making it easy to search and analyze.
    ----------------------------------------------------------------------------------------------------
    Document 3:
    
    TF-IDF (Term Frequency-Inverse Document Frequency)
    Definition: TF-IDF is a statistical measure used to evaluate the importance of a word within a document by considering its frequency and rarity across a corpus.
    Example: Words with high TF-IDF values are often unique and critical for understanding the document.
    Related Keywords: Natural Language Processing (NLP), Information Retrieval, Data Mining
</pre>

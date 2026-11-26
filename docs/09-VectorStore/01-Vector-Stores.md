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

# Vector Stores

- Author: [Youngin Kim](https://github.com/Normalist-K)
- Peer Review: [ro__o_jun](https://github.com/ro-jun)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial provides a comprehensive guide to vector stores, which are specialized databases for indexing and retrieving information using vector representations (embeddings). It highlights their importance in enabling fast, scalable, and semantic search across unstructured data like text, images, and audio.

### Key Sections:
1. **Essentials of Vector Stores** :
   - Focus on the need for semantic search, scalability, and efficient retrieval.
   - Overview of how vector stores outperform traditional keyword-based search systems.

2. **LangChain Interface** :
   - Explanation of core methods such as `add_documents` , `upsert_documents` , `delete_documents` , and `similarity_search` .
   - Introduction to advanced features like parallel document processing.

3. **Search Methods** :  
   - Covers `Keyword-Based Search` , which relies on exact term matching, and `Similarity-Based Search` , which identifies semantically related results using embeddings.  
   - Explains `Score-Based Similarity Search` , which ranks documents based on relevance using metrics such as `cosine similarity` .  
   - Discusses advanced search techniques, including `Sparse` , `Dense` , and `Hybrid Search` , to improve retrieval performance.  
   - Demonstrates the use of LangChain’s `.as_retriever()` method for seamless integration with retrieval pipelines.  

5. **Integration of Popular Vector Stores** :
   - Overview of prominent vector stores such as Chroma, FAISS, Pinecone, Qdrant, Elasticsearch, MongoDB, pgvector, Neo4j, Weaviate, and Milvus.
   - Each store’s strengths and suitable use cases are briefly summarized.

This tutorial serves as a foundational guide to understanding and leveraging vector stores for AI-driven applications. It bridges basic concepts with advanced implementations, offering insights into efficient data retrieval and integration strategies within LangChain’s ecosystem.

### Table of Contents

- [Overview](#overview)
- [Conceptual Guide of Vectore Store](#conceptual-guide-of-vector-store)
- [Integration](#integration)

### References

- [LangChain How-to guides: Vector store](https://python.langchain.com/docs/how_to/#vector-stores)
- [Concept of Vector stores from LangChain](https://python.langchain.com/docs/concepts/vectorstores/)
- [Vector store supported integrations of LangChain](https://python.langchain.com/docs/integrations/vectorstores/)
- [Top 10 Vector Stores](https://blog.langchain.dev/langchain-state-of-ai-2024/)
- [Vector Databases: Comparison for Semantic Search and Retrieval-Augmented Generation](https://jerrysmd.github.io/20231007_vector-database/?utm_source=chatgpt.com)
----

## Conceptual Guide of Vector Store

*Prerequisite: `07-TextSplitter` `08-Embedding`*


**Vector stores** are specialized databases designed to **index** and **retrieve** information using vector representations (embeddings).

They are commonly utilized to search through unstructured data, such as text, images, and audio, by identifying semantically similar content rather than relying on exact keyword matches.



### Why Vector Store Is Essential

1. Fast and Efficient Search

By properly storing and indexing embedding vectors, vector stores allow for the rapid retrieval of relevant information, even when dealing with massive datasets.

2. Scalability for Growing Data

As data continues to expand, vector stores must scale efficiently. A well-structured vector store ensures the system can handle large-scale data without performance issues, supporting seamless growth.

3. Facilitating Semantic Search

Unlike traditional keyword-based search, semantic search retrieves content based on meaning. Vector stores enable this by finding paragraphs or sections that closely align with a user’s query in context. This is a key advantage over databases that store raw text, which are limited to exact keyword matches.

### Interface

LangChain provides a unified interface for interacting with vector stores, allowing users to seamlessly switch between various implementations.

This interface includes core methods for **writing**, **deleting**, and **searching** documents within the vector store.

The main methods are as follows:

- `add_documents` : Adds a list of texts to the vector store.
- `upsert_documents` : Adds new documents to the vector store or updates existing ones if they already exist.
  - In this tutorial, we'll also introduce the `upsert_documents_parallel` method, which enables efficient bulk processing of data when applicable.
- `delete_documents` : Deletes a list of documents from the vector store.
- `similarity_search` : Searches for documents similar to a given query.



### Understanding Search Methods

- **Keyword-Based Search**  
  This method relies on matching exact words or phrases in the query with those in the document. It’s simple but lacks the ability to capture semantic relationships between terms.

- **Similarity-Based Search**  
  Uses vector representations to evaluate how semantically similar the query is to the documents. It provides more accurate results, especially for natural language queries.

- **Score-Based Similarity Search**  
  Assigns a similarity score to each document based on the query. Higher scores indicate stronger relevance. Commonly uses metrics like `cosine similarity` or `distance-based scoring`.


### How Similarity Search Works

- **Concept of Embeddings and Vectors**  
  `Embeddings` are numerical representations of words or documents in a high-dimensional space. They capture semantic meaning, enabling better comparison between query and documents.

- **Similarity Measurement Methods**  
  - **Cosine Similarity**: Measures the cosine of the angle between two vectors. Values closer to 1 indicate higher similarity.  
  - **Euclidean Distance**: Calculates the straight-line distance between two points in vector space. Smaller distances imply higher similarity.

- **Scoring and Ranking Search Results**  
  After calculating similarity, documents are assigned scores. Results are ranked in descending order of relevance based on these scores.

- **Brief Overview of Search Algorithms**  
  - `TF-IDF`: Weights terms based on their frequency in a document relative to their occurrence across all documents.  
  - `BM25`: A refined version of `TF-IDF`, optimized for relevance in information retrieval.  
  - `Neural Search`: Leverages deep learning to generate context-aware embeddings for more accurate results.


### Types of Searches in Vector Stores

- **Similarity Search** : Finds documents with embeddings most similar to the query. Ideal for semantic search applications.

- **Maximal Marginal Relevance (MMR) Search** : Balances relevance and diversity in search results by prioritizing diverse yet relevant documents.

- **Sparse Retriever** : Uses traditional keyword-based methods like `TF-IDF` or `BM25` to retrieve documents. Effective for datasets with limited context.

- **Dense Retriever** : Relies on dense vector embeddings to capture semantic meaning. Common in modern search systems using deep learning.

- **Hybrid Search** : Combines sparse and dense retrieval methods. Balances the precision of dense methods with the broad coverage of sparse methods for optimal results.

### Vector Store as a Retriever
- **Functionality** : By converting a vector store into a retriever using the `.as_retriever()` method, you create a lightweight wrapper that conforms to LangChain’s retriever interface. This enables the use of various retrieval strategies, such as `similarity search` and `maximal marginal relevance (MMR) search`, and allows for customization of retrieval parameters. ￼
- **Use Case** : Ideal for complex applications where the retriever needs to be part of a larger pipeline, such as retrieval-augmented generation (RAG) systems. It facilitates seamless integration with other components in LangChain, enabling functionalities like ensemble retrieval methods and advanced query analysis. ￼

In summary, while direct vector store searches provide basic retrieval capabilities, converting a vector store into a retriever offers enhanced flexibility and integration within LangChain’s ecosystem, supporting more sophisticated retrieval strategies and applications.

## Integration

Here is a brief overview of the vector stores covered in this tutorial:
- `Chroma` : An open-source vector database designed for AI applications, enabling efficient storage and retrieval of embeddings.
- `FAISS` : Developed by Facebook AI, FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. ￼
- `Pinecone` : A managed vector database service that provides high-performance vector similarity search, enabling developers to build scalable AI applications.
- `Qdrant` : Qdrant (read: quadrant ) is a vector similarity search engine. It provides a production-ready service with a convenient API to store, search, and manage vectors with additional payload and extended filtering support. It makes it useful for all sorts of neural network or semantic-based matching, faceted search, and other applications.
- `Elasticsearch` : A distributed, RESTful search and analytics engine that supports vector search, allowing for efficient similarity searches within large datasets.
- `MongoDB` : MongoDB Atlas Vector Search enables efficient storage, indexing, and querying of vector embeddings alongside your operational data, facilitating seamless implementation of AI-driven applications.
- `pgvector (PostgreSQL)` :  An extension for PostgreSQL that adds vector similarity search capabilities, allowing for efficient storage and querying of vector data within a relational database.
- `Neo4j` : A graph database that stores nodes and relationships, with native support for vector search, facilitating complex queries involving both graph and vector data. ￼
- `Weaviate` : An open-source vector database that allows for storing data objects and vector embeddings, supporting various data types and offering semantic search capabilities.
- `Milvus` : A database that stores, indexes, and manages massive embedding vectors generated by machine learning models, designed for high-performance vector similarity search. ￼

These vector stores are integral in building applications that require efficient similarity search and management of high-dimensional data.

| Vector Store             | Delete by ID | Filtering | Search by Vector | Search with Score | Async | Passes Standard Tests | Multi Tenancy | IDs in Add Documents |
|--------------------------|:------------:|:---------:|:----------------:|:-----------------:|:-----:|:---------------------:|:-------------:|:--------------------:|
| Chroma                   |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| Faiss                    |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| Pinecone                 |      ✅      |     ✅    |        ✅        |         ❌        |  ✅   |          ❌           |       ❌       |          ❌          |
| Qdrant                   |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| Elasticsearch            |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| MongoDB-Atlas (MongoDB)                 |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| PGVector (PostgreSQL)    |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |
| Neo4j                    |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ❌       |          ✅          |
| Weaviate                 |      ✅      |     ✅    |        ✅        |         ✅        |  ✅   |          ❌           |       ✅       |          ❌          |
| Milvus                   |      ✅      |     ✅    |        ❌        |         ✅        |  ✅   |          ❌           |       ❌       |          ❌          |


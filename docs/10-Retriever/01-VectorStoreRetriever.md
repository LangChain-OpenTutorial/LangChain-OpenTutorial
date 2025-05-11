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

# VectorStore-backed Retriever

- Author: [Erika Park](https://www.linkedin.com/in/yeonseo-park-094193198/)
- Designer: [Erika Park](https://www.linkedin.com/in/yeonseo-park-094193198/)
- Proofread : [jishin86](https://github.com/jishin86)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/01-VectorStoreRetriever.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/01-VectorStoreRetriever.ipynb)

## Overview
This tutorial provides a comprehensive guide to building and optimizing a **VectorStore-backed retriever** using LangChain. It covers the foundational steps of creating a vector store with FAISS(Facebook AI Similarity Search) and explores advanced retrieval strategies for improving search accuracy and efficiency.

A **VectorStore-backed retriever** is a document retrieval system that leverages a vector store to search for documents based on their vector representations. This approach enables efficient similarity-based search for handling unstructured data.


### RAG (Retrieval-Augmented Generation) Workflow
![rag-flow.png](./img/01-vectorstore-retriever-rag-flow.png)

The diagram above illustrates the  **document search and response generation** workflow within a RAG system. 

The steps include:

1. Document Loading: Importing raw documents.  
2. Text Chunking: Splitting text into manageable chunks.  
3. Vector Embedding: Converting the text into numerical vectors using an embedding model.  
4. Store in Vector Database: Storing the generated embeddings in a vector database for efficient retrieval.

During the query phase:
- Steps: User Query → Embedding → Search in VectorStore → Relevant Chunks Retrieved → LLM Generates Response
- The user's query is transformed into an embedding vector using an embedding model.
- This query embedding is compared against stored document vectors within the vector database to **retrieve the most relevant results**.
- The retrieved chunks are passed to a Large Language Model (LLM), which generates a final response based on the retrieved information.

This tutorial aims to explore and optimize the VectorStore → Relevant Chunks Retrieved → LLM Generates Response stages. It will cover advanced retrieval techniques to improve the accuracy and relevance of the responses.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Initializing and Using VectorStoreRetriever](#initializing-and-using-vectorstoreretriever)
- [Dynamic Configuration (Using ConfigurableField)](#dynamic-configuration-using-configurablefield)
- [Using Separate Query & Passage Embedding Models](#using-separate-query--passage-embedding-models)

### References

- [How to use a vectorstore as a retriever](https://python.langchain.com/docs/how_to/vectorstore_retriever/)
- [Maximum Marginal Relevance (MMR)](https://community.fullstackretrieval.com/retrieval-methods/maximum-marginal-relevance)
- [Upstage-Embeddings](https://console.upstage.ai/docs/capabilities/embeddings)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions, and utilities for tutorials. 
- You can checkout out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_opentutorial",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_core",
        "langchain_upstage",
        "faiss-cpu"
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
        # "OPENAI_API_KEY": "",
        # "LANGCHAIN_API_KEY": "",
        # "UPSTAGE_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "VectorStore Retriever"
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Configuration file to manage the API KEY as an environment variable
from dotenv import load_dotenv

# Load API KEY information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Initializing and Using VectorStoreRetriever

This section demonstrates how to load documents using OpenAI embeddings and create a vector database using FAISS.

- The example below showcases how to use OpenAI embeddings for document loading and FAISS for vector database creation.
- Once the vector database is created, it can be loaded and queried using retrieval methods such as **Similarity Search** and **Maximal Marginal Relevance (MMR)** to search for relevant text within the vector store.

📌 **Creating a Vector Store (Using FAISS)**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load the file using TextLoader
loader = TextLoader("./data/01-vectorstore-retriever-appendix-keywords.txt", encoding="utf-8")
documents = loader.load()

# split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents) # Split into smaller chunks

# Initialize the OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Create a FAISS vector database
db = FAISS.from_documents(split_docs, embeddings)
```

<pre class="custom">Created a chunk of size 351, which is longer than the specified 300
    Created a chunk of size 343, which is longer than the specified 300
    Created a chunk of size 307, which is longer than the specified 300
    Created a chunk of size 316, which is longer than the specified 300
    Created a chunk of size 341, which is longer than the specified 300
    Created a chunk of size 321, which is longer than the specified 300
    Created a chunk of size 303, which is longer than the specified 300
    Created a chunk of size 325, which is longer than the specified 300
    Created a chunk of size 315, which is longer than the specified 300
    Created a chunk of size 304, which is longer than the specified 300
    Created a chunk of size 385, which is longer than the specified 300
    Created a chunk of size 349, which is longer than the specified 300
    Created a chunk of size 376, which is longer than the specified 300
</pre>

📌 **1. Initializing and Using VectorStoreRetriever (```as_retriever``` )**

The ```as_retriever``` method allows you to convert a vector database into a retriever, enabling efficient document search and retrieval from the vector store.

**How It Works**:
* The ```as_retriever()``` method transforms a vector store (like FAISS) into a retriever object, making it compatible with LangChain's retrieval workflows.
* This retriever can then be directly used with RAG pipelines or combined with Large Language Models (LLMs) for building intelligent search systems.

```python
# Basic Retriever Creation (Similarity Search)
retriever = db.as_retriever()
```

**Advanced Retriever Configuration**

The ```as_retriever``` method allows you to configure advanced retrieval strategies, such as **similarity search**, **MMR (Maximal Marginal Relevance)**, and **similarity score threshold-based filtering**.


**Parameters:**

- ```**kwargs``` : Keyword arguments passed to the retrieval function:
   - ```search_type``` : Specifies the search method.
     - ```"similarity"``` : Returns the most relevant documents based on cosine similarity.
     - ```"mmr"``` : Utilizes the Maximal Marginal Relevance algorithm, balancing **relevance** and **diversity**.
     - ```"similarity_score_threshold"``` : Returns documents with a similarity score above a specified threshold.
   - ```search_kwargs``` : Additional search options for fine-tuning results:
     - ```k``` : Number of documents to return (default: ```4``` ).
     - ```score_threshold``` : Minimum similarity score for the ```"similarity_score_threshold"``` search type (e.g., ```0.8``` ).
     - ```fetch_k``` : Number of documents initially retrieved during an MMR search (default: ```20``` ).
     - ```lambda_mult``` : Controls diversity in MMR results (```0``` = maximum diversity, ```1``` = maximum relevance, default: ```0.5``` ).
     - ```filter``` : Metadata filtering for selective document retrieval.


 **Return Value:**

- ```VectorStoreRetriever```: An initialized retriever object that can be directly queried for document search tasks.


**Notes:**
- Supports multiple search strategies (```similarity``` , ```MMR``` , ```similarity_score_threshold``` ).
- MMR improves result diversity while preserving relevance by reducing redundancy in results.
- Metadata filtering enables selective document retrieval based on document properties.
- The ```tags``` parameter can be used to label retrievers for better organization and easier identification.

 **Cautions:**
- Diversity Control with MMR:
  - Adjust both ```fetch_k``` (number of documents initially retrieved) and ```lambda_mult``` (diversity control factor) carefully for optimal balance.
  - ```lambda_mult```
    - Lower values (< 0.5) → Prioritize diversity.
    - Higher values (> 0.5) → Prioritize relevance.
  - set ```fetch_k``` higher than ```k``` for effective diversity control.
- Threshold Settings: 
  - Using a high ```score_threshold``` (e.g., 0.95) can lead to zero results.
- Metadata Filtering: 
  - Ensure the metadata structure is well-defined before applying filters.
- Balanced Configuration:
  - Maintain a proper balance between ```search_type``` and ```search_kwargs``` settings for optimal retrieval performance.


```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={
        "k": 5,  # Return the top 5 most relevant documents
        "score_threshold": 0.7  # Only return documents with a similarity score of 0.7 or higher
    }
)
# Perform the search
query = "Explain the concept of vector search."
results = retriever.invoke(query)

# Display search results
for doc in results:
    print(doc.page_content)
```

<pre class="custom">Semantic Search
    VectorStore
    
    Definition: A vector store is a system for storing data in vector format, often used for search, classification, and data analysis tasks.
    Example: Storing word embeddings in a database for fast retrieval of similar words.
    Related Keywords: Embedding, Database, Vectorization
    Definition: Semantic search is a method of retrieving results based on the meaning of the user's query, going beyond simple keyword matching.
    Example: If a user searches for "solar system planets," the search returns information about related planets like Jupiter and Mars.
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    Definition: Keyword search is the process of finding information based on specific keywords entered by the user. It is commonly used in search engines and database systems as a fundamental search method.
    Example: If a user searches for "coffee shop in Seoul," the search engine returns a list of related coffee shops.
    Related Keywords: Search Engine, Data Retrieval, Information Search
    Definition: FAISS is a high-speed similarity search library developed by Facebook, designed for efficient vector searches in large datasets.
    Example: Searching for similar images in a dataset of millions using FAISS.
    Related Keywords: Vector Search, Machine Learning, Database Optimization
</pre>

### Retriever's ```invoke()``` Method

The ```invoke()``` method is the primary entry point for interacting with a Retriever. It is used to search and retrieve relevant documents based on a given query.

**How It Works** :
1. Query Submission: A user query is provided as input.
2. Embedding Generation: The query is converted into a vector representation (if necessary).
3. Search Process: The retriever searches the vector database using the specified search strategy (similarity, MMR, etc.).
4. Results Return: The method returns a list of relevant document chunks.

 **Parameters:**
- ```input``` (Required):
   - The query string provided by the user.
   - The query is converted into a vector and compared with stored document vectors for similarity-based retrieval.

- ```config``` (Optional):
   - Allows for fine-grained control over the retrieval process.
   - Can be used to specify **tags, metadata insertion, and search strategies**.

- ```**kwargs``` (Optional):
   - Enables direct passing of ```search_kwargs``` for advanced configuration.
   - Example options include:
     - ```k``` : Number of documents to return.
     - ```score_threshold``` : Minimum similarity score for a document to be included.
     - ```fetch_k``` : Number of documents initially retrieved in MMR searches.


 **Return Value:**
- ```List[Document]```:
   - Returns a list of document objects containing the retrieved text and metadata.
   - Each document object includes:
     - ```page_content``` : The main content of the document.
     - ```metadata``` : Associated metadata with the document (e.g., source, tags).


**Usage Example 1: Basic Usage (Synchronous Search)**

```python
docs = retriever.invoke("What is an embedding?")

for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

<pre class="custom">Embedding
    =========================================================
    Definition: Embedding is the process of converting text data, such as words or sentences, into continuous low-dimensional vectors. This allows computers to understand and process text.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    =========================================================
    Semantic Search
    =========================================================
    Deep Learning
    =========================================================
</pre>

**Usage Example 2: Search with Options** ( ```search_kwargs``` )

```python
# search options: top 5 results with a similarity score ≥ 0.7
docs = retriever.invoke(
    "What is a vector database?",
    search_kwargs={"k": 5, "score_threshold": 0.7}
)
for doc in docs:
    print(doc.page_content)
    print("=========================================================")
```

<pre class="custom">VectorStore
    
    Definition: A vector store is a system for storing data in vector format, often used for search, classification, and data analysis tasks.
    Example: Storing word embeddings in a database for fast retrieval of similar words.
    Related Keywords: Embedding, Database, Vectorization
    =========================================================
</pre>

**Usage Example 3: Using** ```config``` **and** ```**kwargs``` **(Advanced Configuration)**

```python
from langchain_core.runnables.config import RunnableConfig

# Create a RunnableConfig with tags and metadata
config = RunnableConfig(
    tags=["retrieval", "faq"],  ## Adding tags for query categorization
    metadata={"project": "vectorstore-tutorial"}  # Project-specific metadata for traceability
)
# Perform a query using advanced configuration settings
docs = retriever.invoke(
    input="What is a DataFrame?", 
    config=config,  # Applying the config with tags and metadata
    search_kwargs={
        "k": 3,                   
        "score_threshold": 0.8   
    }
)
#  Display the search results
for idx, doc in enumerate(docs):
    print(f"\n🔍 [Search Result {idx + 1}]")
    print("📄 Document Content:", doc.page_content)
    print("🗂️ Metadata:", doc.metadata)
    print("=" * 60)
```

<pre class="custom">
    🔍 [Search Result 1]
    📄 Document Content: Definition: A DataFrame is a tabular data structure with rows and columns, commonly used for data analysis and manipulation.
    Example: Pandas DataFrame can store data like an Excel sheet and perform operations like filtering and grouping.
    Related Keywords: Data Analysis, Pandas, Data Manipulation
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    
    🔍 [Search Result 2]
    📄 Document Content: Schema
    
    Definition: A schema defines the structure of a database or file, describing how data is stored and organized.
    Example: A database schema can specify table columns, data types, and constraints.
    Related Keywords: Database, Data Modeling, Data Management
    
    DataFrame
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    
    🔍 [Search Result 3]
    📄 Document Content: Pandas
    
    Definition: Pandas is a Python library for data analysis and manipulation, offering tools for working with structured data.
    Example: Pandas can read CSV files, clean data, and perform statistical analysis.
    Related Keywords: Data Analysis, Python, Data Manipulation
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    
    🔍 [Search Result 4]
    📄 Document Content: Data Mining
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
</pre>

### Max Marginal Relevance (MMR)

The **Maximal Marginal Relevance (MMR)** search method is a document retrieval algorithm designed to reduce redundancy by balancing relevance and diversity when returning results.

**How MMR Works:**
Unlike basic similarity-based searches that return the most relevant documents based solely on similarity scores, MMR considers two critical factors:
1. Relevance: Measures how closely the document matches the user's query.
2. Diversity: Ensures the retrieved documents are distinct from each other to avoid repetitive results.

 **Key Parameters:**
- ```search_type="mmr"```: Activates the MMR retrieval strategy.  
- ```k```: The number of documents returned after applying diversity filtering(default: ```4```).  
- ```fetch_k```: Number of documents initially retrieved before applying diversity filtering (default: ```20```).  
- ```lambda_mult```: Diversity control factor (```0 = max diversity``` , ```1 = max relevance``` , default: ```0.5```).

```python
# MMR Retriever Configuration (Balancing Relevance and Diversity)
retriever = db.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 3,                
        "fetch_k": 10,           
        "lambda_mult": 0.6  # Balancing Similarity and Diversity (0.6: Slight Emphasis on Diversity)
    }
)

query = "What is an embedding?"
docs = retriever.invoke(query)

#  Display the search results
print(f"\n🔎 [Query]: {query}\n")
for idx, doc in enumerate(docs):
    print(f"📄 [Document {idx + 1}]")
    print("📖 Document Content:", doc.page_content)
    print("🗂️ Metadata:", doc.metadata)
    print("=" * 60)
```

<pre class="custom">
    🔎 [Query]: What is an embedding?
    
    📄 [Document 1]
    📖 Document Content: Embedding
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 2]
    📖 Document Content: Definition: Embedding is the process of converting text data, such as words or sentences, into continuous low-dimensional vectors. This allows computers to understand and process text.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 3]
    📖 Document Content: TF-IDF (Term Frequency-Inverse Document Frequency)
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
</pre>

### Similarity Score Threshold Search

**Similarity Score Threshold Search** is a retrieval method where only documents exceeding a predefined similarity score are returned. This approach helps filter out low-relevance results, ensuring that the returned documents are highly relevant to the query.

**Key Features:**
- Relevance Filtering: Returns only documents with a similarity score above the specified threshold.
- Configurable Precision: The threshold is adjustable using the ```score_threshold``` parameter.
- Search Type Activation: Enabled by setting ```search_type="similarity_score_threshold"``` .

This search method is ideal for tasks requiring **highly precise** results, such as fact-checking or answering technical queries.

```python
# Retriever Configuration (Similarity Score Threshold Search)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  
    search_kwargs={
        "score_threshold": 0.6,  
        "k": 5                
    }
)
# Execute the query
query = "What is Word2Vec?"
docs = retriever.invoke(query)

# Display the search results 
print(f"\n🔎 [Query]: {query}\n")
if docs:
    for idx, doc in enumerate(docs):
        print(f"📄 [Document {idx + 1}]")
        print("📖 Document Content:", doc.page_content)
        print("🗂️ Metadata:", doc.metadata)
        print("=" * 60)
else:
    print("⚠️ No relevant documents found. Try lowering the similarity score threshold.")
```

<pre class="custom">
    🔎 [Query]: What is Word2Vec?
    
    📄 [Document 1]
    📖 Document Content: Word2Vec
    
    Definition: Word2Vec is a technique in NLP that maps words into a vector space, representing their semantic relationships based on context.
    Example: In Word2Vec, "king" and "queen" would be represented by vectors close to each other.
    Related Keywords: NLP, Embeddings, Semantic Similarity
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 2]
    📖 Document Content: Definition: Embedding is the process of converting text data, such as words or sentences, into continuous low-dimensional vectors. This allows computers to understand and process text.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 3]
    📖 Document Content: TF-IDF (Term Frequency-Inverse Document Frequency)
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 4]
    📖 Document Content: Tokenizer
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 5]
    📖 Document Content: Semantic Search
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
</pre>

### Configuring ```top_k``` (Adjusting the Number of Returned Documents)

- The parameter ```k``` specifies the number of documents returned during a vector search. It determines how many of the **top-ranked** documents (based on similarity score) will be retrieved from the vector database.

- The number of documents retrieved can be adjusted by setting the ```k``` value within the ```search_kwargs```.  
- For example, setting ```k=1``` will return only the **top 1 most relevant document** based on similarity.

```python
# Retriever Configuration (Return Only the Top 1 Document)
retriever = db.as_retriever(
    search_kwargs={
        "k": 1  # Return only the top 1 most relevant document
    }
)

query = "What is an embedding?"
docs = retriever.invoke(query)

#  Display the search results 
print(f"\n🔎 [Query]: {query}\n")
if docs:
    for idx, doc in enumerate(docs):
        print(f"📄 [Document {idx + 1}]")
        print("📖 Document Content:", doc.page_content)
        print("🗂️ Metadata:", doc.metadata)
        print("=" * 60)
else:
    print("⚠️ No relevant documents found. Try increasing the `k` value.")
```

<pre class="custom">
    🔎 [Query]: What is an embedding?
    
    📄 [Document 1]
    📖 Document Content: Embedding
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
</pre>

## Dynamic Configuration (Using ```ConfigurableField``` )

The ```ConfigurableField``` feature in LangChain allows for **dynamic adjustment** of search configurations, providing flexibility during query execution.

**Key Features:**
- Runtime Search Configuration: Adjust search settings without modifying the core retriever setup.
- Enhanced Traceability: Assign unique identifiers, names, and descriptions to each parameter for improved readability and debugging.
- Flexible Control with ```config```: Search configurations can be passed dynamically using the ```config``` parameter as a dictionary.


**Use Cases:**
- Switching Search Strategies: Dynamically adjust the search type (e.g., ```"similarity"```, ```"mmr"``` ).
- Real-Time Parameter Adjustments: Modify search parameters like ```k``` , ```score_threshold``` , and ```fetch_k``` during query execution.
- Experimentation: Easily test different search strategies and parameter combinations without rewriting code.

```python
from langchain_core.runnables import ConfigurableField 

# Retriever Configuration Using ConfigurableField
retriever = db.as_retriever(search_kwargs={"k": 1}).configurable_fields(
    search_type=ConfigurableField(
        id="search_type", 
        name="Search Type",  # Name for the search strategy
        description="The search type to use",  # Description of the search strategy
    ),
    search_kwargs=ConfigurableField(
        id="search_kwargs",  
        name="Search Kwargs",  # Name for the search parameters
        description="The search kwargs to use",  # Description of the search parameters
    ),
)
```

The following examples demonstrate how to apply dynamic search settings using ```ConfigurableField``` in LangChain.


```python
# ✅ Search Configuration 1: Basic Search (Top 3 Documents)

config_1 = {"configurable": {"search_kwargs": {"k": 3}}}

# Execute the query
docs = retriever.invoke("What is an embedding?", config=config_1)

# Display the search results
print("\n🔎 [Search Results - Basic Configuration (Top 3 Documents)]")
for idx, doc in enumerate(docs):
    print(f"📄 [Document {idx + 1}]")
    print(doc.page_content)
    print("=" * 60)
```

<pre class="custom">
    🔎 [Search Results - Basic Configuration (Top 3 Documents)]
    📄 [Document 1]
    Embedding
    ============================================================
    📄 [Document 2]
    Definition: Embedding is the process of converting text data, such as words or sentences, into continuous low-dimensional vectors. This allows computers to understand and process text.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    ============================================================
    📄 [Document 3]
    Semantic Search
    ============================================================
</pre>

```python
# ✅ Search Configuration 2: Similarity Score Threshold (≥ 0.8)

config_2 = {
    "configurable": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {
            "score_threshold": 0.8,  # Only return documents with a similarity score of 0.8 or higher
        },
    }
}

# Execute the query
docs = retriever.invoke("What is Word2Vec?", config=config_2)

# Display the search results
print("\n🔎 [Search Results - Similarity Score Threshold ≥ 0.8]")
for idx, doc in enumerate(docs):
    print(f"📄 [Document {idx + 1}]")
    print(doc.page_content)
    print("=" * 60)
```

<pre class="custom">
    🔎 [Search Results - Similarity Score Threshold ≥ 0.8]
    📄 [Document 1]
    Word2Vec
    
    Definition: Word2Vec is a technique in NLP that maps words into a vector space, representing their semantic relationships based on context.
    Example: In Word2Vec, "king" and "queen" would be represented by vectors close to each other.
    Related Keywords: NLP, Embeddings, Semantic Similarity
    ============================================================
</pre>

```python
# ✅ Search Configuration 3: MMR Search (Diversity and Relevance Balanced)

config_3 = {
    "configurable": {
        "search_type": "mmr",
        "search_kwargs": {
            "k": 2,            # Return the top 2 most diverse and relevant documents
            "fetch_k": 10,     # Initially fetch the top 10 documents before filtering for diversity
            "lambda_mult": 0.6 # Balance factor: 0.6 (0 = maximum diversity, 1 = maximum relevance)
        },
    }
}
# Execute the query using MMR search
docs = retriever.invoke("What is Word2Vec?", config=config_3)

#  Display the search results
print("\n🔎 [Search Results - MMR (Diversity and Relevance Balanced)]")
for idx, doc in enumerate(docs):
    print(f"📄 [Document {idx + 1}]")
    print(doc.page_content)
    print("=" * 60)
```

<pre class="custom">
    🔎 [Search Results - MMR (Diversity and Relevance Balanced)]
    📄 [Document 1]
    Word2Vec
    
    Definition: Word2Vec is a technique in NLP that maps words into a vector space, representing their semantic relationships based on context.
    Example: In Word2Vec, "king" and "queen" would be represented by vectors close to each other.
    Related Keywords: NLP, Embeddings, Semantic Similarity
    ============================================================
    📄 [Document 2]
    Tokenizer
    ============================================================
</pre>

## Using Separate Query & Passage Embedding Models

By default, a retriever uses the **same embedding model** for both queries and documents. However, certain scenarios can benefit from using different models tailored to the specific needs of queries and documents.

### Why Use Separate Embedding Models?
Using different models for queries and documents can improve retrieval accuracy and search relevance by optimizing each model for its intended purpose:
- Query Embedding Model: Fine-tuned for understanding short and concise search queries.
- Document (Passage) Embedding Model: Optimized for longer text spans with richer context.
  
For instance, **Upstage Embeddings** provides the capability to use distinct models for:  
- Query Embeddings (```solar-embedding-1-large-query```)  
- Document (Passage) Embeddings (```solar-embedding-1-large-passage```)  

In such cases, the query is embedded using the query embedding model, while the documents are embedded using the document embedding model. 

✅ **How to Issue an Upstage API Key**  
- Sign Up & Log In: 
   - Visit [Upstage](https://upstage.ai/) and log in (sign up if you don't have an account).  

- Open API Key Page:
   - Go to the menu bar, select "Dashboards", then navigate to "API Keys".

- Generate API Key:  
   - Click **"Create new key"** → Enter name your key (e.g., ```LangChain-Tutorial```) 

- Copy & Store Safely:  
   - Copy the generated key and keep it secure.  

<img src="./assets/01-vectorstore-retriever-get-upstage-api-key.png" alt="Description" width="1000">


```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_upstage import UpstageEmbeddings

# ✅ 1. Data Loading and Document Splitting
loader = TextLoader("./data/01-vectorstore-retriever-appendix-keywords.txt", encoding="utf-8")
documents = loader.load()

# Split the loaded documents into text chunks 
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# ✅ 2. Document Embedding
doc_embedder = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# ✅ 3. Create a Vector Database
db = FAISS.from_documents(split_docs, doc_embedder)
```

<pre class="custom">Created a chunk of size 351, which is longer than the specified 300
    Created a chunk of size 343, which is longer than the specified 300
    Created a chunk of size 307, which is longer than the specified 300
    Created a chunk of size 316, which is longer than the specified 300
    Created a chunk of size 341, which is longer than the specified 300
    Created a chunk of size 321, which is longer than the specified 300
    Created a chunk of size 303, which is longer than the specified 300
    Created a chunk of size 325, which is longer than the specified 300
    Created a chunk of size 315, which is longer than the specified 300
    Created a chunk of size 304, which is longer than the specified 300
    Created a chunk of size 385, which is longer than the specified 300
    Created a chunk of size 349, which is longer than the specified 300
    Created a chunk of size 376, which is longer than the specified 300
</pre>

The following example demonstrates the process of generating an Upstage embedding for a query, converting the query sentence into a vector, and conducting a vector similarity search.

```python
# ✅ 3. Query Embedding and Vector Search
query_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")

# Convert the query into a vector using the query embedding model
query_vector = query_embedder.embed_query("What is an embedding?")

# ✅ 4. Vector Similarity Search (Return Top 2 Documents)
results = db.similarity_search_by_vector(query_vector, k=2)

# ✅ 5. Display the Search Results
print(f"\n🔎 [Query]: What is an embedding?\n")
for idx, doc in enumerate(results):
    print(f"📄 [Document {idx + 1}]")
    print("📖 Document Content:", doc.page_content)
    print("🗂️ Metadata:", doc.metadata)
    print("=" * 60)
```

<pre class="custom">
    🔎 [Query]: What is an embedding?
    
    📄 [Document 1]
    📖 Document Content: Embedding
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
    📄 [Document 2]
    📖 Document Content: Definition: Embedding is the process of converting text data, such as words or sentences, into continuous low-dimensional vectors. This allows computers to understand and process text.
    Example: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].
    Related Keywords: Natural Language Processing, Vectorization, Deep Learning
    🗂️ Metadata: {'source': './data/01-vectorstore-retriever-appendix-keywords.txt'}
    ============================================================
</pre>

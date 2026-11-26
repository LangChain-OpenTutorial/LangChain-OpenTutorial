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

# Parent Document Retriever

- Author: [Yun Eun](https://github.com/yuneun92)
- Peer Review:
- Proofread : [jishin86](https://github.com/jishin86)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/05-ParentDocumentRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/05-ParentDocumentRetriever.ipynb)
## Overview

This tutorial focuses on the ```ParentDocumentRetriever``` implementation, a tool designed to balance document search and chunking.

When splitting documents for search, two competing needs arise:

1. **Small Chunks** : Needed for accurate meaning representation in embeddings
2. **Context Preservation** : Required for maintaining document coherence

> How It Works

```ParentDocumentRetriever``` manages this balance by:

1. Splitting documents into small searchable chunks
2. Maintaining connections to parent documents via IDs
3. Loading multiple files through ```TextLoader``` objects

> Benefits

1. **Efficient Search:** Quick identification of relevant content
2. **Context Awareness:** Access to broader document context when needed
3. **Flexible Structure:** Works with both complete documents and larger chunks as parent documents

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Full Document Retrieval](#full-document-retrieval)
- [Adjusting Larger Chunk Sizes](#adjusting-larger-chunk-sizes)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial langchain_chroma
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_community",
        "langchain_openai",
        "chromadb",
    ],
    verbose=False,
    upgrade=False,
)

from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
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
        "LANGCHAIN_PROJECT": "Parent-Document-Retriever",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



First, let's load the documents that we'll use as data.

```python
loaders = [
    # load file (It could be multiple files)
    TextLoader("./data/appendix-keywords.txt"),
]
# If your os is window, execute the following line 
# loader = TextLoader("./data/appendix-keywords.txt", encoding="utf-8")

docs = []
for loader in loaders:
    # Load the document using the loader and add it to the docs list.
    docs.extend(loader.load())

```

```python
docs
```




<pre class="custom">[Document(metadata={'source': './data/appendix-keywords.txt'}, page_content='Semantic Search\n\nDefinition: Semantic search refers to a search method that understands the meaning behind user queries, going beyond simple keyword matching to return relevant results.  \nExample: When a user searches for "solar system planets," the search returns information about related planets like Jupiter and Mars.  \nRelated Keywords: Natural Language Processing, Search Algorithms, Data Mining  \n\n\nEmbedding\n\nDefinition: Embedding is the process of converting textual data, such as words or sentences, into low-dimensional continuous vectors. This enables computers to understand and process text.  \nExample: The word "apple" can be represented as a vector like [0.65, -0.23, 0.17].  \nRelated Keywords: Natural Language Processing, Vectorization, Deep Learning  \n\n\nToken\n\nDefinition: A token refers to smaller units of text obtained by breaking it into parts, such as words, sentences, or phrases.  \nExample: The sentence "I go to school" can be split into tokens like "I," "go," "to," and "school."  \nRelated Keywords: Tokenization, Natural Language Processing, Syntax Analysis  \n\n\nTokenizer\n\nDefinition: A tokenizer is a tool used to divide text data into tokens, often as part of preprocessing in natural language processing.  \nExample: The sentence "I love programming." is split into ["I", "love", "programming", "."].  \nRelated Keywords: Tokenization, Natural Language Processing, Syntax Analysis  \n\n\nVectorStore\n\nDefinition: A vector store is a system that stores data transformed into vector formats. It is used in tasks like search, classification, and data analysis.  \nExample: Word embedding vectors are stored in a database for quick access.  \nRelated Keywords: Embedding, Database, Vectorization  \n\n\nSQL\n\nDefinition: SQL (Structured Query Language) is a programming language used to manage data in databases. It supports tasks like querying, modifying, inserting, and deleting data.  \nExample: `SELECT * FROM users WHERE age > 18;` retrieves information about users over 18 years old.  \nRelated Keywords: Database, Query, Data Management  \n\n\nCSV\n\nDefinition: CSV (Comma-Separated Values) is a file format used to store data, where each value is separated by a comma. It is often used for saving and exchanging tabular data.  \nExample: A CSV file with headers "Name, Age, Job" could contain data like "John, 30, Developer."  \nRelated Keywords: Data Format, File Handling, Data Exchange  \n\n\nJSON\n\nDefinition: JSON (JavaScript Object Notation) is a lightweight data interchange format that uses text to represent data objects in a readable manner for both humans and machines.  \nExample: `{"Name": "John", "Age": 30, "Job": "Developer"}` is an example of JSON-formatted data.  \nRelated Keywords: Data Exchange, Web Development, API  \n\n\nTransformer\n\nDefinition: A transformer is a type of deep learning model used in natural language processing for tasks like translation, summarization, and text generation. It is based on the attention mechanism.  \nExample: Google Translate uses transformer models for multilingual translations.  \nRelated Keywords: Deep Learning, Natural Language Processing, Attention  \n\n\nHuggingFace\n\nDefinition: HuggingFace is a library that provides pre-trained models and tools for natural language processing, making it easier for researchers and developers to perform NLP tasks.  \nExample: Using HuggingFace\'s Transformers library to perform sentiment analysis or text generation.  \nRelated Keywords: Natural Language Processing, Deep Learning, Library  \n\n\nDigital Transformation\n\nDefinition: Digital transformation refers to the process of leveraging technology to innovate services, culture, and operations within a business, improving business models and competitiveness through digital technologies.  \nExample: A company adopting cloud computing to revolutionize data storage and processing.  \nRelated Keywords: Innovation, Technology, Business Model  \n\n\nCrawling\n\nDefinition: Crawling is the automated process of visiting web pages to collect data. It is commonly used for search engine optimization and data analysis.  \nExample: Google’s search engine crawls websites to collect and index content.  \nRelated Keywords: Data Collection, Web Scraping, Search Engine  \n\n\nWord2Vec\n\nDefinition: Word2Vec is a natural language processing technique that maps words to vector spaces, representing semantic relationships between words.  \nExample: In a Word2Vec model, "king" and "queen" are represented as vectors close to each other in the vector space.  \nRelated Keywords: Natural Language Processing, Embedding, Semantic Similarity  \n\n\nLLM (Large Language Model)\n\nDefinition: An LLM is a large-scale language model trained on massive text datasets, used for various natural language understanding and generation tasks.  \nExample: OpenAI\'s GPT series is a prominent example of LLMs.  \nRelated Keywords: Natural Language Processing, Deep Learning, Text Generation  \n\n\nFAISS (Facebook AI Similarity Search)\n\nDefinition: FAISS is a high-speed similarity search library developed by Facebook, designed for efficiently searching large sets of vectors for similar ones.  \nExample: Using FAISS to quickly find similar images in a dataset containing millions of image vectors.  \nRelated Keywords: Vector Search, Machine Learning, Database Optimization  \n\n\nOpen Source\n\nDefinition: Open source refers to software where the source code is publicly available for anyone to use, modify, and distribute, promoting collaboration and innovation.  \nExample: The Linux operating system is a widely known open-source project.  \nRelated Keywords: Software Development, Community, Technology Collaboration  \n\n\nStructured Data\nDefinition: Organized data following a predefined format or schema, easily searchable and analyzable in databases and spreadsheets.\nExample: Customer information table stored in a relational database.\nRelated Keywords: database, data analysis, data modeling\n\n\nParser\nDefinition: A tool that analyzes given data (strings, files, etc.) and converts it into a structured format, used in programming language syntax analysis and file data processing.\nExample: Parsing HTML documents to create DOM structure of web pages.\nRelated Keywords: syntax analysis, compiler, data processing\n\n\nTF-IDF (Term Frequency-Inverse Document Frequency)\nDefinition: A statistical measure used to evaluate word importance in documents, considering both word frequency in a document and its rarity across all documents.\nExample: Words that appear infrequently across many documents have high TF-IDF values.\nRelated Keywords: natural language processing, information retrieval, data mining\n\n\nDeep Learning\nDefinition: A machine learning field using artificial neural networks to solve complex problems, focusing on learning high-level data representations.\nExample: Used in image recognition, speech recognition, and natural language processing.\nRelated Keywords: artificial neural networks, machine learning, data analysis\n\n\nSchema\nDefinition: Blueprint defining database or file structure, specifying how data is stored and organized.\nExample: Relational database table schemas define column names, data types, and key constraints.\nRelated Keywords: database, data modeling, data management\n\n\nDataFrame\nDefinition: Table-like data structure with rows and columns, primarily used for data analysis and processing.\nExample: Pandas library DataFrames can contain various data types and facilitate data manipulation and analysis.\nRelated Keywords: data analysis, pandas, data processing\n\n\nAttention Mechanism\nDefinition: A deep learning technique that enables models to focus more on important information, primarily used with sequence data.\nExample: Translation models use attention to focus on important parts of input sentences for accurate translation.\nRelated Keywords: deep learning, natural language processing, sequence modeling\n\n\nPandas\nDefinition: A Python library providing tools for data analysis and manipulation, enabling efficient data analysis tasks.\nExample: Used to read CSV files, clean data, and perform various analyses.\nRelated Keywords: data analysis, Python, data processing\n\n\nGPT (Generative Pretrained Transformer)\nDefinition: A pre-trained generative language model for various text-based tasks, capable of generating natural language based on input text.\nExample: Chatbots generating detailed responses to user questions using GPT models.\nRelated Keywords: natural language processing, text generation, deep learning\n\n\nInstructGPT\nDefinition: An optimized GPT model designed to perform specific tasks based on user instructions, producing more accurate and relevant results.\nExample: Creating email drafts based on user instructions.\nRelated Keywords: artificial intelligence, natural language understanding, instruction-based processing\n\n\nKeyword Search\nDefinition: Process of finding information based on user-input keywords, fundamental to most search engines and database systems.\nExample: Searching "coffee shop Seoul" returns relevant coffee shop listings.\nRelated Keywords: search engine, data search, information retrieval\n\n\nPage Rank\nDefinition: Algorithm evaluating webpage importance by analyzing web page link structures, used in search engine result ranking.\nExample: Google search engine uses PageRank to determine search result rankings.\nRelated Keywords: search engine optimization, web analytics, link analysis\n\n\nData Mining\nDefinition: Process of discovering useful information from large datasets using statistics, machine learning, and pattern recognition.\nExample: Retail stores analyzing customer purchase data to develop sales strategies.\nRelated Keywords: big data, pattern recognition, predictive analytics\n\n\nMultimodal\nDefinition: Technology processing multiple data modes (text, images, sound, etc.) to extract or predict more accurate information through interaction between different data formats.\nExample: Systems analyzing both images and descriptive text for more accurate image classification.\nRelated Keywords: data fusion, artificial intelligence, deep learning')]</pre>



## Full Document Retrieval

In this mode, we aim to search through complete documents. Therefore, we'll only specify the ```child_splitter```.

Later, we'll also specify the ```parent_splitter``` to compare the results.

```python
# Define Child Splitter with chunk size
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Create a Chroma DB collection -- in memory version
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

# Create Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
```

Documents are added using the ```retriever.add_documents(docs, ids=None)``` function:
* If ```ids``` is ```None```, they will be automatically generated.
* Setting ```add_to_docstore=False``` prevents duplicate document additions. However, ```ids``` values are required to check for duplicates.

```python
# Add documents to the retriever. 'docs' is a list of documents, and 'ids' is a list of unique document identifiers.
retriever.add_documents(docs, ids=None, add_to_docstore=True)
```

This code should return two keys because we added two documents.

- Convert the keys returned by the ```store``` object's ```yield_keys()``` method into a list.

```python
# Return all keys from the store as a list.
list(store.yield_keys())
```




<pre class="custom">['739c2480-1aac-4090-ae21-ceebc416d099']</pre>



Let's try calling the vector store search function.

Since we are storing small chunks, we should see small chunks returned in the search results.

Perform similarity search using the ```similarity_search``` method of the vectorstore object.

```python
# Perform similarity search
sub_docs = vectorstore.similarity_search("Word2Vec")

# Print the page_content property of the first element in the sub_docs list.
print(sub_docs[0].page_content)
```

<pre class="custom">Word2Vec
</pre>

Now let's search through the entire retriever. In this process, since it **returns the documents** containing the small chunks, relatively larger documents will be returned.

Use the ```invoke()``` method of the ```retriever``` object to retrieve documents related to the query.

```python
# Retrieve and fetch documents
retrieved_docs = retriever.invoke("Word2Vec")
```

```python
# Print the length of the page content of the retrieved document
print(
    f"Document length: {len(retrieved_docs[0].page_content)}",
    end="\n\n=====================\n\n",
)

# Print a portion of the document
print(retrieved_docs[0].page_content[2000:2500])
```

<pre class="custom">Document length: 10044
    
    =====================
    
     old.  
    Related Keywords: Database, Query, Data Management  
    
    
    CSV
    
    Definition: CSV (Comma-Separated Values) is a file format used to store data, where each value is separated by a comma. It is often used for saving and exchanging tabular data.  
    Example: A CSV file with headers "Name, Age, Job" could contain data like "John, 30, Developer."  
    Related Keywords: Data Format, File Handling, Data Exchange  
    
    
    JSON
    
    Definition: JSON (JavaScript Object Notation) is a lightweight data interchange form
</pre>

## Adjusting Larger Chunk Sizes

Like the previous results, **the entire document may be too large to search through as is** .

In this case, what we actually want to do is first split the raw document into larger chunks, and then split those into smaller chunks.

Then we index the small chunks, but search for larger chunks during retrieval (though still not the entire document).

- Use ```RecursiveCharacterTextSplitter``` to create parent and child documents.

    - Parent documents have ```chunk_size``` set to 1000.
    - Child documents have ```chunk_size``` set to 200, creating smaller sizes than the parent documents.




```python
# Text splitter used to generate parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Text splitter used to generate child documents
# Should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Vector store to be used for indexing child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings()
)
# Storage layer for parent documents
store = InMemoryStore()
```

This is the code to initialize ```ParentDocumentRetriever```:
* The ```vectorstore``` parameter specifies the vector store that stores document vectors.
* The ```docstore``` parameter specifies the document store that stores document data.
* The ```child_splitter``` parameter specifies the document splitter used to split child documents.
* The ```parent_splitter``` parameter specifies the document splitter used to split parent documents.

```ParentDocumentRetriever``` handles hierarchical document structures, separately splitting and storing parent and child documents. This allows effective use of both parent and child documents during retrieval.

```python
retriever = ParentDocumentRetriever(
    # Specify the vector store
    vectorstore=vectorstore,
    # Specify the document store
    docstore=store,
    # Specify the child document splitter
    child_splitter=child_splitter,
    # Specify the parent document splitter
    parent_splitter=parent_splitter,
)
```

Add docs to the ```retriever``` object. This adds new documents to the set of documents that ```retriever``` can search through.

```python
# Add documents to the retriever
retriever.add_documents(docs)
```

Now you can see there are many more documents. These are the larger chunks.

```python
# Generate keys from the store, convert to list, and return the length
len(list(store.yield_keys()))
```




<pre class="custom">12</pre>



```python
# Perform similarity search
sub_docs = vectorstore.similarity_search("Word2Vec")
# Print the page_content property of the first element in the sub_docs list
print(sub_docs[0].page_content)
```

<pre class="custom">Word2Vec
</pre>

Now let's use the ```invoke()``` method of the ```retriever``` object to search for documents.

```python
# Retrieve and fetch documents
retrieved_docs = retriever.invoke("Word2Vec")

# Return the length of the page content of the first retrieved document
print(retrieved_docs[0].page_content)
```

<pre class="custom">Crawling
    
    Definition: Crawling is the automated process of visiting web pages to collect data. It is commonly used for search engine optimization and data analysis.  
    Example: Google’s search engine crawls websites to collect and index content.  
    Related Keywords: Data Collection, Web Scraping, Search Engine  
    
    
    Word2Vec
    
    Definition: Word2Vec is a natural language processing technique that maps words to vector spaces, representing semantic relationships between words.  
    Example: In a Word2Vec model, "king" and "queen" are represented as vectors close to each other in the vector space.  
    Related Keywords: Natural Language Processing, Embedding, Semantic Similarity  
    
    
    LLM (Large Language Model)
</pre>

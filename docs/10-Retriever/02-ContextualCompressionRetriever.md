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

# Contextual Compression Retriever

- Author: [JoonHo Kim](https://github.com/jhboyo)
- Peer Review :
- Proofread : [jishin86](https://github.com/jishin86)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/02-ContextualCompressionRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/02-ContextualCompressionRetriever.ipynb)

## Overview

The ```ContextualCompressionRetriever``` in LangChain is a powerful tool designed to optimize the retrieval process by compressing retrieved documents based on context. This retriever is particularly useful in scenarios where large amounts of data need to be summarized or filtered dynamically, ensuring that only the most relevant information is passed to subsequent processing steps.

Key features of the ContextualCompressionRetriever include:

- Context-Aware Compression: Documents are compressed based on the specific context or query, ensuring relevance and reducing redundancy.
- Flexible Integration: Works seamlessly with other LangChain components, making it easy to integrate into existing pipelines.
- Customizable Compression: Allows for the use of different compression techniques, including summary models and embedding-based methods, to tailor the retrieval process to your needs.

The ```ContextualCompressionRetriever``` is particularly suited for applications like:

- Summarizing large datasets for Q&A systems.
- Enhancing chatbot performance by providing concise and relevant responses.
- Improving efficiency in document-heavy tasks like legal analysis or academic research.

By using this retriever, developers can significantly reduce computational overhead and improve the quality of information presented to end-users.

![](./img/02-contextual-compression-retriever-workflow.png)  


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Basic Retriever Configuration](#basic-retriever-configuration)
- [Contextual Compression](#contextual-compression)
- [Document Filtering Using LLM](#document-filtering-using-llm)
- [Creating a Pipeline (Compressor + Document Converter)](#creating-a-pipeline-compressor--document-converter)


### References

- [How to do retrieval with contextual compression](https://python.langchain.com/docs/how_to/contextual_compression/)
- [LLM ChainFilter](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.document_compressors.chain_filter.LLMChainFilter.html)

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
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_core",
        "faiss-cpu",
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
        "LANGCHAIN_PROJECT": "Contextual Compression Retriever",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



The following function is used to display documents in a visually appealing format.


```python
# Helper function to print documents in a pretty format
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

## Basic Retriever Configuration

Let's start by initializing a simple vector store retriever and saving text documents in chunks.
When a sample question is asked, you can see that the retriever returns 1 to 2 relevant documents along with a few irrelevant ones.

We will follow the following steps to generate a retriever.
1. Generate Loader to load text file using TextLoader
2. Generate text chunks using CharacterTextSplitter and split the text into chunks of 300 characters with no overlap.
3. Generate vector store using FAISS and convert it to retriever
4. Query the retriever to find relevant documents
5. Print the relevant documents


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 1. Generate Loader to lthe text file using TextLoader
loader = TextLoader("./data/appendix-keywords.txt")\

# 2. Generate text chunks using CharacterTextSplitter and split the text into chunks of 300 characters with no overlap.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = loader.load_and_split(text_splitter)

# 3. Generate vector store using FAISS and convert it to retriever
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

# 4. Query the retriever to find relevant documents
docs = retriever.invoke("What is the definition of Multimodal?")

# 5. Print the relevant documents
pretty_print_docs(docs)
```

<pre class="custom">Created a chunk of size 380, which is longer than the specified 300
    Created a chunk of size 343, which is longer than the specified 300
    Created a chunk of size 304, which is longer than the specified 300
    Created a chunk of size 341, which is longer than the specified 300
    Created a chunk of size 349, which is longer than the specified 300
    Created a chunk of size 330, which is longer than the specified 300
    Created a chunk of size 385, which is longer than the specified 300
    Created a chunk of size 349, which is longer than the specified 300
    Created a chunk of size 413, which is longer than the specified 300
    Created a chunk of size 310, which is longer than the specified 300
    Created a chunk of size 391, which is longer than the specified 300
    Created a chunk of size 330, which is longer than the specified 300
    Created a chunk of size 325, which is longer than the specified 300
    Created a chunk of size 349, which is longer than the specified 300
    Created a chunk of size 321, which is longer than the specified 300
    Created a chunk of size 361, which is longer than the specified 300
    Created a chunk of size 437, which is longer than the specified 300
    Created a chunk of size 374, which is longer than the specified 300
    Created a chunk of size 324, which is longer than the specified 300
    Created a chunk of size 412, which is longer than the specified 300
    Created a chunk of size 346, which is longer than the specified 300
    Created a chunk of size 403, which is longer than the specified 300
    Created a chunk of size 331, which is longer than the specified 300
    Created a chunk of size 344, which is longer than the specified 300
    Created a chunk of size 350, which is longer than the specified 300
</pre>

    document 1:
    
    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
    Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
    Relate
    ----------------------------------------------------------------------------------------------------
    document 2:
    
    Semantic Search
    ----------------------------------------------------------------------------------------------------
    document 3:
    
    LLM (Large Language Model)
    ----------------------------------------------------------------------------------------------------
    document 4:
    
    Embedding
    

```python
print(docs[0].page_content)
```

<pre class="custom">Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
    Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
    Relate
</pre>

## Contextual Compression 

The ```DocumentCompressor``` created using ```LLMChainExtractor``` is exactly what is applied to the retriever, which is the ```ContextualCompressionRetriever```.

```ContextualCompressionRetriever``` will compress the documents by removing irrelevant information and focusing on the most relevant information.

Let's see how the retriever works before and after applying ```ContextualCompressionRetriever```.


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Before applying ContextualCompressionRetriever
pretty_print_docs(retriever.invoke("What is the definition of Multimodal?"))
print("="*62)
print("="*15 + "After applying LLMChainExtractor" + "="*15)


# After applying ContextualCompressionRetriever
# 1. Generate LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  

# 2. Generate compressor using LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)

# 3. Generate compression retriever using ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

# 4. Query the compression retriever to find relevant documents
compressed_docs = (
    compression_retriever.invoke( 
        "What is the definition of Multimodal?"
    )
)

# 5. Print the relevant documents
pretty_print_docs(compressed_docs)
```

<pre class="custom">document 1:
    
    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
    Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
    Relate
    ----------------------------------------------------------------------------------------------------
    document 2:
    
    Semantic Search
    ----------------------------------------------------------------------------------------------------
    document 3:
    
    LLM (Large Language Model)
    ----------------------------------------------------------------------------------------------------
    document 4:
    
    Embedding
    ==============================================================
    ===============After applying LLMChainExtractor===============
    document 1:
    
    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
</pre>

## Document Filtering Using LLM


### LLMChainFilter

```LLMChainFilter``` is a simpler yet powerful compressor that uses an LLM chain to decide which documents to filter and which to return from the initially retrieved documents. 

This filter selectively returns documents without altering (compressing) their content.

```python
from langchain.retrievers.document_compressors import LLMChainFilter

# 1. Generate LLMChainFilter object using LLM
_filter = LLMChainFilter.from_llm(llm)

# 2. Generate ContextualCompressionRetriever object using LLMChainFilter and retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=retriever,
)

# 3. Query the compression retriever to find relevant documents
compressed_docs = compression_retriever.invoke(
    "What is the definition of Multimodal?"
)

# 4. Print the relevant documents
pretty_print_docs(compressed_docs)  
```

<pre class="custom">document 1:
    
    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
    Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
    Relate
</pre>

### EmbeddingsFilter

Performing additional LLM calls for each retrieved document is costly and slow. 
The ```EmbeddingsFilter``` provides a more affordable and faster option by embedding both the documents and the query, returning only those documents with embeddings that are sufficiently similar to the query. 

This allows for maintaining the relevance of search results while saving on computational costs and time.
The process involves compressing and retrieving relevant documents using ```EmbeddingsFilter``` and ```ContextualCompressionRetriever```. 

- The ```EmbeddingsFilter``` is used to filter documents that exceed a specified similarity threshold (0.86).

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

# 1. Generate embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 2. Generate EmbedingsFilter object that has similarity threshold of 0.86
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86)

# 3. Generate ContextualCompressionRetriever object using EmbeddingsFilter and retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, 
    base_retriever=retriever
)

# 4. Query the compression retriever to find relevant documents
compressed_docs = compression_retriever.invoke(
    "What is the definition of Multimodal?"
)

# 5. Print the relevant documents
pretty_print_docs(compressed_docs)
```

<pre class="custom">document 1:
    
    Multimodal
    Definition: Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
    Example: A system that analyzes both images and descriptive text to perform more accurate image classification is an example of multimodal technology.
    Relate
</pre>

## Creating a Pipeline (Compressor + Document Converter)

Using ```DocumentCompressorPipeline```, multiple compressors can be sequentially combined.

You can add ```BaseDocumentTransformer``` to the pipeline along with the Compressor, which performs transformations on the document set without performing contextual compression.

For example, ```TextSplitter``` can be used as a document transformer to split documents into smaller pieces, while ```EmbeddingsRedundantFilter``` can be used to filter out duplicate documents based on the embedding similarity between documents (by default, considering documents with a similarity of 0.95 or higher as duplicates).

Below, we first split the documents into smaller chunks, then remove duplicate documents, and filter based on relevance to the query to create a compressor pipeline."


```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

# 1. Generate CharacterTextSplitter object that has chunk size of 300 and chunk overlap of 0
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# 2. Generate EmbeddingsRedundantFilter object using embeddings
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 3. Generate EmbeddingsFilter object that has similarity threshold of 0.86
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86)

# 4. Generate DocumentCompressorPipeline object using splitter, redundant_filter, relevant_filter, and LLMChainExtractor
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        splitter,
        redundant_filter,
        relevant_filter,
        LLMChainExtractor.from_llm(llm),
    ]
)
```

While initializing the  ```ContextualCompressionRetriever```, we use ```pipeline_compressor``` as the ```base_compressor``` and ```retriever``` as the ```base_retriever```.

```python
# 5. Use pipeline_compressor as the base_compressor and retriever as the base_retriever to initialize ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)

# 6. Query the compression retriever to find relevant documents
compressed_docs = compression_retriever.invoke(
    "What is the definition of Multimodal?"
)

# 7. Print the relevant documents
pretty_print_docs(compressed_docs)

```

<pre class="custom">document 1:
    
    Multimodal refers to the technology that combines multiple types of data modes (e.g., text, images, sound) to process and extract richer and more accurate information or predictions.
</pre>

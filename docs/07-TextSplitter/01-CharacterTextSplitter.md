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

# Character Text Splitter

- Author: [hellohotkey](https://github.com/hellohotkey)
- Peer Review : [fastjw](https://github.com/fastjw), [heewung song](https://github.com/kofsitho87)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/01-CharacterTextSplitter.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/01-CharacterTextSplitter.ipynb)
## Overview

Text splitting is a crucial step in document processing with LangChain. 

The ```CharacterTextSplitter``` offers efficient text chunking that provides several key benefits:

- **Token Limits:** Overcomes LLM context window size restrictions
- **Search Optimization:** Enables more precise chunk-level retrieval
- **Memory Efficiency:** Processes large documents effectively
- **Context Preservation:** Maintains textual coherence through ```chunk_overlap```

This tutorial explores practical implementation of text splitting through core methods like ```split_text()``` and ```create_documents()```, including advanced features such as metadata handling.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [CharacterTextSplitter Example](#charactertextsplitter-example)


### References

- [LangChain TextSplitter](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TextSplitter.html)
- [LangChain CharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.CharacterTextSplitter.html)
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
        "langchain_text_splitters",
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
        "LANGCHAIN_PROJECT": "Adaptive-RAG",  # title 과 동일하게 설정해 주세요
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




<pre class="custom">False</pre>



## CharacterTextSplitter Example

Read and store contents from keywords file
* Open ```./data/appendix-keywords.txt``` file and read its contents.
* Store the read contents in the ```file``` variable

```python
with open("./data/appendix-keywords.txt", encoding="utf-8") as f:
   file = f.read()
```

Print the first 500 characters of the file contents.

```python
print(file[:500])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to unders
</pre>

Create ```CharacterTextSplitter``` with parameters:

**Parameters**

* ```separator```: String to split text on (e.g., newlines, spaces, custom delimiters)
* ```chunk_size```: Maximum size of chunks to return
* ```chunk_overlap```: Overlap in characters between chunks
* ```length_function```: Function that measures the length of given chunks
* ```is_separator_regex```: Boolean indicating whether separator should be treated as a regex pattern

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
   separator=" ",           # Splits whenever a space is encountered in text
   chunk_size=250,          # Each chunk contains maximum 250 characters
   chunk_overlap=50,        # Two consecutive chunks share 50 characters
   length_function=len,     # Counts total characters in each chunk
   is_separator_regex=False # Uses space as literal separator, not as regex
)
```

Create document objects from chunks and display the first one

```python
chunks = text_splitter.create_documents([file])
print(chunks[0])
```

<pre class="custom">page_content='Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick'
</pre>

Demonstrate metadata handling during document creation:

* ```create_documents``` accepts both text data and metadata lists
* Each chunk inherits metadata from its source document

```python
# Define metadata for each document
metadatas = [
   {"document": 1},
   {"document": 2},
]

# Create documents with metadata
documents = text_splitter.create_documents(
   [file, file],  # List of texts to split
   metadatas=metadatas,  # Corresponding metadata
)

print(documents[0])  # Display first document with metadata
```

<pre class="custom">page_content='Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick' metadata={'document': 1}
</pre>

Split text using the ```split_text()``` method.
* ```text_splitter.split_text(file)[0]``` returns the first chunk of the split text

```python
# Split the file text and return the first chunk
text_splitter.split_text(file)[0]
```




<pre class="custom">'Semantic Search\n\nDefinition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.\nExample: Vectors of word embeddings can be stored in a database for quick'</pre>



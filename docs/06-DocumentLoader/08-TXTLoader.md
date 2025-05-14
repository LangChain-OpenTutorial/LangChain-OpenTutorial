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

# TXT Loader

- Author: [seofield](https://github.com/seofield)
- Peer Review : [Kane](https://github.com/HarryKane11), [Suhyun Lee](https://github.com/suhyun0115)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/08-TXTLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/08-TXTLoader.ipynb)
## Overview

This tutorial focuses on using LangChain’s ```TextLoader``` to efficiently load and process individual text files. 

You’ll learn how to extract metadata and content, making it easier to prepare text data.


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [TXT Loader](#txt-loader)
- [Automatic Encoding Detection with TextLoader](#automatic-encoding-detection-with-textloader)

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
        "langchain_community",
        "chardet"
    ],
    verbose=False,
    upgrade=False,
)
```

## TXT Loader

Let’s explore how to load files with the **.txt** extension using a loader.

```python
from langchain_community.document_loaders import TextLoader

# Create a text loader
loader = TextLoader("data/appendix-keywords.txt", encoding="utf-8")

# Load the document
docs = loader.load()
print(f"Number of documents: {len(docs)}\n")
print("[Metadata]\n")
print(docs[0].metadata)
print("\n========= [Preview - First 500 Characters] =========\n")
print(docs[0].page_content[:500])
```

<pre class="custom">Number of documents: 1
    
    [Metadata]
    
    {'source': 'data/appendix-keywords.txt'}
    
    ========= [Preview - First 500 Characters] =========
    
    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words
</pre>

## Automatic Encoding Detection with TextLoader

In this example, we explore several strategies for using the TextLoader class to efficiently load large batches of files from a directory with varying encodings.

To illustrate the problem, we’ll first attempt to load multiple text files with arbitrary encodings.

- ```silent_errors```: By passing the ```silent_errors``` parameter to the ```DirectoryLoader```, you can skip files that cannot be loaded and continue the loading process without interruptions.
- ```autodetect_encoding```: Additionally, you can enable automatic encoding detection by passing the ```autodetect_encoding``` parameter to the loader class, allowing it to detect file encodings before failing.


```python
from langchain_community.document_loaders import DirectoryLoader

path = "data/"

text_loader_kwargs = {"autodetect_encoding": True}

loader = DirectoryLoader(
    path,
    glob="**/*.txt",
    loader_cls=TextLoader,
    silent_errors=True,
    loader_kwargs=text_loader_kwargs,
)
docs = loader.load()
```

The **data/appendix-keywords.txt** file and its derivative files with similar names all have different encoding formats.


```python
doc_sources = [doc.metadata["source"] for doc in docs]
doc_sources
```




<pre class="custom">['data/appendix-keywords-CP949.txt',
     'data/appendix-keywords-EUCKR.txt',
     'data/appendix-keywords.txt',
     'data/appendix-keywords-utf8.txt']</pre>



```python
print("[Metadata]\n")
print(docs[0].metadata)
print("\n========= [Preview - First 500 Characters] =========\n")
print(docs[0].page_content[:500])
```

<pre class="custom">[Metadata]
    
    {'source': 'data/appendix-keywords-CP949.txt'}
    
    ========= [Preview - First 500 Characters] =========
    
    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user¡¯s query to return relevant results.
    Example: If a user searches for ¡°planets in the solar system,¡± the system might return information about related planets such as ¡°Jupiter¡± or ¡°Mars.¡±
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such a
</pre>

```python
print("[Metadata]\n")
print(docs[1].metadata)
print("\n========= [Preview - First 500 Characters] =========\n")
print(docs[1].page_content[:500])
```

<pre class="custom">[Metadata]
    
    {'source': 'data/appendix-keywords-EUCKR.txt'}
    
    ========= [Preview - First 500 Characters] =========
    
    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user¡¯s query to return relevant results.
    Example: If a user searches for ¡°planets in the solar system,¡± the system might return information about related planets such as ¡°Jupiter¡± or ¡°Mars.¡±
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such a
</pre>

```python
print("[Metadata]\n")
print(docs[3].metadata)
print("\n========= [Preview - First 500 Characters] =========\n")
print(docs[3].page_content[:500])
```

<pre class="custom">[Metadata]
    
    {'source': 'data/appendix-keywords-utf8.txt'}
    
    ========= [Preview - First 500 Characters] =========
    
    Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching by understanding the meaning of the user’s query to return relevant results.
    Example: If a user searches for “planets in the solar system,” the system might return information about related planets such as “Jupiter” or “Mars.”
    Related Keywords: Natural Language Processing, Search Algorithms, Data Mining
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words
</pre>

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

# 02. RecursiveCharacterTextSplitter

- Author: [fastjw](https://github.com/fastjw)
- Design: [fastjw](https://github.com/fastjw)
- Peer Review : [Wonyoung Lee](https://github.com/BaBetterB), [sohyunwriter](https://github.com/sohyunwriter)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb)

## Overview

This tutorial will show you how to use the `RecursiveCharacterTextSplitter`. 

This is the recommended way to split text.

It works by taking a list of characters as a parameter.

It tries to split the text into smaller pieces in the order of the given character list until the pieces are very small.

By default, the character lists are **['\n\n', '\n', ' ", "']**.

It recursively splits in the following order: **paragraph** -> **sentence** -> **word**.

This means that paragraphs (then sentences, then words) are considered to be the most semantically related pieces of text, so we want to keep them together as much as possible.

1. How the text is split: by a list of characters (**[‘\n\n’, ‘\n’, ‘ “, ”’]**).
2. The chunk size is measured by the number of characters.

### Table of Contents

- [Overview](#overview)
- [RecursiveCharacterTextSplitter Example](#recursivecharactertextsplitter-example)

### References

- [LangChain: Recursively split by character](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
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
        "LANGCHAIN_PROJECT": "RecursiveCharacterTextSplitter", 
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">False</pre>



## RecursiveCharacterTextSplitter Example

Read a file for the `RecursiveCharacterTextSplitter` exercise.

- Open the **appendix-keywords.txt** file and read its contents.
- Save the text to the **file** variable.

```python
# Open the appendix-keywords.txt file to create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # Reads the contents of the file and stores them in the file variable.
```

Output some of the contents of the file read from the file.

```python
# Output the top 500 characters read from the file.
print(file[:500])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to unders
</pre>

Example of using `RecursiveCharacterTextSplitter` to split text into small chunks.

- Set **chunk_size** to 250 to limit the size of each chunk.
- Set **chunk_overlap** to 50 to allow 50 characters of overlap between neighbouring chunks.
- Use the **len** function as **length_function** to calculate the length of the text.
- Set **is_separator_regex** to **False** to disable the use of regular expressions as separators.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set the chunk size to very small. These settings are for illustrative purposes only.
    chunk_size=250,
    # Sets the number of overlapping characters between chunks.
    chunk_overlap=50,
    # Specifies a function to calculate the length of the string.
    length_function=len,
    # Sets whether to use regular expressions as delimiters.
    is_separator_regex=False,
)
```

- Use **text_splitter** to split the **file** text into documents.
- The split documents are stored in the **texts** list.
- Print the first and second documents of the split document via **print(texts[0])** and **print(texts[1])**.

```python
# Split the file text into documents using text_splitter.
texts = text_splitter.create_documents([file])
print(texts[0])  # Outputs the first document in the split document.
print("===" * 20)
print(texts[1])  # Output the second document of the split document.
```

<pre class="custom">page_content='Semantic Search'
    ============================================================
    page_content='Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.'
</pre>

Use the `text_splitter.split_text()` function to split the **file** text.

```python
# Splits the text and returns the first two elements of the split text.
text_splitter.split_text(file)[:2]
```




<pre class="custom">['Semantic Search',
     'Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.\nExample: Vectors of word embeddings can be stored in a database for quick access.']</pre>


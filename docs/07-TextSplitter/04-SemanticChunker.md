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

# SemanticChunker

- Author: [Wonyoung Lee](https://github.com/BaBetterB)
- Design: []()
- Peer Review : [Wooseok Jeong](https://github.com/jeong-wooseok), [sohyunwriter](https://github.com/sohyunwriter)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)



## Overview

This tutorial covers a Text Splitter that splits text based on semantic similarity.

The **Semantic Chunker** is a sophisticated tool within **LangChain** that brings an intelligent approach to document chunking. Rather than simply dividing text at fixed intervals, it analyzes the semantic meaning of content to create more meaningful divisions. 

This process relies on **OpenAI's embedding model** , which evaluates how similar different pieces of text are to each other. The tool offers flexible splitting options, including percentile-based, standard deviation, and interquartile range methods. 

What sets it apart from traditional text splitters is its ability to maintain context by identifying natural break points in the text, ultimately leading to better performance when working with large language models. 

By understanding the actual meaning of the content, it creates more coherent and useful chunks that preserve the original document's context and flow.

 [Greg Kamradt's Notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

The method divides the text into sentence units, then groups them into three sentences, and merges similar sentences in the embedding space.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Creating a Semantic Chunker](#creating-a-semanticchunker)
- [Text Splitting](#text-splitting)
- [Breakpoints](#breakpoints)

### References

- [Greg Kamradt's Notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)


----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

Load sample text and output the content.

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
        "langchain-anthropic",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
        "langchain_experimental",
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
        "LANGCHAIN_PROJECT": "SemanticChunker",  # title
    }
)
```

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it.

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
# Configuration File for Managing API Keys as Environment Variables
from dotenv import load_dotenv

# Load API Key Information
load_dotenv(override=True)
```

```python
# Open the data/appendix-keywords.txt file to create a file object called f.
with open("./data/appendix-keywords.txt", encoding="utf-8") as f:

    file = f.read()  # Read the contents of the file and save it in the file variable.

# Print part of the content read from the file.
print(file[:350])
```

## Creating a SemanticChunker

`SemanticChunker` is one of LangChain's experimental features, which serves to divide text into semantically similar chunks.

This allows you to process and analyze text data more effectively.

Use `SemanticChunker` to divide the text into semantically related chunks.


```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize a semantic chunk splitter using OpenAI embeddings.
text_splitter = SemanticChunker(OpenAIEmbeddings())
```

## Text Splitting

- Use `text_splitter` to divide the `file` text into document units.

```python
chunks = text_splitter.split_text(file)
```

Check the divided chunks.

```python
# Print the first chunk among the divided chunks.
print(chunks[0])
```

You can convert chunks to documents using the `create_documents()` function.


```python
# Split using text_splitter
docs = text_splitter.create_documents([file])
print(
    docs[0].page_content
)  # Print the content of the first document among the divided documents.
```

## Breakpoints
This chunker works by determining when to "split" sentences. 

This is done by examining the embedding differences between two sentences.

If the difference exceeds a certain threshold, the sentences are split.

- Reference video: https://youtu.be/8OJC21T2SL4?si=PzUtNGYJ_KULq3-w&t=2580

### Percentile
The basic splitting method is based on `Percentile`.

In this method, all differences between sentences are calculated, then splitting is done based on the specified percentile.


```python
text_splitter = SemanticChunker(
    # Initialize the semantic chunker using OpenAI's embedding model
    OpenAIEmbeddings(),
    # Set the split breakpoint type to percentile
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

Check the split results.


```python
docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(
        doc.page_content
    )  # Print the content of the first document among the split documents.
    print("===" * 20)
```

Print the length of `docs`.

```python
print(len(docs))  # Print the length of docs.
```

### Standard Deviation

In this method, splitting occurs when there is a difference greater than the specified `breakpoint_threshold_amount` standard deviation.

- Set the `breakpoint_threshold_type` parameter to "standard_deviation" to specify chunk splitting criteria based on standard deviation.

```python
text_splitter = SemanticChunker(
    # Initialize the semantic chunker using OpenAI's embedding model.
    OpenAIEmbeddings(),
    # Use standard deviation as the splitting criterion.
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)
```

Check the split results.

```python
# Split using text_splitter.
docs = text_splitter.create_documents([file])
```

```python
docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(
        doc.page_content
    )  # Print the content of the first document among the split documents.
    print("===" * 20)
```

Print the length of `docs`.

```python
print(len(docs))  # Print the length of docs.
```

### Interquartile

This method uses interquartile range to split chunks.

- Set the `breakpoint_threshold_type` parameter to "interquartile" to specify chunk splitting criteria based on interquartile range.


```python
text_splitter = SemanticChunker(
    # Initialize the semantic chunk splitter using OpenAI's embedding model.
    OpenAIEmbeddings(),
    # Set the breakpoint threshold type to interquartile range.
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=0.5,
)
```

```python
# Split using text_splitter.
docs = text_splitter.create_documents([file])

# Print the results.
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(
        doc.page_content
    )  # Print the content of the first document among the split documents.
    print("===" * 20)
```

Print the length of `docs`.


```python
print(len(docs))  # Print the length of docs.
```

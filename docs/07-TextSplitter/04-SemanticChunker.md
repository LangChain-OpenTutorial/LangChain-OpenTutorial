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
- Peer Review : [Wooseok Jeong](https://github.com/jeong-wooseok), [sohyunwriter](https://github.com/sohyunwriter)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)



## Overview

This tutorial dives into a Text Splitter that uses semantic similarity to split text.

LangChain's ```SemanticChunker``` is a powerful tool that takes document chunking to a whole new level. Unlike traiditional methods that split text at fixed intervals, the ```SemanticChunker``` analyzes the meaning of the content to create more logical divisions.

This approach relies on **OpenAI's embedding model** , calculating how similar different pieces of text are by converting them into numerical representations. The tool offers various splitting options to suit your needs. You can choose from methods based on percentiles, standard deviation, or interquartile range.

What sets the ```SemanticChunker``` apart is its ability to preserve context by identifying natural breaks. This ultimately leads to better performance when working with large language models. 

Since the ```SemanticChunker``` understands the actual content, it generates chunks that are more useful and maintain the flow and context of the original document.

See [Greg Kamradt's notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)


The method breaks down the text into individual sentences first. Then, it groups sementically similar sentences into chunks (e.g., 3 sentences), and finally merges similar sentences in the embedding space.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Creating a Semantic Chunker](#creating-a-semanticchunker)
- [Text Splitting](#text-splitting)
- [Breakpoints](#breakpoints)

### References

- [Greg Kamradt's notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [Greg Kamradt's video](https://youtu.be/8OJC21T2SL4?si=PzUtNGYJ_KULq3-w&t=2580)

----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

Load sample text and output the content.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
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

Alternatively, you can set and load ```OPENAI_API_KEY``` from a ```.env``` file.

**[Note]** This is only necessary if you haven't already set ```OPENAI_API_KEY``` in previous steps.

```python
# Configuration File for Managing API Keys as Environment Variables
from dotenv import load_dotenv

# Load API Key Information
load_dotenv(override=True)
```

Load the sample text and output its content.

```python
# Open the data/appendix-keywords.txt file to create a file object called f.
with open("./data/appendix-keywords.txt", encoding="utf-8") as f:

    file = f.read()  # Read the contents of the file and save it in the file variable.

# Print part of the content read from the file.
print(file[:350])
```

## Creating a ```SemanticChunker```

The ```SemanticChunker``` is an experimental LangChain feature, that splits text into semantically similar chunks.

This approach allows for more effective processing and analysis of text data.

Use the ```SemanticChunker``` to divide the text into semantically related chunks.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize a semantic chunk splitter using OpenAI embeddings.
text_splitter = SemanticChunker(OpenAIEmbeddings())
```

## Text Splitting

Use the ```text_splitter``` with your loaded file (```file```) to split the text into smallar, more manageable unit documents. This process is often referred to as chunking.

```python
chunks = text_splitter.split_text(file)
```

After splitting, you can examine the resulting chunks to see how the text has been divided.

```python
# Print the first chunk among the divided chunks.
print(chunks[0])
```

The ```create_documents()``` function allows you to convert the individual chunks ([```file```]) into proper document objects (```docs```).


```python
# Split using text_splitter
docs = text_splitter.create_documents([file])
print(
    docs[0].page_content
)  # Print the content of the first document among the divided documents.
```

## Breakpoints

This chunking process works by indentifying natural breaks between sentences.

Here's how it decides where to split the text:
1. It calculates the difference between these embeddings for each pair of sentences.
2. When the difference between two sentences exceeds a certain threshold (breakpoint), the ```text_splitter``` identifies this as a natural break and splits the text at that point.

Check out [Greg Kamradt's video](https://youtu.be/8OJC21T2SL4?si=PzUtNGYJ_KULq3-w&t=2580) for more details.



### Percentile-Based Splitting

This method sorts all embedding differences between sentences. Then, it splits the text at a specific percentile (e.g. 70th percentile).

```python
text_splitter = SemanticChunker(
    # Initialize the semantic chunker using OpenAI's embedding model
    OpenAIEmbeddings(),
    # Set the split breakpoint type to percentile
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

Examine the resulting document list (```docs```).


```python
docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(
        doc.page_content
    )  # Print the content of the first document among the split documents.
    print("===" * 20)
```

Use the ```len(docs)``` function to get the number of chunks created.

```python
print(len(docs))  # Print the length of docs.
```

### Standard Deviation Splitting

This method sets a threshold based on a specified number of standard deviations (```breakpoint_threshold_amount```).

To use standard deviation for your breakpoints, set the ```breakpoint_threshold_type``` parameter to ```"standard_deviation"``` when initializing the ```text_splitter```.

```python
text_splitter = SemanticChunker(
    # Initialize the semantic chunker using OpenAI's embedding model.
    OpenAIEmbeddings(),
    # Use standard deviation as the splitting criterion.
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)
```

After splitting, check the ```docs``` list and print its length (```len(docs)```) to see how many chunks were created.

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

```python
print(len(docs))  # Print the length of docs.
```

### Interquartile Range Splitting

This method utilizes the interquartile range (IQR) of the embedding differences to consider breaks, leading to a text split.

Set the ```breakpoint_threshold_type``` parameter to ```"interquartile"``` when initializing the ```text_splitter``` to use the IQR for splitting.

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

Finally, print the length of ```docs``` list (```len(docs)```) to view the number of cunks created.


```python
print(len(docs))  # Print the length of docs.
```

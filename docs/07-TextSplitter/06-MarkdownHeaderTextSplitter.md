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

# MarkdownHeaderTextSplitter

- Author: [HeeWung Song(Dan)](https://github.com/kofsitho87)
- Peer Review : [BokyungisaGod](https://github.com/BokyungisaGod), [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb)
## Overview

This tutorial introduces how to effectively split **Markdown documents** using LangChain's ```MarkdownHeaderTextSplitter```. This tool divides documents into meaningful sections based on Markdown headers, preserving the document's structure for systematic content processing.

Context and structure of documents are crucial for effective text embedding. Simply dividing text isn't enough; maintaining semantic connections is key to generating more comprehensive vector representations. This is particularly true when dealing with large documents, as preserving context can significantly enhance the accuracy of subsequent analysis and search operations.

The ```MarkdownHeaderTextSplitter``` splits documents according to specified header sets, managing the content under each header group as separate chunks. This enables efficient content processing while maintaining the document's structural coherence.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Basic Usage of MarkdownHeaderTextSplitter](#basic-usage-of-markdownheadertextsplitter)
- [Combining with Other Text Splitters](#combining-with-other-text-splitters)

### References

- [Langchain MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)
- [Langchain RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#recursivecharactertextsplitter)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The ```langchain-opentutorial``` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
    ]
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
        "LANGCHAIN_PROJECT": "MarkdownHeaderTextSplitter",
    }
)
```

Alternatively, you can set and load ```OPENAI_API_KEY``` from a ```.env``` file.

**[Note]** This is only necessary if you haven't already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Basic Usage of MarkdownHeaderTextSplitter

The ```MarkdownHeaderTextSplitter``` splits Markdown-formatted text based on headers. Here's how to use it:

- First, the splitter divides the text based on standard Markdown headers (#, ##, ###, etc.).
- Store the Markdown you want to split in a variable called markdown_document.
- You'll need a list called ```headers_to_split_on```. This list uses tuples to define the header levels you want to split on and what you want to call them.
- Now, create a ```markdown_splitter``` object using the ```MarkdownHeaderTextSplitter``` class, and give it that ```headers_to_split_on``` list.
- To actually split the text, call the ```split_text``` method on your ```markdown_splitter``` object, passing in your ```markdown_document```.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Define a markdown document as a string
markdown_document = "# Title\n\n## 1. SubTitle\n\nHi this is Jim\n\nHi this is Joe\n\n### 1-1. Sub-SubTitle \n\nHi this is Lance \n\n## 2. Baz\n\nHi this is Molly"
print(markdown_document)
```

<pre class="custom"># Title
    
    ## 1. SubTitle
    
    Hi this is Jim
    
    Hi this is Joe
    
    ### 1-1. Sub-SubTitle 
    
    Hi this is Lance 
    
    ## 2. Baz
    
    Hi this is Molly
</pre>

```python
headers_to_split_on = [  # Define header levels and their names for document splitting
    (
        "#",
        "Header 1",
    ),  # Header level 1 is marked with '#' and named 'Header 1'
    (
        "##",
        "Header 2",
    ),  # Header level 2 is marked with '##' and named 'Header 2'
    (
        "###",
        "Header 3",
    ),  # Header level 3 is marked with '###' and named 'Header 3'
]

# Create a MarkdownHeaderTextSplitter object to split text based on markdown headers
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# Split markdown_document by headers and store in md_header_splits
md_header_splits = markdown_splitter.split_text(markdown_document)
# Print the split results
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom">Hi this is Jim  
    Hi this is Joe
    {'Header 1': 'Title', 'Header 2': '1. SubTitle'}
    =====================
    Hi this is Lance
    {'Header 1': 'Title', 'Header 2': '1. SubTitle', 'Header 3': '1-1. Sub-SubTitle'}
    =====================
    Hi this is Molly
    {'Header 1': 'Title', 'Header 2': '2. Baz'}
    =====================
</pre>

### Header Retention in Split Output

By default, the ```MarkdownHeaderTextSplitter``` removes headers from the output chunks.

However, you can configure the splitter to retain these headers by setting ```strip_headers``` parameter to ```False```.

Example:

```python
markdown_splitter = MarkdownHeaderTextSplitter(
    # Specify headers to split on
    headers_to_split_on=headers_to_split_on,
    # Set to keep headers in the output
    strip_headers=False,
)
# Split markdown document based on headers
md_header_splits = markdown_splitter.split_text(markdown_document)
# Print the split results
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom"># Title  
    ## 1. SubTitle  
    Hi this is Jim  
    Hi this is Joe
    {'Header 1': 'Title', 'Header 2': '1. SubTitle'}
    =====================
    ### 1-1. Sub-SubTitle  
    Hi this is Lance
    {'Header 1': 'Title', 'Header 2': '1. SubTitle', 'Header 3': '1-1. Sub-SubTitle'}
    =====================
    ## 2. Baz  
    Hi this is Molly
    {'Header 1': 'Title', 'Header 2': '2. Baz'}
    =====================
</pre>

## Combining with Other Text Splitters

After splitting by Markdown headers, you can further process the content within each Markdown group using any desired text splitter.

In this example, we'll use the ```RecursiveCharacterTextSplitter``` to demonstrate how to effectively combine different splitting methods.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

markdown_document = "# Intro \n\n## History \n\nMarkdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\nMarkdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n## Rise and divergence \n\nAs Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\nadditional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n#### Standardization \n\nFrom 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n# Implementations \n\nImplementations of Markdown are available for over a dozen programming languages."
print(markdown_document)
```

<pre class="custom"># Intro 
    
    ## History 
    
    Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] 
    
    Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. 
    
    ## Rise and divergence 
    
    As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for 
    
    additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. 
    
    #### Standardization 
    
    From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. 
    
    # Implementations 
    
    Implementations of Markdown are available for over a dozen programming languages.
</pre>

First, use ```MarkdownHeaderTextSplitter``` to split the Markdown document based on its headers.

```python
headers_to_split_on = [
    ("#", "Header 1"),  # Specify the header level and its name to split on
    # ("##", "Header 2"),
]

# Split the markdown document based on header levels
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)
# Print the split results
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom"># Intro  
    ## History  
    Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9]  
    Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files.  
    ## Rise and divergence  
    As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for  
    additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks.  
    #### Standardization  
    From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort.
    {'Header 1': 'Intro'}
    =====================
    # Implementations  
    Implementations of Markdown are available for over a dozen programming languages.
    {'Header 1': 'Implementations'}
    =====================
</pre>

Now, we'll further split the output of the ```MarkdownHeaderTextSplitter``` using the ```RecursiveCharacterTextSplitter```.

```python
chunk_size = 200  # Specify the size of each split chunk
chunk_overlap = 20  # Specify the number of overlapping characters between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split the document into chunks by characters
splits = text_splitter.split_documents(md_header_splits)
# Print the split results
for header in splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom"># Intro  
    ## History
    {'Header 1': 'Intro'}
    =====================
    Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its
    {'Header 1': 'Intro'}
    =====================
    readers in its source code form.[9]
    {'Header 1': 'Intro'}
    =====================
    Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files.  
    ## Rise and divergence
    {'Header 1': 'Intro'}
    =====================
    As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for
    {'Header 1': 'Intro'}
    =====================
    additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks.  
    #### Standardization
    {'Header 1': 'Intro'}
    =====================
    From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort.
    {'Header 1': 'Intro'}
    =====================
    # Implementations  
    Implementations of Markdown are available for over a dozen programming languages.
    {'Header 1': 'Implementations'}
    =====================
</pre>

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

# Microsoft PowerPoint

- Author: [Kane](https://github.com/HarryKane11)
- Design: [Kane](https://github.com/HarryKane11)
- Peer Review: [architectyou](https://github.com/architectyou), [jhboyo](https://github.com/jhboyo), [fastjw](https://github.com/fastjw)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/07-PowerPointLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/07-PowerPointLoader.ipynb)
## Overview

[Microsoft PowerPoint](https://en.wikipedia.org/wiki/Microsoft_PowerPoint) is a presentation program developed by Microsoft.

This tutorial demonstrates two different approaches to process PowerPoint documents for downstream use:
1. Using ```Unstructured``` to load and parse PowerPoint files into document elements
2. Using ```MarkItDown``` to convert PowerPoint files into markdown format and LangChain Document objects

Both methods enable effective text extraction and processing, with different strengths for various use cases.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Converting PPTX to Langchain Documents Using Unstructured](#converting-pptx-to-langchain-documents-using-unstructured)
- [Converting PPTX to Langchain Documents Using MarkItDown](#converting-pptx-to-langchain-documents-using-markitdown)

### References

- [Unstructured: official documentation](https://docs.unstructured.io/open-source/core-functionality/overview)
- [MarkItDown: GitHub Repository](https://github.com/microsoft/markitdown)
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
        "langchain-community",
        "langchain-core",
        "unstructured",
        "markitdown"
    ],
    verbose=False,
    upgrade=False,
)
```

## Converting PPTX to Langchain Documents Using Unstructured

```Unstructured``` is a robust document processing library that excels at converting various document formats into clean, structured text. <br/>It is well integrated with LangChain's ecosystem and provides reliable document parsing capabilities. 

The library includes:

- Local processing with open-source package
- Remote processing via Unstructured API
- Comprehensive document format support
- Built-in OCR capabilities

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# Initialize UnstructuredPowerPointLoader
loader = UnstructuredPowerPointLoader("data/07-ppt-loader-sample.pptx")

# Load PowerPoint document
docs = loader.load()

# Print number of loaded documents
print(len(docs))
```

<pre class="custom">1
</pre>

```python
print(docs[0].page_content[:100])
```

<pre class="custom">Natural Language Processing with Deep Learning
    
    CS224N/Ling284
    
    Christopher Manning
    
    Lecture 2: Word
</pre>

```Unstructured``` generates various "elements" for different **chunks** of text.

By default, they are combined and returned as a single document, but elements can be easily separated by specifying ```mode="elements"```.

```python
# Create UnstructuredPowerPointLoader with elements mode
loader = UnstructuredPowerPointLoader("data/07-ppt-loader-sample.pptx", mode="elements")

# Load PowerPoint elements
docs = loader.load()

# Print number of elements extracted
print(len(docs))
```

<pre class="custom">498
</pre>

```python
docs[0]
```




<pre class="custom">Document(metadata={'source': 'data/07-ppt-loader-sample.pptx', 'category_depth': 0, 'file_directory': 'data', 'filename': '07-ppt-loader-sample.pptx', 'last_modified': '2025-01-16T21:42:19', 'page_number': 1, 'languages': ['eng'], 'filetype': 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'category': 'Title', 'element_id': 'bb6cdc142e5062b564541bfbc10f7f8c'}, page_content='Natural Language Processing with Deep Learning')</pre>



```python
# Get and display the first element
first_element = docs[0]
print(first_element)

# To see its metadata and content separately, you could do:
print("Content:", first_element.page_content)
print("Metadata:", first_element.metadata)
```

<pre class="custom">page_content='Natural Language Processing with Deep Learning' metadata={'source': 'data/07-ppt-loader-sample.pptx', 'category_depth': 0, 'file_directory': 'data', 'filename': '07-ppt-loader-sample.pptx', 'last_modified': '2025-01-16T21:42:19', 'page_number': 1, 'languages': ['eng'], 'filetype': 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'category': 'Title', 'element_id': 'bb6cdc142e5062b564541bfbc10f7f8c'}
    Content: Natural Language Processing with Deep Learning
    Metadata: {'source': 'data/07-ppt-loader-sample.pptx', 'category_depth': 0, 'file_directory': 'data', 'filename': '07-ppt-loader-sample.pptx', 'last_modified': '2025-01-16T21:42:19', 'page_number': 1, 'languages': ['eng'], 'filetype': 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'category': 'Title', 'element_id': 'bb6cdc142e5062b564541bfbc10f7f8c'}
</pre>

```python
# Print elements with formatted output and enumerate for easy reference
for idx, doc in enumerate(docs[:3], 1):
    print(f"\nElement {idx}/{len(docs)}")
    print(f"Category: {doc.metadata['category']}")
    print("="*50)
    print(f"Content:\n{doc.page_content.strip()}")
    print("="*50)
```

<pre class="custom">
    Element 1/498
    Category: Title
    ==================================================
    Content:
    Natural Language Processing with Deep Learning
    ==================================================
    
    Element 2/498
    Category: Title
    ==================================================
    Content:
    CS224N/Ling284
    ==================================================
    
    Element 3/498
    Category: Title
    ==================================================
    Content:
    Christopher Manning
    ==================================================
</pre>

## Converting PPTX to Langchain Documents Using MarkItDown

[```MarkItDown```](https://github.com/microsoft/markitdown "Visit the GitHub page")
 is an open-source library by Microsoft that converts unstructured documents into structured Markdown, a format that LLMs can easily process and understand. This makes it particularly valuable for RAG (Retrieval Augmented Generation) systems by enabling clean, semantic text representation.

Supporting formats like PDF, PowerPoint, Word, Excel, images (with EXIF/OCR), audio (with transcription), HTML, and more, ```MarkItDown``` preserves semantic structure and handles complex data, such as tables, with precision. This ensures high retrieval quality and enhances LLMs' ability to extract insights from diverse content types.

> ⚠️**Note**: MarkItDown does not interpret the content of images embedded in PowerPoint files. Instead, it extracts the images as-is, leaving their semantic meaning inaccessible to LLMs.

For example, an object in the slide would be processed like this:

```![object #](object#.jpg)```

Installation is straightforward:

```python
# %pip install markitdown
```

### Extracting Text from PPTX Using MarkItDown
In this section, we'll use ```MarkItDown``` to:
* Convert PowerPoint slides to markdown format
* Preserve the semantic structure and visual formatting
* Maintain slide numbers and titles
* Generate clean, readable text output


First, we need to initialize ```MarkItDown``` and run ```convert``` function to load the ```.pptx``` from local.

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("data/07-ppt-loader-sample.pptx")
result_text = result.text_content
print(result_text[:500])
```

<pre class="custom"><!-- Slide number: 1 -->
    
    ![object 2](object2.jpg)
    # Natural Language Processing with Deep Learning
    CS224N/Ling284
    Christopher Manning
    Lecture 2: Word Vectors, Word Senses, and Neural Classifiers
    
    <!-- Slide number: 2 -->
    # Lecture Plan
    10
    Lecture 2: Word Vectors, Word Senses, and Neural Network Classifiers
    Course organization (3 mins)
    Optimization basics (5 mins)
    Review of word2vec and looking at word vectors (12 mins)
    More on word2vec (8 mins)
    Can we capture the essence of word meaning more ef
</pre>

### Convert markdown format to Langchain Document format

The code below processes PowerPoint slides by splitting them into individual Document objects. <br/>Each slide is converted into a Langchain Document object with metadata including the ```slide number``` and ```title```. 

```python
from langchain_core.documents import Document
import re

# Initialize document processing for PowerPoint slides
# Format: <!-- Slide number: X --> where X is the slide number

# Split the input text into individual slides using HTML comment markers
slides = re.split(r'<!--\s*Slide number:\s*(\d+)\s*-->', result_text)

# Initialize list to store Document objects
documents = []

# Process each slide:
# - Start from index 1 since slides[0] is empty from the initial split
# - Step by 2 because the split result alternates between:
#   1. slide number (odd indices)
#   2. slide content (even indices)
# Example: ['', '1', 'content1', '2', 'content2', '3', 'content3']
for i in range(1, len(slides), 2):
    # Extract slide number and content
    slide_number = slides[i]
    content = slides[i + 1].strip() if i + 1 < len(slides) else ""
    
    # Extract slide title from first markdown header if present
    title_match = re.search(r'#\s*(.+?)(?=\n|$)', content)
    title = title_match.group(1).strip() if title_match else ""
    
    # Create Document object with slide metadata
    doc = Document(
        page_content=content,
        metadata={
            "source": "data/07-ppt-loader-sample.pptx",
            "slide_number": int(slide_number),
            "slide_title": title
        }
    )
    documents.append(doc)

documents[:2]
```




<pre class="custom">[Document(metadata={'source': 'data/07-ppt-loader-sample.pptx', 'slide_number': 1, 'slide_title': 'Natural Language Processing with Deep Learning'}, page_content='![object 2](object2.jpg)\n# Natural Language Processing with Deep Learning\nCS224N/Ling284\nChristopher Manning\nLecture 2: Word Vectors, Word Senses, and Neural Classifiers'),
     Document(metadata={'source': 'data/07-ppt-loader-sample.pptx', 'slide_number': 2, 'slide_title': 'Lecture Plan'}, page_content='# Lecture Plan\n10\nLecture 2: Word Vectors, Word Senses, and Neural Network Classifiers\nCourse organization (3 mins)\nOptimization basics (5 mins)\nReview of word2vec and looking at word vectors (12 mins)\nMore on word2vec (8 mins)\nCan we capture the essence of word meaning more effectively by counting? (12m)\nEvaluating word vectors (10 mins)\nWord senses (10 mins)\nReview of classification and how neural nets differ (10 mins)\nIntroducing neural networks (10 mins)\n\nKey Goal: To be able to read and understand word embeddings papers by the end of class')]</pre>



```MarkItDown``` efficiently handles tables in PowerPoint slides by converting them into clean Markdown table syntax. 

This makes tabular data easily accessible for LLMs while preserving the original structure and formatting.

```python
print(documents[15].page_content)
```

<pre class="custom"># Example: Window based co-occurrence matrix
    10
    Window length 1 (more common: 5–10)
    Symmetric (irrelevant whether left or right context)
    
    Example corpus:
    I like deep learning
    I like NLP
    I enjoy flying
    
    | counts | I | like | enjoy | deep | learning | NLP | flying | . |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | I | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
    | like | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
    | enjoy | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
    | deep | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
    | learning | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
    | NLP | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
    | flying | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
    | . | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
</pre>

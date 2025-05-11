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

# GPT4ALL

- Author: [Do Woung Kong](https://github.com/krkrong)
- Peer Review : [Sun Hyoung Lee](https://github.com/LEE1026icarus), [Yongdam Kim](https://github.com/dancing-with-coffee)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/07-GPT4ALLEmbedding.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/07-GPT4ALLEmbedding.ipynb)
## Overview

```GPT4All``` is a local execution-based privacy chatbot that can be used for free.

No GPU or internet connection is required, and ```GPT4All``` offers popular models such as GPT4All Falcon, Wizard, and its own models.

This notebook explains how to use ```GPT4Allembeddings``` with ```LangChain```.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Install Python Binding for GPT4All](#create-a-basic-pdf-based-retrieval-chain)
- [Embed the Textual Data](#query-routing-and-document-evaluation)


### References

- [GPT4All docs](https://docs.gpt4all.io/gpt4all_python_embedding.html#gpt4all.gpt4all.Embed4All)
- [GPT4AllEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.gpt4all.GPT4AllEmbeddings.html#langchain_community.embeddings.gpt4all.GPT4AllEmbeddings)

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    ["langchain_community"],
    verbose=False,
    upgrade=False,
)
```

## Install Python Binding for GPT4All

Before diving into the practical exercises, you need to install the Python bindings for ```GPT4All```.

Python bindings allow a Python program to interface with external libraries or tools, enabling seamless integration and usage of functionalities provided by those external resources.

To install the Python bindings for ```GPT4All```, run the following command:

```python
%pip install --upgrade --quiet  gpt4all > /dev/null
```

<pre class="custom">Note: you may need to restart the kernel to use updated packages.
</pre>

Import the ```GPT4AllEmbeddings``` class from the ```langchain_community.embeddings``` module.

The ```GPT4AllEmbeddings``` class provides functionality to embed text data into vectors using the GPT4All model.

This class implements the embedding interface of the LangChain framework, allowing it to be used seamlessly with LangChain's various features.

```python
from langchain_community.embeddings import GPT4AllEmbeddings
```

GPT4All supports the generation of high-quality embeddings for text documents of arbitrary length using a contrastive learning sentence transformer optimized for CPUs. These embeddings offer a quality comparable to many tasks using OpenAI models.

An instance of the ```GPT4AllEmbeddings``` class is created.

- The ```GPT4AllEmbeddings``` class is an embedding model that uses the GPT4All model to transform text data into vectors.  

- In this code, the ```gpt4all_embd``` variable is assigned an instance of ```GPT4AllEmbeddings```.  

- You can then use ```gpt4all_embd``` to convert text data into vectors.

```python
# Create a GPT4All embedding object
gpt4all_embd = GPT4AllEmbeddings()
```

Assign the string "This is a sample sentence for testing embeddings." to the ```text``` variable.

```python
# Define a sample document text for testing
text = "This is a sample sentence for testing embeddings."
```

## Embed the Textual Data


The process of embedding text data is as follows:

First, the text data is tokenized and converted into numerical form.  

During this step, a pre-trained tokenizer is used to split the text into tokens and map each token to a unique integer.  

Next, the tokenized data is input into an embedding layer, where it is transformed into high-dimensional dense vectors.  

In this process, each token is represented as a vector of real numbers that capture the token's meaning and context.  

Finally, the embedded vectors can be used in various natural language processing tasks.  

For example, they can serve as input data for tasks such as document classification, sentiment analysis, and machine translation, enhancing model performance.  

This process of text data embedding plays a crucial role in natural language processing, making it essential for efficiently processing and analyzing large amounts of text data.

Use the ```embed_query``` method of the ```gpt4all_embd``` object to embed the given text (```text```).  

- The ```text``` variable stores the text to be embedded.  
- The ```gpt4all_embd``` object uses the GPT4All model to perform text embedding.  
- The ```embed_query``` method converts the given text into a vector format and returns it.  
- The embedding result is stored in the ```query_result``` variable.

```python
# Generate query embeddings for the given text.
query_result = gpt4all_embd.embed_query(text)

# Check the dimensions of the embedded space.
len(query_result)
```




<pre class="custom">384</pre>



You can use the ```embed_documents``` function to embed multiple text fragments.

Use the ```embed_documents``` method of the ```gpt4all_embd``` object to embed the ```text``` document.

- Wrap the ```text``` document in a list and pass it as an argument to the ```embed_documents``` method.  
- The ```embed_documents``` method calculates and returns the embedding vector of the document.  
- The resulting embedding vector is stored in the ```doc_result``` variable.

Additionally, these embeddings can be mapped with Nomic's Atlas (https://docs.nomic.ai/index.html) to visualize the data.

```python
# Generate query embeddings for the given text.
doc_result = gpt4all_embd.embed_documents([text])

# Check the dimensions of the embedded space.
len(doc_result[0])
```




<pre class="custom">384</pre>



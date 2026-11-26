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

# HuggingFace Embeddings

- Author: [liniar](https://github.com/namyoungkim)
- Design: [liniar](https://github.com/namyoungkim)
- Peer Review : [byoon](https://github.com/acho98), [Sun Hyoung Lee](https://github.com/LEE1026icarus)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/03-HuggingFaceEmbeddings.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/03-HuggingFaceEmbeddings.ipynb)

## Overview  
- ```Hugging Face``` offers a wide range of **embedding models** for free, enabling various embedding tasks with ease.
- In this tutorial, we’ll use ```langchain_huggingface``` to build a **simple text embedding-based search system.** 
- The following models will be used for **Text Embedding**  

    - 1️⃣ **multilingual-e5-large-instruct:** A multilingual instruction-based embedding model.  
    - 2️⃣ **multilingual-e5-large:** A powerful multilingual embedding model.  
    - 3️⃣ **bge-m3:** Optimized for large-scale text processing.  

![](./img/03-huggingfaceembeddings-workflow.png)  

### Table of Contents  

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation for Embedding-Based Search Tutorial](#data-preparation-for-embedding-based-search-tutorial)
- [Which Text Embedding Model Should You Use?](#which-text-embedding-model-should-you-use) 
- [Similarity Calculation](#similarity-calculation)
- [HuggingFaceEndpointEmbeddings Overview](#huggingfaceendpointembeddings-overview)
- [HuggingFaceEmbeddings Overview](#huggingfaceembeddings-overview)
- [FlagEmbedding Usage Guide](#flagembedding-usage-guide)


### References
- [LangChain: Embedding Models](https://python.langchain.com/docs/concepts/embedding_models)
- [LangChain: Text Embedding](https://python.langchain.com/docs/integrations/text_embedding)
- [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/README.md)
----

## Environment Setup  

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.  

**[Note]**  
- ```langchain-opentutorial``` is a package that provides a set of **easy-to-use environment setup,** **useful functions,** and **utilities for tutorials.**  
- You can check out the [```langchain-opentutorial``` ](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.  

---

### 🛠️ **The following configurations will be set up**  

- **Jupyter Notebook Output Settings**
    - Display standard error ( ```stderr``` ) messages directly instead of capturing them.  
- **Install Required Packages** 
    - Ensure all necessary dependencies are installed.  
- **API Key Setup** 
    - Configure the API key for authentication.  
- **PyTorch Device Selection Setup** 
    - Automatically select the optimal computing device (CPU, CUDA, or MPS).
        - ```{"device": "mps"}``` : Perform embedding calculations using **MPS** instead of GPU. (For Mac users)
        - ```{"device": "cuda"}``` : Perform embedding calculations using **GPU.** (For Linux and Windows users, requires CUDA installation)
        - ```{"device": "cpu"}``` : Perform embedding calculations using **CPU.** (Available for all users)
- **Embedding Model Local Storage Path** 
    - Define a local path for storing embedding models.  

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
        "langchain_huggingface",
        "torch",
        "numpy",
        "scikit-learn",
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
        "LANGCHAIN_PROJECT": "HuggingFace Embeddings",  # Please set it the same as the title
        "HUGGINGFACEHUB_API_TOKEN": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set OPENAI_API_KEY in ```.env``` file and load it.

**[Note]** 
- This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Automatically select the appropriate device
import torch
import platform


def get_device():
    if platform.system() == "Darwin":  # macOS specific
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Using MPS (Metal Performance Shaders) on macOS")
            return "mps"
    if torch.cuda.is_available():
        print("✅ Using CUDA (NVIDIA GPU)")
        return "cuda"
    else:
        print("✅ Using CPU")
        return "cpu"


# Set the device
device = get_device()
print("🖥️ Current device in use:", device)
```

<pre class="custom">✅ Using MPS (Metal Performance Shaders) on macOS
    🖥️ Current device in use: mps
</pre>

```python
# Embedding Model Local Storage Path
import os
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Set the download path to ./cache/
os.environ["HF_HOME"] = "./cache/"
```

## Data Preparation for Embedding-Based Search Tutorial

To perform **embedding-based search,** we prepare both a **Query** and **Documents.**  

1. Query  
- Write a **key question** that will serve as the basis for the search.  

```python
# Query
q = "Please tell me more about LangChain."
```

2. Documents  
- Prepare **multiple documents (texts)** that will serve as the target for the search.  
- Each document will be **embedded** to enable semantic search capabilities.  

```python
# Documents for Text Embedding
docs = [
    "Hi, nice to meet you.",
    "LangChain simplifies the process of building applications with large language models.",
    "The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.",
    "LangChain simplifies the process of building applications with large-scale language models.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]
```

## Which Text Embedding Model Should You Use?
- Leverage the **MTEB leaderboard** and **free embedding models** to confidently select and utilize the **best-performing text embedding models** for your projects! 🚀  

---

### 🚀 **What is MTEB (Massive Text Embedding Benchmark)?**  
- **MTEB** is a benchmark designed to **systematically and objectively evaluate** the performance of text embedding models.  
    - **Purpose:** To **fairly compare** the performance of embedding models.  
    - **Evaluation Tasks:** Includes tasks like **Classification,**  **Retrieval,**  **Clustering,**  and **Semantic Similarity.**  
    - **Supported Models:** A wide range of **text embedding models available on Hugging Face.**  
    - **Results:** Displayed as **scores,**  with top-performing models ranked on the **leaderboard.**  

🔗 [ **MTEB Leaderboard (Hugging Face)** ](https://huggingface.co/spaces/mteb/leaderboard)  

---

### 🛠️ **Models Used in This Tutorial**  

| **Embedding Model** | **Description** |
|----------|----------|
| 1️⃣ **multilingual-e5-large-instruct** | Offers strong multilingual support with consistent results. |
| 2️⃣ **multilingual-e5-large** | A powerful multilingual embedding model. |
| 3️⃣ **bge-m3** | Optimized for large-scale text processing, excelling in retrieval and semantic similarity tasks. |

1️⃣ **multilingual-e5-large-instruct**
![](./img/03-huggingfaceembeddings-leaderboard-01.png)

2️⃣ **multilingual-e5-large**
![](./img/03-huggingfaceembeddings-leaderboard-02.png)

3️⃣ **bge-m3**
![](./img/03-huggingfaceembeddings-leaderboard-03.png)

## Similarity Calculation

**Similarity Calculation Using Vector Dot Product**  
- Similarity is determined using the **dot product** of vectors.  

- **Similarity Calculation Formula:**  

$$ \text{similarities} = \mathbf{query} \cdot \mathbf{documents}^T $$  

---

### 📐 **Mathematical Significance of the Vector Dot Product**  

**Definition of Vector Dot Product**  

The **dot product** of two vectors, $\mathbf{a}$ and $\mathbf{b}$, is mathematically defined as:  

$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i $$  

---

**Relationship with Cosine Similarity**  

The **dot product** also relates to **cosine similarity** and follows this property:  

$$ \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta $$  

Where:  
- $\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ represent the **magnitudes** (**norms,**  specifically Euclidean norms) of vectors $\mathbf{a}$ and $\mathbf{b}$.  
- $\theta$ is the **angle between the two vectors.**  
- $\cos \theta$ represents the **cosine similarity** between the two vectors.  

---

**🔍 Interpretation of Vector Dot Product in Similarity**  

When the **dot product value is large** (a large positive value):  
- The **magnitudes** ($\|\mathbf{a}\|$ and $\|\mathbf{b}\|$) of the two vectors are large.  
- The **angle** ($\theta$) between the two vectors is small ( **$\cos \theta$ approaches 1** ).  

This indicates that the two vectors point in a **similar direction** and are **more semantically similar,**  especially when their magnitudes are also large.  

---

### 📏 **Calculation of Vector Magnitude (Norm)**  

**Definition of Euclidean Norm**  

For a vector $\mathbf{a} = [a_1, a_2, \ldots, a_n]$, the **Euclidean norm** $\|\mathbf{a}\|$ is calculated as:  

$$ \|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2} $$  

This **magnitude** represents the **length** or **size** of the vector in multi-dimensional space.  

---

Understanding these mathematical foundations helps ensure precise similarity calculations, enabling better performance in tasks like **semantic search,**  **retrieval systems,**  and **recommendation engines.**  🚀

----
### Similarity calculation between ```embedded_query``` and ```embedded_document``` 
- ```embed_documents``` : For embedding multiple texts (documents)
- ```embed_query``` : For embedding a single text (query)

We've implemented a method to search for the most relevant documents using **text embeddings.** 
- Let's use ```search_similar_documents(q, docs, hf_embeddings)``` to find the most relevant documents.

```python
import numpy as np


def search_similar_documents(q, docs, hf_embeddings):
    """
    Search for the most relevant documents based on a query using text embeddings.

    Args:
        q (str): The query string for which relevant documents are to be found.
        docs (list of str): A list of document strings to compare against the query.
        hf_embeddings: An embedding model object with `embed_query` and `embed_documents` methods.

    Returns:
        tuple:
            - embedded_query (numpy.ndarray): The embedding vector of the query.
            - embedded_documents (numpy.ndarray): The embedding matrix of the documents.

    Workflow:
        1. Embed the query string into a numerical vector using `embed_query`.
        2. Embed each document into numerical vectors using `embed_documents`.
        3. Calculate similarity scores between the query and documents using the dot product.
        4. Sort the documents based on their similarity scores in descending order.
        5. Print the query and display the sorted documents by their relevance.
        6. Return the query and document embeddings for further analysis if needed.
    """
    # Embed the query and documents using the embedding model
    embedded_query = hf_embeddings.embed_query(q)
    embedded_documents = hf_embeddings.embed_documents(docs)

    # Calculate similarity scores using dot product
    similarity_scores = np.array(embedded_query) @ np.array(embedded_documents).T

    # Sort documents by similarity scores in descending order
    sorted_idx = similarity_scores.argsort()[::-1]

    # Display the results
    print(f"[Query] {q}\n" + "=" * 40)
    for i, idx in enumerate(sorted_idx):
        print(f"[{i}] {docs[idx]}")
        print()

    # Return embeddings for potential further processing or analysis
    return embedded_query, embedded_documents
```

## HuggingFaceEndpointEmbeddings Overview

**HuggingFaceEndpointEmbeddings** is a feature in the **LangChain** library that leverages **Hugging Face’s Inference API endpoint** to generate text embeddings seamlessly.

---

### 📚 **Key Concepts**

1. **Hugging Face Inference API**  
   - Access pre-trained embedding models via Hugging Face’s API.  
   - No need to download models locally; embeddings are generated directly through the API.  

2. **LangChain Integration**  
   - Easily integrate embedding results into LangChain workflows using its standardized interface.  

3. **Use Cases**  
   - Text-query and document similarity calculation  
   - Search and recommendation systems  
   - Natural Language Understanding (NLU) applications  

---

### ⚙️ **Key Parameters**

- ```model``` : The Hugging Face model ID (e.g., ```BAAI/bge-m3``` )  
- ```task``` : The task to perform (usually ```"feature-extraction"``` )  
- ```api_key``` : Your Hugging Face API token  
- ```model_kwargs``` : Additional model configuration parameters  

---

### 💡 **Advantages**  
- **No Local Model Download:** Instant access via API.  
- **Scalability:** Supports a wide range of pre-trained Hugging Face models.  
- **Seamless Integration:** Effortlessly integrates embeddings into LangChain workflows.  

---

### ⚠️ **Caveats**  
- **API Support:** Not all models support API inference.  
- **Speed & Cost:** Free APIs may have slower response times and usage limitations.  

---

With **HuggingFaceEndpointEmbeddings,**  you can easily integrate Hugging Face’s powerful embedding models into your **LangChain workflows** for efficient and scalable NLP solutions. 🚀

---
Let’s use the ```intfloat/multilingual-e5-large-instruct``` model via the API to search for the most relevant documents using text embeddings.

- [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

```python
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

model_name = "intfloat/multilingual-e5-large-instruct"

hf_endpoint_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
```

Search for the most relevant documents based on a query using text embeddings.

```python
%%time
# Embed the query and documents using the embedding model
embedded_query = hf_endpoint_embeddings.embed_query(q)
embedded_documents = hf_endpoint_embeddings.embed_documents(docs)
```

<pre class="custom">CPU times: user 7.18 ms, sys: 2.32 ms, total: 9.5 ms
    Wall time: 1.21 s
</pre>

```python
# Calculate similarity scores using dot product
similarity_scores = np.array(embedded_query) @ np.array(embedded_documents).T

# Sort documents by similarity scores in descending order
sorted_idx = similarity_scores.argsort()[::-1]
```

```python
# Display the results
print(f"[Query] {q}\n" + "=" * 40)
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] {docs[idx]}")
    print()
```

<pre class="custom">[Query] Please tell me more about LangChain.
    ========================================
    [0] LangChain simplifies the process of building applications with large language models.
    
    [1] LangChain simplifies the process of building applications with large-scale language models.
    
    [2] The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.
    
    [3] Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    [4] Hi, nice to meet you.
    
</pre>

```python
print("[HuggingFace Endpoint Embedding]")
print(f"Model: \t\t{model_name}")
print(f"Document Dimension: \t{len(embedded_documents[0])}")
print(f"Query Dimension: \t{len(embedded_query)}")
```

<pre class="custom">[HuggingFace Endpoint Embedding]
    Model: 		intfloat/multilingual-e5-large-instruct
    Document Dimension: 	1024
    Query Dimension: 	1024
</pre>

We can verify that the dimensions of ```embedded_documents``` and ```embedded_query``` are consistent.  

You can also perform searches using the ```search_similar_documents``` method we implemented earlier.  
From now on, let's use this method for our searches.  

```python
%%time
embedded_query, embedded_documents = search_similar_documents(q, docs, hf_endpoint_embeddings)
```

<pre class="custom">[Query] Please tell me more about LangChain.
    ========================================
    [0] LangChain simplifies the process of building applications with large language models.
    
    [1] LangChain simplifies the process of building applications with large-scale language models.
    
    [2] The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.
    
    [3] Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    [4] Hi, nice to meet you.
    
    CPU times: user 7.25 ms, sys: 3.26 ms, total: 10.5 ms
    Wall time: 418 ms
</pre>

## HuggingFaceEmbeddings Overview

- **HuggingFaceEmbeddings** is a feature in the **LangChain** library that enables the conversion of text data into vectors using **Hugging Face embedding models.** 
- This class downloads and operates Hugging Face models **locally** for efficient processing.

---

### 📚 **Key Concepts**

1. **Hugging Face Pre-trained Models**  
   - Leverages pre-trained embedding models provided by Hugging Face.  
   - Downloads models locally for direct embedding operations.  

2. **LangChain Integration**  
   - Seamlessly integrates with LangChain workflows using its standardized interface.  

3. **Use Cases**  
   - Text-query and document similarity calculation  
   - Search and recommendation systems  
   - Natural Language Understanding (NLU) applications  

---

### ⚙️ **Key Parameters**

- ```model_name``` : The Hugging Face model ID (e.g., ```sentence-transformers/all-MiniLM-L6-v2``` )
- ```model_kwargs``` : Additional model configuration parameters (e.g., GPU/CPU device settings)
- ```encode_kwargs``` : Extra settings for embedding generation

---

### 💡 **Advantages**  
- **Local Embedding Operations:** Perform embeddings locally without requiring an internet connection.  
- **High Performance:** Utilize GPU settings for faster embedding generation.  
- **Model Variety:** Supports a wide range of Hugging Face models.  

---

### ⚠️ **Caveats**  
- **Local Storage Requirement:** Pre-trained models must be downloaded locally.  
- **Environment Configuration:** Performance may vary depending on GPU/CPU device settings.  

---

With **HuggingFaceEmbeddings,** you can efficiently leverage **Hugging Face's powerful embedding models** in a **local environment,** enabling flexible and scalable NLP solutions. 🚀

---
Let's download the embedding model locally, perform embeddings, and search for the most relevant documents.

```intfloat/multilingual-e5-large-instruct``` 

- [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings_e5_instruct = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device},  # mps, cuda, cpu
    encode_kwargs={"normalize_embeddings": True},
)
```


<pre class="custom">modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]</pre>



    config_sentence_transformers.json:   0%|          | 0.00/128 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/140k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/690 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/1.18k [00:00<?, ?B/s]



    sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/271 [00:00<?, ?B/s]


```python
%%time
embedded_query, embedded_documents = search_similar_documents(q, docs, hf_embeddings_e5_instruct)
```

<pre class="custom">[Query] Please tell me more about LangChain.
    ========================================
    [0] LangChain simplifies the process of building applications with large language models.
    
    [1] LangChain simplifies the process of building applications with large-scale language models.
    
    [2] The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.
    
    [3] Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    [4] Hi, nice to meet you.
    
    CPU times: user 326 ms, sys: 120 ms, total: 446 ms
    Wall time: 547 ms
</pre>

```python
print(f"Model: \t\t{model_name}")
print(f"Document Dimension: \t{len(embedded_documents[0])}")
print(f"Query Dimension: \t{len(embedded_query)}")
```

<pre class="custom">Model: 		intfloat/multilingual-e5-large-instruct
    Document Dimension: 	1024
    Query Dimension: 	1024
</pre>

---
```intfloat/multilingual-e5-large``` 

- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "intfloat/multilingual-e5-large"

hf_embeddings_e5_large = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device},  # mps, cuda, cpu
    encode_kwargs={"normalize_embeddings": True},
)
```


<pre class="custom">modules.json:   0%|          | 0.00/387 [00:00<?, ?B/s]</pre>



    README.md:   0%|          | 0.00/160k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/57.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/690 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/2.24G [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/418 [00:00<?, ?B/s]



    sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/280 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/201 [00:00<?, ?B/s]


```python
%%time
embedded_query, embedded_documents = search_similar_documents(q, docs, hf_embeddings_e5_large)
```

<pre class="custom">[Query] Please tell me more about LangChain.
    ========================================
    [0] LangChain simplifies the process of building applications with large-scale language models.
    
    [1] LangChain simplifies the process of building applications with large language models.
    
    [2] The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.
    
    [3] Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    [4] Hi, nice to meet you.
    
    CPU times: user 84.1 ms, sys: 511 ms, total: 595 ms
    Wall time: 827 ms
</pre>

```python
print(f"Model: \t\t{model_name}")
print(f"Document Dimension: \t{len(embedded_documents[0])}")
print(f"Query Dimension: \t{len(embedded_query)}")
```

<pre class="custom">Model: 		intfloat/multilingual-e5-large
    Document Dimension: 	1024
    Query Dimension: 	1024
</pre>

---
```BAAI/bge-m3``` 

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

```python
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3"
model_kwargs = {"device": device}  # mps, cuda, cpu
encode_kwargs = {"normalize_embeddings": True}

hf_embeddings_bge_m3 = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
```


<pre class="custom">modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]</pre>



    config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/15.8k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]



    sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]


```python
%%time
embedded_query, embedded_documents = search_similar_documents(q, docs, hf_embeddings_bge_m3)
```

<pre class="custom">[Query] Please tell me more about LangChain.
    ========================================
    [0] LangChain simplifies the process of building applications with large language models.
    
    [1] LangChain simplifies the process of building applications with large-scale language models.
    
    [2] The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively.
    
    [3] Hi, nice to meet you.
    
    [4] Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    CPU times: user 81.1 ms, sys: 1.29 s, total: 1.37 s
    Wall time: 1.5 s
</pre>

```python
print(f"Model: \t\t{model_name}")
print(f"Document Dimension: \t{len(embedded_documents[0])}")
print(f"Query Dimension: \t{len(embedded_query)}")
```

<pre class="custom">Model: 		BAAI/bge-m3
    Document Dimension: 	1024
    Query Dimension: 	1024
</pre>

## FlagEmbedding Usage Guide

- **FlagEmbedding** is an advanced embedding framework developed by **BAAI (Beijing Academy of Artificial Intelligence).**
- It supports **various embedding approaches** and is primarily used with the **BGE (BAAI General Embedding) model.**
- FlagEmbedding excels in tasks such as **semantic search**, **natural language processing (NLP)**, and **recommendation systems.**

---

### 📚 **Core Concepts of FlagEmbedding**

1️⃣ ```Dense Embedding``` 
- Definition: Represents the overall meaning of a text as a single high-density vector.  
- Advantages: Effectively captures semantic similarity.  
- Use Cases: Semantic search, document similarity computation.  

2️⃣ ```Lexical Embedding``` 
- Definition: Breaks text into word-level components, emphasizing word matching.  
- Advantages: Ensures precise matching of specific words or phrases.  
- Use Cases: Keyword-based search, exact word matching.  

3️⃣ ```Multi-Vector Embedding``` 
- Definition: Splits a document into multiple vectors for representation.  
- Advantages: Allows more granular representation of lengthy texts or diverse topics.  
- Use Cases: Complex document structure analysis, detailed topic matching.  

---

FlagEmbedding offers a **flexible and powerful toolkit** for leveraging embeddings across a wide range of **NLP tasks and semantic search applications.** 🚀

The following code is used to control **tokenizer parallelism** in Hugging Face's ```transformers``` library:

- ```TOKENIZERS_PARALLELISM = "true"```  → **Optimized for speed,** suitable for large-scale data processing.  
- ```TOKENIZERS_PARALLELISM = "false"```  → **Ensures stability,** prevents conflicts and race conditions.  

```python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # "false"
```

```python
# install FlagEmbedding
%pip install -qU FlagEmbedding
```

### ⚙️ **Key Parameter**

```BGEM3FlagModel``` 
-  ```model_name``` : The Hugging Face **model ID** (e.g., ```BAAI/bge-m3``` ).
-  ```use_fp16``` : When set to **True,** reduces **memory usage** and improves **encoding speed.**

```bge_embeddings.encode``` 
- ```batch_size``` : Defines the **number of documents** to process at once.  
- ```max_length``` : Sets the **maximum token length** for encoding documents.  
   - Increase for longer documents to ensure full content encoding.  
   - Excessively large values may **degrade performance.**
- ```return_dense``` : When set to **True**, returns **Dense Vectors** only.  
- ```return_sparse``` : When set to **True**, returns **Sparse Vectors.**
- ```return_colbert_vecs``` : When set to **True,** returns **ColBERT-style vectors.**



### 1️⃣ **Dense Vector Embedding Example**
- Definition: Represents the overall meaning of a text as a single high-density vector.  
- Advantages: Effectively captures semantic similarity.  
- Use Cases: Semantic search, document similarity computation.  

```python
from FlagEmbedding import BGEM3FlagModel

model_name = "BAAI/bge-m3"

bge_embeddings = BGEM3FlagModel(
    model_name,
    use_fp16=True,  # Enabling fp16 improves encoding speed with minimal precision trade-off.
)

# Encode documents with specified parameters
embedded_documents_dense_vecs = bge_embeddings.encode(
    sentences=docs,
    batch_size=12,
    max_length=8192,  # Reduce this value if your documents are shorter to speed up encoding.
)["dense_vecs"]

# Query Encoding
embedded_query_dense_vecs = bge_embeddings.encode(
    sentences=[q],
    batch_size=12,
    max_length=8192,  # Reduce this value if your documents are shorter to speed up encoding.
)["dense_vecs"]
```


<pre class="custom">Fetching 30 files:   0%|          | 0/30 [00:00<?, ?it/s]</pre>



    imgs/mkqa.jpg:   0%|          | 0.00/608k [00:00<?, ?B/s]



    imgs/.DS_Store:   0%|          | 0.00/6.15k [00:00<?, ?B/s]



    imgs/long.jpg:   0%|          | 0.00/485k [00:00<?, ?B/s]



    imgs/bm25.jpg:   0%|          | 0.00/132k [00:00<?, ?B/s]



    imgs/miracl.jpg:   0%|          | 0.00/576k [00:00<?, ?B/s]



    imgs/nqa.jpg:   0%|          | 0.00/158k [00:00<?, ?B/s]



    .gitattributes:   0%|          | 0.00/1.63k [00:00<?, ?B/s]



    colbert_linear.pt:   0%|          | 0.00/2.10M [00:00<?, ?B/s]



    imgs/others.webp:   0%|          | 0.00/21.0k [00:00<?, ?B/s]



    long.jpg:   0%|          | 0.00/127k [00:00<?, ?B/s]



    onnx/Constant_7_attr__value:   0%|          | 0.00/65.6k [00:00<?, ?B/s]



    onnx/config.json:   0%|          | 0.00/698 [00:00<?, ?B/s]



    model.onnx:   0%|          | 0.00/725k [00:00<?, ?B/s]



    model.onnx_data:   0%|          | 0.00/2.27G [00:00<?, ?B/s]



    onnx/tokenizer_config.json:   0%|          | 0.00/1.17k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]



    sparse_linear.pt:   0%|          | 0.00/3.52k [00:00<?, ?B/s]


    You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    

```python
embedded_documents_dense_vecs
```




<pre class="custom">array([[-0.0271  ,  0.003561, -0.0506  , ...,  0.00911 , -0.04565 ,
             0.02028 ],
           [-0.02242 , -0.01398 , -0.00946 , ...,  0.01851 ,  0.01907 ,
            -0.01917 ],
           [ 0.01386 , -0.02118 ,  0.01807 , ..., -0.01463 ,  0.04373 ,
            -0.011856],
           [-0.02365 , -0.008675, -0.000806, ...,  0.01537 ,  0.01438 ,
            -0.02342 ],
           [-0.01289 , -0.007313, -0.0121  , ..., -0.00561 ,  0.03787 ,
             0.006016]], dtype=float16)</pre>



```python
embedded_query_dense_vecs
```




<pre class="custom">array([[-0.02156 , -0.01993 , -0.01706 , ..., -0.01994 ,  0.0318  ,
            -0.003395]], dtype=float16)</pre>



```python
# docs embedding dimension
embedded_documents_dense_vecs.shape
```




<pre class="custom">(5, 1024)</pre>



```python
# query embedding dimension
embedded_query_dense_vecs.shape
```




<pre class="custom">(1, 1024)</pre>



```python
# Calculating Similarity Between Documents and Query
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(
    embedded_query_dense_vecs, embedded_documents_dense_vecs
)
most_similar_idx = similarities.argmax()

# Display the Most Similar Document
print(f"Question: {q}")
print(f"Most similar document: {docs[most_similar_idx]}")
```

<pre class="custom">Question: Please tell me more about LangChain.
    Most similar document: LangChain simplifies the process of building applications with large language models.
</pre>

```python
from FlagEmbedding import BGEM3FlagModel

model_name = "BAAI/bge-m3"

bge_embeddings = BGEM3FlagModel(
    model_name,
    use_fp16=True,  # Enabling fp16 improves encoding speed with minimal precision trade-off.
)

# Encode documents with specified parameters
embedded_documents_dense_vecs_default = bge_embeddings.encode(
    sentences=docs, return_dense=True
)["dense_vecs"]

# Query Encoding
embedded_query_dense_vecs_default = bge_embeddings.encode(
    sentences=[q], return_dense=True
)["dense_vecs"]
```


<pre class="custom">Fetching 30 files:   0%|          | 0/30 [00:00<?, ?it/s]</pre>


    You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    

```python
# Calculating Similarity Between Documents and Query
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(
    embedded_query_dense_vecs_default, embedded_documents_dense_vecs_default
)
most_similar_idx = similarities.argmax()

# Display the Most Similar Document
print(f"Question: {q}")
print(f"Most similar document: {docs[most_similar_idx]}")
```

<pre class="custom">Question: Please tell me more about LangChain.
    Most similar document: LangChain simplifies the process of building applications with large language models.
</pre>

### 2️⃣ **Sparse(Lexical) Vector Embedding Example**

**Sparse Embedding (Lexical Weight)**
- **Sparse embedding** is an embedding method that utilizes **high-dimensional vectors where most values are zero.**
- The approach using **lexical weight** generates embeddings by considering the **importance of each word.**

**How It Works**  
1. Calculate the **lexical weight** for each word. Techniques like **TF-IDF** or **BM25** can be used.
2. For each word in a document or query, assign a value to the corresponding dimension of the **sparse vector** based on its lexical weight.
3. As a result, documents and queries are represented as **high-dimensional vectors where most values are zero.** 

**Advantages**  
- Directly reflects the **importance of words.** 
- Enables **precise matching** of specific words or phrases.  
- **Faster computation** compared to dense embeddings.  

```python
from FlagEmbedding import BGEM3FlagModel

model_name = "BAAI/bge-m3"

bge_embeddings = BGEM3FlagModel(
    model_name,
    use_fp16=True,  # Enabling fp16 improves encoding speed with minimal precision trade-off.
)

# Encode documents with specified parameters
embedded_documents_sparse_vecs = bge_embeddings.encode(
    sentences=docs, return_sparse=True
)

# Query Encoding
embedded_query_sparse_vecs = bge_embeddings.encode(sentences=[q], return_sparse=True)
```


<pre class="custom">Fetching 30 files:   0%|          | 0/30 [00:00<?, ?it/s]</pre>


    You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    

```python
lexical_scores_0 = bge_embeddings.compute_lexical_matching_score(
    embedded_query_sparse_vecs["lexical_weights"][0],
    embedded_documents_sparse_vecs["lexical_weights"][0],
)

lexical_scores_1 = bge_embeddings.compute_lexical_matching_score(
    embedded_query_sparse_vecs["lexical_weights"][0],
    embedded_documents_sparse_vecs["lexical_weights"][1],
)

lexical_scores_2 = bge_embeddings.compute_lexical_matching_score(
    embedded_query_sparse_vecs["lexical_weights"][0],
    embedded_documents_sparse_vecs["lexical_weights"][2],
)

lexical_scores_3 = bge_embeddings.compute_lexical_matching_score(
    embedded_query_sparse_vecs["lexical_weights"][0],
    embedded_documents_sparse_vecs["lexical_weights"][3],
)

lexical_scores_4 = bge_embeddings.compute_lexical_matching_score(
    embedded_query_sparse_vecs["lexical_weights"][0],
    embedded_documents_sparse_vecs["lexical_weights"][4],
)
```

```python
print(f"question: {q}")
print("====================")
for i, doc in enumerate(docs):
    print(doc, f": {eval(f'lexical_scores_{i}')}")
```

<pre class="custom">question: Please tell me more about LangChain.
    ====================
    Hi, nice to meet you. : 0.0118865966796875
    LangChain simplifies the process of building applications with large language models. : 0.2313995361328125
    The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively. : 0.18797683715820312
    LangChain simplifies the process of building applications with large-scale language models. : 0.2268962860107422
    Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses. : 0.002368927001953125
</pre>

### 3️⃣ **Multi-Vector(ColBERT) Embedding Example**

**ColBERT** (Contextualized Late Interaction over BERT) is an efficient approach for **document retrieval.** 
- This method uses a **multi-vector strategy** to represent both documents and queries with multiple vectors.  

**How It Works**  
1. Generate a **separate vector** for each **token in a document,** resulting in multiple vectors per document.  
2. Similarly, generate a **separate vector** for each **token in a query.** 
3. During retrieval, calculate the **similarity** between each query token vector and all document token vectors.  
4. Aggregate these similarity scores to produce a **final retrieval score.**  

**Advantages**  
- Enables **fine-grained token-level matching.**  
- Captures **contextual embeddings** effectively.  
- Performs efficiently even with **long documents.** 

```python
from FlagEmbedding import BGEM3FlagModel

model_name = "BAAI/bge-m3"

bge_embeddings = BGEM3FlagModel(
    model_name,
    use_fp16=True,  # Enabling fp16 improves encoding speed with minimal precision trade-off.
)

# Encode documents with specified parameters
embedded_documents_colbert_vecs = bge_embeddings.encode(
    sentences=docs, return_colbert_vecs=True
)

# Query Encoding
embedded_query_colbert_vecs = bge_embeddings.encode(
    sentences=[q], return_colbert_vecs=True
)
```


<pre class="custom">Fetching 30 files:   0%|          | 0/30 [00:00<?, ?it/s]</pre>


    You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    

```python
colbert_scores_0 = bge_embeddings.colbert_score(
    embedded_query_colbert_vecs["colbert_vecs"][0],
    embedded_documents_colbert_vecs["colbert_vecs"][0],
)

colbert_scores_1 = bge_embeddings.colbert_score(
    embedded_query_colbert_vecs["colbert_vecs"][0],
    embedded_documents_colbert_vecs["colbert_vecs"][1],
)

colbert_scores_2 = bge_embeddings.colbert_score(
    embedded_query_colbert_vecs["colbert_vecs"][0],
    embedded_documents_colbert_vecs["colbert_vecs"][2],
)

colbert_scores_3 = bge_embeddings.colbert_score(
    embedded_query_colbert_vecs["colbert_vecs"][0],
    embedded_documents_colbert_vecs["colbert_vecs"][3],
)

colbert_scores_4 = bge_embeddings.colbert_score(
    embedded_query_colbert_vecs["colbert_vecs"][0],
    embedded_documents_colbert_vecs["colbert_vecs"][4],
)
```

```python
print(f"question: {q}")
print("====================")
for i, doc in enumerate(docs):
    print(doc, f": {eval(f'colbert_scores_{i}')}")
```

<pre class="custom">question: Please tell me more about LangChain.
    ====================
    Hi, nice to meet you. : 0.509117841720581
    LangChain simplifies the process of building applications with large language models. : 0.7039894461631775
    The LangChain English tutorial is structured based on LangChain's official documentation, cookbook, and various practical examples to help users utilize LangChain more easily and effectively. : 0.6632840037345886
    LangChain simplifies the process of building applications with large-scale language models. : 0.7057777643203735
    Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses. : 0.38082367181777954
</pre>

### 💡 **Advantages of FlagEmbedding**  

- **Diverse Embedding Options:** Supports the **Dense,** **Lexical,** and **Multi-Vector** approaches.  
- **High-Performance Models:** Utilizes powerful pre-trained models like **BGE.**  
- **Flexibility:** Choose the optimal embedding method based on your **use case.**  
- **Scalability:** Capable of performing embeddings on **large-scale datasets.**  

---

### ⚠️ **Considerations**  

- **Model Size:** Some models may require **significant storage capacity.**  
- **Resource Requirements:** **GPU usage is recommended** for large-scale vector computations.  
- **Configuration Needs:** Optimal performance may require **parameter tuning.**   

---

### 📊 **FlagEmbedding Vector Comparison**  

| **Embedding Type** | **Strengths**         | **Use Cases**              |
|---------------------|-----------------------|----------------------------|
| **Dense Vector**   | Emphasizes semantic similarity | Semantic search, document matching |
| **Lexical Vector** | Precise word matching        | Keyword search, exact matches      |
| **Multi-Vector**   | Captures complex meanings    | Long document analysis, topic classification |

---

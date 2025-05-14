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

 # Text Splitting Methods in NLP

- Author: [Ilgyun Jeong](https://github.com/johnny9210)
- Peer Review : [JoonHo Kim](https://github.com/jhboyo), [Sunyoung Park (architectyou)](https://github.com/Architectyou)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/03-TextSplittingMethods.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/03-TextSplittingMethods.ipynb)

## Overview
Text splitting is a crucial preprocessing step in Natural Language Processing (NLP). This tutorial covers various text splitting methods and tools, exploring their advantages, disadvantages, and appropriate use cases.

Main approaches to text splitting:

1. **Token-based Splitting**
   - Tiktoken: OpenAI's high-performance BPE tokenizer
   - Hugging Face tokenizers: Tokenizers for various pre-trained models
   
2. **Sentence-based Splitting**
   - SentenceTransformers: Splits text while maintaining semantic coherence
   - NLTK: Natural language processing based sentence and word splitting
   - spaCy: Text splitting utilizing advanced language processing capabilities

3. **Language-specific Tools**
   - KoNLPy: Specialized splitting tool for Korean text processing

Each tool has its unique characteristics and advantages:
- ```Tiktoken``` offers fast processing speed and compatibility with OpenAI models
- ```SentenceTransformers``` provides meaning-based sentence splitting
- ```NLTK``` and ```spaCy``` implement linguistic rule-based splitting
- ```KoNLPy``` specializes in Korean morphological analysis and splitting

Through this tutorial, you will understand the characteristics of each tool and learn to choose the most suitable text splitting method for your project.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Basic Usage of Tiktoken](#basic-usage-of-tiktoken)
- [Basic Usage of TokenTextSplitter](#basic-usage-of-tokentextsplitter)
- [Basic Usage of spaCy](#basic-usage-of-spaCy)
- [Basic Usage of SentenceTransformers](#basic-usage-of-sentencetransformers)
- [Basic Usage of NLTK](#basic-usage-of-NLTK)
- [Basic Usage of KoNLPy](#basic-usage-of-KoNLPy)
- [Basic Usage of Hugging Face tokenizers](#basic-usage-of-Hugging-Face-tokenizers)

### References

- [LangChain: How to split text by tokens](https://python.langchain.com/docs/how_to/split_by_token/)
- [Langchain TokenTextSplitter](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TokenTextSplitter.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [langchain-opentutorial](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_text_splitters",
        "tiktoken",
        "spacy",
        "sentence-transformers",
        "nltk",
        "konlpy",
    ],
    verbose=False,
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
        "LANGCHAIN_PROJECT": "TokenTextSplitter",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Basic Usage of ```tiktoken```

tiktoken is a fast BPE tokenizer created by OpenAI.

- Open the file ./data/appendix-keywords.txt and read its contents.
- Store the read content in the file variable.

```python
# Open the file data/appendix-keywords.txt and create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = (
        f.read()
    )  # Read the contents of the file and store them in the file variable.
```

Print a portion of the content read from the file.

```python
# Print a portion of the content read from the file.
print(file[:500])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to unders
</pre>

Use the CharacterTextSplitter to split the text.

- Initialize the text splitter using the from_tiktoken_encoder method, which is based on the Tiktoken encoder.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    # Set the chunk size to 300.
    chunk_size=300,
    # Ensure there is no overlap between chunks.
    chunk_overlap=50,
)
# Split the file text into chunks.
texts = text_splitter.split_text(file)
```

Print the number of divided chunks.

```python
print(len(texts))  # Output the number of divided chunks.
```

<pre class="custom">10
</pre>

Print the first element of the texts list.

```python
# Print the first element of the texts list.
print(texts[0])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to understand and process the text.
    Example: Represent the word “apple” as a vector such as [0.65, -0.23, 0.17].
    Related keywords: natural language processing, vectorization, deep learning
    
    Token
    
    Definition: A token is a breakup of text into smaller units. These can typically be words, sentences, or phrases.
    Example: Split the sentence “I am going to school” into “I am”, “to school”, and “going”.
    Associated keywords: tokenization, natural language processing, parsing
    
    Tokenizer
</pre>

Reference
- When using CharacterTextSplitter.from_tiktoken_encoder, the text is split solely by CharacterTextSplitter, and the Tiktoken tokenizer is only used to measure and merge the divided text. (This means that the split text might exceed the chunk size as measured by the Tiktoken tokenizer.)
- When using RecursiveCharacterTextSplitter.from_tiktoken_encoder, the divided text is ensured not to exceed the chunk size allowed by the language model. If a split text exceeds this size, it is recursively divided. Additionally, you can directly load the Tiktoken splitter, which guarantees that each split is smaller than the chunk size.

## Basic Usage of ```TokenTextSplitter```

Use the TokenTextSplitter class to split the text into token-based chunks.

```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=200,  # Set the chunk size to 10.
    chunk_overlap=50,  # Set the overlap between chunks to 0.
)

# Split the state_of_the_union text into chunks.
texts = text_splitter.split_text(file)
print(texts[0])  # Print the first chunk of the divided text.
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to understand and process the text.
    Example: Represent the word “apple” as a vector such as [0.65, -0.23, 0.17].
    Related keywords: natural language processing, vectorization, deep learning
    
    Token
    
    Definition: A token is a breakup of text into smaller units. These can typically be words, sentences, or phrases.
    Example: Split the sentence “I am going to school
</pre>

## Basic Usage of ```spaCy```

spaCy is an open-source software library for advanced natural language processing, written in Python and Cython programming languages.

Another alternative to NLTK is using the spaCy tokenizer.

1. How the text is divided: The text is split using the spaCy tokenizer.
2. How the chunk size is measured: It is measured by the number of characters.

Download the en_core_web_sm model.

```python
!python -m spacy download en_core_web_sm --quiet
```

<pre class="custom">✔ Download and installation successful
    You can now load the package via spacy.load('en_core_web_sm')
</pre>

Open the appendix-keywords.txt file and read its contents.

```python
# Open the file data/appendix-keywords.txt and create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = (
        f.read()
    )  # Read the contents of the file and store them in the file variable.
```

Verify by printing a portion of the content.

```python
# Print a portion of the content read from the file.
print(file[:350])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embed
</pre>

Create a text splitter using the SpacyTextSplitter class.


```python
import warnings
from langchain_text_splitters import SpacyTextSplitter

# Ignore  warning messages.
warnings.filterwarnings("ignore")

# Create the SpacyTextSplitter.
text_splitter = SpacyTextSplitter(
    chunk_size=200,  # Set the chunk size to 200.
    chunk_overlap=50,  # Set the overlap between chunks to 50.
)
```

Use the **split_text** method of the **text_splitter** object to split the ```file``` text.

```python
# Split the file text using the text_splitter.
texts = text_splitter.split_text(file)
print(texts[0])  # Print the first element of the split text.
```

<pre class="custom">Created a chunk of size 215, which is longer than the specified 200
    Created a chunk of size 241, which is longer than the specified 200
    Created a chunk of size 225, which is longer than the specified 200
    Created a chunk of size 211, which is longer than the specified 200
    Created a chunk of size 231, which is longer than the specified 200
    Created a chunk of size 230, which is longer than the specified 200
    Created a chunk of size 219, which is longer than the specified 200
    Created a chunk of size 214, which is longer than the specified 200
    Created a chunk of size 215, which is longer than the specified 200
    Created a chunk of size 203, which is longer than the specified 200
    Created a chunk of size 211, which is longer than the specified 200
    Created a chunk of size 218, which is longer than the specified 200
    Created a chunk of size 230, which is longer than the specified 200
</pre>

    Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format.
    
    It is used for search, classification, and other data analysis tasks.
    

## Basic Usage of ```SentenceTransformers```

SentenceTransformersTokenTextSplitter is a text splitter specialized for sentence-transformer models.

Its default behavior is to split text into chunks that fit within the token window of the sentence-transformer model being used.


```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Create a sentence splitter and set the overlap between chunks to 50.
splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=50)
```

Check the sample text.

```python
# Open the data/appendix-keywords.txt file and create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # Read the file content and store it in the variable file.

# Print a portion of the content read from the file.
print(file[:350])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embed
</pre>

The following code counts the number of tokens in the text stored in the ```file``` variable, excluding the count of start and stop tokens, and prints the result.

```python
count_start_and_stop_tokens = 2  # Set the number of start and stop tokens to 2.

# Subtract the count of start and stop tokens from the total number of tokens in the text.
text_token_count = splitter.count_tokens(text=file) - count_start_and_stop_tokens
print(text_token_count)  # Print the calculated number of tokens in the text.
```

<pre class="custom">2231
</pre>

Use the ```splitter.split_text()``` function to split the text stored in the ```text_to_split``` variable into chunks.

```python
text_chunks = splitter.split_text(text=file)  # Split the text into chunks.
```

Split the text into chunks.


```python
# Print the 0th chunk.
print(text_chunks[1])  # Print the second chunk from the divided text chunks.
```

<pre class="custom">a database for quick access. related keywords : embedding, database, vectorization, vectorization sql definition : sql ( structured query language ) is a programming language for managing data in a database. you can query, modify, insert, delete, and more data. example : select * from users where age > 18 ; looks up information about users who are 18 years old or older. associated keywords : database, query, data management, data management csv definition : csv ( comma - separated values ) is a file format for storing data, where each data value is separated by a comma. it is used for simple storage and exchange of tabular data. example : a csv file with the headers name, age, and occupation might contain data such as hong gil - dong, 30, developer. related keywords : data format, file processing, data exchange json definition : json ( javascript object notation ) is a lightweight data interchange format that represents data objects using text that is readable to both humans and machines. example : { “ name ” : “ honggildong ”, ‘ age ’ : 30, “ occupation ” : “ developer " } is data in json format. related keywords : data exchange, web development, apis transformer definition : transformers are a type of deep learning model used in natural language processing, mainly for translation, summarization, text generation, etc. it is based on the attention mechanism. example : google translator uses transformer models to perform translations between different languages. related keywords : deep learning, natural language processing, attention huggingface definition : huggingface is a library that provides a variety of pre - trained models and tools for natural language processing. it helps researchers and developers to easily perform nlp tasks. example : you can use huggingface ' s transformers library to perform tasks such as sentiment analysis, text generation
</pre>

## Basic Usage of ```NLTK```

The Natural Language Toolkit (NLTK) is a library and a collection of programs for English natural language processing (NLP), written in the Python programming language.

Instead of simply splitting by "\n\n", NLTK can be used to split text based on NLTK tokenizers.
1. Text splitting method: The text is split using the NLTK tokenizer.
2.	Chunk size measurement: The size is measured by the number of characters.
3.	nltk (Natural Language Toolkit) is a Python library for natural language processing.
4.	It supports various NLP tasks such as text preprocessing, tokenization, morphological analysis, and part-of-speech tagging.

Before using NLTK, you need to run nltk.download('punkt_tab').

The reason for running nltk.download('punkt_tab') is to allow the NLTK (Natural Language Toolkit) library to download the necessary data files required for tokenizing text.

Specifically, punkt_tab is a tokenization model capable of splitting text into words or sentences for multiple languages, including English.

```python
import nltk

nltk.download("punkt_tab")
```

<pre class="custom">[nltk_data] Downloading package punkt_tab to
    [nltk_data]     /Users/ilgyun/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt_tab.zip.
</pre>




    True



Verify the sample text.


```python
# Open the data/appendix-keywords.txt file and create a file object named f.
with open("./data/appendix-keywords_kr.txt") as f:
    file = (
        f.read()
    )  # Read the contents of the file and store them in the file variable.

# Print a portion of the content read from the file.
print(file[:350])
```

<pre class="custom">Semantic Search
    
    정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.
    예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다.
    연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝
    
    Embedding
    
    정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
    예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.
    연관키워드: 자연어 처
</pre>

- Create a text splitter using the NLTKTextSplitter class.
- Set the chunk_size parameter to 1000 to split the text into chunks of up to 1000 characters.

```python
from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(
    chunk_size=200,  # Set the chunk size to 200.
    chunk_overlap=50,  # Set the overlap between chunks to 50.
)
```

Use the split_text method of the text_splitter object to split the ```file``` text.

```python
# Split the file text using the text_splitter.
texts = text_splitter.split_text(file)
print(texts[0])  # Print the first element of the split text.
```

<pre class="custom">Semantic Search
    
    정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.
    
    예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다.
</pre>

## Basic Usage of ```KoNLPy```

KoNLPy (Korean NLP in Python) is a Python package for Korean Natural Language Processing (NLP).

Tokenization involves the process of dividing text into smaller, more manageable units called tokens. 
These tokens often represent meaningful elements such as words, phrases, symbols, or other components crucial for further processing and analysis.

In languages like English, tokenization typically involves separating words based on spaces and punctuation.
The effectiveness of tokenization largely depends on the tokenizer's understanding of the language structure, ensuring the generation of meaningful tokens.

Tokenizers designed for English lack the ability to comprehend the unique semantic structure of other languages, such as Korean, and therefore cannot be effectively used for Korean text processing.

### Korean Tokenization Using ```KoNLPy```'s ```Kkma``` Analyzer

For Korean text, KoNLPy includes a morphological analyzer called Kkma (Korean Knowledge Morpheme Analyzer).

Kkma provides detailed morphological analysis for Korean text.
It breaks sentences into words and further decomposes words into their morphemes while identifying the part of speech for each token.
It can also split text blocks into individual sentences, which is particularly useful for processing lengthy texts.

### Considerations When Using ```Kkma```
Kkma is known for its detailed analysis. However, this precision can affect processing speed.
Therefore, Kkma is best suited for applications that prioritize analytical depth over rapid text processing.
- KoNLPy is a Python package for Korean Natural Language Processing, offering features such as morphological analysis, part-of-speech tagging, and syntactic parsing.

Verify the sample text.

```python
# Open the data/appendix-keywords.txt file and create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # Read the file content and store it in the variable file.

# Print a portion of the content read from the file.
print(file[:350])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embed
</pre>

This is an example of splitting Korean text using KonlpyTextSplitter.

```python
from langchain_text_splitters import KonlpyTextSplitter

# Create a text splitter object using KonlpyTextSplitter.
text_splitter = KonlpyTextSplitter()
```

Use the text_splitter to split the ```file``` content into sentences.

```python
texts = text_splitter.split_text(file)  # Split the file content into sentences.
print(texts[0])  # Print the first sentence from the divided text.
```

<pre class="custom">Semantic Search Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks. Example: Vectors of word embeddings can be stored in a database for quick access. Related keywords: embedding, database, vectorization, vectorization Embedding Definition: Embedding is the process of converting textual data, such as words or sentences, into a low-dimensional, continuous vector. This allows computers to understand and process the text. Example: Represent the word “apple” as a vector such as [0.65, -0.23, 0.17]. Related keywords: natural language processing, vectorization, deep learning Token Definition: A token is a breakup of text into smaller units. These can typically be words, sentences, or phrases. Example: Split the sentence “I am going to school” into “I am”, “to school”, and “going”. Associated keywords: tokenization, natural language processing, parsing Tokenizer Definition: A tokenizer is a tool that splits text data into tokens. It is used to preprocess data in natural language processing. Example: Split the sentence “I love programming.” into [ “I”, “love”, “programming”, “.”]. Associated keywords: tokenization, natural language processing, parsing VectorStore Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks. Example: Vectors of word embeddings can be stored in a database for quick access. Related keywords: embedding, database, vectorization, vectorization SQL Definition: SQL(Structured Query Language) is a programming language for managing data in a database. You can query, modify, insert, delete, and more data. Example: SELECT * FROM users WHERE age > 18; looks up information about users who are 18 years old or older. Associated keywords: database, query, data management, data management CSV Definition: CSV(Comma-Separated Values) is a file format for storing data, where each data value is separated by a comma. It is used for simple storage and exchange of tabular data. Example: A CSV file with the headers Name, Age, and Occupation might contain data such as Hong Gil-dong, 30, Developer. Related keywords: data format, file processing, data exchange JSON Definition: JSON(JavaScript Object Notation) is a lightweight data interchange format that represents data objects using text that is readable to both humans and machines. Example: { “Name”: “HongGilDong”, ‘Age’: 30, “Occupation”: “Developer"} is data in JSON format. Related keywords: data exchange, web development, APIs Transformer Definition: Transformers are a type of deep learning model used in natural language processing, mainly for translation, summarization, text generation, etc. It is based on the Attention mechanism. Example: Google Translator uses transformer models to perform translations between different languages. Related keywords: Deep learning, Natural language processing, Attention HuggingFace Definition: HuggingFace is a library that provides a variety of pre-trained models and tools for natural language processing. It helps researchers and developers to easily perform NLP tasks. Example: You can use HuggingFace's Transformers library to perform tasks such as sentiment analysis, text generation, and more. Related keywords: natural language processing, deep learning, libraries Digital Transformation Definition: Digital transformation is the process of leveraging technology to transform a company's services, culture, and operations. It focuses on improving business models and increasing competitiveness through digital technologies. Example: When a company adopts cloud computing to revolutionize data storage and processing, it's an example of digital transformation. Related keywords: transformation, technology, business model Crawling Definition: Crawling is the process of visiting web pages in an automated way to collect data. It is often used for search engine optimization or data analysis. Example: When the Google search engine visits websites on the internet to collect and index content, it is crawling. Related keywords: data collection, web scraping, search engine Word2Vec Definition: Word2Vec is a natural language processing technique that maps words to a vector space to represent semantic relationships between words. It generates vectors based on the contextual similarity of words. Example: In a Word2Vec model, “king” and “queen” are represented as vectors in close proximity to each other. Related keywords: natural language processing, embeddings, semantic similarity LLM (Large Language Model) Definition: LLM refers to large-scale language models trained on large amounts of textual data. These models are used for a variety of natural language understanding and generation tasks. Example: OpenAI's GPT series is a typical large-scale language model. Related keywords: natural language processing, deep learning, text generation FAISS (Facebook AI Similarity Search) Definition: FAISS is a fast similarity search library developed by Facebook, specifically designed to efficiently search for similar vectors in large vector sets. Example: FAISS can be used to quickly find similar images among millions of image vectors. Related keywords: vector search, machine learning, database optimization Open Source Definition: Open source refers to software whose source code is publicly available and can be freely used, modified, and distributed by anyone. This plays an important role in fostering collaboration and innovation. Example: The Linux operating system is a prominent open source project. Related keywords: software development, community, technical collaboration Structured Data Definition: Structured data is data that is organized according to a set format or schema. It can be easily searched and analyzed in databases, spreadsheets, etc. Example: A table of customer information stored in a relational database is an example of structured data. Related keywords: database, data analysis, data modeling, data modeling Parser Definition: A parser is a tool that analyzes given data (strings, files, etc.) and converts it into a structured form. It is used for parsing programming languages or processing file data. Example: Parsing an HTML document to generate the DOM structure of a web page is an example of parsing. Associated keywords: parsing, compiler, data processing TF-IDF (Term Frequency-Inverse Document Frequency) Definition: TF-IDF is a statistical measure used to evaluate the importance of a word within a document. It takes into account the frequency of the word within the document and the sparsity of the word in the entire document set. Example: A word that occurs infrequently in many documents has a high TF-IDF value. Related keywords: natural language processing, information retrieval, data mining Deep Learning Definition: Deep learning is a branch of machine learning that uses artificial neural networks to solve complex problems. It focuses on learning high-level representations from data. Examples: Deep learning models are used in image recognition, speech recognition, natural language processing, and more. Related keywords: Artificial neural networks, machine learning, data analytics Schema Definition: A schema defines the structure of a database or file, providing a blueprint for how data is stored and organized. Example: The table schema of a relational database defines column names, data types, key constraints, and more. Related keywords: database, data modeling, data management, data management DataFrame Definition: A DataFrame is a table-like data structure with rows and columns, primarily used for data analysis and processing. Example: In the Pandas library, a DataFrame can have columns of different data types and facilitates data manipulation and analysis. Related keywords: data analytics, Pandas, data processing Attention mechanisms Definition: Attention mechanisms are techniques that allow deep learning to pay more “attention” to important information. They are often used with sequential data (e .g., text, time series data). Example: In a translation model, the Attention mechanism focuses more on the important parts of the input sentence to produce an accurate translation. Associated keywords: deep learning, natural language processing, sequence modeling Pandas Definition: Pandas is a library that provides data analysis and manipulation tools for the Python programming language. It enables you to perform data analysis tasks efficiently. Example: You can use Pandas to read CSV files, cleanse data, and perform various analyses. Related keywords: Data analysis, Python, Data processing GPT (Generative Pretrained Transformer) Definition: GPTs are generative language models pre-trained on large datasets and utilized for a variety of text-based tasks. It can generate natural language based on input text. Example: A chatbot that generates detailed answers to user-supplied questions can use a GPT model. Related keywords: natural language processing, text generation, deep learning InstructGPT Definition: InstructGPT is a GPT model optimized to perform specific tasks based on user instructions. The model is designed to produce more accurate and relevant results. Example: If a user provides a specific instruction, such as “draft an email,” InstructGPT will create an email based on relevant content. Related keywords: artificial intelligence, natural language understanding, command-based processing Keyword Search Definition: Keyword search is the process of finding information based on keywords entered by a user. It is the primary search method used by most search engines and database systems. Example: If a user searches for “coffee shops Seoul”, a list of relevant coffee shops is returned. Related keywords: search engine, data search, information search Page Rank Definition: PageRank is an algorithm that evaluates the importance of a web page and is primarily used to determine its ranking in search engine results. It is evaluated by analyzing the link structure between web pages. Example: The Google search engine uses the PageRank algorithm to rank search results. Related keywords: search engine optimization, web analytics, link analysis Data Mining Definition: Data mining is the process of uncovering useful information from large amounts of data. It leverages techniques such as statistics, machine learning, and pattern recognition. Example: When a retailer analyzes customer purchase data to create a sales strategy, it's an example of data mining. Related keywords: big data, pattern recognition, predictive analytics Multimodal (Multimodal) Definition: Multimodal is a technique that combines and processes multiple modes of data (e .g., text, images, sound, etc.). It is used to extract or predict richer and more accurate information through the interaction between different forms of data. Example: A system that analyzes images and descriptive text together to perform more accurate image classification is an example of multimodal technology. Related keywords: data fusion, artificial intelligence, deep learning
</pre>

## Basic Usage of ```Hugging Face tokenizers```

Hugging Face provides various tokenizers.

This code demonstrates calculating the token length of a text using one of Hugging Face's tokenizers, GPT2TokenizerFast.

The text splitting approach is as follows:

- The text is split at the character level.

The chunk size measurement is determined as follows:

- It is based on the number of tokens calculated by the Hugging Face tokenizers.
- A tokenizer object is created using the GPT2TokenizerFast class.
- from_pretrained method is called to load the pre-trained gpt2 tokenizer model.

```python
from transformers import GPT2TokenizerFast

# Load the GPT-2 tokenizer.
hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
```

```python
# Open the data/appendix-keywords.txt file and create a file object named f.
with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # Read the file content and store it in the variable file.

# Print a portion of the content read from the file.
print(file[:350])
```

<pre class="custom">Semantic Search
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    Embedding
    
    Definition: Embed
</pre>

from_huggingface_tokenizer method is used to initialize a text splitter with a Hugging Face tokenizers (tokenizer).

```python
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    # Use the Hugging Face tokenizers to create a CharacterTextSplitter object.
    hf_tokenizer,
    chunk_size=300,
    chunk_overlap=50,
)
# Split the file text into chunks.
texts = text_splitter.split_text(file)
```

Check the split result of the first element

```python
print(texts[1])  # Print the first element of the texts list.
```

<pre class="custom">Tokenizer
    
    Definition: A tokenizer is a tool that splits text data into tokens. It is used to preprocess data in natural language processing.
    Example: Split the sentence “I love programming.” into [“I”, “love”, “programming”, “.”].
    Associated keywords: tokenization, natural language processing, parsing
    
    VectorStore
    
    Definition: A vector store is a system that stores data converted to vector format. It is used for search, classification, and other data analysis tasks.
    Example: Vectors of word embeddings can be stored in a database for quick access.
    Related keywords: embedding, database, vectorization, vectorization
    
    SQL
    
    Definition: SQL(Structured Query Language) is a programming language for managing data in a database. You can query, modify, insert, delete, and more data.
    Example: SELECT * FROM users WHERE age > 18; looks up information about users who are 18 years old or older.
    Associated keywords: database, query, data management, data management
    
    CSV
</pre>

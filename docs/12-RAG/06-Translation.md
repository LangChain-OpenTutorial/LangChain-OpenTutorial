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

# Translation

- Author: [Wonyoung Lee](https://github.com/BaBetterB)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BaBetterB/LangChain-OpenTutorial/blob/main/12-RAG/06-Translation.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)


## Overview

This tutorial compares two approaches to translating Chinese text into English using LangChain.

The first approach utilizes a single LLM (e.g. GPT-4) to generate a straightforward translation. The second approach employs Retrieval-Augmented Generation (RAG), which enhances translation accuracy by retrieving relevant documents.

The tutorial evaluates the translation accuracy and performance of each method, helping users choose the most suitable approach for their needs.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Translation using LLM](#translation-using-llm)
- [Translation using RAG](#translation-using-rag)
- [Evaluation of translation results](#evaluation-of-translation-results)


### References

- [LangChain OpenAIEmbeddings API](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [NLTK](https://www.nltk.org/)
- [TER](https://machinetranslate.org/ter)
- [BERTScore](https://arxiv.org/abs/1904.09675)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Chinese Source](https://cn.chinadaily.com.cn/)



----

 


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [ ```langchain-opentutorial``` ](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

Load sample text and output the content.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

<pre class="custom">
    [notice] A new release of pip is available: 24.2 -> 25.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package


package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "load_dotenv",
        "langchain_openai",
        "faiss-cpu",
        "sacrebleu",
        "bert_score",
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
        "LANGCHAIN_PROJECT": "Translation",  # title
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
# Configuration File for Managing API Keys as Environment Variables
from dotenv import load_dotenv

# Load API Key Information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Translation using LLM

Translation using LLM refers to using a large language model (LLM), such as GPT-4, to translate text from one language to another. 
The model processes the input text and generates a direct translation based on its pre-trained knowledge. This approach is simple, fast, and effective.



```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator.",
        ),
        (
            "human",
            "Please translate the following Chinese document into natural and accurate English."
            "Consider the context and vocabulary to ensure smooth and fluent sentences.:.\n\n"
            "**Chinese Original Text:** {chinese_text}\n\n**English Translation:**",
        ),
    ]
)

translation_chain = RunnableSequence(prompt, llm)

chinese_text = "人工智能正在改变世界，各国都在加紧研究如何利用这一技术提高生产力。"

response = translation_chain.invoke({"chinese_text": chinese_text})

print("Chinese_text:", chinese_text)
print("Translation:", response.content)
```

<pre class="custom">Chinese_text: 人工智能正在改变世界，各国都在加紧研究如何利用这一技术提高生产力。
    Translation: Artificial intelligence is transforming the world, and countries are intensifying their research on how to leverage this technology to enhance productivity.
</pre>

## Translation using RAG 

Translation using RAG (Retrieval-Augmented Generation) enhances translation accuracy by combining a pre-trained LLM with a retrieval mechanism. This approach first retrieves relevant documents or data related to the input text and then utilizes this additional context to generate a more precise and contextually accurate translation.


### Simple Search Implementation Using FAISS

In this implementation, we use a vector database to store and retrieve embedded representations of entire sentences. Instead of relying solely on predefined knowledge in the LLM, our approach allows the model to retrieve semantically relevant sentences from the vector database, improving the translation's accuracy and fluency.

**FAISS (Facebook AI Similarity Search)**

FAISS is a library developed by Facebook AI for efficient similarity search and clustering of dense vectors. It is widely used for approximate nearest neighbor (ANN) search in large-scale datasets.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(model="gpt-4o-mini")

file_path = "data/news_cn.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"file not found!!: {file_path}")

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()


# Vectorizing Sentences Individually
sentences = []
for doc in docs:
    text = doc.page_content
    sentence_list = text.split("。")  # Splitting Chinese sentences based on '。'
    sentences.extend(
        [sentence.strip() for sentence in sentence_list if sentence.strip()]
    )


# Store sentences in the FAISS vector database
vector_store = FAISS.from_texts(sentences, embedding=embeddings)

# Search vectors using keywords "人工智能"
search_results = vector_store.similarity_search("人工智能", k=3)

# check result
print("Search result")
for idx, result in enumerate(search_results, start=1):
    print(f"{idx}. {result.page_content}")
```

<pre class="custom">Search result
    1. 当地球员并非专业人士，而是农民、建筑工人、教师和学生，对足球的热爱将他们凝聚在一起
    2. ”卡卡说道
    3. “足球让我们结识新朋友，连接更广阔的世界
</pre>

### Let's compare translation using LLM and translation using RAG.

First, write the necessary functions.

```python
import re
import nltk
from nltk.tokenize import sent_tokenize
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence


llm = ChatOpenAI(model="gpt-4o-mini")


# Download the necessary data for sentence tokenization in NLTK (requires initial setup)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# Document Search Function (Used in RAG)
def retrieve_relevant_docs(query, vector_store, k=3):
    """
    Searches for relevant documents using vector similarity search.

    Parameters:
        query (str): The keyword or sentence to search for.
        vector_store (FAISS): The vector database.
        k (int): The number of top matching documents to retrieve (default: 3).

    Returns:
        list: A list of retrieved document texts.
    """
    search_results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in search_results]


# Translation using only LLM
def translate_with_llm(chinese_text):
    """
    Translates Chinese text into English using GPT-4o-mini.

    Parameters:
        chinese_text (str): The input Chinese sentence to be translated.

    Returns:
        str: The translated English sentence.
    """
    prompt_template_llm = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation expert. Translate the following Chinese sentence into English:",
            ),
            ("user", f'Chinese sentence: "{chinese_text}"'),
            ("user", "Please provide an accurate translation."),
        ]
    )

    translation_chain_llm = RunnableSequence(prompt_template_llm, llm)

    return translation_chain_llm.invoke({"chinese_text": chinese_text})


# RAG-based Translation
def translate_with_rag(chinese_text, vector_store):
    """
    Translates Chinese text into English using the RAG approach.
    It first retrieves relevant documents and then uses them for context-aware translation.

    Parameters:
        chinese_text (str): The input Chinese sentence to be translated.
        vector_store (FAISS): The vector database for document retrieval.

    Returns:
        str: The translated English sentence with contextual improvements.
    """
    retrieved_docs = retrieve_relevant_docs(chinese_text, vector_store)

    # Add retrieved documents as context

    context = "\n".join(retrieved_docs)

    # Construct prompt template (Using RAG)

    prompt_template_rag = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation expert. Below is the Chinese text that needs to be translated into English. Additionally, the following context has been provided from relevant documents that might help you in producing a more accurate and context-aware translation.",
            ),
            ("system", f"Context (Relevant Documents):\n{context}"),
            ("user", f'Chinese sentence: "{chinese_text}"'),
            (
                "user",
                "Please provide a translation that is both accurate and reflects the context from the documents provided.",
            ),
        ]
    )

    translation_chain_rag = RunnableSequence(prompt_template_rag, llm)

    # Request translation using RAG

    return translation_chain_rag.invoke({"chinese_text": chinese_text})


# Load Chinese text from a file and split it into sentences, returning them as a list.
def chinese_text_from_file_loader(path):
    """
    Loads Chinese text from a file and splits it into individual sentences.

    Parameters:
        path (str): File path.

    Returns:
        list: List of Chinese sentences.
    """
    # Load data
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    return split_chinese_sentences_from_docs(docs)


# Split sentences from a list of documents and return them as a list
def split_chinese_sentences_from_docs(docs):
    """
    Extracts sentences from a list of documents.

    Parameters:
        docs (list): List of document objects.

    Returns:
        list: List of extracted sentences.
    """
    sentences = []

    for doc in docs:
        text = doc.page_content
        sentences.extend(split_chinese_sentences(text))

    return sentences


# Use regular expressions to split sentences and punctuation together.
# Then, combine the sentences and punctuation back and return them
def split_chinese_sentences(text):
    """
    Splits Chinese text into sentences based on punctuation marks (。！？).

    Parameters:
        text (str): The input Chinese text.

    Returns:
        list: List of separated sentences.
    """
    # Separate sentences and punctuation,
    sentence_list = re.split(r"([。！？])", text)

    # Combine the sentences and punctuation back to restore them.
    merged_sentences = [
        "".join(x) for x in zip(sentence_list[0::2], sentence_list[1::2])
    ]

    # Remove empty sentences and return the result.
    return [sentence.strip() for sentence in merged_sentences if sentence.strip()]


def count_chinese_sentences(docs):
    """
    Counts the number of sentences in a given Chinese text.

    Parameters:
        docs (str or list): Input text data.

    Returns:
        list: List of split sentences.
    """
    if isinstance(docs, str):
        sentences = split_chinese_sentences(docs)

    print(f"Total number of sentences: {len(sentences)}")
    return sentences


def split_english_sentences_from_docs(docs):
    """
    Splits English text into sentences using NLTK's sentence tokenizer.

    Parameters:
        docs (list): The input English text.

    Returns:
        list: List of separated sentences.
    """
    sentences = []

    for doc in docs:
        text = doc.page_content
        sentences.extend(split_english_sentences(text))
    return sentences


# Use NLTK's sent_tokenize() to split sentences accurately.
# By default, it recognizes periods (.), question marks (?), and exclamation marks (!) to separate sentences.
def split_english_sentences(text):
    """
    Splits English text into sentences using NLTK's sentence tokenizer.

    Parameters:
        text (str): The input English text.

    Returns:
        list: List of separated sentences.
    """
    return sent_tokenize(text)


def count_paragraphs_and_sentences(docs):
    """
    Counts the number of paragraphs and sentences in a given text.

    Parameters:
        docs (str): Input text data.

    Returns:
        int: Total number of sentences.
    """
    if isinstance(docs, str):

        paragraphs = paragraphs = re.split(r"\n\s*\n", docs.strip())
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        sentences = [sent for para in paragraphs for sent in sent_tokenize(para)]

        print(f"Total number of paragraphs : {len(paragraphs)}")
        print(f"Total number of sentences  : {len(sentences)}")
    return len(sentences)
```

<pre class="custom">[nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\herme\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
</pre>

**Use the written functions to perform the comparison.**

```python
# Download the 'punkt_tab' data if it's not available.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

sentences = chinese_text_from_file_loader("data/comparison_cn.txt")


chinese_text = ""


for sentence in sentences:

    chinese_text += sentence


# LLM


llm_translation = translate_with_llm(chinese_text)


# RAG


rag_translation = translate_with_rag(chinese_text, vector_store)


print("\ninput chinese text")
count_chinese_sentences(chinese_text)
print(chinese_text)


print("\nTranslation using LLM")


count_paragraphs_and_sentences(llm_translation.content)


print(llm_translation.content)


print("\nTranslation using RAG")


count_paragraphs_and_sentences(rag_translation.content)


print(rag_translation.content)
```

<pre class="custom">
    input chinese text
    Total number of sentences: 15
    数据领域迎来国家标准。10月8日，国家发改委等部门发布关于印发《国家数据标准体系建设指南》(以下简称《指南》)的通知。为“充分发挥标准在激活数据要素潜能、做强做优做大数字经济等方面的规范和引领作用”，国家发展改革委、国家数据局、中央网信办、工业和信息化部、财政部、国家标准委组织编制了《国家数据标准体系建设指南》。《指南》提出，到2026年底，基本建成国家数据标准体系，围绕数据流通利用基础设施、数据管理、数据服务、训练数据集、公共数据授权运营、数据确权、数据资源定价、企业数据范式交易等方面制修订30项以上数据领域基础通用国家标准，形成一批标准应用示范案例，建成标准验证和应用服务平台，培育一批具备数据管理能力评估、数据评价、数据服务能力评估、公共数据授权运营绩效评估等能力的第三方标准化服务机构。《指南》明确，数据标准体系框架包含基础通用、数据基础设施、数据资源、数据技术、数据流通、融合应用、安全保障等7个部分。数据基础设施方面，标准涉及存算设施中的数据算力设施、数据存储设施，网络设施中的5G网络数据传输、光纤数据传输、卫星互联网数据传输，此外还有流通利用设施。数据流通方面，标准包括数据产品、数据确权、数据资源定价、数据流通交易。融合应用方面，标准涉及工业制造、农业农村、商贸流通、交通运输、金融服务、科技创新、文化旅游(文物)、卫生健康、应急管理、气象服务、城市治理、绿色低碳。安全保障方面，标准涉及数据基础设施安全，数据要素市场安全，数据流通安全。数据资源中的数据治理标准包括数据业务规划、数据质量管理、数据调查盘点、数据资源登记；训练数据集方面的标准包括训练数据集采集处理、训练数据集标注、训练数据集合成。在组织保障方面，将指导建立全国数据标准化技术组织，加快推进急用、急需数据标准制修订工作，强化与有关标准化技术组织、行业、地方及相关社团组织之间的沟通协作、协调联动，以标准化促进数据产业生态建设。同时还将完善标准试点政策配套，搭建数据标准化公共服务平台，开展标准宣贯，选择重点地方、行业先行先试，打造典型示范。探索推动数据产品第三方检验检测，深化数据标准实施评价管理。在人才培养方面，将打造标准配套的数据人才培训课程，形成一批数据标准化专业人才。优化数据国际标准化专家队伍，支持参与国际标准化活动，强化国际交流。
    
    Translation using LLM
    Total number of paragraphs : 1
    Total number of sentences  : 16
    The data field welcomes national standards. On October 8, the National Development and Reform Commission and other departments issued a notice regarding the release of the "Guidelines for the Construction of a National Data Standard System" (hereinafter referred to as the "Guidelines"). To "fully leverage the role of standards in activating the potential of data elements and strengthening, optimizing, and expanding the digital economy," the National Development and Reform Commission, the National Data Bureau, the Central Cyberspace Affairs Commission, the Ministry of Industry and Information Technology, the Ministry of Finance, and the National Standardization Administration organized the preparation of the "Guidelines." The "Guidelines" propose that by the end of 2026, a national data standard system will be basically established, with the development and revision of more than 30 basic general national standards in the areas of data circulation and utilization infrastructure, data management, data services, training datasets, public data authorized operation, data rights confirmation, data resource pricing, and enterprise data paradigm transactions. A number of standard application demonstration cases will be formed, a standard verification and application service platform will be built, and a number of third-party standardized service institutions capable of data management capability assessment, data evaluation, data service capability assessment, and public data authorized operation performance assessment will be cultivated. The "Guidelines" clarify that the framework of the data standard system includes seven parts: basic general standards, data infrastructure, data resources, data technology, data circulation, integrated applications, and security assurance. In terms of data infrastructure, the standards cover data computing facilities and data storage facilities in computing storage infrastructure, as well as 5G network data transmission, optical fiber data transmission, satellite internet data transmission, and circulation utilization facilities in network infrastructure. Regarding data circulation, the standards include data products, data rights confirmation, data resource pricing, and data circulation transactions. In terms of integrated applications, the standards involve industrial manufacturing, agriculture and rural areas, trade and circulation, transportation, financial services, technological innovation, cultural tourism (cultural relics), health, emergency management, meteorological services, urban governance, and green low-carbon initiatives. Concerning security assurance, the standards address data infrastructure security, data element market security, and data circulation security. The data governance standards within data resources include data business planning, data quality management, data survey and inventory, and data resource registration; standards related to training datasets encompass the collection and processing of training datasets, dataset labeling, and dataset synthesis. In terms of organizational support, guidance will be provided to establish a national data standardization technical organization, accelerate the revision of urgently needed data standards, and strengthen communication and collaboration with relevant standardization technical organizations, industries, localities, and related associations to promote the construction of the data industry ecosystem through standardization. Additionally, policies for standard pilot projects will be improved, a public service platform for data standardization will be established, standard promotion activities will be conducted, and key localities and industries will be selected for pioneering trials to create typical demonstrations. Efforts will be made to explore third-party inspection and testing of data products and deepen the evaluation and management of data standard implementation. In terms of talent cultivation, training courses for data talents that complement the standards will be developed to cultivate a group of professionals in data standardization. The international standardization expert team will be optimized to support participation in international standardization activities and strengthen international exchanges.
    
    Translation using RAG
    Total number of paragraphs : 5
    Total number of sentences  : 19
    The data sector is welcoming national standards. On October 8, the National Development and Reform Commission (NDRC) and other departments issued a notice regarding the release of the "Guidelines for the Construction of a National Data Standard System" (hereinafter referred to as the "Guidelines"). This initiative aims to "fully leverage the regulatory and guiding role of standards in activating the potential of data elements, strengthening, optimizing, and expanding the digital economy." The NDRC, the National Data Bureau, the Cyberspace Administration of China, the Ministry of Industry and Information Technology, the Ministry of Finance, and the National Standardization Administration jointly organized the development of the "Guidelines."
    
    The "Guidelines" propose that by the end of 2026, a national data standard system will be essentially established. This will involve the formulation and revision of over 30 foundational and general national standards in areas such as data circulation and utilization infrastructure, data management, data services, training data sets, public data authorization operations, data rights confirmation, data resource pricing, and enterprise data paradigm transactions. The aim is to create a number of standard application demonstration cases, establish a standard verification and application service platform, and cultivate a group of third-party standardized service organizations capable of assessing data management capabilities, data evaluation, data service capabilities, and public data authorization operation performance.
    
    The "Guidelines" clarify that the framework of the data standard system consists of seven components: foundational general standards, data infrastructure, data resources, data technology, data circulation, integrated applications, and security guarantees. In terms of data infrastructure, the standards address data computing facilities and data storage facilities within computing and storage infrastructure, as well as network facilities including 5G data transmission, fiber-optic data transmission, and satellite internet data transmission, along with circulation and utilization facilities. Regarding data circulation, standards encompass data products, data rights confirmation, data resource pricing, and data circulation transactions. In terms of integrated applications, standards relate to industrial manufacturing, agriculture and rural development, commerce and circulation, transportation, financial services, technological innovation, cultural tourism (cultural relics), health care, emergency management, meteorological services, urban governance, and green low-carbon initiatives. Security guarantees cover the security of data infrastructure, the safety of the data elements market, and the security of data circulation.
    
    Data governance standards within data resources include data business planning, data quality management, data inventory and survey, and data resource registration. Standards related to training data sets include collection and processing of training data sets, data labeling, and training data set synthesis. In terms of organizational support, there will be guidance to establish a national data standardization technical organization, accelerate the formulation and revision of urgently needed data standards, and strengthen communication and collaboration with relevant standardization technical organizations, industries, localities, and related associations to promote the construction of a data industry ecosystem through standardization. 
    
    Additionally, there will be improvements to the supporting policies for standard pilot projects, the establishment of a public service platform for data standardization, the promotion of standards, and the selection of key localities and industries for pilot testing to create typical models. The exploration of third-party inspection and testing of data products will be encouraged, along with the deepening of data standard implementation evaluation and management. In terms of talent development, there will be initiatives to create training courses for data talent that complement standards, aiming to cultivate a group of professionals in data standardization. The optimization of the team of international standardization experts in data will be supported to encourage participation in international standardization activities and to strengthen international exchanges.
</pre>

## Evaluation of translation results

Evaluating machine translation quality is essential to ensure the accuracy and fluency of translated text. In this tutorial, we use two key metrics, TER and BERTScore, to assess the quality of translations produced by both a general LLM-based translation system and a RAG-based translation system.

By combining TER and BERTScore, we achieve a comprehensive evaluation of translation quality.
TER measures the structural differences and required edits between translations and reference texts.
BERTScore captures the semantic similarity between translations and references.
This dual evaluation approach allows us to effectively compare LLM and RAG translations, helping determine which method provides more accurate, fluent, and natural translations.


**TER (Translation Edit Rate)**

TER quantifies how much editing is required to transform a system-generated translation into the reference translation. It accounts for insertions, deletions, substitutions, and Shifts (word reordering).

Interpretation:
Lower TER indicates a better translation (fewer modifications needed).
Higher TER suggests that the translation deviates significantly from the reference

**BERTScore - Contextual Semantic Evaluation**

BERTScore evaluates translation quality by computing semantic similarity scores between reference and candidate translations. It utilizes contextual embeddings from a pre-trained BERT model, unlike traditional n-gram-based methods that focus solely on word overlap.

Interpretation:
Higher BERTScore (closer to 1.0) indicates better semantic similarity between the candidate and reference translations.
Lower scores indicate less semantic alignment with the reference translation.

Since Chinese and English are grammatically very different languages, there can be significant differences in word order and sentence structure. As a result, the TER score may be relatively high, while BERTScore can serve as a more important evaluation metric.

By leveraging both TER and BERTScore, we can effectively analyze the strengths and weaknesses of LLM-based and RAG-based translation methods.

```python
import nltk
import sacrebleu
import bert_score


# Download the 'punkt' data if it's not available.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# TER Score Calculation
def calculate_ter(reference, candidate):
    ter_metric = sacrebleu.metrics.TER()
    return round(ter_metric.corpus_score([candidate], [[reference]]).score, 3)


# BERTScore Calculation
def calculate_bert_score(reference, candidate):
    try:
        P, R, F1 = bert_score.score([candidate], [reference], lang="en")
        return round(F1.mean().item(), 3)
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return None


sentences = chinese_text_from_file_loader("data/comparison_cn.txt")

# Store sentences in the FAISS vector database
vector_store = FAISS.from_texts(sentences, embedding=embeddings)


# Execute translation
translated_results = []
for idx, sentence in enumerate(sentences, start=1):
    llm_translation = translate_with_llm(sentence)
    rag_translation = translate_with_rag(sentence, vector_store)

    # Evaluate translation quality (LLM)
    ter_llm = calculate_ter(sentence, llm_translation.content)
    bert_llm = calculate_bert_score(sentence, llm_translation.content)

    # Evaluate translation quality (RAG)
    ter_rag = calculate_ter(sentence, rag_translation.content)
    bert_rag = calculate_bert_score(sentence, rag_translation.content)

    translated_results.append(
        {
            "source_text": sentence,
            "llm_translation": llm_translation.content,
            "rag_translation": rag_translation.content,
            "TER LLM": ter_llm,
            "BERTScore LLM": bert_llm,
            "TER RAG": ter_rag,
            "BERTScore RAG": bert_rag,
        }
    )


# Since Chinese and English are grammatically very different languages, there can be significant differences in word order and sentence structure. As a result, the TER score may be relatively high, while BERTScore can serve as a more important evaluation metric.
# Sort in descending order based on BERTScore LLM and extract the top 5.
top_5_bert_llm = sorted(
    translated_results, key=lambda x: x["BERTScore LLM"], reverse=True
)[:5]
# Display results in a transposed format
for idx, result in enumerate(top_5_bert_llm, start=1):
    print(f"**Top {idx}**")
    print("-" * 60)
    print(f"Source Text       | {result['source_text']}")
    print(f"LLM Translation   | {result['llm_translation']}")
    print(f"RAG Translation   | {result['rag_translation']}")
    print(f"TER Score (LLM)   | {result['TER LLM']}")
    print(f"BERTScore (LLM)   | {result['BERTScore LLM']}")
    print(f"TER Score (RAG)   | {result['TER RAG']}")
    print(f"BERTScore (RAG)   | {result['BERTScore RAG']}")
    print("-" * 60, "\n")
```

<pre class="custom">[nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\herme\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    c:\Users\herme\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-9y5W8e20-py3.11\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
</pre>

    **Top 1**
    ------------------------------------------------------------
    Source Text       | 数据领域迎来国家标准。
    LLM Translation   | The translation of the Chinese sentence "数据领域迎来国家标准。" is "The data field welcomes national standards."
    RAG Translation   | "The data sector is ushering in national standards."
    TER Score (LLM)   | 1400.0
    BERTScore (LLM)   | 0.923
    TER Score (RAG)   | 800.0
    BERTScore (RAG)   | 0.782
    ------------------------------------------------------------ 
    
    **Top 2**
    ------------------------------------------------------------
    Source Text       | 在人才培养方面，将打造标准配套的数据人才培训课程，形成一批数据标准化专业人才。
    LLM Translation   | In terms of talent development, we will create standardized and supportive training programs for data professionals, forming a group of standardized professionals in data.
    RAG Translation   | "In terms of talent development, we will create standardized training courses for data professionals, aiming to cultivate a group of specialized personnel in data standardization."
    TER Score (LLM)   | 2400.0
    BERTScore (LLM)   | 0.764
    TER Score (RAG)   | 2500.0
    BERTScore (RAG)   | 0.772
    ------------------------------------------------------------ 
    
    **Top 3**
    ------------------------------------------------------------
    Source Text       | 数据流通方面，标准包括数据产品、数据确权、数据资源定价、数据流通交易。
    LLM Translation   | In terms of data circulation, the standards include data products, data rights confirmation, data resource pricing, and data circulation transactions.
    RAG Translation   | In terms of data circulation, the standards include data products, data ownership rights, data resource pricing, and data circulation transactions.
    TER Score (LLM)   | 2000.0
    BERTScore (LLM)   | 0.762
    TER Score (RAG)   | 2000.0
    BERTScore (RAG)   | 0.761
    ------------------------------------------------------------ 
    
    **Top 4**
    ------------------------------------------------------------
    Source Text       | 安全保障方面，标准涉及数据基础设施安全，数据要素市场安全，数据流通安全。
    LLM Translation   | In terms of security guarantees, the standards involve data infrastructure security, data factor market security, and data circulation security.
    RAG Translation   | In terms of security guarantees, the standards cover the safety of data infrastructure, the security of the data factor market, and the security of data circulation.
    TER Score (LLM)   | 1900.0
    BERTScore (LLM)   | 0.76
    TER Score (RAG)   | 2600.0
    BERTScore (RAG)   | 0.761
    ------------------------------------------------------------ 
    
    **Top 5**
    ------------------------------------------------------------
    Source Text       | 优化数据国际标准化专家队伍，支持参与国际标准化活动，强化国际交流。
    LLM Translation   | "Optimize the team of international standardization experts in data, support participation in international standardization activities, and strengthen international exchanges."
    RAG Translation   | "Optimize the team of international experts in data standardization, support participation in international standardization activities, and strengthen international communication."
    TER Score (LLM)   | 1900.0
    BERTScore (LLM)   | 0.758
    TER Score (RAG)   | 1900.0
    BERTScore (RAG)   | 0.764
    ------------------------------------------------------------ 
    
    

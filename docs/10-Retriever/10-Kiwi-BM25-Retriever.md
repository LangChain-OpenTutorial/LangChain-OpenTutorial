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

# Kiwi BM25 Retriever

- Author: [JeongGi Park](https://github.com/jeongkpa)
- Design: []()
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb)

## Overview
This document explores the use of `kiwipiepy` for Korean morphological analysis and demonstrates its integration within the `LangChain` framework. 
It highlights methods to tokenize text, compare retrieval models like `BM25` and `FAISS`, and analyze relationships between queries and documents using metrics such as cosine similarity. 
Additionally, it emphasizes the role of these techniques in enhancing workflows like text analysis and information retrieval.

Since this tutorial covers Korean morphological analysis, the output primarily contains Korean text, reflecting the language structure being analyzed
For international users, we provide English translations alongside Korean examples in this tutorial.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Korean Word Retriever Tuning](#Korean-Word-Retriever-Tuning)
- [Testing with Various Sentences](#Testing-with-Various-Sentences)
- [Experiment: Compare Search Results Using Different Retrievers](#Experiment-Compare-Search-Results-Using-Different-Retrievers)
- [Conclusion](#conclusion)

### References
- [kiwipiepy](https://github.com/bab2min/kiwipiepy)
- [fiass](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
- [openai-embeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain-openai",
        "langchain",
        "python-dotenv",
        "langchain-core",
        "kiwipiepy",
        "rank_bm25",        
        "langchain-community",
        "faiss-cpu",

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
        "LANGCHAIN_PROJECT": "Kiwi-BM25-Retriever",
    },
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

[Note] If you are using a `.env` file, proceed as follows.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



### Why Korean Tokenization?

- In Korean, words are morphologically rich. For instance, “안녕하세요” is tokenized into:
  - Token(form='안녕', tag='NNG')
  - Token(form='하', tag='XSA')
  - Token(form='세요', tag='EF')

- Compared to English tokenization at the word level (e.g., “Hello” remains one word), Korean often splits into multiple morphemes (어근, 접사, 어미 등).
- Kiwi provides detailed POS tagging such as NNG(일반 명사), XSA(형용사 파생 접사), EF(종결 어미) to reflect these language-specific nuances.


## Korean Word Retriever Tuning

Install the Korean morphological analyzer library, `kiwipiepy`.

[Project Link for kiwipiepy](https://github.com/bab2min/kiwipiepy)

```python
from kiwipiepy import Kiwi

kiwi = Kiwi()
```

Perform Tokenization

```python
kiwi.tokenize("안녕하세요? 형태소 분석기 키위입니다")
# "안녕하세요? 형태소 분석기 키위입니다." it means "Hi, this is Kiwi, the morphological analyser."
```




<pre class="custom">[Token(form='안녕', tag='NNG', start=0, len=2),
     Token(form='하', tag='XSA', start=2, len=1),
     Token(form='세요', tag='EF', start=3, len=2),
     Token(form='?', tag='SF', start=5, len=1),
     Token(form='형태소', tag='NNG', start=7, len=3),
     Token(form='분석기', tag='NNG', start=11, len=3),
     Token(form='키위', tag='NNG', start=15, len=2),
     Token(form='이', tag='VCP', start=17, len=1),
     Token(form='ᆸ니다', tag='EF', start=17, len=3)]</pre>



## Testing with Various Sentences

BM25 is a traditional ranking function based on term frequency and inverse document frequency. It works well when exact keyword matches are important.


FAISS uses vector embeddings to capture semantic similarity. By combining BM25 with FAISS in an ensemble, we can leverage the lexical match benefits from BM25 and the semantic understanding from FAISS.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Sample documents for retriever testing
docs = [
    Document(
        page_content="금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다."
    ),
    Document(
        page_content="금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다."
    ),
    Document(
        page_content="저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다."
    ),
    Document(
        page_content="금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다."
    ),
    Document(
        page_content="금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다."
    ),
    Document(
        page_content="금보험은 저축성과를 극대화합니다. 특히 노후 대비 저축에 유리하게 구성되어 있습니다."
    ),
    Document(
        page_content="금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요."
    ),
]

# 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
# Financial insurance is a financial product designed for long term asset management and risk coverage.

# 금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다.
# Financial savings insurance allows individuals to accumulate a lump sum through regular savings, and also offers life insurance benefits.

# 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
# Savings financial insurance helps individuals gather a lump sum through savings and finance, and also provides death benefit coverage.

# 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
# Financial savings livestock insurance is a special financial product designed for long term savings, which also includes provisions for livestock products.

# 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
# Financial 'carpet bombing' insurance focuses on risk coverage rather than savings. It is suitable for customers willing to take on high risk.

# 금보험은 저축성과를 극대화합니다. 특히 노후 대비 저축에 유리하게 구성되어 있습니다.
# Gold insurance maximizes returns on savings. It is especially advantageous for retirement savings.

# 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
# Hey, Mr. 'Financial Bo,' please refrain from harsh words and consider saving money. I'm not sure why you're in such a hurry.

```

```python
# Print tokenized documents
for doc in docs:
    print(" ".join([token.form for token in kiwi.tokenize(doc.page_content)]))

```

<pre class="custom">금융 보험 은 장기 적 이 ᆫ 자산 관리 와 위험 대비 를 목적 으로 고안 되 ᆫ 금융 상품 이 ᆸ니다 .
    금융 저축 보험 은 규칙 적 이 ᆫ 저축 을 통하 어 목돈 을 마련 하 ᆯ 수 있 으며 , 생명 보험 기능 도 겸비 하 고 있 습니다 .
    저축 금융 보험 은 저축 과 금융 을 통하 어 목돈 마련 에 도움 을 주 는 보험 이 ᆸ니다 . 또한 , 사망 보장 기능 도 제공 하 ᆸ니다 .
    금융 저 축산물 보험 은 장기 적 이 ᆫ 저축 목적 과 더불 어 , 축산물 제공 기능 을 갖추 고 있 는 특별 금융 상품 이 ᆸ니다 .
    금융 단 폭격 보험 은 저축 은 커녕 위험 대비 에 초점 을 맞추 ᆫ 상품 이 ᆸ니다 . 높 은 위험 을 감수 하 고자 하 는 고객 에게 적합 하 ᆸ니다 .
    금 보험 은 저축 성과 를 극대 화 하 ᆸ니다 . 특히 노후 대비 저축 에 유리 하 게 구성 되 어 있 습니다 .
    금융 보 씨 험하 ᆫ 말 좀 하 지 말 시 고 , 저축 이나 좀 하 시 던가요 . 뭐 가 그리 급하 시 ᆫ지 모르 겠 네요 .
</pre>

```python
# Create a tokenization function

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

```

## Experiment: Compare Search Results Using Different Retrievers

In this section, we compare how different retrieval methods rank documents when given the same query. We are using:

* `BM25`: A traditional ranking function based on term frequency (TF) and inverse document frequency (IDF).
* `Kiwi BM25`: `BM25` with an added benefit of kiwipiepy tokenization, enabling more accurate splitting of Korean words into morphemes (especially important for Korean queries).
* `FAISS`: A vector-based retriever using embeddings (in this case, `OpenAIEmbeddings`). It captures semantic similarity, so it’s less reliant on exact keyword matches and more on meaning.
* `Ensemble`: A combination of BM25 (or `Kiwi BM25`) and `FAISS`, weighted to leverage both the lexical matching strengths of `BM25` and the semantic understanding of FAISS.

### Key points of Comparison

**Exact Keyword Matching vs. Semantic Matching**

* `BM25` (and `Kiwi BM25`) excel in finding documents that share exact terms or closely related morphological variants.
* `FAISS` retrieves documents that may not have exact lexical overlap but are semantically similar (e.g., synonyms or paraphrases).

**Impact of Korean Morphological Analysis**

* Korean often merges stems and endings into single words (“안녕하세요” → “안녕 + 하 + 세요”). `Kiwi BM25` handles this by splitting the query and documents more precisely.
* This can yield more relevant results when dealing with conjugated verbs, particles, or compound nouns.

**Ensemble Approaches**

* By combining lexical (`BM25`) and semantic (`FAISS`) retrievers, we can produce a more balanced set of results.
* The weighting (e.g., 70:30 or 30:70) can be tuned to emphasize one aspect over the other.
* Using MMR (Maximal Marginal Relevance) ensures diversity in the retrieved results, reducing redundancy.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize BM25 retriever using raw documents
bm25 = BM25Retriever.from_documents(docs)

# Initialize BM25 retriever with a custom preprocessing function (e.g., Kiwi tokenizer)
kiwi_bm25 = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)

# Initialize FAISS retriever with OpenAI embeddings
faiss = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()

# Create an ensemble retriever combining BM25 and FAISS with a 70:30 weighting
bm25_faiss_73 = EnsembleRetriever(
    retrievers=[bm25, faiss],  # List of retrieval models to combine
    weights=[0.7, 0.3],        # Weighting for BM25 (70%) and FAISS (30%) results
    search_type="mmr",        # Use MMR (Maximal Marginal Relevance) to diversify search results
)

# Create an ensemble retriever combining BM25 and FAISS with a 30:70 weighting
bm25_faiss_37 = EnsembleRetriever(
    retrievers=[bm25, faiss],  # List of retrieval models to combine
    weights=[0.3, 0.7],        # Weighting for BM25 (30%) and FAISS (70%) results
    search_type="mmr",        # Use MMR (Maximal Marginal Relevance) to diversify search results
)

# Create an ensemble retriever combining Kiwi BM25 and FAISS with a 70:30 weighting
kiwibm25_faiss_73 = EnsembleRetriever(
    retrievers=[kiwi_bm25, faiss],  # List of retrieval models to combine
    weights=[0.7, 0.3],             # Weighting for Kiwi BM25 (70%) and FAISS (30%) results
    search_type="mmr",             # Use MMR (Maximal Marginal Relevance) to diversify search results
)

# Create an ensemble retriever combining Kiwi BM25 and FAISS with a 30:70 weighting
kiwibm25_faiss_37 = EnsembleRetriever(
    retrievers=[kiwi_bm25, faiss],  # List of retrieval models to combine
    weights=[0.3, 0.7],             # Weighting for Kiwi BM25 (30%) and FAISS (70%) results
    search_type="mmr",             # Use MMR (Maximal Marginal Relevance) to diversify search results
)

# Dictionary to store all retrievers for easy access
retrievers = {
    "bm25": bm25,  # Standard BM25 retriever
    "kiwi_bm25": kiwi_bm25,  # BM25 retriever with Kiwi tokenizer
    "faiss": faiss,  # FAISS retriever with OpenAI embeddings
    "bm25_faiss_73": bm25_faiss_73,  # Ensemble retriever (BM25:70%, FAISS:30%)
    "bm25_faiss_37": bm25_faiss_37,  # Ensemble retriever (BM25:30%, FAISS:70%)
    "kiwi_bm25_faiss_73": kiwibm25_faiss_73,  # Ensemble retriever (Kiwi BM25:70%, FAISS:30%)
    "kiwi_bm25_faiss_37": kiwibm25_faiss_37,  # Ensemble retriever (Kiwi BM25:30%, FAISS:70%)
}

```

```python
# Function to print search results from multiple retrievers
def print_search_results(retrievers, query):
    """
    Prints the top search result from each retriever for a given query.
    
    Args:
        retrievers (dict): A dictionary of retriever instances.
        query (str): The search query.
    """
    print(f"Query: {query}")
    for name, retriever in retrievers.items():
        # Retrieve and print the top search result for each retriever
        print(f"{name}\t: {retriever.invoke(query)[0].page_content}")
    print("===" * 20)

```

### Display Search Results

금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.

-> Financial insurance is a financial product designed for long term asset management and risk coverage


금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다.

-> Financial savings insurance allows individuals to accumulate a lump sum through regular savings, and also offers life insurance benefits


저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.

-> Savings financial insurance helps individuals gather a lump sum through savings and finance, and also provides death benefit coverage


금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.

-> Financial savings livestock insurance is a special financial product designed for long term savings, which also includes provisions for livestock products


금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.

-> Financial 'carpet bombing' insurance focuses on risk coverage rather than savings. It is suitable for customers willing to take on high risk


금보험은 저축성과를 극대화합니다. 특히 노후 대비 저축에 유리하게 구성되어 있습니다.

-> Gold insurance maximizes returns on savings. It is especially advantageous for retirement savings


금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.

-> Hey, Mr. 'Financial Bo,' please refrain from harsh words and consider saving money. I'm not sure why you're in such a hurry.


```python
print_search_results(retrievers, "금융보험")
```

<pre class="custom">Query: 금융보험
    bm25	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    kiwi_bm25	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    faiss	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    bm25_faiss_73	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    bm25_faiss_37	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    kiwi_bm25_faiss_73	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    kiwi_bm25_faiss_37	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    ============================================================
</pre>

```python
print_search_results(retrievers, "금융 보험")
```

<pre class="custom">Query: 금융 보험
    bm25	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    kiwi_bm25	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    faiss	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    bm25_faiss_73	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    bm25_faiss_37	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    kiwi_bm25_faiss_73	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    kiwi_bm25_faiss_37	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    ============================================================
</pre>

```python
print_search_results(retrievers, "금융저축보험")
```

<pre class="custom">Query: 금융저축보험
    bm25	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    kiwi_bm25	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    faiss	: 금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다.
    bm25_faiss_73	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    bm25_faiss_37	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    kiwi_bm25_faiss_73	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    kiwi_bm25_faiss_37	: 금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다.
    ============================================================
</pre>

```python
print_search_results(retrievers, "축산물 보험")
```

<pre class="custom">Query: 축산물 보험
    bm25	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    kiwi_bm25	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    faiss	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    bm25_faiss_73	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    bm25_faiss_37	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    kiwi_bm25_faiss_73	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    kiwi_bm25_faiss_37	: 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.
    ============================================================
</pre>

```python
print_search_results(retrievers, "저축금융보험")
```

<pre class="custom">Query: 저축금융보험
    bm25	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    kiwi_bm25	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    faiss	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    bm25_faiss_73	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    bm25_faiss_37	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    kiwi_bm25_faiss_73	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    kiwi_bm25_faiss_37	: 저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다.
    ============================================================
</pre>

```python
print_search_results(retrievers, "금융보씨 개인정보 조회")
```

<pre class="custom">Query: 금융보씨 개인정보 조회
    bm25	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    kiwi_bm25	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    faiss	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    bm25_faiss_73	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    bm25_faiss_37	: 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.
    kiwi_bm25_faiss_73	: 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.
    kiwi_bm25_faiss_37	: 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.
    ============================================================
</pre>

## Conclusion

By running the code and observing the top documents returned for each query, you’ll see how each retriever type has its strengths:

`BM25` / `Kiwi BM25`: Great for precise keyword matching, beneficial for Korean morphological nuances.

`FAISS`: Finds semantically related documents even if the wording differs.

`Ensemble`: Balances both worlds, often achieving better overall coverage for a wide range of queries.


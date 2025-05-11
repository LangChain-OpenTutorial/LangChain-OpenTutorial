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

# Arxiv Loader

- Author: [Sunyoung Park (architectyou)](https://github.com/architectyou)
- Peer Review : [ppakyeah](https://github.com/ppakyeah)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/10-ArxivLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/10-ArxivLoader.ipynb)


## Overview

[```arXiv```](https://arxiv.org/) is an open access archive for 2 million scholarly articles in the fields of physics, 

mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems 

science, and economics.

[API Documentation](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.arxiv.ArxivLoader.html#langchain_community.document_loaders.arxiv.ArxivLoader)


To access the Arxiv document loader, you need to install ```arxiv```, ```PyMuPDF``` and ```langchain-community``` integration packages.

```PyMuPDF``` converts PDF files downloaded from arxiv.org into text format.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Arxiv Loader Instantiate](#arxiv-loader-instantiate)
- [Load](#load)
- [Lazy Load](#lazy-load)
- [Asynchronous Load](#asynchronous-load)
- [Use summaries of articles as docs](#use-summaries-of-articles-as-docs)

### References

- [ArxivLoader API Documentation](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.arxiv.ArxivLoader.html#langchain_community.document_loaders.arxiv.ArxivLoader)
- [Arxiv API Acess Documentation](https://info.arxiv.org/help/api/index.html)

---

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
        "arxiv",
        "pymupdf",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 23.3.2 -> 24.3.1
    [notice] To update, run: pip install --upgrade pip
</pre>

## Arxiv-Loader-Instantiate

You can make arxiv loader instance to load documents from arxiv.org.

Initialize with search query to find documents in the Arixiv.org.
Supports all arguments of ```ArxivAPIWrapper``` .

```python
from langchain_community.document_loaders import ArxivLoader

### Enter the research topic you want to search for in the Query parameter
loader = ArxivLoader(
    query="Chain of thought",
    load_max_docs=2,  # max number of documents
    load_all_available_meta=True,  # load all available metadata
)
```

### Load

Use ```Load``` method to load documents from arxiv.org with ```ArxivLoader``` instance.

```python
# Print the first document's content and metadata
docs = loader.load()
print(docs[0].page_content[:100])
print(docs[0].metadata)
```

<pre class="custom">Contrastive Chain-of-Thought Prompting
    Yew Ken Chia∗1,
    Guizhen Chen∗1, 2
    Luu Anh Tuan2
    Soujanya Pori
    {'Published': '2023-11-15', 'Title': 'Contrastive Chain-of-Thought Prompting', 'Authors': 'Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing', 'Summary': 'Despite the success of chain of thought in enhancing language model\nreasoning, the underlying process remains less well understood. Although\nlogically sound reasoning appears inherently crucial for chain of thought,\nprior studies surprisingly reveal minimal impact when using invalid\ndemonstrations instead. Furthermore, the conventional chain of thought does not\ninform language models on what mistakes to avoid, which potentially leads to\nmore errors. Hence, inspired by how humans can learn from both positive and\nnegative examples, we propose contrastive chain of thought to enhance language\nmodel reasoning. Compared to the conventional chain of thought, our approach\nprovides both valid and invalid reasoning demonstrations, to guide the model to\nreason step-by-step while reducing reasoning mistakes. To improve\ngeneralization, we introduce an automatic method to construct contrastive\ndemonstrations. Our experiments on reasoning benchmarks demonstrate that\ncontrastive chain of thought can serve as a general enhancement of\nchain-of-thought prompting.', 'entry_id': 'http://arxiv.org/abs/2311.09277v1', 'published_first_time': '2023-11-15', 'comment': None, 'journal_ref': None, 'doi': None, 'primary_category': 'cs.CL', 'categories': ['cs.CL'], 'links': ['http://arxiv.org/abs/2311.09277v1', 'http://arxiv.org/pdf/2311.09277v1']}
</pre>

- If ```load_all_available_meta``` is False, only partial metadata is displayed, not the complete metadata.

### Lazy Load

When loading large amounts of documents, If you can perform downstream tasks on a subset of all loaded documents, you can ```lazy_load``` documents one at a time to minimize memory usage.

```python
docs = []
docs_lazy = loader.lazy_load()

# append docs to docs list
# async variant : docs_lazy = await loader.lazy_load()

for doc in docs_lazy:
    docs.append(doc)

print(docs[0].page_content[:100])
print(docs[0].metadata)
```

<pre class="custom">Contrastive Chain-of-Thought Prompting
    Yew Ken Chia∗1,
    Guizhen Chen∗1, 2
    Luu Anh Tuan2
    Soujanya Pori
    {'Published': '2023-11-15', 'Title': 'Contrastive Chain-of-Thought Prompting', 'Authors': 'Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing', 'Summary': 'Despite the success of chain of thought in enhancing language model\nreasoning, the underlying process remains less well understood. Although\nlogically sound reasoning appears inherently crucial for chain of thought,\nprior studies surprisingly reveal minimal impact when using invalid\ndemonstrations instead. Furthermore, the conventional chain of thought does not\ninform language models on what mistakes to avoid, which potentially leads to\nmore errors. Hence, inspired by how humans can learn from both positive and\nnegative examples, we propose contrastive chain of thought to enhance language\nmodel reasoning. Compared to the conventional chain of thought, our approach\nprovides both valid and invalid reasoning demonstrations, to guide the model to\nreason step-by-step while reducing reasoning mistakes. To improve\ngeneralization, we introduce an automatic method to construct contrastive\ndemonstrations. Our experiments on reasoning benchmarks demonstrate that\ncontrastive chain of thought can serve as a general enhancement of\nchain-of-thought prompting.', 'entry_id': 'http://arxiv.org/abs/2311.09277v1', 'published_first_time': '2023-11-15', 'comment': None, 'journal_ref': None, 'doi': None, 'primary_category': 'cs.CL', 'categories': ['cs.CL'], 'links': ['http://arxiv.org/abs/2311.09277v1', 'http://arxiv.org/pdf/2311.09277v1']}
</pre>

```python
len(docs)
```




<pre class="custom">3</pre>



### Asynchronous Load

Use ```aload``` method to load documents from arxiv.org asynchronously.

```python
docs = await loader.aload()
print(docs[0].page_content[:100])
print(docs[0].metadata)
```

<pre class="custom">Contrastive Chain-of-Thought Prompting
    Yew Ken Chia∗1,
    Guizhen Chen∗1, 2
    Luu Anh Tuan2
    Soujanya Pori
    {'Published': '2023-11-15', 'Title': 'Contrastive Chain-of-Thought Prompting', 'Authors': 'Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, Lidong Bing', 'Summary': 'Despite the success of chain of thought in enhancing language model\nreasoning, the underlying process remains less well understood. Although\nlogically sound reasoning appears inherently crucial for chain of thought,\nprior studies surprisingly reveal minimal impact when using invalid\ndemonstrations instead. Furthermore, the conventional chain of thought does not\ninform language models on what mistakes to avoid, which potentially leads to\nmore errors. Hence, inspired by how humans can learn from both positive and\nnegative examples, we propose contrastive chain of thought to enhance language\nmodel reasoning. Compared to the conventional chain of thought, our approach\nprovides both valid and invalid reasoning demonstrations, to guide the model to\nreason step-by-step while reducing reasoning mistakes. To improve\ngeneralization, we introduce an automatic method to construct contrastive\ndemonstrations. Our experiments on reasoning benchmarks demonstrate that\ncontrastive chain of thought can serve as a general enhancement of\nchain-of-thought prompting.', 'entry_id': 'http://arxiv.org/abs/2311.09277v1', 'published_first_time': '2023-11-15', 'comment': None, 'journal_ref': None, 'doi': None, 'primary_category': 'cs.CL', 'categories': ['cs.CL'], 'links': ['http://arxiv.org/abs/2311.09277v1', 'http://arxiv.org/pdf/2311.09277v1']}
</pre>

## Use Summaries of Articles as Docs

Use ```get_summaries_as_docs``` method to get summaries of articles as docs.

```python
from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(
    query="reasoning"
)

docs = loader.get_summaries_as_docs()
print(docs[0].page_content[:100])
print(docs[0].metadata)
```

<pre class="custom">Large language models (LLMs) have demonstrated impressive reasoning
    abilities, but they still strugg
    {'Entry ID': 'http://arxiv.org/abs/2410.13080v1', 'Published': datetime.date(2024, 10, 16), 'Title': 'Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models', 'Authors': 'Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, Shirui Pan'}
</pre>

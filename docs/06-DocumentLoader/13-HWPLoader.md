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

# HWP (Hangeul) Loader

- Author: [Sunyoung Park (architectyou)](https://github.com/Architectyou)
- Peer Review : [Suhyun Lee](https://github.com/suhyun0115), [Kane](https://github.com/HarryKane11)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/13-HWPLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/13-HWPLoader.ipynb)
## Overview

HWP is Hangeul Word Processor developed by **Hancom** , and it is Korea's representative office software.

It uses the **.hwp** file extension and is widely used in Businesses, Schools, and Government Institutions, and more.

Therefore, if you're a developer in South Korea, you've likely had (or will have) experience dealing with **.hwp** documents.

Unfortunately, it's not yet integrated with LangChain, so we'll need to use a custom-implemented ```HWPLoader``` with ```langchain-teddynote``` and ```langchain-opentutorial``` .


In this tutorial, we'll implement a ```HWPLoader``` that can load **.hwp** files and extract text from them.


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [HWP Loader Instantiate](#hwp-loader-instantiate)
- [Loader](#loader)

### References

- [Hancom Developer Forum](https://developer.hancom.com/)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial langchain-teddynote
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain-teddynote",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 23.3.2 -> 24.3.1
    [notice] To update, run: pip install --upgrade pip
</pre>

## HWP Loader Instantiate

You can instantiate HWP Loader with ```HWPLoader``` class.

```python
from langchain_teddynote.document_loaders import HWPLoader

loader = HWPLoader(file_path="data/Regulations_of_the_Establishment_and_Operation_of_the_National_Artificial_Intelligence_Committee.hwp")
```

### Loader

You can load the document with ```load``` method.

```python
docs = loader.load()

print(docs[0].page_content[:1000])
```

<pre class="custom">Regulations on the Establishment and Operation of the National Artifical Intelligence Committee[Effective Augst 6, 2024] [Presidential Decree No. 34787, Enacted August 6, 2024]Regulations on the Establishment and Operation of the National Artificial Intelligence Committee Ministry of Government Legislation-  /  - National Statutory Information Center    Reason for Enactment [Enactment]◇ Purpose  To establish the National Artificial Intelligence Committee under the President to strengthen national competitiveness, protect national interests, and improve the quality of life for citizens by promoting the artificial intelligence industry and creating a trustworthy AI usage environment.◇ Main Contents  A. Establishment and Functions of the National AI Committee (Article 2)    1) The National AI Committee shall be established under the President to efficiently deliberate and coordinate major policies for promoting the AI industry and establishing a foundation of trust in AI.    2) The Commit
</pre>

```python
len(docs) # Check the number of documents
```




<pre class="custom">1</pre>



```python
print(docs[0].metadata) # Information about the document
```

<pre class="custom">{'source': 'data/Regulations_of_the_Establishment_and_Operation_of_the_National_Artificial_Intelligence_Committee.hwp'}
</pre>

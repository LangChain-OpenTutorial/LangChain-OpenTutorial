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

# LangChain Hub

- Author: [ChangJun Lee](https://www.linkedin.com/in/cjleeno1/)
- Design: []()
- Peer Review: [musangk](https://github.com/musangk), [ErikaPark](https://github.com/ErikaPark), [jeong-wooseok](https://github.com/jeong-wooseok)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/03-LangChain-Hub.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/03-LangChain-Hub.ipynb)

## Overview

This is an example of retrieving and executing prompts from LangChain Hub.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Register Your Own Prompt to Prompt Hub]()

### References

- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [LangChain Core Output Parsers](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.list.CommaSeparatedListOutputParser.html#)
- [Python List Tutorial](https://docs.python.org/3.13/tutorial/datastructures.html)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- You can check LangChain Hub prompts at the address below.
  - You can retrieve prompts by using the prompt repo ID, and you can also get prompts for specific versions by adding the commit ID.
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

You can check LangChain Hub prompts at the address below.

You can retrieve prompts using the prompt repo ID, and you can also get prompts for specific versions by adding the commit ID.

## **Getting Prompts from Hub**

```python
from langchain import hub 

# Get the latest version of the prompt
prompt = hub.pull("rlm/rag-prompt")
```

```python
# Print the prompt content
print(prompt)
```

> input_variables=['context', 'question'] metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]

```python
# To get a specific version of prompt, specify the version hash
prompt = hub.pull("rlm/rag-prompt:50442af1")
prompt
```

> ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

## **Register Your Own Prompt to Prompt Hub**

```python
from langchain.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    "Summarize the following text based on the given content. Please write the answer in Korean\n\nCONTEXT: {context}\n\nSUMMARY:"
)
prompt
```

> ChatPromptTemplate(input_variables=['context'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], template='Summarize the following text based on the given content. Please write the answer in Korean\n\nCONTEXT: {context}\n\nSUMMARY:'))])

```python
from langchain import hub

# Upload the prompt to the hub
hub.push("teddynote/simple-summary-korean", prompt)
```

The following is the output after successfully uploading to Hub.

`ID/PromptName/Hash`

> Output: 'https://smith.langchain.com/hub/teddynote/simple-summary-korean/0e296563'

```python
from langchain import hub

# Get the prompt from the hub
pulled_prompt = hub.pull("teddynote/simple-summary-korean")
```

```python
# Print the prompt content
print(pulled_prompt)
```

> input_variables=['context'] metadata={'lc_hub_owner': 'teddynote', 'lc_hub_repo': 'simple-summary-korean', 'lc_hub_commit_hash': '0e296563564b581e5ad77089b035596246c2b96046f8db0503355dd3c275d056'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], template='Summarize the following text based on the given content. Please write the answer in Korean\n\nCONTEXT: {context}\n\nSUMMARY:'))]

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
- Peer Review: [musangk](https://github.com/musangk), [ErikaPark](https://github.com/ErikaPark), [jeong-wooseok](https://github.com/jeong-wooseok)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)

- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/03-LangChain-Hub.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/03-LangChain-Hub.ipynb)
## Overview

This is an example of retrieving and executing prompts from LangChain Hub.

LangChain Hub is a repository that collects prompts frequently used across various projects. This enables developers to efficiently search for, retrieve, and execute these prompts whenever needed, thereby streamlining their workflow.

- **Prompt Search and Categorization**: Developers can easily find the desired prompts using keyword-based search and categorization.
- **Reusability**: Once created, a prompt can be reused across multiple projects, reducing development time.
- **Real-time Execution**: Retrieved prompts can be executed immediately through LangChain to view the results in real time.
- **Extensibility and Customization**: In addition to the default prompts provided, users have the flexibility to add and modify prompts according to their needs.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Getting Prompts from Hub](#getting-prompts-from-hub)
- [Register Your Own Prompt to Prompt Hub](#register-your-own-prompt-to-prompt-hub)

### References

- [LangChain Hub](https://python.langchain.com/api_reference/langchain/hub.html#langchain-hub)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- You can check LangChain Hub prompts at the address below.
  - You can retrieve prompts by using the prompt repo ID, and you can also get prompts for specific versions by adding the commit ID.
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

You can check LangChain Hub prompts at the address below.

You can retrieve prompts using the prompt repo ID, and you can also get prompts for specific versions by adding the commit ID.



```python
%%capture --no-stderr
%pip install langchain-opentutorial langchain langchainhub
```

```python
# Install required packages 
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchainhub"
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
        # Get an API key for your Personal organization if you have not yet. The hub will not work with your non-personal organization's api key!
        # If you already have LANGCHAIN_API_KEY set to a personal organization’s api key from LangSmith, you can skip this.
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Personal Prompts for LangChain",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Getting Prompts from Hub

- Retrieve and execute prompts directly from LangChain Hub to accelerate your workflow.
- How to seamlessly integrate available prompts into your projects.


```python
from langchain import hub 

# Get the latest version of the prompt
prompt = hub.pull("rlm/rag-prompt")
```

```python
# Print the prompt content
print(prompt)
```

<pre class="custom">input_variables=['context', 'question'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]
</pre>

```python
# To get a specific version of prompt, specify the version hash
prompt = hub.pull("rlm/rag-prompt:50442af1")
prompt
```




<pre class="custom">ChatPromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})])</pre>



## Register Your Own Prompt to Prompt Hub

- Registering your own prompt to Prompt Hub allows developers to share custom prompts with the community, making them reusable across various projects.
- This feature enhances prompt standardization and efficient management, streamlining development and fostering collaboration.

```python
from langchain.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    "Summarize the following text based on the given content. Please write the answer in Korean\n\nCONTEXT: {context}\n\nSUMMARY:"
)
prompt
```




<pre class="custom">ChatPromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template='Summarize the following text based on the given content. Please write the answer in Korean\n\nCONTEXT: {context}\n\nSUMMARY:'), additional_kwargs={})])</pre>



```python
from langchain import hub

# Upload the prompt to the hub
hub.push("cjlee/simple-summary-korean-1", prompt)
```




<pre class="custom">'https://smith.langchain.com/prompts/simple-summary-korean-1/3635fdf1?organizationId=f03a1307-d0da-5ea5-9ee0-4fc021a0d5b2'</pre>



The following is the output after successfully uploading to Hub.

ID/PromptName/Hash

> [Output](https://smith.langchain.com/hub/teddynote/simple-summary-korean/0e296563)

```python
from langchain import hub

# Get the prompt from the hub
pulled_prompt = hub.pull("teddynote/simple-summary-korean")
```

```python
# Print the prompt content
print(pulled_prompt)
```

<pre class="custom">input_variables=['context'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'teddynote', 'lc_hub_repo': 'simple-summary-korean', 'lc_hub_commit_hash': 'b7e31df5666de7758d72fd038875973520d141548280185ee5b5ba846f015308'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template='주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:'), additional_kwargs={})]
</pre>

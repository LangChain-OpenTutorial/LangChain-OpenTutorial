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

# ConversationEntityMemory

- Author: [ulysyszh](https://github.com/ulysyszh)
- Design: [ulysyszh](https://github.com/ulysyszh)
- Peer Review: [rlatjcj](https://github.com/rlatjcj), [gyjong](https://github.com/gyjong)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/04_ConversationEntityMemory.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/04_ConversationEntityMemory.ipynb)


## Overview

`ConversationEntityMemory` allows the conversation system to retain facts about specific entities mentioned during the dialogue.

In this case, Extracting information about entities from the conversation using a large language model.

Accumulating knowledge about these entities over time using the same large language model.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Entity Memory Conversation Example](#entity-memory-conversation-example)
- [Retrieving Entity Memory](#retrieving-entity-memory)

### References

- [LangChain Python API Reference > langchain: 0.3.13 > memory > ConversationEntityMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.entity.ConversationEntityMemory.html)
----

## Environment Setup
Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.


```python
## Environment Setup
!pip install langchain langchain-community langchain-opentutorial langchain-openai
```

<pre class="custom">Requirement already satisfied: langchain in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (0.3.13)
    Requirement already satisfied: langchain-community in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (0.3.13)
    Requirement already satisfied: langchain-opentutorial in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (0.0.3)
    Requirement already satisfied: langchain-openai in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (0.2.14)
    Requirement already satisfied: PyYAML>=5.3 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (6.0.2)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (2.0.36)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (3.11.11)
    Requirement already satisfied: langchain-core<0.4.0,>=0.3.26 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (0.3.28)
    Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.3 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (0.3.4)
    Requirement already satisfied: langsmith<0.3,>=0.1.17 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (0.2.7)
    Requirement already satisfied: numpy<2,>=1.22.4 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (1.26.4)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (2.10.4)
    Requirement already satisfied: requests<3,>=2 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (2.32.3)
    Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain) (9.0.0)
    Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-community) (0.6.7)
    Requirement already satisfied: httpx-sse<0.5.0,>=0.4.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-community) (0.4.0)
    Requirement already satisfied: pydantic-settings<3.0.0,>=2.4.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-community) (2.7.1)
    Requirement already satisfied: openai<2.0.0,>=1.58.1 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-openai) (1.58.1)
    Requirement already satisfied: tiktoken<1,>=0.7 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-openai) (0.8.0)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (0.2.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.18.3)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain-community) (3.23.2)
    Requirement already satisfied: typing-inspect<1,>=0.4.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain-community) (0.9.0)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (1.33)
    Requirement already satisfied: packaging<25,>=23.2 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (24.2)
    Requirement already satisfied: typing-extensions>=4.7 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (4.12.2)
    Requirement already satisfied: httpx<1,>=0.23.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (0.27.2)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (3.10.13)
    Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (1.0.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain-openai) (4.7.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain-openai) (1.9.0)
    Requirement already satisfied: jiter<1,>=0.4.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain-openai) (0.8.2)
    Requirement already satisfied: sniffio in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain-openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain-openai) (4.67.1)
    Requirement already satisfied: annotated-types>=0.6.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
    Requirement already satisfied: pydantic-core==2.27.2 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.2)
    Requirement already satisfied: python-dotenv>=0.21.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from pydantic-settings<3.0.0,>=2.4.0->langchain-community) (1.0.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (2024.12.14)
    Requirement already satisfied: greenlet!=0.4.17 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from SQLAlchemy<3,>=1.4->langchain) (3.1.1)
    Requirement already satisfied: regex>=2022.1.18 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from tiktoken<1,>=0.7->langchain-openai) (2024.11.6)
    Requirement already satisfied: httpcore==1.* in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from httpx<1,>=0.23.0->langsmith<0.3,>=0.1.17->langchain) (1.0.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<0.3,>=0.1.17->langchain) (0.14.0)
    Requirement already satisfied: jsonpointer>=1.9 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.26->langchain) (3.0.0)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /home/a/.cache/pypoetry/virtualenvs/langchain-opentutorial-ZHleb-h7-py3.11/lib/python3.11/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain-community) (1.0.0)
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "ConversationEntityMemory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it.

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Entity Memory Conversation Example

This example demonstrates how to use `ConversationEntityMemory` to store and manage information about entities mentioned during a conversation. The conversation accumulates ongoing knowledge about these entities while maintaining a natural flow.


```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory.entity import ConversationEntityMemory
```

```python
from langchain.prompts import PromptTemplate

entity_memory_conversation_template = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template="""
You are an assistant to a human, powered by a large language model trained by OpenAI.

You assist with various tasks, from answering simple questions to providing detailed discussions on a wide range of topics. You can generate human-like text, allowing natural conversations and coherent, relevant responses.

You constantly learn and improve, processing large amounts of text to provide accurate and informative responses. You can use personalized information provided in the context below, along with your own generated knowledge.

Context:
{entities}

Current conversation:
{history}
Last line:
Human: {input}
You:
""",
)

print(entity_memory_conversation_template)
```

<pre class="custom">input_variables=['entities', 'history', 'input'] input_types={} partial_variables={} template='\nYou are an assistant to a human, powered by a large language model trained by OpenAI.\n\nYou assist with various tasks, from answering simple questions to providing detailed discussions on a wide range of topics. You can generate human-like text, allowing natural conversations and coherent, relevant responses.\n\nYou constantly learn and improve, processing large amounts of text to provide accurate and informative responses. You can use personalized information provided in the context below, along with your own generated knowledge.\n\nContext:\n{entities}\n\nCurrent conversation:\n{history}\nLast line:\nHuman: {input}\nYou:\n'
</pre>

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

conversation = ConversationChain(
    llm=llm,
    prompt=entity_memory_conversation_template,
    memory=ConversationEntityMemory(llm=llm),
)
```

## Retrieving Entity Memory
Let's examine the conversation history stored in memory using the `memory.entity_store.store` method to verify memory retention.

```python
# Input conversation
response = conversation.predict(
    input=(
        "Amelia is an award-winning landscape photographer who has traveled around the globe capturing natural wonders. "
        "David is a wildlife conservationist dedicated to protecting endangered species. "
        "They are planning to open a nature-inspired photography gallery and learning center that raises funds for conservation projects."
    )
)

# Print the assistant's response
print(response)
```

<pre class="custom">That sounds like a fantastic initiative! Combining Amelia's stunning landscape photography with David's passion for wildlife conservation could create a powerful platform for raising awareness and funds. What kind of exhibits or programs are they considering for the gallery and learning center?
</pre>

```python
# Print the entity memory
conversation.memory.entity_store.store
```




<pre class="custom">{'Amelia': 'Amelia is an award-winning landscape photographer who has traveled around the globe capturing natural wonders and is planning to open a nature-inspired photography gallery and learning center with David, a wildlife conservationist, to raise funds for conservation projects.',
     'David': 'David is a wildlife conservationist dedicated to protecting endangered species, and he is planning to open a nature-inspired photography gallery and learning center with Amelia that raises funds for conservation projects.'}</pre>



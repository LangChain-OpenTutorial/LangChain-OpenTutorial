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
- Peer Review: [rlatjcj](https://github.com/rlatjcj), [gyjong](https://github.com/gyjong)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni) 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/04-ConversationEntityMemory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/04-ConversationEntityMemory.ipynb)

## Overview

```ConversationEntityMemory``` allows the conversation system to retain facts about specific entities mentioned during the dialogue.

It extracts information about entities from the conversation (using an LLM) and 
accumulates knowledge about these entities over time (also using an LLM)


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
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.


```python
## Environment Setup
%%capture --no-stderr
%pip install langchain langchain-opentutorial langchain-community langchain-openai
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
        "LANGCHAIN_PROJECT": "ConversationEntityMemory",
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



## Entity Memory Conversation Example

This example demonstrates how to use ```ConversationEntityMemory``` to store and manage information about entities mentioned during a conversation. The conversation accumulates ongoing knowledge about these entities while maintaining a natural flow.


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

## Retrieving Entity Memory
Let's examine the conversation history stored in memory using the ```memory.entity_store.store``` method to verify memory retention.

```python
# Print the entity memory
conversation.memory.entity_store.store
```




<pre class="custom">{'Amelia': 'Amelia is an award-winning landscape photographer who has traveled around the globe capturing natural wonders and is planning to open a nature-inspired photography gallery and learning center with David, a wildlife conservationist, to raise funds for conservation projects.',
     'David': 'David is a wildlife conservationist dedicated to protecting endangered species, and he is planning to open a nature-inspired photography gallery and learning center with Amelia that raises funds for conservation projects.'}</pre>



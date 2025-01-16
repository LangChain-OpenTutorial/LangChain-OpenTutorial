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

# ConversationTokenBufferMemory

- Author: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Design: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Peer Review : [Wooseok Jeong](https://github.com/jeong-wooseok), [JeongGi Park](https://www.linkedin.com/in/jeonggipark/)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/03-ConversationTokenBufferMemory.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/03-ConversationTokenBufferMemory.ipynb)

## Overview

`ConversationTokenBufferMemory` stores recent conversation history in a buffer memory and determines when to flush conversation content based on **token length** rather than the number of conversations.

Key parameters:
- `max_token_limit`: Sets the maximum token length for storing conversation content
- `return_messages`: When True, returns the messages in chat format. When False, returns a string
- `human_prefix`: Prefix to add before human messages (default: "Human")
- `ai_prefix`: Prefix to add before AI messages (default: "AI")


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Limit Maximum Token Length to 50](#limit-maximum-token-length-to-50)
- [Setting Maximum Token Length to 150](#setting-maximum-token-length-to-150)

### References

- [LangChain Python API Reference > langchain: 0.3.13 > memory > ConversationTokenBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.token_buffer.ConversationTokenBufferMemory.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain-anthropic",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
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
        "LANGCHAIN_PROJECT": "ConversationTokenBufferMemory",
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



## Limit Maximum Token Length to 50

This section demonstrates how to limit the conversation memory to 50 tokens

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI


# Create LLM model
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Configure memory
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=50,
    return_messages=True,  # Limit maximum token length to 50
)
```


    ---------------------------------------------------------------------------

    OpenAIError                               Traceback (most recent call last)

    Cell In[1], line 6
          2 from langchain_openai import ChatOpenAI
          5 # Create LLM model
    ----> 6 llm = ChatOpenAI(model_name="gpt-4o-mini")
          8 # Configure memory
          9 memory = ConversationTokenBufferMemory(
         10     llm=llm,
         11     max_token_limit=50,
         12     return_messages=True,  # Limit maximum token length to 50
         13 )
    

    File ~/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-0Vwtx6mm-py3.11/lib/python3.11/site-packages/langchain_core/load/serializable.py:125, in Serializable.__init__(self, *args, **kwargs)
        123 def __init__(self, *args: Any, **kwargs: Any) -> None:
        124     """"""
    --> 125     super().__init__(*args, **kwargs)
    

        [... skipping hidden 1 frame]
    

    File ~/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-0Vwtx6mm-py3.11/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:578, in BaseChatOpenAI.validate_environment(self)
        576         self.http_client = httpx.Client(proxy=self.openai_proxy)
        577     sync_specific = {"http_client": self.http_client}
    --> 578     self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
        579     self.client = self.root_client.chat.completions
        580 if not self.async_client:
    

    File ~/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-0Vwtx6mm-py3.11/lib/python3.11/site-packages/openai/_client.py:110, in OpenAI.__init__(self, api_key, organization, project, base_url, websocket_base_url, timeout, max_retries, default_headers, default_query, http_client, _strict_response_validation)
        108     api_key = os.environ.get("OPENAI_API_KEY")
        109 if api_key is None:
    --> 110     raise OpenAIError(
        111         "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
        112     )
        113 self.api_key = api_key
        115 if organization is None:
    

    OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable


```python
# Add arbitrary conversations
memory.save_context(
    inputs={
        "human": "Hello, I recently purchased a machine tool from your company. Could you tell me how to install it?"
    },
    outputs={
        "ai": "Hello! Thank you for your purchase. Could you please tell me the machine model number?"
    },
)
memory.save_context(
    inputs={"human": "Yes, the model number is XG-200."},
    outputs={
        "ai": "Thank you. I'll help you with the installation guide for the XG-200 model. First, please check the power supply status at the installation site. The machine requires 220V power."
    },
)
memory.save_context(
    inputs={"human": "I've checked the power. What's the next step?"},
    outputs={
        "ai": "Good. Next, please place the machine on a flat and stable surface. Then, proceed with cable connections according to the provided user manual."
    },
)
memory.save_context(
    inputs={"human": "How do I make the connections?"},
    outputs={
        "ai": "Please refer to page 5 of the manual. There are detailed instructions for cable connections. If you have any difficulties with this process, I'll be happy to help further."
    },
)
memory.save_context(
    inputs={"human": "What should I do after the installation is complete?"},
    outputs={
        "ai": "Once the installation is complete, please turn on the power and perform the initial operation test. The test procedure is explained on page 10 of the manual. If there are any issues with the machine or if you need additional support, please don't hesitate to contact us."
    },
)
memory.save_context(
    inputs={"human": "Thank you, this has been very helpful!"},
    outputs={
        "ai": "We're always ready to help. If you have any additional questions or need support, please feel free to ask. Have a great day!"
    },
)
```

```python
# Check the conversation history
memory.load_memory_variables({})["history"]
```




<pre class="custom">[HumanMessage(content='Thank you, this has been very helpful!', additional_kwargs={}, response_metadata={}),
     AIMessage(content="We're always ready to help. If you have any additional questions or need support, please feel free to ask. Have a great day!", additional_kwargs={}, response_metadata={})]</pre>



## Setting Maximum Token Length to 150

Let's check how the conversation is stored when we set the maximum token length to **150**.

```python
# Memory configuration
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=150,
    return_messages=True,  # Limit maximum token length to 150
)
```

```python
# Add arbitrary conversations
memory.save_context(
    inputs={
        "human": "Hello, I recently purchased a machine tool from your company. Could you tell me how to install it?"
    },
    outputs={
        "ai": "Hello! Thank you for your purchase. Could you please tell me the machine model number?"
    },
)
memory.save_context(
    inputs={"human": "Yes, the model number is XG-200."},
    outputs={
        "ai": "Thank you. I'll help you with the installation guide for the XG-200 model. First, please check the power supply status at the installation site. The machine requires 220V power."
    },
)
memory.save_context(
    inputs={"human": "I've checked the power. What's the next step?"},
    outputs={
        "ai": "Good. Next, please place the machine on a flat and stable surface. Then, proceed with cable connections according to the provided user manual."
    },
)
memory.save_context(
    inputs={"human": "How do I make the connections?"},
    outputs={
        "ai": "Please refer to page 5 of the manual. There are detailed instructions for cable connections. If you have any difficulties with this process, I'll be happy to help further."
    },
)
memory.save_context(
    inputs={"human": "What should I do after the installation is complete?"},
    outputs={
        "ai": "Once the installation is complete, please turn on the power and perform the initial operation test. The test procedure is explained on page 10 of the manual. If there are any issues with the machine or if you need additional support, please don't hesitate to contact us."
    },
)
memory.save_context(
    inputs={"human": "Thank you, this has been very helpful!"},
    outputs={
        "ai": "We're always ready to help. If you have any additional questions or need support, please feel free to ask. Have a great day!"
    },
)
```

```python
# Check the conversation history
memory.load_memory_variables({})["history"]
```




<pre class="custom">[HumanMessage(content='What should I do after the installation is complete?', additional_kwargs={}, response_metadata={}),
     AIMessage(content="Once the installation is complete, please turn on the power and perform the initial operation test. The test procedure is explained on page 10 of the manual. If there are any issues with the machine or if you need additional support, please don't hesitate to contact us.", additional_kwargs={}, response_metadata={}),
     HumanMessage(content='Thank you, this has been very helpful!', additional_kwargs={}, response_metadata={}),
     AIMessage(content="We're always ready to help. If you have any additional questions or need support, please feel free to ask. Have a great day!", additional_kwargs={}, response_metadata={})]</pre>



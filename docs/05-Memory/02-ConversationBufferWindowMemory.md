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

# ConversationBufferWindowMemory

- Author: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Design: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Peer Review : [Wooseok Jeong](https://github.com/jeong-wooseok), [JeongGi Park](https://www.linkedin.com/in/jeonggipark/)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/02-ConversationBufferWindowMemory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/02-ConversationBufferWindowMemory.ipynb)
## Overview

This tutorial introduces ```ConversationBufferWindowMemory```, which maintains a sliding window of recent conversation interactions over time.

Instead of utilizing all conversation content, ```ConversationBufferWindowMemory``` retains only the **most recent K** interactions.

This prevents the memory buffer from growing too large.


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Online Bank Account Opening Conversation Example](#online-bank-account-opening-conversation-example)
- [Retrieving Conversation History](#retrieving-conversation-history)

### References

- [LangChain Python API Reference > langchain: 0.3.13 > memory > ConversationBufferWindowMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)
----

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
        "LANGCHAIN_PROJECT": "ConversationBufferWindowMemory",
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



## Online Bank Account Opening Conversation Example

This example demonstrates using ```ConversationBufferWindowMemory``` to simulate a virtual banking assistant conversation. The flow covers a typical online bank account opening process, from initial greeting to account creation confirmation, retaining only the most recent interactions in memory.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True)

memory.save_context(
    inputs={
        "human": "Hello, I'd like to open a bank account online. How do I get started?"
    },
    outputs={
        "ai": "Hello! I'm glad you want to open an account. First, could you please prepare your ID for verification?"
    },
)
memory.save_context(
    inputs={"human": "Yes, I have my ID ready. What should I do next?"},
    outputs={
        "ai": "Thank you. Please take clear photos of both sides of your ID and upload them. Then we'll proceed with the identity verification process."
    },
)
memory.save_context(
    inputs={
        "human": "I've uploaded the photos. How does the identity verification work?"
    },
    outputs={
        "ai": "I've confirmed your uploaded photos. Now, please proceed with phone verification. You'll receive a verification code via text message - please enter that code."
    },
)
memory.save_context(
    inputs={
        "human": "I've entered the verification code. How do I proceed with opening the account?"
    },
    outputs={
        "ai": "Your identity has been verified. Now, please select the type of account you want and enter the required information. You can choose the type of deposit, currency, and other options."
    },
)
memory.save_context(
    inputs={"human": "I've entered all the information. What's the next step?"},
    outputs={
        "ai": "I've confirmed your information. The account opening process is almost complete. Finally, please agree to the terms of service and confirm the account opening."
    },
)
memory.save_context(
    inputs={"human": "I've completed all the steps. Has the account been opened?"},
    outputs={
        "ai": "Yes, your account has been successfully opened. Your account number and related information have been sent to your registered email. Please let us know if you need any further assistance. Thank you!"
    },
)
```

## Retrieving Conversation History

Let's examine the stored conversation history using the ```load_memory_variables()``` method to verify our window-based memory retention.

```python
# Check the conversation history
memory.load_memory_variables({})["history"]
```




<pre class="custom">[HumanMessage(content="I've entered all the information. What's the next step?", additional_kwargs={}, response_metadata={}),
     AIMessage(content="I've confirmed your information. The account opening process is almost complete. Finally, please agree to the terms of service and confirm the account opening.", additional_kwargs={}, response_metadata={}),
     HumanMessage(content="I've completed all the steps. Has the account been opened?", additional_kwargs={}, response_metadata={}),
     AIMessage(content='Yes, your account has been successfully opened. Your account number and related information have been sent to your registered email. Please let us know if you need any further assistance. Thank you!', additional_kwargs={}, response_metadata={})]</pre>



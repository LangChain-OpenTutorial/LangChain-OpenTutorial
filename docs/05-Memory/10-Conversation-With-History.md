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

# Conversation With History

- Author: [3dkids](https://github.com/3dkids), [Joonha Jeon](https://github.com/realjoonha)
- Peer Review : [Teddy Lee](https://github.com/teddylee777), [Shinar12](https://github.com/Shinar12), [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung), [Sunyoung Park (architectyou)](https://github.com/Architectyou)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/10-Conversation-With-History.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/10-Conversation-With-History.ipynb)

## Overview

This tutorial covers how to create a multi-turn ```Chain``` that remembers previous conversations, using LangChain.<br>
It includes managing conversation history, defining a ```ChatPromptTemplate```, and utilizing an LLM for chain creation.<br>
The conversation history is managed using ```chat_history```.



### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Creating a Chain that Remembers Previous Conversations](#creating-a-chain-that-remembers-previous-conversations)
- [Creating a Chain to Record Conversations](#creating-a-chain-to-record-conversations-chain_with_history)

### References

- [LangChain: MessagesPlaceholder](https://python.langchain.com/docs/concepts/prompt_templates/#messagesplaceholder)
- [LangChain: chatmessagehistory](https://python.langchain.com/docs/versions/migrating_memory/chat_history/#chatmessagehistory)
- [LangChain: runnablewithmessagehistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#runnablewithmessagehistory)
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
    ["langchain_core", "langchain_community", "langchain_openai"],
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
        "LANGCHAIN_PROJECT": "ConversationWithHistory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Alternatively, environment variables can also be set using a ```.env``` file.

**[Note]**

- This is not necessary if you've already set the environment variables in the previous step.

```python
from dotenv import load_dotenv

load_dotenv()
```

## Creating a Chain that Remembers Previous Conversations

```MessagesPlaceholder``` is a class in LangChain used to handle conversation history. It is primarily utilized in chatbots or multi-turn conversation systems to store and reuse previous conversation content.

### Key Roles  
**Inserting Conversation History** :  
- Used to insert prior conversations (e.g., question-and-answer history) into the prompt.  
- This allows the model to understand the context of the conversation and generate appropriate responses.  

**Managing Variables** :  
- Manages conversation history within the prompt using a specific key (e.g., ```chat_history```).  
- It is linked to a user-defined variable name.  

### Usage  
```MessagesPlaceholder(variable_name="chat_history")```  
- Here, ```chat_history``` is the variable name where conversation history is stored.  
- As the conversation progresses, ```chat_history``` is continually updated with pairs of questions and responses.


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Define the prompt.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Question-Answering chatbot. Please provide answers to the given questions.",
        ),
        # Use "chat_history" as the key for conversation history without modifying it if possible.
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),  # Use user input as a variable.
    ]
)

# Create the LLM.
llm = ChatOpenAI(model_name="gpt-4o")

# Create a basic chain.
chain = prompt | llm | StrOutputParser()
```

## Creating a Chain to Record Conversations (```chain_with_history```)

In this step, we create a system that manages **session-based conversation history** and generates an **executable chain**.

- **Conversation History Management** : The ```store``` dictionary saves and retrieves conversation history (```ChatMessageHistory```) by session ID. If a session does not exist, a new one is created.  
- **Chain Execution** : ```RunnableWithMessageHistory``` combines conversation history and the chain to generate responses based on user questions and conversation history. This structure is designed to effectively manage multi-turn conversations.


```python
# A dictionary to store session history.
store = {}


# A function to retrieve session history based on the session ID.
def get_session_history(session_ids):
    print(f"[Conversation session ID]: {session_ids}")
    if session_ids not in store:  # When the session ID is not in the store.
        # Create a new ChatMessageHistory object and save it in the store.
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the given session ID.
```

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # A function to retrieve session history.
    input_messages_key="question",  # The key where the user's question will be inserted into the template variable.
    history_messages_key="chat_history",  # The key for the message in the history.
)
```

Execute the first question.

```python
chain_with_history.invoke(
    # Question input.
    {"question": "My name is Teddy."},
    # Record conversations based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation session ID]: abc123
</pre>




    "Hello, Teddy! Do you have a question or something specific you'd like to discuss?"



Execute the next question.

```python
chain_with_history.invoke(
    # Question input.
    {"question": "What's my name?"},
    # Record conversations based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation session ID]: abc123
</pre>




    'Your name is Teddy.'



Below is a case where a new session is created when the ```session_id``` is different.

```python
chain_with_history.invoke(
    # Question input.
    {"question": "What's my name?"},
    # Record conversations based on the session ID.
    config={"configurable": {"session_id": "abc1234"}},
)
```

<pre class="custom">[Conversation session ID]: abc1234
</pre>




    "I'm sorry, but I don't have access to personal information about you, including your name."



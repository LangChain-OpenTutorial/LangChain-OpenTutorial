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

# Memory Using SQLite

- Author: [Heesun Moon](https://github.com/MoonHeesun)
- Peer Review: [harheem](https://github.com/harheem), [gyjong](https://github.com/gyjong)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/09-Memory-Using-SQLite.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/09-Memory-Using-SQLite.ipynb)
## Overview

This tutorial explains the ```SQLChatMessageHistory``` class, which allows storing chat history in any database supported by ```SQLAlchemy```.

```Structured Query Language (SQL)``` is a domain-specific language used in programming and designed for managing data held in a Relational Database Management System (RDBMS), or for stream processing in a Relational Data Stream Management System (RDSMS). It is particularly useful for handling structured data, including relationships between entities and variables.

```SQLAlchemy``` is an open-source **SQL** toolkit and Object-Relational Mapper (ORM) for the Python programming language, released under the MIT License.

To use a database other than ```SQLite```, please make sure to install the appropriate database driver first.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Usage](#Usage)
- [Chaining](#Chaining)

### References

- [Wikipedia: SQL](https://en.wikipedia.org/wiki/SQL)
- [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain_opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_community",
        "langchain_openai",
        "langchain_core",
        "SQLAlchemy",
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
        "LANGCHAIN_PROJECT": "MemoryUsingSQLite",
    }
)
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```

## Usage

To use the storage, you need to provide only the following 2 things:

1. ```session_id``` - A unique identifier for the session, such as a user name, email, chat ID, etc.

2. ```connection``` - A string that specifies the database connection. This string will be passed to SQLAlchemy's ```create_engine``` function.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Initialize chat history with session ID and database connection.
chat_message_history = SQLChatMessageHistory(
    session_id="sql_history", connection="sqlite:///sqlite.db"
)
```

```python
# Add a user message
chat_message_history.add_user_message(
    "Hello, nice to meet you! My name is Heesun :) I'm a LangChain developer. I look forward to working with you!"
)
# Add an AI message
chat_message_history.add_ai_message(
    "Hi, Heesun! Nice to meet you. I look forward to working with you too!"
)
```

Now, let's check the stored conversation history.

```python
chat_message_history.messages
```




<pre class="custom">[HumanMessage(content="Hello, nice to meet you! My name is Heesun :) I'm a LangChain developer. I look forward to working with you!", additional_kwargs={}, response_metadata={}),
     AIMessage(content='Hi, Heesun! Nice to meet you. I look forward to working with you too!', additional_kwargs={}, response_metadata={})]</pre>



You can also clear the session memory from db:

```python
# Clear the session memory
chat_message_history.clear()
chat_message_history.messages
```




<pre class="custom">[]</pre>



### Adding Metadata

**Metadata** can be added by directly creating ```HumanMessage``` and ```AIMessage``` objects. This approach enables flexible data handling and logging.

**Parameters**:
- ```additional_kwargs``` - Stores custom tags or metadata, such as priority or task type.

- ```response_metadata``` - Captures AI response details, including model, timestamp, and token count.

These fields enhance debugging and task tracking through detailed data storage.

```python
from langchain_core.messages import HumanMessage

# Add a user message with additional metadata.
user_message = HumanMessage(
    content="Can you help me summarize this text?",
    additional_kwargs={"task": "summarization"},
)

# Add the message to chat history.
chat_message_history.add_message(user_message)
```

```python
chat_message_history.messages
```




<pre class="custom">[HumanMessage(content='Can you help me summarize this text?', additional_kwargs={'task': 'summarization'}, response_metadata={})]</pre>



```python
from langchain_core.messages import AIMessage

# Add an AI message with response metadata.
ai_message = AIMessage(
    content="Sure! Here's the summary of the provided text.",
    response_metadata={"model": "gpt-4", "token_count": 30, "response_time": "150ms"},
)

# Add the message to chat history.
chat_message_history.add_message(ai_message)
```

```python
chat_message_history.messages
```




<pre class="custom">[HumanMessage(content='Can you help me summarize this text?', additional_kwargs={'task': 'summarization'}, response_metadata={}),
     AIMessage(content="Sure! Here's the summary of the provided text.", additional_kwargs={}, response_metadata={'model': 'gpt-4', 'token_count': 30, 'response_time': '150ms'})]</pre>



## Chaining

You can easily integrate this chat history class with [LCEL Runnables](https://wikidocs.net/235884).

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
```

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # Placeholder for chat history
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Chaining
chain = prompt | ChatOpenAI(model_name="gpt-4o") | StrOutputParser()
```

The following shows how to create a function that returns chat history from ```sqlite.db```.

```python
def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )
```

Set ```config_fields``` to provide reference information when retrieving conversation details.

```python
from langchain_core.runnables.utils import ConfigurableFieldSpec

config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]
```

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    # Set parameters for retrieving chat history
    history_factory_config=config_fields,
)
```

Set the ```"user_id"``` and ```"conversation_id"``` key-value pairs under the ```"configurable"``` key.

```python
# Config settings
config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}
```

Let's ask a question about the name. If there is any previously saved conversation history, it will provide the correct response.  

- Use the ```invoke``` method of the ```chain_with_history``` object to generate an answer to the question.  
- Pass a question dictionary and ```config``` settings to the ```invoke``` method as inputs.  

```python
# Execute by passing the question and config
chain_with_history.invoke(
    {"question": "Hi, nice to meet you. My name is Heesun."}, config
)
```




<pre class="custom">'Hi Heesun! Nice to meet you again. How can I help you today?'</pre>



```python
# Execute a follow-up question
chain_with_history.invoke({"question": "What is my name?"}, config)
```




<pre class="custom">'Your name is Heesun.'</pre>



This time, set the same ```user_id``` but use a different value for ```conversation_id```.

```python
# Config settings
config = {"configurable": {"user_id": "user1", "conversation_id": "conversation2"}}

# Execute by passing the question and config
chain_with_history.invoke({"question": "What is my name?"}, config)
```




<pre class="custom">"I'm sorry, but I don't have access to personal information, so I don't know your name. If you'd like, you can tell me your name, and I can address you by it."</pre>



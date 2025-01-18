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

# RunnableWithMessageHistory

- Author: [Secludor](https://github.com/Secludor)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb)

## Overview

`RunnableWithMessageHistory` is a powerful tool in LangChain's Expression Language (LCEL) **for managing conversation history** in chatbots, virtual assistants, and other conversational AI applications. This class seamlessly integrates with existing LangChain components **to handle message history management and updates automatically.**

### Key Features

**Message History Management**
- Maintains conversation context across multiple interactions.
- Automatically tracks and stores chat messages.
- Enables contextual responses based on previous conversations.

**Flexible Input/Output Support**
- Handles both message objects and Python dictionaries.
- Supports various input formats including:
  - Single messages
  - Message sequences
  - Dictionary inputs with custom keys
- Provides consistent output handling regardless of input format.

**Session Management**
- Manages conversations through unique identifiers:
  - Simple session IDs
  - Combined user and conversation IDs
- Maintains separate conversation threads for different users or contexts.
  - Ensures conversation continuity within the same session.

**Storage Options**
- In-memory storage for development and testing.
- Persistent storage support (Redis, files, etc.) for production.
- Easy integration with various storage backends.

**Advantages Over Legacy Approaches**
- More flexible than the older ConversationChain.
- Better state management capabilities.
- Improved integration with modern LangChain components.

### Summary
`RunnableWithMessageHistory` serves as the new standard for conversation management in LangChain, offering:
- Simplified conversation state management.
- Enhanced user experience through context preservation.
- Flexible configuration options for different use cases.

### Table of Contents

- [Overview](#overview)  
- [Environment Setup](#environment-setup)  
- [Getting Started with RunnableWithMessageHistory](#getting-started-with-runnablewithmessagehistory)
- [In-Memory Conversation History](#in-memory-conversation-history)
- [Example of Runnables with Using Veriety Keys](#example-of-runnables-with-using-veriety-keys)
- [Persistent Storage](#persistent-storage)
- [Using Redis for Persistence](#using-redis-for-persistence)

### References

- [LangChain Core API Documentation - RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory)
- [LangChain Documentation - Message History](https://python.langchain.com/docs/how_to/message_history/)
- [LangChain Memory Integrations](https://integrations.langchain.com/memory)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_openai",
        "langchain_core",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "08-RunnableWithMessageHistory",
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



## Getting Started with `RunnableWithMessageHistory`
Message history management is crucial for conversational applications and complex data processing tasks. To effectively implement message history with `RunnableWithMessageHistory`, you need two key components: 

1. **Creating a `Runnable`**
  - An object that primarily interacts with `BaseChatMessageHistory`, such as Retriever and Chain.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant skilled in {ability}. Keep responses under 20 words.",
        ),
        # Use the conversation history as a variable, with 'history' as the key for MessageHistory.
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),  # Use user input as a variable
    ]
)
runnable = (
    prompt | model
)  # Create a runnable object by connecting the prompt and the model
```

2. **Message History Manager (callable)**

- A callable that returns an instance of `BaseChatMessageHistory` .
- Provides message storage, retrieval, and update capabilities.
- Maintains conversation context for contextual responses.

### Implementation Options

LangChain offers multiple ways to implement message history management. You can explore various storage options and integration methods in the [memory integrations](https://integrations.langchain.com/memory) page.

This tutorial covers two primary implementation approaches:

1. **In-Memory ChatMessageHistory**
   - Manages message history in memory.
   - Ideal for development and simple applications.
   - Provides fast access speeds.
   - Message history is lost on application restart.

2. **Persistent Storage with RedisChatMessageHistory**
   - Enables permanent message storage using Redis.
   - High-performance, open-source in-memory data structure store.
   - Suitable for distributed environments.
   - Ideal for complex applications and long-running services.

- Consider these factors when selecting a message history management approach:
   - Application requirements
   - Expected traffic volume
   - Message data importance
   - Retention period requirements

While in-memory implementation offers simplicity and speed, persistent storage solutions like Redis are more appropriate when data durability is required.

## In-Memory Conversation History

In-memory conversation history provides a simple and fast way to manage chat message history during development and testing. This approach stores conversation data in memory, offering quick access but without persistence across application restarts.

### Core Configuration Parameters

**Required Components**
- `runnable`: The chain or model to execute (e.g., ChatOpenAI, Chain)
- `get_session_history`: Function returning a `BaseChatMessageHistory` instance
- `input_messages_key`: Specifies the key for user input in `invoke()` calls
- `history_messages_key`: Defines the key for accessing conversation history

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # Initialize empty store for message histories.


# function for getting session logs by session_id.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    print(session_id)
    if session_id not in store:  # if session_id isn't in the store,
        # create new ChatMessageHistory and put it in the store.
        store[session_id] = ChatMessageHistory()
    return store[session_id]  # return the session logs for session_id.


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### Default Session Implementation
`RunnableWithMessageHistory` uses `session_id` as its default identifier for managing conversation threads. This is evident in its core implementation:

```python
if history_factory_config:
    _config_specs = history_factory_config
else:
    # If not provided, then we'll use the default session_id field
    _config_specs = [
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for a session.",
            default="",
            is_shared=True,
        ),
    ]
```
### Using Session Management
To utilize session management, you must specify a session ID in your invoke call:

```python
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">abc123
</pre>




    AIMessage(content='Cosine is a trigonometric function representing the adjacent side over hypotenuse in a right triangle.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 31, 'total_tokens': 53, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_5f20662549', 'finish_reason': 'stop', 'logprobs': None}, id='run-4afc17da-4dd9-46ff-888c-fff45266820e-0', usage_metadata={'input_tokens': 31, 'output_tokens': 22, 'total_tokens': 53, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



When using the same `session_id`, the conversation can continue because it retrieves the previous thread's content (this continuous conversation is called a session):

```python
# Call with message history
with_message_history.invoke(
    # Set ability and input
    {"ability": "math", "input": "Please translate the previous content to Korean"},
    # Specify configuration options
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">abc123
</pre>




    AIMessage(content='코사인은 직각 삼각형에서 인접 변을 빗변으로 나눈 값을 나타내는 삼각 함수입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 67, 'total_tokens': 96, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_5f20662549', 'finish_reason': 'stop', 'logprobs': None}, id='run-a3207321-ffe4-4c85-b89c-d418967f1ee2-0', usage_metadata={'input_tokens': 67, 'output_tokens': 29, 'total_tokens': 96, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



However, if you specify a different `session_id`, the response won't be accurate because there's no conversation history. 

(In the example below, since `session_id`: def234 doesn't exist, you can see an irrelevant response)

```python
# New session_id means no previous conversation memory
with_message_history.invoke(
    # Pass math ability and input message
    {"ability": "math", "input": "Please translate the previous content to Korean"},
    # Set a new session_id
    config={"configurable": {"session_id": "def234"}},
)
```

<pre class="custom">def234
</pre>




    AIMessage(content='이전 내용을 한국어로 번역해드릴 수 없습니다. 수학 관련 질문이 있으면 도와드리겠습니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 28, 'prompt_tokens': 33, 'total_tokens': 61, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-277d6cc5-ca04-40a5-a2a3-ff67e31e72ec-0', usage_metadata={'input_tokens': 33, 'output_tokens': 28, 'total_tokens': 61, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



The configuration parameters used for tracking message history can be customized by passing a list of `ConfigurableFieldSpec` objects through the `history_factory_config` parameter.

Setting a new `history_factory_config` will override the existing `session_id` configuration.

The example below uses two parameters: `user_id` and `conversation_id`.

```python
from langchain_core.runnables import ConfigurableFieldSpec

store = {}  # Initialize empty store for message histories.


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    # Return session history for given user_id and conversation_id combination.
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


# Configure RunnableWithMessageHistory with custom identifiers
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[  # replacing the "session_id"
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for conversation.",
            default="",
            is_shared=True,
        ),
    ],
)
```

 Using the Custom Configuration 

```python
with_message_history.invoke(
    {"ability": "math", "input": "what is cosine?"},
    config={"configurable": {"user_id": "123", "conversation_id": "abc"}},
)
```




<pre class="custom">AIMessage(content='Cosine is a trigonometric function representing the adjacent side over the hypotenuse in a right triangle.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 30, 'total_tokens': 53, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-ebd536b5-e599-4c68-b8b8-e011d43b1d0c-0', usage_metadata={'input_tokens': 30, 'output_tokens': 23, 'total_tokens': 53, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



## Example of Runnables with Using Veriety Keys

### Messages Input with Dictionary Output

 This example demonstrates how to handle message inputs and dictionary outputs in `RunnableWithMessageHistory`.

**Important** : By omitting `input_messages_key="input"` , we configure the system to accept `Message` objects as input.

```python
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel

# Create a chain that outputs dictionary with message.
chain = RunnableParallel(
    {"output_message": ChatOpenAI(model_name="gpt-4o", temperature=0)}
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Create new ChatMessageHistory if session doesn't exist.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Create RunnableWithMessageHistory with message history capability.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    # input_messages_key="input" is omitted to accept Message objects
    output_messages_key="output_message",  # Specify output format as dictionary
)

# Execute chain with message input
with_message_history.invoke(
    [HumanMessage(content="what is the definition of cosine?")],
    config={"configurable": {"session_id": "abc123"}},
)
```




<pre class="custom">{'output_message': AIMessage(content='The cosine of an angle in a right-angled triangle is defined as the ratio of the length of the adjacent side to the length of the hypotenuse. In mathematical terms, for an angle \\( \\theta \\):\n\n\\[\n\\cos(\\theta) = \\frac{\\text{Adjacent side}}{\\text{Hypotenuse}}\n\\]\n\nIn the context of the unit circle, which is a circle with a radius of 1 centered at the origin of a coordinate plane, the cosine of an angle \\( \\theta \\) is the x-coordinate of the point where the terminal side of the angle intersects the circle. This definition extends the concept of cosine to all real numbers, not just angles in right triangles.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 144, 'prompt_tokens': 14, 'total_tokens': 158, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_b7d65f1a5b', 'finish_reason': 'stop', 'logprobs': None}, id='run-6e321281-a1c6-4ea9-a4d4-3b7752534c74-0', usage_metadata={'input_tokens': 14, 'output_tokens': 144, 'total_tokens': 158, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}</pre>



```python
# Continue conversation with same session ID
with_message_history.invoke(
    [HumanMessage(content="Please translate the previous content to Korean!")],
    config={"configurable": {"session_id": "abc123"}},
)
```




<pre class="custom">{'output_message': AIMessage(content='코사인의 정의는 직각삼각형에서 한 각의 코사인은 그 각의 인접 변의 길이를 빗변의 길이로 나눈 비율로 정의됩니다. 수학적으로, 각 \\( \\theta \\)에 대해 다음과 같이 표현됩니다:\n\n\\[\n\\cos(\\theta) = \\frac{\\text{인접 변}}{\\text{빗변}}\n\\]\n\n단위원에서의 맥락에서는, 단위원은 좌표 평면의 원점에 중심을 두고 반지름이 1인 원입니다. 각 \\( \\theta \\)의 코사인은 각의 종단선이 원과 만나는 점의 x좌표입니다. 이 정의는 코사인의 개념을 직각삼각형의 각뿐만 아니라 모든 실수로 확장합니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 174, 'prompt_tokens': 173, 'total_tokens': 347, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_5f20662549', 'finish_reason': 'stop', 'logprobs': None}, id='run-4eeff02e-73fe-4b80-aeb3-25a9815c8a4d-0', usage_metadata={'input_tokens': 173, 'output_tokens': 174, 'total_tokens': 347, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}</pre>



This configuration allows:
- Direct input of `Message` objects
- Dictionary output format
- Session-based conversation history
- Seamless continuation of conversations using session IDs


### `Message` Objects as Input and Output

**Direct Message Object Handling**
- Important: Omitting `output_messages_key="output_message"` configures the system to return `Message` objects as output.

```python
with_message_history = RunnableWithMessageHistory(
    ChatOpenAI(model_name="gpt-4o", temperature=0),  # Use ChatOpenAI language model
    get_session_history,  # Function to retrieve conversation history
    # input_messages_key="input",     # Omit to accept Message objects as input
    # output_messages_key="output_message"  # Omit to return Message objects as output
)
```

```python
# Invoke with Message object input
with_message_history.invoke(
    [HumanMessage(content="What is the meaning of cosine?")],
    config={"configurable": {"session_id": "def123"}},
)
```




<pre class="custom">AIMessage(content='The term "cosine" can refer to a few different concepts depending on the context, but it is most commonly associated with trigonometry in mathematics. Here are the primary meanings:\n\n1. **Trigonometric Function**: In trigonometry, the cosine of an angle in a right triangle is defined as the ratio of the length of the adjacent side to the length of the hypotenuse. If \\(\\theta\\) is an angle in a right triangle, then:\n   \\[\n   \\cos(\\theta) = \\frac{\\text{Adjacent side}}{\\text{Hypotenuse}}\n   \\]\n   The cosine function is one of the fundamental trigonometric functions, along with sine and tangent.\n\n2. **Unit Circle Definition**: In the context of the unit circle, which is a circle with a radius of 1 centered at the origin of a coordinate plane, the cosine of an angle \\(\\theta\\) is the x-coordinate of the point on the unit circle that is reached by moving counterclockwise from the positive x-axis by an angle \\(\\theta\\).\n\n3. **Cosine Function in Calculus**: The cosine function, often written as \\(\\cos(x)\\), is a periodic function with a period of \\(2\\pi\\). It is an even function, meaning \\(\\cos(-x) = \\cos(x)\\), and it is used extensively in calculus and analysis.\n\n4. **Cosine Similarity**: In the context of data analysis and machine learning, cosine similarity is a measure of similarity between two non-zero vectors. It is calculated as the cosine of the angle between the two vectors, which is the dot product of the vectors divided by the product of their magnitudes. It is often used in text analysis and information retrieval.\n\nThese are the primary contexts in which the term "cosine" is used, each with its own specific meaning and application.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 389, 'prompt_tokens': 14, 'total_tokens': 403, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-0ba9374f-a492-4770-b1df-a0bf740b57e0-0', usage_metadata={'input_tokens': 14, 'output_tokens': 389, 'total_tokens': 403, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



### Dictionary with Single Key for All Messages

**Using a Single Key for Input/Output**
- This approach uses one key for all message inputs and outputs
- Utilizes `itemgetter("input_messages")` to extract input messages


```python
from operator import itemgetter

with_message_history = RunnableWithMessageHistory(
    itemgetter("input_messages")
    | ChatOpenAI(model_name="gpt-4o", temperature=0),  # Extract and process messages.
    get_session_history,  # Session history management
    input_messages_key="input_messages",  # Specify input message key
)
```

```python
# Invoke with dictionary input
with_message_history.invoke(
    {"input_messages": "What is the meaning of cosine?"},
    config={"configurable": {"session_id": "xyz123"}},
)
```




<pre class="custom">AIMessage(content='The term "cosine" refers to a trigonometric function that is fundamental in mathematics, particularly in the study of triangles and periodic phenomena. In the context of a right-angled triangle, the cosine of an angle is defined as the ratio of the length of the adjacent side to the length of the hypotenuse. Mathematically, for an angle \\( \\theta \\), it is expressed as:\n\n\\[\n\\cos(\\theta) = \\frac{\\text{Adjacent side}}{\\text{Hypotenuse}}\n\\]\n\nIn the unit circle, which is a circle with a radius of one centered at the origin of a coordinate plane, the cosine of an angle \\( \\theta \\) is the x-coordinate of the point where the terminal side of the angle intersects the circle.\n\nCosine is also an even function, meaning that \\( \\cos(-\\theta) = \\cos(\\theta) \\), and it has a range of values from -1 to 1. It is periodic with a period of \\( 2\\pi \\), meaning that \\( \\cos(\\theta + 2\\pi) = \\cos(\\theta) \\).\n\nCosine is widely used in various fields such as physics, engineering, and computer science, particularly in wave analysis, signal processing, and the study of oscillations.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 266, 'prompt_tokens': 14, 'total_tokens': 280, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-7616dbf4-706f-4f2b-ba66-a32f863db461-0', usage_metadata={'input_tokens': 14, 'output_tokens': 266, 'total_tokens': 280, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



This configuration allows for:
- Direct message object handling
- Simplified input/output processing
- Flexible message format conversion
- Consistent session management

## Persistent Storage

 Persistent storage is **a mechanism that maintains data even when a program terminates or the system reboots.** This can be implemented through databases, file systems, or other non-volatile storage devices.

It is **essential for preserving data long-term** in applications. It enables:
- State preservation across sessions
- User preference retention
- **Continuous operation without data loss**
- Recovery from previous execution points


### Implementation Options

`RunnableWithMessageHistory` offers flexible storage options:

- Independent of how `get_session_history` retrieves chat message history
- Supports local file system (see example [here](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/server.py))
- Integrates with various storage providers (see [memory integrations](https://integrations.langchain.com/memory))


## Using Redis for Persistence

 1. **Installation** 

```python
%pip install -qU redis
```

<pre class="custom">Note: you may need to restart the kernel to use updated packages.
</pre>

2. **Redis Server Setup**

Launch a local Redis Stack server using Docker: 

```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
Configuration options:
- -d: Run in daemon mode (background)
- -p {port}:6379: Redis server port mapping
- -p 8001:8001: RedisInsight UI port mapping
- redis/redis-stack:latest: Latest Redis Stack image

**Troubleshooting**
- Verify Docker is running
- Check port availability (terminate occupying processes or use different ports)

3. **Redis Connection**
- Set up the Redis connection URL: `"redis://localhost:{port}/0"`


```python
REDIS_URL = "redis://localhost:6379/0"
```

### Implementing Redis Message History

To update the message history implementation, define a new callable that returns an instance of `RedisChatMessageHistory`:


```python
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory


def get_message_history(session_id: str) -> RedisChatMessageHistory:
    # Return RedisChatMessageHistory instance based on session ID.
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,  # Runnable object
    get_message_history,  # Message history retrieval
    input_messages_key="input",  # Key for input messages
    history_messages_key="history",  # Key for history messages
)
```

### Testing Conversation Continuity

**First Interaction**
- You can call it in the same way as before

```python
# Initial query with new session ID
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "redis123"}},
)
```




<pre class="custom">AIMessage(content='Cosine is the ratio of the length of the adjacent side to the hypotenuse in a right triangle.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 214, 'total_tokens': 237, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_5f20662549', 'finish_reason': 'stop', 'logprobs': None}, id='run-08fb9a1f-b16d-4e19-9c2d-404052d3b111-0', usage_metadata={'input_tokens': 214, 'output_tokens': 23, 'total_tokens': 237, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



**Continuing the Conversation**
- You can perform the second call using the same `session_id`.

```python
# Second query using same session ID
with_message_history.invoke(
    {"ability": "math", "input": "Please translate the previous response to Korean"},
    config={"configurable": {"session_id": "redis123"}},
)
```




<pre class="custom">AIMessage(content='코사인은 직각삼각형에서 인접 변의 길이를 빗변의 길이로 나눈 비율입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 251, 'total_tokens': 280, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_5f20662549', 'finish_reason': 'stop', 'logprobs': None}, id='run-8256bc74-2094-4091-b34e-075bf5d973ca-0', usage_metadata={'input_tokens': 251, 'output_tokens': 29, 'total_tokens': 280, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



**Testing with Different Session**
- This time, I will ask the question using a different `session_id`.

```python
# Query with different session ID
with_message_history.invoke(
    {"ability": "math", "input": "Please translate the previous response to Korean"},
    config={"configurable": {"session_id": "redis456"}},
)
```




<pre class="custom">AIMessage(content='수학에 능숙한 도우미입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 12, 'prompt_tokens': 101, 'total_tokens': 113, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-7625665b-73b6-43c4-aebf-28a465054aa9-0', usage_metadata={'input_tokens': 101, 'output_tokens': 12, 'total_tokens': 113, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



Note: The last response will be inaccurate since there's no conversation history for the new session ID "redis456".

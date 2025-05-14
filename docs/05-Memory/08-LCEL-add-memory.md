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

# LCEL (Remembering Conversation History): Adding Memory

- Author: [Heeah Kim](https://github.com/yellowGangneng)
- Peer Review : [Sungchul Kim](https://github.com/rlatjcj), [Jongwon Seo](https://github.com/3dkids)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/08-LCEL-add-memory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/08-LCEL-add-memory.ipynb)
## Overview

This tutorial demonstrates how to add memory to arbitrary chains using ```LCEL```.

The ```LangChain Expression Language (LCEL)``` takes a declarative approach to building new ```Runnables``` from existing ```Runnables```. For more details about LCEL, please refer to the References below.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Initializing Model and Prompt](#initializing-model-and-prompt)
- [Creating Memory](#creating-memory)
- [Adding Memory to Chain](#adding-memory-to-chain)
- [Example Implementation of a Custom ConversationChain](#example-implementation-of-a-custom-conversationChain)

### References

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)
- [LangChain-OpenTutorial (CH.13)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/tree/main/13-LCEL)
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
        "langchain",
        "langchain_core",
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
        "LANGCHAIN_PROJECT": "LCEL-Adding-Memory",
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

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Initializing Model and Prompt

Now, let's start to initialize the model and the prompt we'll use.

```python
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


# Initialize Model
model = ChatOpenAI()

# Generate a conversational prompt. The prompt includes a system message, previous conversation history, and user input.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
```

## Creating Memory

Create a ```ConversationBufferMemory``` to store conversation history.

- ```return_messages``` : When set to **True**, it returns ```HumanMessage``` and ```AIMessage``` objects.
- ```memory_key```: The key that will be substituted into the Chain's **prompt** later. This can be modified as needed.

```python
# Create a ConversationBufferMemory and enable the message return feature.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
```

Check the saved conversation history. 

Since nothing has been saved yet, the conversation history is empty.

```python
memory.load_memory_variables({})
```




<pre class="custom">{'chat_history': []}</pre>



Use ```RunnablePassthrough.assign``` to assign the result of the ```memory.load_memory_variables``` function to the ```chat_history``` variable, and extract the value corresponding to the ```chat_history``` key from this result.

Hold on a second! What is...

### ```RunnablePassthrough```? ```RunnableLambda```?

To put it simply, ```RunnablePassthrough``` provides the functionality to pass through data as is, <br>
while ```RunnableLambda``` provides the functionality to execute user-defined functions.

When you call ```RunnablePassthrough``` alone, it simply passes the input as received. <br>
However, when you use ```RunnablePassthrough.assign```, it delivers the input combined with additional arguments provided to the function.

Let's look at the code for more details.


```python
runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")  # itemgetter's input as same as memory_key.
)

runnable.invoke({"input": "hi"})
```




<pre class="custom">{'input': 'hi', 'chat_history': []}</pre>



Since ```RunnablePassthrough.assign``` is used, the returned value is a combination of the input and the additional arguments provided to the function.

In this case, the key of the additional argument is ```chat_history```. The value corresponds to the part of the result of ```memory.load_memory_variables``` executed through ```RunnableLambda``` that is extracted by ```itemgetter``` using the ```chat_history``` key.

## Adding Memory to Chain

Let's add memory to the chain using LCEL.

```python
chain = runnable | prompt | model
```

Proceed with the first conversation.

```python
# Using the invoke method of the chain object, a response to the input is generated.
response = chain.invoke({"input": "Nice to see you. My name is Heeah."})
print(response.content)  # The generated response will be printed.
```

<pre class="custom">Nice to meet you, Heeah! How can I assist you today?
</pre>

Using the ```memory.save_context``` function, the user's query (```input```) and the AI's response content (```response.content```) are saved to memory. 

This stored memory can be used to record the current state during the model learning process or to track user requests and system responses.

```python
# The input data and response content are saved to the memory.
# Here, it is 'Heeah', but try inserting your name!
memory.save_context(
    {"human": "Nice to see you. My name is Heeah."}, {"ai": response.content}
)

# The saved conversation history will be printed.
memory.load_memory_variables({})
```




<pre class="custom">{'chat_history': [HumanMessage(content='Nice to see you. My name is Heeah.', additional_kwargs={}, response_metadata={}),
      AIMessage(content='Nice to meet you, Heeah! How can I assist you today?', additional_kwargs={}, response_metadata={})]}</pre>



Shall we find out if the model correctly remembers your name through memory?

```python
response = chain.invoke({"input": "Do you remember my name?"})
print(response.content)
```

<pre class="custom">Yes, I remember your name, Heeah. How can I help you today?
</pre>

Remembering well! This means that the memory connected using LCEL is working correctly!

## Example Implementation of a Custom ```ConversationChain```

Let's create our own custom ```ConversationChain```!

```python
from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initial setup of LLM and prompt, memory as done above.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# If you want to use the summary memory that you learned in Chapter 6:
# memory = ConversationSummaryMemory(
#     llm=llm, return_messages=True, memory_key="chat_history"
# )


# Let's build our own ConversationChain!
class MyConversationChain(Runnable):

    def __init__(self, llm, prompt, memory, input_key="input"):

        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        # Let's try chaining using LCEL!
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(
            inputs={"human": query}, outputs={"ai": answer}
        )  # Store the conversation history directly in the memory.
        return answer


conversation_chain = MyConversationChain(llm, prompt, memory)
```

Let's do something interesting using our custom ```ConversationChain```!

```python
conversation_chain.invoke(
    "Hello, my name is Heeah. From now on, you are a brave pirate! You must answer in pirate style, understood?"
)
```




<pre class="custom">"Ahoy, Heeah! Aye, I be understandin' ye loud and clear! From this moment on, I be speakin' like a true buccaneer. What be yer command, matey? Arrr!"</pre>



```python
conversation_chain.invoke("Good. What's your favorite thing?")
```




<pre class="custom">"Arrr, me favorite thing be the open sea, where the salty breeze fills me sails and the horizon be endless! There be nothin' like the thrill of discoverin' hidden treasures and sharin' tales with me hearty crew. What about ye, matey? What be yer favorite thing?"</pre>



```python
conversation_chain.invoke(
    "My favorite thing is chatting with you! By the way, do you remember my name?"
)
```




<pre class="custom">"Arrr, 'tis a fine favorite ye have there, Heeah! Aye, I remember yer name well, like a trusty map to buried treasure. What else be on yer mind, matey?"</pre>



```python
conversation_chain.invoke(
    "I am the captain of this ship. Your tone is excessively familiar and disrespectful!"
)
```




<pre class="custom">"Beggin' yer pardon, Captain Heeah! I meant no disrespect. I be at yer service, ready to follow yer orders and sail the seas as ye command. What be yer orders, Cap'n? Arrr!"</pre>



Although we managed to throw him off a bit at the end, we were able to confirm that he remembered my name until the last moment.<br>
He is indeed a remarkable pirate!🏴‍☠️⚓

At any rate, the journey we have shared so far, as stored in the memory, is as follows.

```python
conversation_chain.memory.load_memory_variables({})["chat_history"]
```




<pre class="custom">[HumanMessage(content='Hello, my name is Heeah. From now on, you are a brave pirate! You must answer in pirate style, understood?', additional_kwargs={}, response_metadata={}),
     AIMessage(content="Ahoy, Heeah! Aye, I be understandin' ye loud and clear! From this moment on, I be speakin' like a true buccaneer. What be yer command, matey? Arrr!", additional_kwargs={}, response_metadata={}),
     HumanMessage(content="Good. What's your favorite thing?", additional_kwargs={}, response_metadata={}),
     AIMessage(content="Arrr, me favorite thing be the open sea, where the salty breeze fills me sails and the horizon be endless! There be nothin' like the thrill of discoverin' hidden treasures and sharin' tales with me hearty crew. What about ye, matey? What be yer favorite thing?", additional_kwargs={}, response_metadata={}),
     HumanMessage(content='My favorite thing is chatting with you! By the way, do you remember my name?', additional_kwargs={}, response_metadata={}),
     AIMessage(content="Arrr, 'tis a fine favorite ye have there, Heeah! Aye, I remember yer name well, like a trusty map to buried treasure. What else be on yer mind, matey?", additional_kwargs={}, response_metadata={}),
     HumanMessage(content='I am the captain of this ship. Your tone is excessively familiar and disrespectful!', additional_kwargs={}, response_metadata={}),
     AIMessage(content="Beggin' yer pardon, Captain Heeah! I meant no disrespect. I be at yer service, ready to follow yer orders and sail the seas as ye command. What be yer orders, Cap'n? Arrr!", additional_kwargs={}, response_metadata={})]</pre>



Now, create your own journey using the custom ```ConversationChain``` with LCEL! 

Thank you for your hard work!🎉🎉🎉

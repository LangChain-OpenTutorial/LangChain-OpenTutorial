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

# Conversation-With-History

- Author: [Sunworl Kim](https://github.com/sunworl)
- Design:
- Peer Review:
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

This tutorial provides a comprehensive guide to implementing **conversational AI systems** with memory capabilities using LangChain in two main approaches.

**1. Creating a chain to record conversations**

- Creates a simple question-answering **chatbot** using ChatOpenAI.

- Implements a system to store and retrieve conversation history based on session IDs.

- Uses **RunnableWithMessageHistory** to incorporate chat history into the chain.


**2. Creating a RAG chain that retrieves information from documents and records conversations**

- Builds a more complex system that combines document retrieval with conversational AI. 

- Processes a **PDF document**, creates embeddings, and sets up a vector store for efficient retrieval.

- Implements a **RAG chain** that can answer questions based on the document content and previous conversation history.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Creating a Chain that remembers previous conversations](#creating-a-chain-that-remembers-previous-conversations)
  - [1. Add conversation history to the general Chain](#1-add-conversation-history-to-the-general-chain)
  - [2. RAG + RunnableWithMessageHistory](#2-rag--runnablewithmessagehistory)


### References

- [Langchain Python API : RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
- [Langchain docs : Build a Chatbot](https://python.langchain.com/docs/tutorials/chatbot/) 
----

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
        "langchain",
        "langchain_core",
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
        "LANGCHAIN_PROJECT": "Conversation-With-History"  
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Creating a Chain that remembers previous conversations

Background knowledge needed to understand this content : [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#runnablewithmessagehistory)

## 1. Add conversation history to the general Chain

- Use `MessagesPlaceholder` to include conversation history.

- Define a prompt that takes user input for questions.

- Create a `ChatOpenAI` instance that uses OpenAI's `ChatGPT` model.

- Build a chain by connecting the prompt, language model, and output parser.

- Use `StrOutputParser` to convert the model's output into a string.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Defining the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Question-Answering chatbot. Please provide an answer to the given question.",
        ),
        # Please use the key 'chat_history' for conversation history without changing it if possible!
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),  # Use user input as a variable
    ]
)

# Generating an LLM
llm = ChatOpenAI()

# Creating a regular Chain
chain = prompt | llm | StrOutputParser()
```

Creating a chain that records conversations (chain_with_history)

- Create a dictionary to store session records.

- Define a function to retrieve session records based on session ID. If the session ID is not in the store, create a new `ChatMessageHistory` object.

- Create a `RunnableWithMessageHistory` object to manage conversation history.


```python
# Dictionary to store session records
store = {}

# Function to retrieve session records based on session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")
    if session_ids not in store:  # If the session ID is not in the store
        # Create a new ChatMessageHistory object and save it to the store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the corresponding session ID


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to retrieve session history
    input_messages_key="question",  # Key for the template variable that will contain the user's question
    history_messages_key="chat_history",  # Key for the history messages
)
```

Execute the first question.

```python
chain_with_history.invoke(
    # Input question
    {"question": "My name is Jack."},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation Session ID]: abc123
</pre>




    'Hello Jack! How can I help you today?'



Execute the question in continuation.

```python
chain_with_history.invoke(
    # Input question
    {"question": "What is my name?"},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation Session ID]: abc123
</pre>




    'Your name is Jack.'



## 2. RAG + RunnableWithMessageHistory

Implement a PDF document-based question-answering (QA) system.

First, create a regular RAG Chain, However, make sure to include `{chat_history}` in the prompt for step 6.

- (step 1) Use `PDFPlumberLoader` to load PDF files.

- (step 2)  Split documents into smaller chunks using `RecursiveCharacterTextSplitter`.

- (step 3)  Generate vector representations of text chunks using `OpenAIEmbeddings`.

- (step 4)  Store embeddings and make them searchable using `FAISS`.

- (step 5) Create a `retriever` to search for relevant information in the vector database.

- (step 6)  Generate a prompt template for question-answering tasks, including previous conversation history, questions, and context, with instructions to answer.

- (step 7)  Initialize the `GPT-4o` model using `ChatOpenAI`.

- (step 8)  Construct a chain that connects retrieval, prompt processing, and language model inference.

Retrieve relevant context for user questions and generate answers based on this context.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

# Step 1: Load Documents
loader = PDFPlumberLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf") 
docs = loader.load()

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Create DB and Save
# Create the vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Create Retriever
# Retrieve and generate information contained in the documents.
retriever = vectorstore.as_retriever()

# Step 6: Create Prompt
# Generate the prompt.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

#Previous Chat History:
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# Step 7: Create Language Model (LLM)
# Generate the model (LLM).
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Step 8: Create Chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Defining a function to save the conversation.

- The `store` dictionary is used to save conversation histories according to `session ids`, and the `get_session_history` function retrieves session records. 

- A `RunnableWithMessageHistory` object is created to add conversation history management functionality to the `RAG chain`, processing user questions and conversation histories. 

```python
# Dictionary to store session records
store = {}

# Function to retrieve session records based on session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")
    if session_ids not in store:  # If the session ID is not in the store
        # Create a new ChatMessageHistory object and save it to the store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  

# Create a RAG chain that records conversations
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Function to retrieve session history
    input_messages_key="question",  # Key for the template variable that will contain the user's question
    history_messages_key="chat_history",  # Key for the history messages
)
```

Execute the first question.

```python
rag_with_history.invoke(
    # Input question
    {"question": "What are the three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy?"},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "rag123"}},
)
```

<pre class="custom">[Conversation Session ID]: rag123
</pre>




    "The three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy are: (1) compliance with the law, (2) fulfillment of ethical principles, and (3) robustness."



Execute the subsequent question.

```python
rag_with_history.invoke(
    # Input question
    {"question": "Please translate the previous answer into Spanish."},
    # Record the conversation based on the session ID.
    config={"configurable": {"session_id": "rag123"}},
)
```

<pre class="custom">[Conversation Session ID]: rag123
</pre>




    'Los tres componentes clave necesarios para lograr una "IA confiable" en el enfoque europeo de la política de IA son: (1) cumplimiento de la ley, (2) cumplimiento de principios éticos y (3) robustez.'



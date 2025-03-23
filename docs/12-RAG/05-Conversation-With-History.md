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
- Peer Review: [Yun Eun](https://github.com/yuneun92)
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/12-RAG/05-Conversation-With-History.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/12-RAG/05-Conversation-With-History.ipynb)

## Overview

This tutorial provides a comprehensive guide to implementing **conversational AI systems** with memory capabilities using LangChain in two main approaches.

**1. Creating a chain to record conversations**

- Creates a simple question-answering **chatbot** using ```ChatOpenAI```.

- Implements a system to store and retrieve conversation history based on session IDs.

- Uses ```RunnableWithMessageHistory``` to incorporate chat history into the chain.


**2. Creating a RAG chain that retrieves information from documents and records conversations**

- Builds a more complex system that combines document retrieval with conversational AI. 

- Processes a **PDF document** , creates embeddings, and sets up a vector store for efficient retrieval.

- Implements a **RAG chain** that can answer questions based on the document content and previous conversation history.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Creating a Chain that remembers previous conversations](#creating-a-chain-that-remembers-previous-conversations)
  - [1. Adding Chat History to the Core Chain](#1-adding-chat-history-to-the-core-chain)
  - [2. Implementing RAG with Conversation History Management](#2-implementing-rag-with-conversation-history-management)


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

### 1. Adding Chat History to the Core Chain

- Implement `MessagesPlaceholder` to incorporate conversation history

- Define a prompt template that handles user input queries

- Initialize a `ChatOpenAI` instance configured to use the **ChatGPT** model

- Construct a chain by connecting the prompt template, language model, and output parser

- Implement `StrOutputParser` to format the model's response as a string

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define the chat prompt template with system message and history placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Question-Answering chatbot. Please provide an answer to the given question.",
        ),
        # Note: Keep 'chat_history' as the key name for maintaining conversation context
        MessagesPlaceholder(variable_name="chat_history"),
        # Format user question as input variable {question}
        ("human", "#Question:\n{question}"),
    ]
)

# Initialize the ChatGPT language model
llm = ChatOpenAI()

# Build the processing chain: prompt -> LLM -> string output
chain = prompt | llm | StrOutputParser()
```

Creating a Chain with Conversation History (```chain_with_history```)

- Initialize a dictionary to store conversation session records

- Create the function `get_session_history` that retrieves chat history by session ID and creates a new `ChatMessageHistory` instance if none exists

- Instantiate a `RunnableWithMessageHistory` object to handle persistent conversation history


```python
# Initialize an empty dictionary to store conversation sessions
store = {}

# Get or create chat history for a given session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")
    
    if session_ids not in store:     
        # Initialize new chat history for this session
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return existing or newly created chat history

# Configure chain with conversation history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  
    input_messages_key="question",  # User input variable name
    history_messages_key="chat_history",  # Conversation history variable name
)
```

Process the initial input.

```python
chain_with_history.invoke(

    # User input message
    {"question": "My name is Jack."},
    
    # Configure session ID for conversation tracking
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation Session ID]: abc123
</pre>




    'Hello Jack! How can I help you today?'



Handle Subsequent Query.

```python
chain_with_history.invoke(

    # User follow-up question
    {"question": "What is my name?"},

    # Use same session ID to maintain conversation context
    config={"configurable": {"session_id": "abc123"}},
)
```

<pre class="custom">[Conversation Session ID]: abc123
</pre>




    'Your name is Jack.'



### 2. Implementing RAG with Conversation History Management

Build a PDF-based Question Answering system that incorporates conversational context.

Create a standard RAG Chain, ensuring to include `{chat_history}` in the prompt template at step 6.

- (step 1) Load PDF documents using `PDFPlumberLoader`

- (step 2) Segment documents into manageable chunks with `RecursiveCharacterTextSplitter`

- (step 3) Create vector embeddings of text chunks using `OpenAIEmbeddings`

- (step 4) Index and store embeddings in a `FAISS` vector database

- (step 5) Implement a `retriever` to query relevant information from the vector database

- (step 6) Design a QA prompt template that incorporates **conversation history** , user queries, and retrieved context with response instructions

- (step 7) Initialize a `ChatOpenAI` instance configured to use the `GPT-4o` model

- (step 8) Build the complete chain by connecting the retriever, prompt template, and language model

The system retrieves relevant document context for user queries and generates contextually informed responses.

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

loader = PDFPlumberLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

retriever = vectorstore.as_retriever()

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

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

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

**Implementing Conversation History Management**

- Initialize the `store` dictionary to maintain conversation histories indexed by session IDs, and create the `get_session_history` function to retrieve or create session records

- Create a `RunnableWithMessageHistory` instance to enhance the RAG chain with conversation tracking capabilities, handling both user queries and historical context

```python
# Dictionary for storing session records
store = {}

# Retrieve session records by session ID
def get_session_history(session_ids):
    print(f"[Conversation Session ID]: {session_ids}")

    if session_ids not in store:
        # Initialize new ChatMessageHistory and store it
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  

# Create RAG chain with conversation history tracking
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Session history retrieval function
    input_messages_key="question",  # Template variable key for user question
    history_messages_key="chat_history",  # Key for conversation history
)
```

Process the first user input.

```python
rag_with_history.invoke(

    # User query for analysis
    {"question": "What are the three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy?"},

    # Session configuration for conversation tracking
    config={"configurable": {"session_id": "rag123"}},

)
```

<pre class="custom">[Conversation Session ID]: rag123
</pre>




    "The three key components necessary to achieve 'trustworthy AI' in the European approach to AI policy are: (1) compliance with the law, (2) fulfillment of ethical principles, and (3) robustness."



Execute the subsequent question.

```python
rag_with_history.invoke(

    # Request for translation of previous response
    {"question": "Please translate the previous answer into Spanish."},

    # Session configuration for maintaining conversation context
    config={"configurable": {"session_id": "rag123"}},
    
)
```

<pre class="custom">[Conversation Session ID]: rag123
</pre>




    'Los tres componentes clave necesarios para lograr una "IA confiable" en el enfoque europeo de la política de IA son: (1) cumplimiento de la ley, (2) cumplimiento de principios éticos y (3) robustez.'



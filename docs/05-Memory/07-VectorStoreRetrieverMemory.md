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

# VectorStoreRetrieverMemory

- Author: [Harheem Kim](https://github.com/harheem)
- Peer Review:
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/07-VectorStoreRetrieverMemory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/07-VectorStoreRetrieverMemory.ipynb)
## Overview

```VectorStoreRetrieverMemory``` stores memory in a vector store and queries the top-K most 'relevant' documents whenever called.
This differs from most of the other memory classes in that it does not explicitly track the order of conversation.

In this tutorial, we'll explore the practical application of ```VectorStoreRetrieverMemory``` through a simulated interview scenario. Through this example, we'll see how ```VectorStoreRetrieverMemory``` searches for information based on semantic relevance rather than chronological order of conversations.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Initializing the Vector Store](#initializing-the-vector-store)
- [Saving Interview Conversations](#saving-interview-conversations)
- [Retrieving Relevant Conversations](#retrieving-relevant-conversations)

### References

- [LangChain Python API Reference > langchain: 0.3.13 > memory > VectorStoreRetrieverMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.vectorstore.VectorStoreRetrieverMemory.html)
- [Faiss](https://github.com/facebookresearch/faiss)
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
    ["langchain_community", "langchain_openai", "langchain_core", "faiss-cpu"],
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
        "LANGCHAIN_PROJECT": "VectorStoreRetrieverMemory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Initializing the Vector Store

Next, we'll set up our vector store using FAISS. FAISS is an efficient similarity search library that will help us store and retrieve conversation embeddings:

```python
import faiss
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS

# Initialize OpenAI embeddings
embeddings_model = OpenAIEmbeddings()

# Create FAISS index
embedding_size = 1536  # Size for OpenAI embeddings
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
```

This setup creates an in-memory vector store that will maintain our conversation embeddings for quick similarity searches.

## Saving Interview Conversations

Now, we'll create our memory system and populate it with example interview conversations.

Note that by setting ```k=1```, we ensure that only the single most relevant conversation is returned. (In real applications, you might want to increase this value to provide more context.):

```python
from langchain.memory import VectorStoreRetrieverMemory

# We set k=1 to show that vector lookups still can return semantically relevant information
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Save arbitrary conversations
memory.save_context(
    inputs={
        "human": "Hello, thank you for coming to the interview today. Please introduce yourself."
    },
    outputs={
        "ai": "Hello. I'm a junior developer who majored in Computer Science. In college, I mainly used Java and Python, and recently, I participated in a web development project where I gained experience developing services for real users."
    },
)
memory.save_context(
    inputs={"human": "What was your role in the project?"},
    outputs={
        "ai": "My role was as a backend developer. I was responsible for processing user data and developing server logic, implementing RESTful APIs for communication with the frontend. I also participated in database design."
    },
)
memory.save_context(
    inputs={
        "human": "If you faced any difficulties in team projects, how did you resolve them?"
    },
    outputs={
        "ai": "We had some communication issues at the beginning of the project. To resolve this, our team held regular meetings to share each person's progress. Also, when problems arose, we actively shared opinions and worked to find reasonable solutions."
    },
)
memory.save_context(
    inputs={"human": "What do you consider your strengths as a developer?"},
    outputs={
        "ai": "My strengths are quick learning ability and problem-solving skills. I can quickly acquire new technologies and tools, and when faced with complex problems, I can propose creative solutions. Also, I value teamwork and consider collaboration with colleagues important."
    },
)
```

<pre class="custom">/var/folders/b5/3pdyzl995m5_37d6rr9vfbh00000gn/T/ipykernel_31365/1922969020.py:5: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory = VectorStoreRetrieverMemory(retriever=retriever)
</pre>

## Retrieving Relevant Conversations

Let's see how the system retrieves relevant information based on queries:

```python
# Query about educational background
print("Query: What was the interviewee's major?")
print(
    memory.load_memory_variables({"prompt": "What was the interviewee's major?"})[
        "history"
    ]
)
```

<pre class="custom">Query: What was the interviewee's major?
    human: Hello, thank you for coming to the interview today. Please introduce yourself.
    ai: Hello. I'm a junior developer who majored in Computer Science. In college, I mainly used Java and Python, and recently, I participated in a web development project where I gained experience developing services for real users.
</pre>

```python
# Query about project experience
print("Query: What was the interviewee's role in the project?")
print(
    memory.load_memory_variables(
        {"human": "What was the interviewee's role in the project?"}
    )["history"]
)
```

<pre class="custom">Query: What was the interviewee's role in the project?
    human: What was your role in the project?
    ai: My role was as a backend developer. I was responsible for processing user data and developing server logic, implementing RESTful APIs for communication with the frontend. I also participated in database design.
</pre>

This approach is particularly valuable when building systems that need to reference past conversations contextually, such as in customer service bots, educational assistants, or any application where maintaining context-aware conversation history is important.

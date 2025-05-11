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

# ConversationKGMemory

- Author: [Secludor](https://github.com/Secludor)
- Design: [Secludor](https://github.com/Secludor)
- Peer Review : [ulysyszh](https://github.com/ulysyszh), [Jinu Cho](https://github.com/jinucho)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/05-ConversationKGMemory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/05-ConversationKGMemory.ipynb)
## Overview

Unlike ```ConversationEntityMemory```, which manages information about entities in a key-value format for individual entities, ```ConversationKGMemory```(Conversation Knowledge Graph Memory) is a module that manages relationships between entities in a graph format.

It extracts and structures **knowledge triplets** (subject-relationship-object) to identify and store complex relationships between entities, and allows exploration of entity connectivity through **graph structure** .

This helps the model understand relationships between different entities and better respond to queries based on complex networks and historical context.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Conversation Knowledge Graph Memory](#conversation-knowledge-graph-memory)
- [Applying KG Memory to Chain](#applying-kg-memory-to-chain)
- [Applying KG Memory with LCEL](#applying-kg-memory-with-lcel)

### References

- [LangChain Python API Reference>langchain-community: 0.3.13>memory>ConversationKGMemory](https://python.langchain.com/api_reference/community/memory/langchain_community.memory.kg.ConversationKGMemory.html)
- [LangChain Python API Reference>langchain-community: 0.2.16>NetworkxEntityGraph](https://python.langchain.com/v0.2/api_reference/community/graphs/langchain_community.graphs.networkx_graph.NetworkxEntityGraph.html#langchain_community.graphs.networkx_graph.NetworkxEntityGraph)
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
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "05-ConversationKGMemory",  # title 과 동일하게 설정해 주세요
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Conversation Knowledge Graph Memory

```ConversationKGMemory``` is a memory module that stores and manages information extracted from conversations in a graph structure. 

This example demonstrates the following key features:
- Storing conversation context (```save_context```)
- (Reference) Getting a list of entity names in the graph sorted by causal dependence. (```get_topological_sort```)
- Extracting entities from current conversation (```get_current_entities```)
- Extracting knowledge triplets (```get_knowledge_triplets```)
- Retrieving stored memory (```load_memory_variables```)

The following example shows the process of extracting entities and relationships from a conversation about a new designer, Shelly Kim, and storing them in a graph format.

```python
from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory
```

```python
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

memory = ConversationKGMemory(llm=llm, return_messages=True)
memory.save_context(
    {"input": "This is Shelly Kim who lives in Pangyo."},
    {"output": "Hello Shelly, nice to meet you! What kind of work do you do?"},
)
memory.save_context(
    {"input": "Shelly Kim is our company's new designer."},
    {
        "output": "That's great! Welcome to our team. I hope you'll enjoy working with us."
    },
)
```

### (Reference) get_topological_sort() → List[str]

You can use the ```get_topological_sort``` method to view all entities stored in the knowledge graph in topological order:

This method:
- Uses ```NetworkX``` library to analyze the knowledge graph structure.
- Performs topological sorting based on directed edges.
- Returns a list of entities in dependency order.

The order reflects the relationships between entities in the conversation, showing how they are connected in the knowledge graph.

```python
memory.kg.get_topological_sort()
```




<pre class="custom">['Shelly Kim', 'Pangyo', "our company's new designer"]</pre>



### get_current_entities(input_string: str) → List[str]

Here's how the ```get_current_entities``` method works:

**1. Entity Extraction Chain Creation**
- Creates an ```LLMChain``` using the ```entity_extraction_prompt``` template.
- This prompt is designed to extract proper nouns from the last line of the conversation.

**2. Context Processing**
- Retrieves the last **k*2** messages from the buffer. (default : k=2)
- Generates conversation history string using ```human_prefix``` and ```ai_prefix```.

**3. Entity Extraction**
- Extracts proper nouns from the input string "Who is Shelly Kim?"
- Primarily recognizes words starting with capital letters as proper nouns.
- In this case, "Shelly Kim" is extracted as an entity.

This method **only extracts entities from the question itself** , while the previous conversation context is used only for reference.

```python
memory.get_current_entities({"input": "Who is Shelly Kim?"})
```




<pre class="custom">['Shelly Kim']</pre>



### get_knowledge_triplets(input_string: str) → List[KnowledgeTriple]

The ```get_knowledge_triplets``` method operates as follows:

**1. Knowledge Triple Extraction Chain**
- Creates an ```LLMChain``` using the ```knowledge_triplet_extraction_prompt``` template.
- Designed to extract triples in ( **subject-relation-object** ) format from given text.

**2. Memory Search**
- Searches for information related to "Shelly" from previously stored conversations.
- Stored context:
  - "This is Shelly Kim who lives in Pangyo."
  - "Shelly Kim is our company's new designer."

**3. Triple Extraction**
- Generates the following triples from the retrieved information:
  - (Shelly Kim, lives in, Pangyo)
  - (Shelly Kim, is, designer)
  - (Shelly Kim, works at, our company)

This method extracts relationship information in **triple format** from all stored conversation content **related to a specific entity** .

```python
memory.get_knowledge_triplets({"input": "Shelly"}), "\n", memory.get_knowledge_triplets(
    {"input": "Pangyo"}
), "\n", memory.get_knowledge_triplets(
    {"input": "designer"}
), "\n", memory.get_knowledge_triplets(
    {"input": "Langchain"}
)
```




<pre class="custom">([KnowledgeTriple(subject='Shelly Kim', predicate='lives in', object_='Pangyo'),
      KnowledgeTriple(subject='Shelly Kim', predicate='is', object_="company's new designer")],
     '\n',
     [KnowledgeTriple(subject='Shelly Kim', predicate='lives in', object_='Pangyo')],
     '\n',
     [KnowledgeTriple(subject='Shelly Kim', predicate='is a', object_='designer')],
     '\n',
     [])</pre>



### load_memory_variables(inputs: Dict[str, Any]) → Dict[str, Any]

The ```load_memory_variables``` method operates through the following steps:

**1. Entity Extraction**
- Extracts entities (e.g., "Shelly Kim") from the input "Who is Shelly Kim?"
- Internally uses the ```get_current_entities``` method.

**2. Knowledge Retrieval**
- Searches for all knowledge triplets related to the extracted entities.
- Queries the graph for information previously stored via ```save_context``` method.

**3. Information Formatting**
- Converts found triplets into system messages.
- Returns a list of message objects due to the ```return_messages=True``` setting.

This method retrieves relevant information from the stored knowledge graph and returns it in a structured format, which can then be used as context for subsequent conversations with the language model.

```python
memory.load_memory_variables({"input": "Who is Shelly Kim?"})
```




<pre class="custom">{'history': [SystemMessage(content="On Shelly Kim: Shelly Kim lives in Pangyo. Shelly Kim is our company's new designer.", additional_kwargs={}, response_metadata={})]}</pre>



## Applying KG Memory to Chain

This section demonstrates how to use ```ConversationKGMemory``` with **ConversationChain** .

(The class **ConversationChain** was deprecated in LangChain 0.2.7 and will be removed in 1.0. If you want, you can skip to [Applying KG Memory with LCEL](#applying-kg-memory-with-lcel))

```python
from langchain_community.memory.kg import ConversationKGMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. 
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

conversation_with_kg = ConversationChain(
    llm=llm, prompt=prompt, memory=ConversationKGMemory(llm=llm)
)
```

<pre class="custom">C:\Users\Caelu\AppData\Local\Temp\ipykernel_5648\1729312250.py:21: LangChainDeprecationWarning: The class `ConversationChain` was deprecated in LangChain 0.2.7 and will be removed in 1.0. Use :meth:`~RunnableWithMessageHistory: https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html` instead.
      conversation_with_kg = ConversationChain(
</pre>

Let's initialize the conversation with some basic information.

```python
conversation_with_kg.predict(
    input="My name is Teddy. Shelly is a coworker of mine, and she's a new designer at our company."
)
```




<pre class="custom">"Hi Teddy! It's great to meet you. It sounds like you and Shelly are working together in a creative environment. Being a new designer, Shelly must be bringing fresh ideas and perspectives to your team. How has it been working with her so far?"</pre>



Let's query the memory for information about Shelly.

```python
conversation_with_kg.memory.load_memory_variables({"input": "who is Shelly?"})
```




<pre class="custom">{'history': 'On Shelly: Shelly is a coworker of Teddy. Shelly is a new designer. Shelly works at our company.'}</pre>



You can also reset the memory by ```memory.clear()```.

```python
conversation_with_kg.memory.clear()
conversation_with_kg.memory.load_memory_variables({"input": "who is Shelly?"})
```




<pre class="custom">{'history': ''}</pre>



## Applying KG Memory with LCEL

Let's examine the memory after having a conversation using a custom **ConversationChain** with ```ConversationKGMemory``` by LCEL

```python
from operator import itemgetter
from langchain_community.memory.kg import ConversationKGMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. 
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:
{history}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationKGMemory(llm=llm, return_messages=True, memory_key="history")


class ConversationChain:
    def __init__(self, prompt, llm, memory):
        self.memory = memory
        self.chain = (
            RunnablePassthrough()
            | RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables)
                | itemgetter("history")
            )
            | prompt
            | llm
        )

    def invoke(self, input_dict):
        response = self.chain.invoke(input_dict)
        self.memory.save_context(input_dict, {"output": response.content})
        return response


conversation_with_kg = ConversationChain(prompt, llm, memory)
```

Let's initialize the conversation with some basic information.

```python
response = conversation_with_kg.invoke(
    {
        "input": "My name is Teddy. Shelly is a coworker of mine, and she's a new designer at our company."
    }
)
response.content
```




<pre class="custom">"Hi Teddy! It's nice to meet you. It sounds like you and Shelly are working together at your company. How's everything going with the new designer on board?"</pre>



Let's query the memory for information about Shelly.

```python
conversation_with_kg.memory.load_memory_variables({"input": "who is Shelly?"})
```




<pre class="custom">{'history': [SystemMessage(content='On Shelly: Shelly is a coworker of Teddy. Shelly is a new designer. Shelly works at our company.', additional_kwargs={}, response_metadata={})]}</pre>



You can also reset the memory by ```memory.clear()```.

```python
conversation_with_kg.memory.clear()
conversation_with_kg.memory.load_memory_variables({"input": "who is Shelly?"})
```




<pre class="custom">{'history': []}</pre>



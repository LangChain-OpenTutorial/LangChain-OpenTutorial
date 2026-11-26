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

# Title 

- Author: [touseefahmed96](https://github.com/touseefahmed96)
- Peer Review: [BAEM1N](https://github.com/baem1n)
- Proofread: [BAEM1N](https://github.com/baem1n)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/17-LangGraph/01-Core-Features/02-LangGraph-ChatBot(groq).ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/17-LangGraph/01-Core-Features/02-LangGraph-ChatBot(groq).ipynb)

## Overview
This tutorial covers how to create a simple chatbot using `LangGraph`.

LangGraph is an open-source framework for building and managing AI agents and multi-agent systems. It simplifies state management, agent interactions, and error handling, enabling developers to create robust applications powered by Large Language Models (LLMs).

LangGraph enables agentic applications by defining three core components:
- `Nodes`: Individual computation steps or functions.
- `States`: Context or memory maintained during computations.
- `Edges`: Connections between nodes, guiding the flow of computation.

In this tutorial, We’ll build a simple chatbot using LangGraph. The chatbot will respond directly to user messages. To start, we’ll create a `StateGraph`, which defines the chatbot’s structure as a state machine

### Table of Contents

- [Overview](#overview) 
- [Environment Setup](#environment-setup)
- [Graph Creation](#lets-create-a-graph)
- [Chatbot Implementation](#implementing-the-chatbot-logic)
- [Chatbot State Building](#building-the-chatbot-state-machine)
- [Chatbot State Visualization](#visualizing-the-chatbot-state-machine)
- [Start the Chat](#running-the-chatbot)


### References
- [LangGraph StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langgraph",
        "langchain_groq",
        "langchain_community",
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

If you want to use `Groq api` you can get it from here [GROQ_API_KEY](https://console.groq.com/keys) or you can also use `Openai api`.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            # "OPENAI_API_KEY": "",
            # "LANGCHAIN_API_KEY": "",
            # "GROQ_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "Basic-LangGraph-Chatbot", 
        }
    )
```

Let's setup `ChatGroq` with `Gemma2-9b-It` model.

```python
from langchain_groq import ChatGroq

llm = ChatGroq( model_name = "Gemma2-9b-It") 
```

Also you can use `ChatOpenAI` with `gpt-4o` model comment the last cell and uncomment the below cell.

```python
# from langchain_openai import ChatOpenAI

# # Load the model
# llm = ChatOpenAI(model_name="gpt-4o")
```

## Lets Create a Graph

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```

#### Key Components:
1. State: A `TypedDict` that holds the chatbot's state, specifically a list of messages.
2. StateGraph: The core structure that defines the chatbot's behavior as a state machine.

#### Code Implementation:

```python
# Define the State for the chatbot
class State(TypedDict):
    messages : Annotated[list,add_messages] # List of messages in the conversation

# Initialize the StateGraph    
graph_builder = StateGraph(State)
```

#### Explanation:
- State: The `State` class defines the chatbot's memory, which is a list of messages (`messages`). The `Annotated` type hints that `add_messages` will handle message updates.
- StateGraph: The `StateGraph` object (`graph_builder`) will manage the chatbot's state transitions and logic.

### Implementing the Chatbot Logic

The `chatbot` function defines the core behavior of the chatbot. It takes the current `state` (which contains the conversation history) and generates a response using a Large Language Model (LLM).

#### Key Points:
- Input: The `state` object contains the conversation history (`messages`).
- Output: The function returns an updated `state` with the chatbot's response appended to the `messages` list.
- LLM Integration: The `llm.invoke()` method is used to generate a response based on the conversation history.

#### Code Implementation:

```python
def chatbot(state: State):
    # Generate a response using the LLM and update the state
    return {"messages": llm.invoke(state['messages'])}
```

#### Explanation:
- `state: State`: The function accepts the current state, which includes the list of messages.
- `llm.invoke(state['messages'])`: The LLM processes the conversation history and generates a response.
- Return Value: The function returns a dictionary with the updated `messages` list, including the chatbot's response.

### Building the Chatbot State Machine

Now that we've defined the `chatbot` function, we'll integrate it into the `StateGraph` by adding nodes and edges. This defines the flow of the chatbot's state machine.

#### Key Steps:
1. Add Node: Register the `chatbot` function as a node in the graph.
2. Add Edges: Define the flow of the state machine:
   - From `START` to the `chatbot` node.
   - From the `chatbot` node to `END`.
3. Compile the Graph: Finalize the graph structure so it can be executed.

#### Code Implementation:

```python
# Add the chatbot function as a node in the graph
graph_builder.add_node("chatbot", chatbot)

# Define the flow of the state machine
graph_builder.add_edge(START, "chatbot")  # Start the conversation with the chatbot
graph_builder.add_edge("chatbot", END)    # End the conversation after the chatbot responds

# Compile the graph to finalize the state machine
graph = graph_builder.compile()
```

#### Explanation:
- `add_node("chatbot", chatbot)`: Registers the `chatbot` function as a node named `"chatbot"`.
- `add_edge(START, "chatbot")`: Specifies that the conversation starts with the `chatbot` node.
- `add_edge("chatbot", END)`: Specifies that the conversation ends after the chatbot responds.
- `compile()`: Finalizes the graph structure, making it ready for execution.

### Visualizing the Chatbot State Machine

To better understand the structure of the chatbot's state machine, we can visualize the graph using Mermaid diagrams. This step is optional but highly useful for debugging and understanding the flow of the application.

#### Key Points:
- Mermaid Diagram: The `graph.get_graph().draw_mermaid_png()` method generates a visual representation of the graph.
- IPython Display: The `IPython.display` module is used to render the diagram directly in the notebook.

#### Code Implementation:

```python
from IPython.display import Image, display

# Attempt to visualize the graph as a Mermaid diagram
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # Handle cases where visualization fails (e.g., missing dependencies)
    pass
```


    
![png](./img/output_22_0.png)
    


#### Explanation:
- `graph.get_graph().draw_mermaid_png()`: Generates a Mermaid diagram of the graph and converts it to a PNG image.
- `display(Image(...))`: Renders the PNG image in the notebook.
- `try-except` Block: Ensures the code doesn't break if visualization fails (e.g., due to missing dependencies or unsupported environments).

### Running the Chatbot

This section implements the chatbot's interaction loop, where the user can input messages, and the chatbot responds. The loop continues until the user types `quit` or `q`.

#### Key Features:
- User Input: The chatbot listens for user input and processes it using the compiled graph.
- Streaming Responses: The `graph.stream()` method processes the input and generates responses in real-time.
- Exit Condition: The loop exits when the user types `quit` or `q`.

#### Code Implementation:

```python
while True:
    # Get user input
    user_input = input("User: ")
    print("================================")
    print("User: ", user_input)
    # Exit the loop if the user types "quit" or "q"
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        print("================================")
        break
    
    # Process the user input using the graph
    for event in graph.stream({'messages': ("user", user_input)}):
        # Print the event values (debugging purposes)
        # print(event.values())
        
        # Extract and display the chatbot's response
        for value in event.values():
            # print(value['messages'])
            print("Assistant:", value["messages"].content)
```

<pre class="custom">================================
    User:  hey
    Assistant: Hey there! 👋 What can I do for you today?
    
    ================================
    User:  what is EAG
    Assistant: EAG can stand for a few different things. To figure out which one you're looking for, I need more context. 
    
    Could you tell me what field or industry this EAG relates to? For example:
    
    * **Business/Finance:** EAG might refer to **Entertainment Arts Group**.
    * **Technology:** It could stand for **Enhanced Audio Gateway**.
    * **Science/Medicine:** EAG could be **Electroencephalographic Activity**.
    
    Please provide more information so I can give you a precise answer! 
    
    
    
    ================================
    User:  q
    Good Bye
    ================================
</pre>

#### Explanation:
- `while True`: Creates an infinite loop for continuous interaction.
- `input("User: ")`: Prompts the user for input.
- `graph.stream({'messages': ("user", user_input)})`: Processes the user input through the graph and streams the results.
- `event.values()`: Extracts the chatbot's response from the event.
- `value['messages'].content`: Displays the chatbot's response.

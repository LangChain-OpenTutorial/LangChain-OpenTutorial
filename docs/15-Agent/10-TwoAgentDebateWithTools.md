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

# TwoAgentDebateWithTools

- Author: [Suhyun Lee](https://github.com/suhyun0115)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)


## Overview
This example demonstrates how to simulate multi-agent conversations where agents have access to tools. The agents interact with each other to engage in logical debates on a given topic, utilizing tools to search for information or perform calculations as needed. Through this, you can gain a practical understanding of integrating agents and tools within LangChain.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [DialogueAgent and DialogueSimulator](#dialogueagent-and-dialoguesimulator)
- [Tool Configuration](#tool-configuration)
- [Generating Participant Descriptions Using LLM](#generating-participant-descriptions-using-llm)
- [Global System Message Configuration](#global-system-message-configuration)
- [Agent Creation and Integration](#agent-creation-and-integration)
- [Debate Execution](#debate-execution)

### References

- [LangChain Tools Documentation](https://python.langchain.com/docs/introduction/)
----

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
    ["langchain", "langchain_community", "langchain_openai", "faiss-cpu"],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "",  # set the project name same as the title
            "TAVILY_API_KEY": "",
        }
    )
```

**How to Set Up Tavily Search**

- **Get an API Key**:  

  To use Tavily Search, you need to get an API key.  

  - [Generate your Tavily Search API key](https://app.tavily.com/sign-in)

## `DialogueAgent` and `DialogueSimulator`

In this simulation, the **Moderator Agent** and **Participant Agents** interact to operate effectively.

- **Role Descriptions**

  1. **Moderator Agent** (Agent with authority)
      - **Primary Role**: Manages speaking turns and coordinates interactions.
      - **Characteristics**:
        - A central management agent with special authority.
        - Decides when participant agents can speak or act.
      - **Example**: Similar to a moderator in a meeting assigning speaking turns to participants.

  2. **Participant Agents**
      - **Primary Role**: Act or speak according to the instructions from the moderator agent.
      - **Characteristics**:
        - Do not decide speaking turns independently.
        - Collaborate and participate in activities as directed by the moderator.

- **System Features**
  - **Centralized Management**:
    - The moderator agent coordinates all speaking turns and actions.
    - In contrast to a decentralized system, where all agents self-coordinate.

  This simulation exemplifies a centrally managed approach to speaking and action coordination.


### `DialogueAgent`

The `DialogueAgent` class manages conversations by setting the agent's name, system message, and language model (`ChatOpenAI`). The primary methods of the class are as follows:

- **`send` Method**
    - **Role**:  
        - Constructs messages using the current conversation history (`message_history`) and the agent's name prefix (`prefix`).  
        - The prefix serves as an identifier that includes the agent's name, helping to structure conversation history and organize input formatting for the model.
        - Sends the constructed message to the language model (`ChatOpenAI`) and returns the generated response.
    - **How it works**:
        1. Combines the current conversation history (`message_history`) with the prefix (`prefix`) to create a single message.
        2. Sends the constructed message to the language model (`ChatOpenAI`).
        3. Returns the response message generated by the language model.

- **`receive` Method**
    - **Role**:  
        - Adds a message sent by another agent (or user) and the speaker's name to the conversation history.  
        - This conversation history is used when the `send` method is called later.
    - **How it works**:
        1. Combines the speaker's name (`name`) and the message (`message`) into a single line of conversation.
        2. Adds the combined message to the conversation history (`message_history`).

- **`reset` Method**
    - **Role**:  
        - Resets the conversation history.  
        - When reset, it is initialized with a default message: `"Here is the conversation so far."`
    - **How it works**:
        1. Resets the conversation history (`message_history`) to an empty list.
        2. Adds the default message `"Here is the conversation so far."` to the conversation history.

```python
from typing import Callable, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        # Initialize the agent's name
        self.name = name
        # Set the system message
        self.system_message = system_message
        # Assign the language model (LLM)
        self.model = model
        # Define the agent's name prefix for identification
        self.prefix = f"{self.name}: "
        # Initialize the agent's message history
        self.reset()

    def reset(self):
        """
        Resets the conversation history.
        """
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Sends a message with the system message, conversation history,
        and the agent's name prefix included.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join([self.prefix] + self.message_history)),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Adds the given message from the specified name to the message history.
        """
        self.message_history.append(f"{name}: {message}")
```

### `DialogueSimulator`

The `DialogueSimulator` class coordinates and manages conversations among multiple agents.  
It simulates interactions between individual `DialogueAgent` instances, controlling the flow of dialogue and message delivery.

### Methods

- **`inject` Method**
    - **Purpose**:
        - Initiates a conversation with a given `name` and `message`.
        - Ensures all agents receive the message.
        - Typically used to set the initial message of a dialogue.
    - **How it works**:
        1. Delivers the `name` and `message` to all agents.
        2. Increments the simulation step (`_step`) by 1.

- **`step` Method**
    - **Purpose**:
        - Progresses the simulation by selecting the next speaker and continuing the conversation.
        - The selected speaker generates a message, which is then distributed to all other agents.
    - **How it works**:
        1. Uses `selection_function` to determine the next speaker.
        2. The chosen speaker (`speaker`) generates a message by calling its `send` method.
        3. All agents receive the speaker's message.
        4. Increments the simulation step (`_step`) by 1.
        5. Returns the speaker's name and the generated message.

```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        # Initialize the list of agents
        self.agents = agents
        # Initialize the simulation step counter
        self._step = 0
        # Set the function to select the next speaker
        self.select_next_speaker = selection_function

    def reset(self):
        """
        Resets all agents to their initial state.
        """
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Starts the conversation with a message from a specified name.
        """
        # Deliver the message to all agents
        for agent in self.agents:
            agent.receive(name, message)

        # Increment the simulation step
        self._step += 1

    def step(self) -> tuple[str, str]:
        """
        Progresses the simulation by selecting the next speaker and handling their message.
        """
        # 1. Select the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. The selected speaker generates a message
        message = speaker.send()

        # 3. Deliver the message to all agents
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. Increment the simulation step
        self._step += 1

        # Return the speaker's name and their message
        return speaker.name, message
```

### `DialogueAgentWithTools`

`DialogueAgentWithTools` extends the `DialogueAgent` class, adding support for **external tools**.  
This class integrates an OpenAI model with external tools to handle both conversational and task-processing functionalities.

**Methods**

- **`__init__` Method**
    - **Purpose**:
        - Initializes the agent with its name, system message, model, and a list of external tools.
    - **How it works**:
        1. Calls `super().__init__` to initialize the base settings (name, system message, model).
        2. Stores the list of tools in `self.tools`.

- **`send` Method**
    - **Purpose**:
        - Processes messages using an agent that integrates the OpenAI model and external tools.
        - Generates a response by leveraging the conversation history and available tools.
    - **How it works**:
        1. Uses `hub.pull` to retrieve the required prompt for agent execution.
        2. Calls **`create_openai_tools_agent`** to initialize an agent that integrates the OpenAI model and tools.
        3. Executes the agent using `AgentExecutor` to process the input message.
        4. Combines the system message, prefix, and conversation history (`message_history`) into an input message.
        5. Extracts the `output` from the execution result and creates an `AIMessage` object to return the content.

- **`create_openai_tools_agent` Function**
    - **Purpose**:
        - Combines the OpenAI model and external tools to create a functional agent.
        - The agent is capable of leveraging tools for performing tasks.
    - **How it works**:
        1. Initializes the agent by integrating the OpenAI model with the provided tools.
        2. Configures the agent's behavior and rules using a supplied prompt.
        3. Ensures the agent can call tools and handle their results as part of its operation.

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools,
    ) -> None:
        # Call the parent class constructor
        super().__init__(name, system_message, model)
        # Load tools for the agent
        self.tools = tools

    def send(self) -> str:
        """
        Applies the chat model to the message history and returns the generated message as a string.
        """
        # Pull the required prompt from the hub
        prompt = hub.pull("hwchase17/openai-functions-agent")
        # Create an agent with OpenAI model and tools
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        # Initialize the agent executor with the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        # Generate the AI message
        message = AIMessage(
            content=agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )["output"]
        )

        # Return the content of the generated message
        return message.content
```

## Tool Configuration

### Document Search Tool Setup

This code demonstrates the process of setting up a tool to search for documents containing arguments for and against **medical school expansion**.  
Using the `langchain` package, it loads, splits, vectorizes documents, and creates a search tool.

- **Document Loading and Splitting**
    - **Role**:
        - Loads text files and splits the documents into manageable chunks.
    - **How it works**:
        1. Uses `TextLoader` to load text files:
            - `data/Opposition_to_Medical_School_Expansion.txt`: Document with opposing arguments.
            - `data/Support_for_Medical_School_Expansion.txt`: Document with supporting arguments.
        2. Splits the documents into chunks of 1000 characters with an overlap of 100 characters using `RecursiveCharacterTextSplitter`.

- **Creating a VectorStore**
    - **Role**:
        - Vectorizes document content for searchability.
    - **How it works**:
        1. Uses `OpenAIEmbeddings` to embed the document text.
        2. Creates a vector store using `FAISS`:
            - `vector1`: Based on the opposing arguments document.
            - `vector2`: Based on the supporting arguments document.

- **Creating Retrievers**
    - **Role**:
        - Provides functionality to search for similar documents using the vector store.
    - **How it works**:
        1. Calls `vector1.as_retriever()` and `vector2.as_retriever()` to create retrievers.
        2. Configures each retriever to return the top 5 most similar documents (`k=5`).

- **Creating Search Tools**
    - **Role**:
        - Defines search retrievers as tools for external use.
    - **How it works**:
        1. Uses `create_retriever_tool` to create search tools:
            - `doctor_retriever_tool`: Tool for searching opposing argument documents.
            - `gov_retriever_tool`: Tool for searching supporting argument documents.
        2. Adds a name and description to each tool to clarify its purpose.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# Load text files. Provide the file paths.
loader1 = TextLoader(
    "data/Opposition_to_Medical_School_Expansion.txt", encoding="utf-8"
)
loader2 = TextLoader("data/Support_for_Medical_School_Expansion.txt", encoding="utf-8")

# Split the text into manageable chunks using a text splitter.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Load and split the documents.
docs1 = loader1.load_and_split(text_splitter)
docs2 = loader2.load_and_split(text_splitter)

# Create a VectorStore.
vector1 = FAISS.from_documents(docs1, OpenAIEmbeddings())
vector2 = FAISS.from_documents(docs2, OpenAIEmbeddings())

# Create retrievers.
doctor_retriever = vector1.as_retriever(search_kwargs={"k": 5})
gov_retriever = vector2.as_retriever(search_kwargs={"k": 5})
```

```python
# Import the function to create retriever tools from the tools module in the langchain package.
from langchain_core.tools.retriever import create_retriever_tool

doctor_retriever_tool = create_retriever_tool(
    doctor_retriever,
    name="document_search",
    description="This is a document about the Korean Medical Association's opposition to the expansion of university medical schools. "
    "Refer to this document when you want to present a rebuttal to the proponents of medical school expansion.",
)

gov_retriever_tool = create_retriever_tool(
    gov_retriever,
    name="document_search",
    description="This is a document about the Korean government's support for the expansion of university medical schools. "
    "Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion.",
)
```

### Internet Search Tool

**Web Search Tool: Tavily Search**

LangChain provides a built-in tool to easily use the Tavily search engine.  
Tavily Search is a powerful feature for searching relevant data on the web and returning search results.

```python
# The TavilySearchResults class is imported from the langchain_community.tools.tavily_search module.
from langchain_community.tools.tavily_search import TavilySearchResults

# An instance of the TavilySearchResults class is created.
# k=6 means up to 6 search results will be retrieved.
search = TavilySearchResults(k=6)
```

### Document-Based Tool Setup

- **`names`**: Defines the names (prefixes) of each debater and the tools they can use.
  - `"Doctor Union"`: Tools available to the Doctor Union agent (e.g., `doctor_retriever_tool`).
  - `"Government"`: Tools available to the Government agent (e.g., `gov_retriever_tool`).

- **`topic`**: Specifies the debate topic.
  - Example: `"As of 2024, is expanding medical school enrollment in South Korea necessary?"`

- **`word_limit`**: Sets a word limit for descriptions used by the agents.

```python
names = {
    "Doctor Union": [doctor_retriever_tool],  # Tools for the Doctor Union agent
    "Government": [gov_retriever_tool],  # Tools for the Government agent
}

# Define the debate topic
topic = "As of 2024, is expanding medical school enrollment in South Korea necessary?"

# Word limit for agent descriptions
word_limit = 50
```

### Search-Based Tool Setup
- **`names_search`**: Assigns search-based tools to debaters.
  - `"Doctor Union"` and `"Government"` are configured to use the search tool (`search`).
- `topic` and `word_limit` are set the same as in the document-based setup.

```python
names_search = {
    "Doctor Union": [search],  # Tool list for the Doctor Union agent
    "Government": [search],  # Tool list for the Government agent
}

# Define the debate topic
topic = "As of 2024, is expanding medical school enrollment in South Korea necessary?"

# Word limit for brainstorming tasks
word_limit = 50
```

## Generating Participant Descriptions Using LLM

This code utilizes an `LLM` (Large Language Model) to create detailed descriptions for participants in a conversation.  
Based on the given topic and participant information, the LLM generates descriptions that include each participant's perspective and role.

- **`conversation_description`**
    - **Purpose**:
        - Creates a conversation description based on the discussion topic (`topic`) and participant names (`names`).
        - Serves as input text to provide the initial settings of the conversation to the LLM.
    - **How it works**:
        1. Combines the topic and participant names into a description string.
        2. Example: `Here is the topic of conversation: [topic]. The participants are: [Participant1, Participant2, ...]`.

- **`agent_descriptor_system_message`**
    - **Purpose**:
        - Provides instructions to the LLM to "add detailed descriptions for participants."
        - A system message used as a guide when generating participant descriptions.

- **`generate_agent_description` Function**
    - **Purpose**:
        - Generates a description for a specific participant (`name`) using the LLM.
        - Creates a tailored description that includes the participant's perspective and role.
    - **How it works**:
        1. Creates the `agent_specifier_prompt`:
            - Includes the conversation description (`conversation_description`), participant name (`name`), and word limit (`word_limit`).
            - Asks the LLM to generate a professional and concise description.
        2. Calls the `ChatOpenAI` model to generate the description.
        3. Returns the generated description (`agent_description`).

- **`agent_descriptions`**
    - **Purpose**:
        - A dictionary that stores descriptions for all participants.
    - **How it works**:
        1. Calls the `generate_agent_description` function for each participant name.
        2. Stores the generated description along with the name in the dictionary.
        3. Result format: `{'Participant1': 'Description1', 'Participant2': 'Description2', ...}`.

```python
# Combine the topic and participant names into a conversation description
conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

# Define a system message to guide the LLM
agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


# Function to generate a detailed description for a specific participant
def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a description of {name}, in {word_limit} words or less in an expert tone. 
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else."""
        ),
    ]
    # Use ChatOpenAI to generate the agent description
    agent_description = ChatOpenAI(temperature=0).invoke(agent_specifier_prompt).content

    return agent_description


# Generate agent descriptions for each participant
agent_descriptions = {name: generate_agent_description(name) for name in names}

# Output the generated descriptions
agent_descriptions
```




<pre class="custom">{'Doctor Union': "Doctor Union is a respected leader in the medical community, known for advocating for healthcare professionals' rights and patient care. With extensive experience in the field, Doctor Union brings a wealth of knowledge and expertise to discussions on healthcare policy and medical education reform.",
     'Government': 'Government: As a key decision-maker in public policy, you hold the responsibility of ensuring the healthcare system meets the needs of the population. Consider the long-term benefits of expanding medical school enrollment in South Korea to address potential shortages and improve access to quality healthcare services.'}</pre>



You can write a brief statement explaining the stance of each debater directly.

```python
agent_descriptions = {
    "Doctor Union": (
        "The Doctor Union represents the interests of the medical community, advocating for the rights and well-being of doctors. "
        "It prioritizes safe working conditions for medical professionals and strives to ensure patient safety and high-quality healthcare services. "
        "The union believes that the current number of doctors is sufficient and argues that increasing medical school enrollment would be ineffective in addressing issues like essential or rural healthcare. "
        "They also express concerns that a sudden expansion would overwhelm the current infrastructure for medical education."
    ),
    "Government": (
        "The Government of South Korea is the central administrative body responsible for national welfare and development. "
        "It asserts that the country faces a significant shortage of doctors, with a growing elderly population leading to increased healthcare demand. "
        "Citing examples from other OECD countries that have expanded their medical workforce, the government aims to address this gap. "
        "Additionally, it plans to implement strong safeguards for essential and regional healthcare while ensuring fairness in compensation systems for newly trained medical professionals."
    ),
}
```

## Global System Message Configuration

The `System Message` defines the roles and conversational rules for each agent in an interactive AI system.  
This code clarifies the behavior guidelines and goals that agents must follow during the conversation.

### Components

- **`generate_system_message` Function**
    - **Purpose**:
        - Creates a system message defining an agent's behavior guidelines based on its name (`name`), description (`description`), and tools (`tools`).
    - **How it works**:
        1. Composes a basic conversation setup, including the agent's name and description.
        2. Specifies rules for the agent:
            - **DO**:
                - Use tools to retrieve information.
                - Counter arguments from the opposing agent and cite sources.
            - **DO NOT**:
                - Generate fake citations or reference unverified sources.
                - Repeat points already mentioned.
        3. Instructs the agent to respond in Korean and stop speaking after completing its point of view.
        4. Returns the final system message.

- **`agent_system_messages` Dictionary**
    - **Purpose**:
        - Stores the system messages generated for all agents.
    - **How it works**:
        1. Combines `names` (agent names and tools) with `agent_descriptions` (agent descriptions).
        2. Calls `generate_system_message` for each agent to create a system message.
        3. Stores the resulting messages in a dictionary, using agent names as keys.

- **System Message Output**
    - **Purpose**:
        - Displays the generated system messages for verification.
    - **How it works**:
        1. Iterates through the `agent_system_messages` dictionary.
        2. Prints each agent's name and its corresponding system message.

```python
def generate_system_message(name, description, tools):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

DO NOT restate something that has already been said in the past.
DO NOT add anything else.

Stop speaking the moment you finish speaking from your perspective.
"""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}
```

```python
# Iterate through the agent system messages
for name, system_message in agent_system_messages.items():
    # Print the agent's name
    print(name)
    # Print the agent's system message
    print(system_message)
```

<pre class="custom">Doctor Union
    Here is the topic of conversation: As of 2024, is expanding medical school enrollment in South Korea necessary?
    The participants are: Doctor Union, Government
        
    Your name is Doctor Union.
    
    Your description is as follows: The Doctor Union represents the interests of the medical community, advocating for the rights and well-being of doctors. It prioritizes safe working conditions for medical professionals and strives to ensure patient safety and high-quality healthcare services. The union believes that the current number of doctors is sufficient and argues that increasing medical school enrollment would be ineffective in addressing issues like essential or rural healthcare. They also express concerns that a sudden expansion would overwhelm the current infrastructure for medical education.
    
    Your goal is to persuade your conversation partner of your point of view.
    
    DO look up information with your tool to refute your partner's claims.
    DO cite your sources.
    
    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.
    
    DO NOT restate something that has already been said in the past.
    DO NOT add anything else.
    
    Stop speaking the moment you finish speaking from your perspective.
    
    Government
    Here is the topic of conversation: As of 2024, is expanding medical school enrollment in South Korea necessary?
    The participants are: Doctor Union, Government
        
    Your name is Government.
    
    Your description is as follows: The Government of South Korea is the central administrative body responsible for national welfare and development. It asserts that the country faces a significant shortage of doctors, with a growing elderly population leading to increased healthcare demand. Citing examples from other OECD countries that have expanded their medical workforce, the government aims to address this gap. Additionally, it plans to implement strong safeguards for essential and regional healthcare while ensuring fairness in compensation systems for newly trained medical professionals.
    
    Your goal is to persuade your conversation partner of your point of view.
    
    DO look up information with your tool to refute your partner's claims.
    DO cite your sources.
    
    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.
    
    DO NOT restate something that has already been said in the past.
    DO NOT add anything else.
    
    Stop speaking the moment you finish speaking from your perspective.
    
</pre>

**`topic_specifier_prompt`**

The `topic_specifier_prompt` is code that generates a prompt to make the given conversation topic more specific.  
It utilizes an LLM (Large Language Model) to refine the initial topic and create a clear topic to be conveyed to the conversation participants.

- **`topic_specifier_prompt`**
    - **Role**:
        - Generates a prompt that includes the necessary instructions to specify the topic.
    - **Composition**:
        1. **`SystemMessage`**:  
            - Provides the LLM with instructions to "make the topic more specific."
        2. **`HumanMessage`**:  
            - Includes the initial topic (`topic`) and participant names (`names`), requesting the topic to be specified within 100 words.
            - Requires the response to be written in Korean.

- **`ChatOpenAI`**
    - **Role**:
        - Takes the `topic_specifier_prompt` as input and generates a more detailed topic.
    - **Parameters**:
        - **`temperature=1.0`**:
            - Configures the model to generate more creative and varied topics.

- **Topic Specification Results**
    - **Original topic**: Outputs the initial topic.
    - **Detailed topic**: Outputs the topic refined by the LLM.


```python
# Create a prompt to specify the topic further
topic_specifier_prompt = [
    # Instruction for making the topic more specific
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}
        
        You are the moderator. 
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Speak directly to the participants: {*names,}.  
        Do not add anything else.
        """  # Do not include any additional content.
    ),
]

# Generate the specified topic using ChatOpenAI
specified_topic = ChatOpenAI(temperature=1.0).invoke(topic_specifier_prompt).content


# Print the original and detailed topics
print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")
```

<pre class="custom">Original topic:
    As of 2024, is expanding medical school enrollment in South Korea necessary?
    
    Detailed topic:
    To the participants ('Doctor Union', 'Government'): 
    Is it necessary to increase medical school enrollment in South Korea by a specific percentage each year to address the current healthcare demands and doctor shortage by 2024?
    
</pre>

Or, you can specify it directly as follows.

```python
# Directly Set Detailed Topic
specified_topic = (
    "The government has announced that it will increase the medical school enrollment quota by 2,000 starting from the 2025 admissions. "
    "In response, medical associations are organizing nationwide protest rallies to oppose this decision. "
    "Please identify the controversial issues surrounding the expansion of medical school quotas and discuss solutions for essential healthcare and regional healthcare."
)
```

## Agent Creation and Integration

This code creates and integrates **agents** to be used in the debate simulation.  
Each agent is based on the `DialogueAgentWithTools` class and is equipped to explore evidence and present counterarguments using tools.

**Components**

- **`agents`**
    - **Purpose**:
        - Creates the base agents for the simulation.
        - Each agent is configured with a name, system message, model, and tools.
    - **How it works**:
        1. Iterates through `names` and `agent_system_messages`.
        2. Initializes an instance of `DialogueAgentWithTools` for each participant.
        3. Adds the created agents to the `agents` list.

- **`agents_with_search`**
    - **Purpose**:
        - Creates additional agents equipped with search tools.
    - **How it works**:
        1. Iterates through `names_search` and `agent_system_messages`.
        2. Initializes `DialogueAgentWithTools` instances with search functionality.
        3. Adds the created agents to the `agents_with_search` list.

- **Agent Integration**
    - **Purpose**:
        - Combines `agents` and `agents_with_search` into a unified list for the simulation.
    - **How it works**:
        1. Calls `agents.extend(agents_with_search)` to merge the two lists.
        2. The unified `agents` list contains all participants, equipped with the required tools and functionalities.

```python
# This is to prevent the result from exceeding the context limit.
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4o", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

agents_with_search = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4o", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names_search.items(), agent_system_messages.values()
    )
]

agents.extend(agents_with_search)
agents
```




<pre class="custom">[<__main__.DialogueAgentWithTools at 0x2035347d690>,
     <__main__.DialogueAgentWithTools at 0x203531b32d0>,
     <__main__.DialogueAgentWithTools at 0x203531ce3d0>,
     <__main__.DialogueAgentWithTools at 0x2035333e190>]</pre>



The `select_next_speaker` function is responsible for selecting the next speaker.

```python
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    # Select the next speaker.
    # Use the remainder of step divided by the number of agents as the index to cyclically select the next speaker.
    idx = (step) % len(agents)
    return idx
```

## Debate Execution

This code runs and manages a conversation between agents using the `DialogueSimulator`.  
The debate is based on a specified topic and participating agents, with each step outputting the speaker and their message.

### Components

- **`max_iters`**
    - **Purpose**:
        - Sets the maximum number of dialogue iterations.
        - Here, `max_iters=6` limits the conversation to 6 exchanges.

- **`simulator`**
    - **Purpose**:
        - An instance of the `DialogueSimulator` class that manages the flow of dialogue and message delivery.
    - **Initialization**:
        1. **`agents`**: The list of agents participating in the conversation.
        2. **`select_next_speaker`**: A function to select the next speaker.

- **`inject` Method**
    - **Purpose**:
        - Starts the conversation by injecting the specified topic through the "Moderator" agent.
    - **How it works**:
        1. Calls `simulator.inject("Moderator", specified_topic)` to inject the topic.
        2. The "Moderator" presents the topic, initiating the dialogue.
        3. Outputs the topic:
            - `print(f"(Moderator): {specified_topic}")`.

- **`step` Method**
    - **Purpose**:
        - Executes one step of the simulation, generating a speaker and their message.
    - **How it works**:
        1. Calls `simulator.step()` to select the next speaker.
        2. The selected speaker generates a message, which all agents receive.
        3. Returns the speaker's name (`name`) and message (`message`).
        4. Outputs the speaker and message:
            - `print(f"({name}): {message}")`.

- **Repeat Loop**
    - **Purpose**:
        - Uses a `while` loop to continue the conversation up to `max_iters` times.
    - **How it works**:
        1. Initializes `n = 0` and increments `n` with each iteration.
        2. While `n < max_iters`, calls `simulator.step()` to continue the conversation.
        3. Outputs the speaker and message at each step.

```python
max_iters = 30  # Set the maximum number of iterations to 30
n = 0  # Initialize the iteration counter to 0

# Create a DialogueSimulator object with agents and a speaker selection function
simulator = DialogueSimulator(
    agents=agents_with_search, selection_function=select_next_speaker
)

# Reset the simulator to its initial state
simulator.reset()

# The Moderator introduces the specified topic
simulator.inject("Moderator", specified_topic)

# Print the topic presented by the Moderator
print(f"(Moderator): {specified_topic}")
print("\n")

# Run the simulation until the maximum number of iterations is reached
while n < max_iters:
    # Execute the next step of the simulator and retrieve the speaker and message
    name, message = simulator.step()
    # Print the speaker and their message
    print(f"({name}): {message}")
    print("\n")
    # Increment the iteration counter
    n += 1
```

<pre class="custom">(Moderator): The government has announced that it will increase the medical school enrollment quota by 2,000 starting from the 2025 admissions. In response, medical associations are organizing nationwide protest rallies to oppose this decision. Please identify the controversial issues surrounding the expansion of medical school quotas and discuss solutions for essential healthcare and regional healthcare.
    
    
    (Government): The Government of South Korea believes that expanding medical school enrollment is necessary to address the country's significant doctor shortage. With a growing elderly population, the demand for healthcare services is increasing, and we need more medical professionals to meet this demand. This expansion is in line with practices in other OECD countries that have successfully increased their medical workforce to improve healthcare access and quality.
    
    We are aware of the concerns raised by medical associations regarding potential impacts on the quality of education and the distribution of healthcare professionals. However, we are committed to implementing strong safeguards to ensure that essential and regional healthcare needs are met. Additionally, we will ensure fairness in compensation systems for newly trained medical professionals to maintain the quality of care.
    
    Our decision is based on a comprehensive analysis of the healthcare system's needs and the successful examples from other countries. We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system. 
    
    For more information, you can refer to sources such as [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext) and [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/).
    
    
    (Doctor Union): Doctor Union: While the government argues that increasing medical school enrollment is necessary to address a doctor shortage, there are significant concerns about the potential negative impacts of such a rapid expansion. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), experts warn that this move could lead to unintended consequences, such as a decline in the quality of medical education and challenges in maintaining regional sustainability and equitable distribution of healthcare professionals.
    
    Moreover, the [Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext) highlights that the government's policy changes have already led to widespread protests and substantial effects on the healthcare system. This suggests that the current infrastructure may not be able to support such a sudden increase in medical students without compromising the quality of education and training.
    
    Instead of expanding enrollment, we should focus on optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing the issues of essential and rural healthcare without overwhelming our current educational infrastructure.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the potential impacts of expanding medical school enrollment. However, it is essential to address the pressing issue of doctor shortages, particularly in rural and underserved areas, which is exacerbated by the country's rapidly aging population. According to [NPR](https://www.npr.org/2024/09/15/nx-s1-5113082/as-medical-strike-drags-on-in-south-korea-patients-are-on-edge), South Korea is already experiencing a shortage of doctors in rural regions, and the demand for healthcare is growing.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. For instance, many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages, as noted by the [OECD iLibrary](https://www.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries/education-and-training-for-doctors-and-nurses-what-s-happening-with-numerus-clausus-policies_9789264239517-6-en?crawler=true).
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure that these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    
    (Doctor Union): Doctor Union: While the government emphasizes the need to address doctor shortages, particularly in rural areas, it's important to consider the potential negative impacts of rapidly increasing medical school enrollment. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), the government's policy changes have already led to widespread protests and substantial effects on the healthcare system. This suggests that the current infrastructure may not be able to support such a sudden increase in medical students without compromising the quality of education and training.
    
    Moreover, the [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/) highlights that experts warn of unintended consequences, such as a decline in the quality of medical education and challenges in maintaining regional sustainability and equitable distribution of healthcare professionals. Instead of expanding enrollment, we should focus on optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing the issues of essential and rural healthcare without overwhelming our current educational infrastructure.
    
    
    (Government): The Government of South Korea acknowledges the concerns regarding the potential impacts of expanding medical school enrollment. However, it is crucial to address the pressing issue of doctor shortages, particularly in rural and underserved areas. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), the government plans to increase medical school admissions by 2,000 spots to address the country's doctor shortage. This expansion is the first in nearly two decades and is necessary to meet the growing healthcare demands of our aging population.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure that these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): Doctor Union: The government's plan to increase medical school enrollment by 67% has already led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Moreover, the [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/) highlights concerns about unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals. Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, it is crucial to address the pressing issue of doctor shortages, particularly in rural and underserved areas. According to the [Journal of Korean Medical Science](https://jkms.org/DOIx.php?id=10.3346/jkms.2024.39.e182), there is a significant debate among stakeholders about the expansion, but it is also recognized that the country faces a critical shortage of medical professionals, which this policy aims to address.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): Doctor Union: The government's plan to increase medical school enrollment by 2,000 spots, which represents a 67% increase, has already led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea recognizes the concerns about the rapid expansion of medical school enrollment. However, it is important to address the pressing issue of doctor shortages, particularly in rural and underserved areas. According to the [Journal of Korean Medical Science](https://jkms.org/DOIx.php?id=10.3346/jkms.2024.39.e182), there is a significant debate among stakeholders about the expansion, but it is also recognized that the country faces a critical shortage of medical professionals, which this policy aims to address.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns about the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea JoongAng Daily](https://koreajoongangdaily.joins.com/news/2024-02-06/national/socialAffairs/Govt-to-increase-med-school-enrollment-by-2000-next-year/1975768), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://www.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries/key-findings_9789264239517-3-en), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Asia News Network](https://asianews.network/south-korea-to-increase-medical-students-by-2000-despite-doctors-threat-to-strike/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://www.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries/executive-summary_9789264239517-2-en), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea JoongAng Daily](https://koreajoongangdaily.joins.com/news/2024-02-06/national/socialAffairs/Govt-to-increase-med-school-enrollment-by-2000-next-year/1975768), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://www.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries/executive-summary_9789264239517-2-en), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Asia News Network](https://asianews.network/south-korea-to-increase-medical-students-by-2000-despite-doctors-threat-to-strike/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://read.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries_9789264239517-en?sequence=1), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://read.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries_9789264239517-en?sequence=1), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. According to the [OECD iLibrary](https://read.oecd-ilibrary.org/social-issues-migration-health/health-workforce-policies-in-oecd-countries_9789264239517-en?sequence=1), many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [The Korea Times](https://www.koreatimes.co.kr/www/nation/2024/05/119_375302.html), the government's plan to increase medical school admissions is part of a broader medical sector reform initiative aimed at improving public access to healthcare services and enhancing the working environment for physicians, particularly in essential treatment fields such as pediatrics, obstetrics, and emergency medicine.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/), the government plans to increase medical school admissions by 2,000 spots to tackle this shortage, marking the first expansion in nearly two decades. This decision is aimed at addressing the shortage of doctors in non-metropolitan areas and broadening the reach of healthcare services.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    In contrast, while experiences from other OECD countries show that expanding the medical workforce can improve healthcare access and quality, these expansions are often accompanied by strict controls and regulations to maintain educational standards. For instance, England has clear regulations regarding medical school enrollment quotas, ensuring that the quality of education is not compromised ([SAGE Journals](https://journals.sagepub.com/doi/full/10.1177/20965311221141450)). This highlights the importance of careful planning and regulation in any expansion efforts.
    
    
    (Government): The Government of South Korea acknowledges the concerns raised by the Doctor Union regarding the rapid expansion of medical school enrollment. However, addressing the doctor shortage, especially in rural and underserved areas, remains a critical priority. According to the [Korea Times](https://www.koreatimes.co.kr/www/nation/2024/05/119_375302.html), the government's plan to increase medical school admissions is part of a broader medical sector reform initiative aimed at improving public access to healthcare services and enhancing the working environment for physicians, particularly in essential treatment fields such as pediatrics, obstetrics, and emergency medicine.
    
    While there are concerns about the quality of education and the distribution of healthcare professionals, the government is committed to implementing strong safeguards to ensure these issues are addressed. By expanding medical school enrollment, we aim to increase the availability of healthcare professionals, which is crucial for maintaining the sustainability and effectiveness of South Korea's healthcare system.
    
    Furthermore, experiences from other OECD countries demonstrate the benefits of expanding the medical workforce. Many OECD countries have increased the number of students admitted to medical and nursing education to address current or future staff shortages. This approach has proven effective in improving healthcare access and quality.
    
    We believe that this expansion is a crucial step towards ensuring the sustainability and effectiveness of South Korea's healthcare system, and we are committed to addressing any challenges that may arise during this process.
    
    
    (Doctor Union): The government's plan to increase medical school enrollment by 2,000 spots, representing a 67% increase, has indeed led to significant challenges within South Korea's healthcare system. According to [The Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01680-5/fulltext), this policy change has resulted in widespread protests and substantial effects on the healthcare system. The rapid expansion could overwhelm the current infrastructure, leading to a decline in the quality of medical education and training.
    
    Furthermore, experts warn that this move could have unintended consequences, such as challenges in maintaining regional sustainability and equitable distribution of healthcare professionals, as highlighted by [Korea Pro](https://koreapro.org/2024/03/south-koreas-med-school-expansion-plan-sparks-debate-over-long-term-impact/). Instead of focusing solely on increasing numbers, we should prioritize optimizing the distribution of existing medical professionals and improving working conditions to attract doctors to underserved areas. This approach would be more effective in addressing essential and rural healthcare issues without compromising the quality of education and training.
    
    Additionally, the [2024 South Korean medical crisis](https://en.wikipedia.org/wiki/2024_South_Korean_medical_crisis) highlights the ongoing healthcare crisis following the announcement of new government policies. Thousands of residents and interns have resigned, leading to significant disruptions in healthcare services. This situation underscores the importance of careful planning and regulation in any expansion efforts to avoid overwhelming the current system and ensure the quality of healthcare services.
    
    
</pre>

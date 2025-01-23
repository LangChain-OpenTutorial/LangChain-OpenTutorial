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

# Agent-with-Toolkits-File-Management

- Author: [Secludor](https://github.com/Secludor)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

When configuring an agent using LangChain, one of the biggest advantages is **the integration of various features through third-party tools** .

Among them, Toolkits provide a variety of integrated tools.

In this tutorial, we will learn how to manage local files using the `FileManagementToolkit`.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Supplies](#supplies)
- [How to Use FileManagementToolkit](#how-to-use-filemanagementtoolkit)

### References

- [Agent Toolkits](https://api.python.langchain.com/en/latest/community/agent_toolkits.html)
    - [FileManagementToolkit](https://api.python.langchain.com/en/latest/community/agent_toolkits/langchain_community.agent_toolkits.file_management.toolkit.FileManagementToolkit.html#langchain_community.agent_toolkits.file_management.toolkit.FileManagementToolkit)
- [Tools](https://python.langchain.com/docs/integrations/tools/)
    - [File System](https://python.langchain.com/docs/integrations/tools/filesystem/)

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
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

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
        }
    )

# set the project name same as the title
set_env(
    {
        "LANGCHAIN_PROJECT": "08-Agent-with-Toolkits-File-Management",
    }
)
```

## Supplies
These are the materials needed for the practice.

### GoogleNews

The `GoogleNews` class is the utility for fetching and parsing news from Google News RSS feeds. Here's a concise explanation of its key features:

**Core Functionality**

- **News Retrieval**
    - Fetches news articles from Google News RSS feeds
    - Supports both latest news and keyword-based searches
    - Returns structured data in a consistent format (URL and content)

- **Main Methods**
    - `search_latest()`: Retrieves the most recent news articles
    - `search_by_keyword()`: Searches news based on specific keywords

**Usage Example**

```python
# Initialize the news client
news = GoogleNews()

# Get latest news
latest = news.search_latest(k=3)

# Search by keyword
results = news.search_by_keyword("artificial intelligence", k=5)
```

The class handles URL encoding, RSS feed parsing, and error cases automatically, making it ideal for integration with news monitoring systems or AI agents that need access to current news content.

```python
import feedparser
from urllib.parse import quote
from typing import List, Dict, Optional


class GoogleNews:
    """
    A class for searching and retrieving Google News results.
    Provides methods to fetch latest news and search by keywords using Google News RSS feeds.
    """

    def __init__(self):
        """
        Initialize the GoogleNews class.
        Sets up the base URL for Google News RSS feed.
        """
        self.base_url = "https://news.google.com/rss"

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Fetch news from the given URL.

        Args:
            url (str): The URL to fetch news from
            k (int): Maximum number of news items to fetch (default: 3)

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing news titles and links
        """
        news_data = feedparser.parse(url)
        return [
            {"title": entry.title, "link": entry.link}
            for entry in news_data.entries[:k]
        ]

    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
         Process and organize the news list.

        Args:
            news_list (List[Dict[str, str]]): List of dictionaries containing news information

        Returns:
            List[Dict[str, str]]: List of dictionaries containing URLs and content
        """
        if not news_list:
            print("No news found for the given keyword.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_latest(self, k: int = 3) -> List[Dict[str, str]]:
        """
        Search for the latest news.

        Args:
            k (int): Maximum number of news items to retrieve (default: 3)

        Returns:
            List[Dict[str, str]]: List of dictionaries containing URLs and content
        """
        url = f"{self.base_url}?hl=en&gl=US&ceid=US:en"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

    def search_by_keyword(
        self, keyword: Optional[str] = None, k: int = 3
    ) -> List[Dict[str, str]]:
        """
         Search news by keyword.

        Args:
            keyword (Optional[str]): Keyword to search for (default: None)
            k (int): Maximum number of news items to retrieve (default: 3)

        Returns:
            List[Dict[str, str]]: List of dictionaries containing URLs and content
        """
        if keyword:
            encoded_keyword = quote(keyword)
            url = f"{self.base_url}/search?q={encoded_keyword}&hl=en&gl=US&ceid=US:en"
        else:
            url = f"{self.base_url}?hl=en&gl=US&ceid=US:en"  # latest headlines
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)
```

### AgentStreamParser
The AgentStreamParser is a utility class designed to handle and process the output stream from AI agents. Here's a concise explanation of its key features:

**Core Functionality**
- **Stream Processing**
    - Parses and processes agent actions, observations, and results in real-time
    - Handles three main types of events:
        1. Tool calls (when agents use tools)
        2. Observations (agent's findings)
        3. Final results (agent's conclusions)

- **Callback System**
    - The class uses three main callbacks:
        - Tool callbacks for monitoring tool usage
        - Observation callbacks for tracking agent findings
        - Result callbacks for handling final outputs

**Usage Example**

```python
# Create parser with default callbacks
parser = AgentStreamParser()

# Process agent output
parser.process_agent_steps({
    "actions": [...],  # Agent actions
    "steps": [...],    # Observation steps
    "output": "..."    # Final result
})
```

This parser is particularly useful for debugging agent behavior, logging agent actions, and maintaining a clear record of the agent's decision-making process.

```python
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction


# Callback function executed when a tool is called.
def tool_callback(tool) -> None:
    print("[Tool Call]")
    print(f"Tool: {tool.get('tool')}")  # Print the name of the tool used.
    if tool_input := tool.get("tool_input"):  # If there are input values for the tool
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # Print the key and value of the input.
    print(f"Log: {tool.get('log')}")  # Print the tool execution log.


# Callback function to print observation results.
def observation_callback(observation) -> None:
    print("[Observation]")
    print(
        f"Observation: {observation.get('observation')}"
    )  # Print the observation content.


# Callback function to print the final result.
def result_callback(result: str) -> None:
    print("[Final Answer]")
    print(result)  # Print the final answer.


@dataclass
class AgentCallbacks:
    """
    A dataclass containing callback functions for the agent.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): Callback function called when using tools
        observation_callback (Callable[[Dict[str, Any]], None]): Callback function called when processing observations
        result_callback (Callable[[str], None]): Callback function called when processing final results
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    A class for parsing and processing agent stream outputs.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """
        Initialize an AgentStreamParser object.

        Args:
            callbacks (AgentCallbacks, optional): Callback functions to use during parsing. Defaults to AgentCallbacks().
        """
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        Process agent steps.

        Args:
            step (Dict[str, Any]): Agent step information to process
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """
        Process agent actions.

        Args:
            actions (List[Any]): List of actions to process
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """
        Process tool calls.

        Args:
            action (Any): Tool call action to process
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """
        Process observation results.

        Args:
            observations (List[Any]): List of observation results to process
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """
        Process the final result.

        Args:
            result (str): Final result to process
        """
        self.callbacks.result_callback(result)
        self.output = result
```

## How to Use FileManagementToolkit

`FileManagementToolkit` is a toolkit for local file management operations that:
- Automates file management tasks
- Enables AI agents to manipulate files safely
- Provides comprehensive file operation tools

### Security Considerations
When using `FileManagementToolkit`, implement these security measures:
- Limit directory access using `root_dir`
- Configure filesystem permissions
- Use `selected_tools` to restrict available operations
- Run agents in sandboxed environments

### Main Components
**File Management Tools**
- `CopyFileTool` : Create a copy of a file in a specified location.
- `DeleteFileTool` : Delete a file.
- `FileSearchTool` : Recursively search for files in a subdirectory that match the regex pattern.
- `MoveFileTool` : Move or rename a file from one location to another.
- `ReadFileTool` : Read file from disk.
- `WriteFileTool` : Write file to disk.
- `ListDirectoryTool` : List files and directories in a specified folder.

**Settings**
- `root_dir` : Set the root directory of workflows.
- `selected_tools` : Select the tools you want to use.

**Dynamic Tool Creation**
- `get_tools` : create instances of the selected tools.

### 1. Basic Setup
The `FileManagementToolkit` provides essential file operation capabilities with security considerations. Let's explore how to set it up and use it safely.

```python
from langchain_community.agent_toolkits import FileManagementToolkit

# Set the working directory to a directory named 'tmp'.
working_directory = "tmp"

# Create a FileManagementToolkit object.
# Initialize toolkit with root directory
toolkit = FileManagementToolkit(root_dir=str(working_directory))

# Call the toolkit.get_tools() method to retrieve all available file management tools.
available_tools = toolkit.get_tools()

# Display available tools
print("[Available File Management Tools]")
for tool in available_tools:
    print(f"- {tool.name}: {tool.description}")
```

<pre class="custom">[Available File Management Tools]
    - copy_file: Create a copy of a file in a specified location
    - file_delete: Delete a file
    - file_search: Recursively search for files in a subdirectory that match the regex pattern
    - move_file: Move or rename a file from one location to another
    - read_file: Read file from disk
    - write_file: Write file to disk
    - list_directory: List files and directories in a specified folder
</pre>

**Selective Tool Access**
-  For better security, you can restrict available tools: 

```python
# Initialize toolkit with selected tools only
tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["read_file", "file_delete", "write_file", "list_directory"],
).get_tools()
tools
```




<pre class="custom">[ReadFileTool(root_dir='tmp'),
     DeleteFileTool(root_dir='tmp'),
     WriteFileTool(root_dir='tmp'),
     ListDirectoryTool(root_dir='tmp')]</pre>



### 2. File Operations
 Let's explore basic file operations using the toolkit's tools. Each operation demonstrates a core file management functionality. 

```python
# Unpack tools for easier access
read_tool, delete_tool, write_tool, list_tool = tools



# Create a new file with content
write_tool.invoke({"file_path": "example.txt", "text": "Hello World!"})
```




<pre class="custom">'File written successfully to example.txt.'</pre>



```python
# Check current directory contents
print(list_tool.invoke({}))
```

<pre class="custom">example.txt
</pre>

```python
# Remove the created file
print(delete_tool.invoke({"file_path": "example.txt"}))
```

<pre class="custom">File deleted successfully: example.txt.
</pre>

```python
# Verify file removal
print(list_tool.invoke({}))
```

<pre class="custom">No files found in directory .
</pre>

### 3. Advanced Usage (News Articles)
Below, we combine file management with news retrieval to create and organize news articles: 

```python
from langchain_core.tools import tool
from typing import List, Dict
import re


# Function to clean filenames by removing invalid characters and making them filesystem-safe
def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename (str): Original filename
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing periods or spaces for additional safety
    sanitized = sanitized.strip(". ")
    return sanitized


# Tool for writing news articles to files with proper formatting and sanitized filenames
@tool
def write_news_file(title: str, content: str, url: str) -> str:
    """
    Write news content to a file with a sanitized filename.

    Args:
        title: News title
        content: News content
        url: News URL
    Returns:
        str: Result message
    """
    # Create safe filename from the title
    safe_filename = sanitize_filename(title) + ".txt"

    # Format the content with title, body, and URL
    formatted_content = f"Title: {title}\nContent: {content}\nURL: {url}"

    # Get the write tool and create the file
    write_tool = next(tool for tool in tools if tool.name == "write_file")
    write_tool.invoke({"file_path": safe_filename, "text": formatted_content})

    return f"Created file: {safe_filename}"


# Define the latest news search tool.
@tool
def latest_news(k: int = 5) -> List[Dict[str, str]]:
    """Look up latest news"""
    # Create a GoogleNews object.
    news_tool = GoogleNews()
    # Search for the latest news and return the results. k indicates the number of news items to return.
    return news_tool.search_latest(k=k)


# Use the FileManagementToolkit to retrieve file management tools.
tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=[
        "copy_file",
        "file_delete",
        "file_search",
        "move_file",
        "read_file",
        "write_file",
        "list_directory",
    ],
).get_tools()

# Add custom tools
tools.extend([latest_news, write_news_file])

# Confirm which tools are available
tools
```




<pre class="custom">[CopyFileTool(root_dir='tmp'),
     DeleteFileTool(root_dir='tmp'),
     FileSearchTool(root_dir='tmp'),
     MoveFileTool(root_dir='tmp'),
     ReadFileTool(root_dir='tmp'),
     WriteFileTool(root_dir='tmp'),
     ListDirectoryTool(root_dir='tmp'),
     StructuredTool(name='latest_news', description='Look up latest news', args_schema=<class 'langchain_core.utils.pydantic.latest_news'>, func=<function latest_news at 0x000002225925F880>),
     StructuredTool(name='write_news_file', description='Write news content to a file with a sanitized filename.\n\nArgs:\n    title: News title\n    content: News content\n    url: News URL\nReturns:\n    str: Result message', args_schema=<class 'langchain_core.utils.pydantic.write_news_file'>, func=<function write_news_file at 0x000002225925E700>)]</pre>



 Set up the agent with appropriate tools and configuration: 

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Dictionary to store session history
store = {}

# Create a prompt describing tool usage
# The prompt provides the agent with text describing the tasks the model should perform (names and roles of tools).
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `latest_news` tool to find latest news. "
            "For each article, use the `write_news_file` tool to save it safely. "
            "The write_news_file tool will automatically handle filename sanitization.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create an agent configured to call the tools
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor for structured operation
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)


# Retrieve or create a session’s chat history
def get_session_history(session_ids):
    if session_ids not in store:  # If session_id is not in store
        # Create a new ChatMessageHistory object and store it
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return the session history for the specified session ID


# Create a runnable agent with a chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # Chat session_id
    get_session_history,
    # Key for input messages in the prompt
    input_messages_key="input",
    # Key for history messages in the prompt
    history_messages_key="chat_history",
)

agent_stream_parser = AgentStreamParser()
```

**Example Operations**

1. Fetch and Save News

```python
# Request the agent to fetch and store news articles
result = agent_with_chat_history.stream(

    {

        "input": "Search for the latest 5 news articles, create a file for each news article with the title as the filename (.txt), "

        "and include the content and URL of the news in the file."

    },

    config={"configurable": {"session_id": "abc123"}},

)


print("Agent Execution Result:")

for step in result:

    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">Agent Execution Result:
    [Tool Call]
    Tool: latest_news
    k: 5
    Log: 
    Invoking: `latest_news` with `{'k': 5}`
    
    
    
    [Observation]
    Observation: [{'url': 'https://news.google.com/rss/articles/CBMipwFBVV95cUxNMjdwdGtGZ1BHUTAza0d4Q25YTWRTeTl6ZFExN2VaOXZMTUV1UXdkeEFMTmdsV3VqZHBFTXp2ZkxxakcwY1l2bFpyQmQ5SElSRTdYRjYtLUZFV2tnOWMxVUtsbjVTSVMwTDBUbGl4Y2U4UnNsU2JDVDNWQ0Y0aVhuOWpxdVpwVWNTcDNjeHRRMElJQzM1YkFoM05KX2pWdFNjdVk3VUhqcw?oc=5', 'content': 'Question on ASEAN stumped Hegseth at Senate hearing. What is it and why is it important? - The Associated Press'}, {'url': 'https://news.google.com/rss/articles/CBMiswFBVV95cUxPVnNhNDh4aHpfLTJ5SlJqbkcxa3hLSGxMQUdKNm1FVlRyMDROZXp5U0hQRGhKSDFWXzBtYXhBemZESFdfcjFsOHgtOVdJd3BFQm13c3FyMnRYUWpxV19RNVVsVzJIbUVzbDlSWGxBSWlmX1pkVEV1RXhTWmdIYWJZdTJOZHE4VU5lc1ZHNDhJMmFiWGw5R19HQVRDWEJndGhpYVpOQXlPejlMUTJ4V3ZUcTFGYw?oc=5', 'content': 'Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn - CNN'}, {'url': 'https://news.google.com/rss/articles/CBMimgFBVV95cUxQdDAycFllRjA4dzRFbDZxUDZDbklKaEJkcEoyVG5GMUZscndoamY5Z24zMVpBLWZ6b2hOQThMNUVCUDdJOFNjM082MTBfR0ZHMjFMbUJybGRUYnlQLUxqRk9jbHpHQ0hRR2xCR2NoMExOMHh3WjRiSktPbldSTmlsbFNYVnlBMlYtWTdFaVhKcTg4em14WC1aSFl3?oc=5', 'content': 'The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before - CNN'}, {'url': 'https://news.google.com/rss/articles/CBMigAFBVV95cUxPMXJ6aDBPcmlsOVhyeFU2TzNBZExzNVR2NUZVTXhhVVJoX0xiWUtKa1BqbzN2cHVXNm1kMjkyMmRxWWZOWDg3WEtYbGthMmxHcFJfcXNFVDJvZ05hV0V0ckRNRXBIWHZMOXlybFFlekVkWEQzdk5FN21XZXhGRWs0ZA?oc=5', 'content': "South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff - Axios"}, {'url': 'https://news.google.com/rss/articles/CBMinwFBVV95cUxNVGJxWEdTM2w2Y3piNWlmRzliTk5JcHhoN1VhVGpSa2RXckV2bnkxVzVEcjFWb0R4Qkdxbnp3MEUtRDRaLWNIanQyYm00ZGdWM3hqdlNEZXYtTzNydWVMQUVDMFZ1VldGemFiRWNVX0NYLUlYY1ZYRGU0ajlrZnFPSVhXckwwR1JqSEYxUlNDc3ZXNU1DNS1HMFNLTi1BVDA?oc=5', 'content': 'Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights - The Associated Press'}]
    [Tool Call]
    Tool: write_news_file
    title: Question on ASEAN stumped Hegseth at Senate hearing
    content: Question on ASEAN stumped Hegseth at Senate hearing. What is it and why is it important? - The Associated Press
    url: https://news.google.com/rss/articles/CBMipwFBVV95cUxNMjdwdGtGZ1BHUTAza0d4Q25YTWRTeTl6ZFExN2VaOXZMTUV1UXdkeEFMTmdsV3VqZHBFTXp2ZkxxakcwY1l2bFpyQmQ5SElSRTdYRjYtLUZFV2tnOWMxVUtsbjVTSVMwTDBUbGl4Y2U4UnNsU2JDVDNWQ0Y0aVhuOWpxdVpwVWNTcDNjeHRRMElJQzM1YkFoM05KX2pWdFNjdVk3VUhqcw?oc=5
    Log: 
    Invoking: `write_news_file` with `{'title': 'Question on ASEAN stumped Hegseth at Senate hearing', 'content': 'Question on ASEAN stumped Hegseth at Senate hearing. What is it and why is it important? - The Associated Press', 'url': 'https://news.google.com/rss/articles/CBMipwFBVV95cUxNMjdwdGtGZ1BHUTAza0d4Q25YTWRTeTl6ZFExN2VaOXZMTUV1UXdkeEFMTmdsV3VqZHBFTXp2ZkxxakcwY1l2bFpyQmQ5SElSRTdYRjYtLUZFV2tnOWMxVUtsbjVTSVMwTDBUbGl4Y2U4UnNsU2JDVDNWQ0Y0aVhuOWpxdVpwVWNTcDNjeHRRMElJQzM1YkFoM05KX2pWdFNjdVk3VUhqcw?oc=5'}`
    
    
    
    [Tool Call]
    Tool: write_news_file
    title: Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn
    content: Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn - CNN
    url: https://news.google.com/rss/articles/CBMiswFBVV95cUxPVnNhNDh4aHpfLTJ5SlJqbkcxa3hLSGxMQUdKNm1FVlRyMDROZXp5U0hQRGhKSDFWXzBtYXhBemZESFdfcjFsOHgtOVdJd3BFQm13c3FyMnRYUWpxV19RNVVsVzJIbUVzbDlSWGxBSWlmX1pkVEV1RXhTWmdIYWJZdTJOZHE4VU5lc1ZHNDhJMmFiWGw5R19HQVRDWEJndGhpYVpOQXlPejlMUTJ4V3ZUcTFGYw?oc=5
    Log: 
    Invoking: `write_news_file` with `{'title': 'Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn', 'content': 'Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn - CNN', 'url': 'https://news.google.com/rss/articles/CBMiswFBVV95cUxPVnNhNDh4aHpfLTJ5SlJqbkcxa3hLSGxMQUdKNm1FVlRyMDROZXp5U0hQRGhKSDFWXzBtYXhBemZESFdfcjFsOHgtOVdJd3BFQm13c3FyMnRYUWpxV19RNVVsVzJIbUVzbDlSWGxBSWlmX1pkVEV1RXhTWmdIYWJZdTJOZHE4VU5lc1ZHNDhJMmFiWGw5R19HQVRDWEJndGhpYVpOQXlPejlMUTJ4V3ZUcTFGYw?oc=5'}`
    
    
    
    [Tool Call]
    Tool: write_news_file
    title: The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before
    content: The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before - CNN
    url: https://news.google.com/rss/articles/CBMimgFBVV95cUxQdDAycFllRjA4dzRFbDZxUDZDbklKaEJkcEoyVG5GMUZscndoamY5Z24zMVpBLWZ6b2hOQThMNUVCUDdJOFNjM082MTBfR0ZHMjFMbUJybGRUYnlQLUxqRk9jbHpHQ0hRR2xCR2NoMExOMHh3WjRiSktPbldSTmlsbFNYVnlBMlYtWTdFaVhKcTg4em14WC1aSFl3?oc=5
    Log: 
    Invoking: `write_news_file` with `{'title': 'The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before', 'content': 'The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before - CNN', 'url': 'https://news.google.com/rss/articles/CBMimgFBVV95cUxQdDAycFllRjA4dzRFbDZxUDZDbklKaEJkcEoyVG5GMUZscndoamY5Z24zMVpBLWZ6b2hOQThMNUVCUDdJOFNjM082MTBfR0ZHMjFMbUJybGRUYnlQLUxqRk9jbHpHQ0hRR2xCR2NoMExOMHh3WjRiSktPbldSTmlsbFNYVnlBMlYtWTdFaVhKcTg4em14WC1aSFl3?oc=5'}`
    
    
    
    [Tool Call]
    Tool: write_news_file
    title: South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff
    content: South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff - Axios
    url: https://news.google.com/rss/articles/CBMigAFBVV95cUxPMXJ6aDBPcmlsOVhyeFU2TzNBZExzNVR2NUZVTXhhVVJoX0xiWUtKa1BqbzN2cHVXNm1kMjkyMmRxWWZOWDg3WEtYbGthMmxHcFJfcXNFVDJvZ05hV0V0ckRNRXBIWHZMOXlybFFlekVkWEQzdk5FN21XZXhGRWs0ZA?oc=5
    Log: 
    Invoking: `write_news_file` with `{'title': "South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff", 'content': "South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff - Axios", 'url': 'https://news.google.com/rss/articles/CBMigAFBVV95cUxPMXJ6aDBPcmlsOVhyeFU2TzNBZExzNVR2NUZVTXhhVVJoX0xiWUtKa1BqbzN2cHVXNm1kMjkyMmRxWWZOWDg3WEtYbGthMmxHcFJfcXNFVDJvZ05hV0V0ckRNRXBIWHZMOXlybFFlekVkWEQzdk5FN21XZXhGRWs0ZA?oc=5'}`
    
    
    
    [Tool Call]
    Tool: write_news_file
    title: Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights
    content: Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights - The Associated Press
    url: https://news.google.com/rss/articles/CBMinwFBVV95cUxNVGJxWEdTM2w2Y3piNWlmRzliTk5JcHhoN1VhVGpSa2RXckV2bnkxVzVEcjFWb0R4Qkdxbnp3MEUtRDRaLWNIanQyYm00ZGdWM3hqdlNEZXYtTzNydWVMQUVDMFZ1VldGemFiRWNVX0NYLUlYY1ZYRGU0ajlrZnFPSVhXckwwR1JqSEYxUlNDc3ZXNU1DNS1HMFNLTi1BVDA?oc=5
    Log: 
    Invoking: `write_news_file` with `{'title': 'Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights', 'content': 'Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights - The Associated Press', 'url': 'https://news.google.com/rss/articles/CBMinwFBVV95cUxNVGJxWEdTM2w2Y3piNWlmRzliTk5JcHhoN1VhVGpSa2RXckV2bnkxVzVEcjFWb0R4Qkdxbnp3MEUtRDRaLWNIanQyYm00ZGdWM3hqdlNEZXYtTzNydWVMQUVDMFZ1VldGemFiRWNVX0NYLUlYY1ZYRGU0ajlrZnFPSVhXckwwR1JqSEYxUlNDc3ZXNU1DNS1HMFNLTi1BVDA?oc=5'}`
    
    
    
    [Observation]
    Observation: Created file: Question on ASEAN stumped Hegseth at Senate hearing.txt
    [Observation]
    Observation: Created file: Live updates_ Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    [Observation]
    Observation: Created file: The great social media migration_ Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    [Observation]
    Observation: Created file: South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    [Observation]
    Observation: Created file: Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    [Final Answer]
    I have successfully created files for the latest 5 news articles. Here are the titles of the files:
    
    1. **Question on ASEAN stumped Hegseth at Senate hearing.txt**
    2. **Live updates: Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt**
    3. **The great social media migration: Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt**
    4. **South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt**
    5. **Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt**
    
    Each file contains the respective article's content and URL. If you need any further assistance, feel free to ask!
</pre>

You can check the contents of the `tmp` folder and see that the files have been created as shown below.

![08-agent-with-toolkits-file-management-demonstration-00](./img/08-agent-with-toolkits-file-management-demonstration-00.png)

2. Modify Filenames

```python
result = agent_with_chat_history.stream(
    {
        "input": "Change the filenames of the previously created files by adding a suitable emoji at the beginning of the title. "
        "Also, make sure the filenames are neat."
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent Execution Result:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">Agent Execution Result:
    [Tool Call]
    Tool: list_directory
    Log: 
    Invoking: `list_directory` with `{}`
    
    
    
    [Observation]
    Observation: Live updates_ Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    Question on ASEAN stumped Hegseth at Senate hearing.txt
    South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    The great social media migration_ Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    [Tool Call]
    Tool: move_file
    source_path: Question on ASEAN stumped Hegseth at Senate hearing.txt
    destination_path: 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    Log: 
    Invoking: `move_file` with `{'source_path': 'Question on ASEAN stumped Hegseth at Senate hearing.txt', 'destination_path': '🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt'}`
    
    
    
    [Tool Call]
    Tool: move_file
    source_path: Live updates_ Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    destination_path: 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    Log: 
    Invoking: `move_file` with `{'source_path': 'Live updates_ Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt', 'destination_path': '🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt'}`
    
    
    
    [Tool Call]
    Tool: move_file
    source_path: The great social media migration_ Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    destination_path: 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    Log: 
    Invoking: `move_file` with `{'source_path': 'The great social media migration_ Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt', 'destination_path': '🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt'}`
    
    
    
    [Tool Call]
    Tool: move_file
    source_path: South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    destination_path: 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    Log: 
    Invoking: `move_file` with `{'source_path': "South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt", 'destination_path': "🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt"}`
    
    
    
    [Tool Call]
    Tool: move_file
    source_path: Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    destination_path: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    Log: 
    Invoking: `move_file` with `{'source_path': 'Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt', 'destination_path': '⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt'}`
    
    
    
    [Observation]
    Observation: File moved successfully from Question on ASEAN stumped Hegseth at Senate hearing.txt to 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt.
    [Observation]
    Observation: File moved successfully from Live updates_ Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt to 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt.
    [Observation]
    Observation: File moved successfully from The great social media migration_ Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt to 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt.
    [Observation]
    Observation: File moved successfully from South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt to 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt.
    [Observation]
    Observation: File moved successfully from Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt to ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt.
    [Final Answer]
    The filenames of the news articles have been successfully updated with suitable emojis at the beginning. Here are the new filenames:
    
    1. **🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt**
    2. **🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt**
    3. **🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt**
    4. **🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt**
    5. **⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt**
    
    If you need any more changes or assistance, feel free to let me know!
</pre>

You can check the contents of the `tmp` folder and see that the filenames have been changed as shown below.

![08-agent-with-toolkits-file-management-demonstration-01](./img/08-agent-with-toolkits-file-management-demonstration-01.png)

3. Organize Files

```python
result = agent_with_chat_history.stream(
    {
        "input": "Create a `news` folder and then copy all previously created files into that folder. "
        "Ensure that the contents are copied as well."
    },
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent Execution Result:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">Agent Execution Result:
    [Tool Call]
    Tool: list_directory
    Log: 
    Invoking: `list_directory` with `{}`
    
    
    
    [Observation]
    Observation: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    [Tool Call]
    Tool: write_file
    file_path: news/.keep
    text: 
    append: False
    Log: 
    Invoking: `write_file` with `{'file_path': 'news/.keep', 'text': '', 'append': False}`
    
    
    
    [Observation]
    Observation: File written successfully to news/.keep.
    [Tool Call]
    Tool: copy_file
    source_path: 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    destination_path: news/🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    Log: 
    Invoking: `copy_file` with `{'source_path': '🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt', 'destination_path': 'news/🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt'}`
    
    
    
    [Tool Call]
    Tool: copy_file
    source_path: 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    destination_path: news/🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    Log: 
    Invoking: `copy_file` with `{'source_path': '🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt', 'destination_path': 'news/🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt'}`
    
    
    
    [Tool Call]
    Tool: copy_file
    source_path: 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    destination_path: news/🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    Log: 
    Invoking: `copy_file` with `{'source_path': '🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt', 'destination_path': 'news/🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt'}`
    
    
    
    [Tool Call]
    Tool: copy_file
    source_path: 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    destination_path: news/🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    Log: 
    Invoking: `copy_file` with `{'source_path': "🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt", 'destination_path': "news/🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt"}`
    
    
    
    [Tool Call]
    Tool: copy_file
    source_path: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    destination_path: news/⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    Log: 
    Invoking: `copy_file` with `{'source_path': '⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt', 'destination_path': 'news/⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt'}`
    
    
    
    [Observation]
    Observation: File copied successfully from 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt to news/🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt.
    [Observation]
    Observation: File copied successfully from 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt to news/🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt.
    [Observation]
    Observation: File copied successfully from 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt to news/🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt.
    [Observation]
    Observation: File copied successfully from 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt to news/🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt.
    [Observation]
    Observation: File copied successfully from ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt to news/⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt.
    [Final Answer]
    I have created a `news` folder and successfully copied all the previously created news files into that folder. The contents have been preserved in the copied files. If you need any further assistance or modifications, just let me know!
</pre>

You can check the contents of the `tmp` folder and see that the `news` folder has been created and the files have been copied as shown below.

![08-agent-with-toolkits-file-management-demonstration-02](./img/08-agent-with-toolkits-file-management-demonstration-02.png)

```python
result = agent_with_chat_history.stream(
    {"input": "Delete all .txt files except for those in the news folder."},
    config={"configurable": {"session_id": "abc123"}},
)

print("Agent Execution Result:")
for step in result:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">Agent Execution Result:
    [Tool Call]
    Tool: file_search
    pattern: *.txt
    Log: 
    Invoking: `file_search` with `{'pattern': '*.txt'}`
    
    
    
    [Observation]
    Observation: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    news\⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    news\🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    news\🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    news\🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    news\🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    [Tool Call]
    Tool: file_delete
    file_path: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt
    Log: 
    Invoking: `file_delete` with `{'file_path': '⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt'}`
    
    
    
    [Tool Call]
    Tool: file_delete
    file_path: 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt
    Log: 
    Invoking: `file_delete` with `{'file_path': "🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt"}`
    
    
    
    [Tool Call]
    Tool: file_delete
    file_path: 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt
    Log: 
    Invoking: `file_delete` with `{'file_path': '🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt'}`
    
    
    
    [Tool Call]
    Tool: file_delete
    file_path: 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt
    Log: 
    Invoking: `file_delete` with `{'file_path': '🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt'}`
    
    
    
    [Tool Call]
    Tool: file_delete
    file_path: 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt
    Log: 
    Invoking: `file_delete` with `{'file_path': '🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt'}`
    
    
    
    [Observation]
    Observation: File deleted successfully: ⚖️ Supreme Court will hear Texas anti-pornography law that challengers say violates free-speech rights.txt.
    [Observation]
    Observation: File deleted successfully: 🇰🇷 South Korea's impeached president Yoon Suk Yeol arrested after hours-long standoff.txt.
    [Observation]
    Observation: File deleted successfully: 🇸🇬 Question on ASEAN stumped Hegseth at Senate hearing.txt.
    [Observation]
    Observation: File deleted successfully: 🌐 The great social media migration Sudden influx of US users to RedNote connects Chinese and Americans like never before.txt.
    [Observation]
    Observation: File deleted successfully: 🔥 Live updates Los Angeles wildfires, Palisades and Eaton fires, Auto fire in Ventura continue to burn.txt.
    [Final Answer]
    All .txt files outside of the `news` folder have been successfully deleted. The files in the `news` folder remain intact. If you need any further assistance or modifications, feel free to ask!
</pre>

You can check the contents of the `tmp` folder and see that all files except for the `news` folder have been deleted, as shown below.

![08-agent-with-toolkits-file-management-demonstration-03](./img/08-agent-with-toolkits-file-management-demonstration-03.png)

### 4. Error Handling Examples
To ensure robust file operations, wrap your tool calls with a safe operation function: 

```python
def safe_file_operation(operation_func, **kwargs):
    """
    Wrapper function for safe file operations with comprehensive error handling.
    """
    try:
        # Tools expect a single input dictionary
        return operation_func(input=kwargs)
    except FileNotFoundError:
        return "Error: File not found"
    except PermissionError:
        return "Error: Permission denied"
    except IsADirectoryError:
        return "Error: Path is a directory, not a file"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Example usage with correct argument structure
def demonstrate_file_operations():
    # Writing to a protected directory
    result1 = safe_file_operation(
        write_tool.invoke, file_path="protected/example.txt", text="Test content"
    )
    print(f"Writing to protected directory: {result1}")

    # Reading non-existent file
    result2 = safe_file_operation(read_tool.invoke, file_path="nonexistent.txt")
    print(f"Reading non-existent file: {result2}")


# Run the demonstrations
demonstrate_file_operations()
```

<pre class="custom">Writing to protected directory: File written successfully to protected/example.txt.
    Reading non-existent file: Error: no such file or directory: nonexistent.txt
</pre>

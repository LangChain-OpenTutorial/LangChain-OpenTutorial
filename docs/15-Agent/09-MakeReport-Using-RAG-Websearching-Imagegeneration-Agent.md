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

# Make Report Using RAG, Web searching, Image generation Agent

- Author: [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb)

## Overview
In this tutorial, we showcase how to use **three different agents** in LangChain to create a comprehensive report. Specifically, we combine:

1. **Web Searching Agent:**  
   - Performs web searches (via a custom tool) to gather additional real-time information.  

2. **RAG (Retrieval-Augmented Generation) Agent:**  
   - Uses a local PDF (e.g., Tesla-related) loaded and chunked into a VectorStore.  
   - Provides relevant context from the PDF using retrieval tools.  

3. **Image Generation Agent:**  
   - Utilizes the **DALL·E** tool to generate images based on text prompts.  

These agents collect data from a PDF, supplement it with web search results, and enrich the final report with generated images, all while demonstrating **streaming outputs** in real time.

By the end of this tutorial, you will learn how to:

- **Integrate** multiple agents (Web Searching, RAG, Image Generation) in a single **LangChain** pipeline.
- **Generate** and **update** a Markdown report (`report.md` and `report-final.md`) using the agents’ outputs.
- **Observe** and process streaming outputs using a **custom generator** and **callback** system.

### Table of Contents
- [Overview](#overview)  
- [Environment Setup](#environment-setup)  
- [Implementing Multiple Agents](#implementing-multiple-agents)  
- [Using the Agent in a Streaming Workflow](#using-the-agent-in-a-streaming-workflow)  

### References

- [LangChain ChatOpenAI API Reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)  
- [LangChain Tools Documentation](https://python.langchain.com/docs/integrations/tools/)  
- [TavilySearchResults Documentation](https://python.langchain.com/docs/integrations/tools/tavily_search/)  
- [DALL·E API Wrapper Documentation](https://python.langchain.com/docs/integrations/tools/dalle_image_generator/)  

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
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langchain_community",
        "langchain_text_splitters",
        "faiss-cpu",
        "pymupdf",
        "pydantic",
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
        "LANGCHAIN_PROJECT": "09-MakeReport_Using_RAG_Websearching_Imagegeneration",
        "TAVILY_API_KEY": "",
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



## Implementing Multiple Agents

In the following sections, we set up a **LangChain** pipeline with three different agents:
1. **Web Searching Agent** to gather live data,
2. **RAG (Retrieval-Augmented Generation) Agent** to pull context from a PDF,
3. **Image Generation Agent** for creating a final image.

We will demonstrate how each agent can be combined to generate a streaming report with real-time parsing and callback-driven outputs.


### Adding a Web Searching Agent

Below, we import a sample tool `TavilySearchResults` for performing web searches. 
This will serve as our **Web Searching Agent** to gather real-time information 
based on user queries.

To use the Tavily Search API, you need to obtain an API key.

- You can obtain your API key by visiting the following link: [Tavily Search API Registration](https://app.tavily.com/home).

You can set `TAVILY_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `TAVILY_API_KEY` in previous steps.

```python
import os

# Set the environment variable directly
os.environ["TAVILY_API_KEY"] = "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Access the API key
print(f"Tavily API Key: {os.getenv('TAVILY_API_KEY')}")
```

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Create an instance of TavilySearchResults with k=6 for retrieving up to 6 search results
search = TavilySearchResults(k=6)
```

### Data Loading and Vector Store (RAG)

Next, we set up the **RAG (Retrieval-Augmented Generation) Agent**. 
Below, we load a PDF file (e.g., `shsconf_icdeba2023_02022.pdf`), split it into chunks, 
and create a **VectorStore** using **FAISS**. We then initialize a **retriever** 
to query those chunks.

Document Used for Practice  

**Tesla's Revenue Forecast Based on Business Model and Financial Statement Analysis**  

**Author:** Chenhao Fang  
**Institution:** Intelligent Accounting Management Institute, Guangdong University of Finance and Economics  
**Link:** [Tesla's revenue forecast base on business model and financial statement analysis ](https://www.shs-conferences.org/articles/shsconf/pdf/2024/01/shsconf_icdeba2023_02022.pdf)  
**File Name:** shsconf_icdeba2023_02022.pdf

_Please copy the downloaded file to the data folder for practice._  


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# Example PDF file path (modify according to your environment)
pdf_file_path = "data/shsconf_icdeba2023_02022.pdf"

# Load the PDF using PyMuPDFLoader
loader = PyMuPDFLoader(pdf_file_path)

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = loader.load_and_split(text_splitter)

# Create FAISS VectorStore
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Create a retriever from the VectorStore
retriever = vector.as_retriever()
```

### Wrapping the Retriever as a Tool

We wrap our **retriever** in a LangChain tool so it can be invoked by an agent. 
Here, we define a prompt template to format the retrieved documents.


```python
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

document_prompt = PromptTemplate.from_template(
    "<document><content>{page_content}</content><page>{page}</page><filename>{source}</filename></document>"
)

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search for information in Tesla PDF file",
    document_prompt=document_prompt,
)
```

### Adding a DALL·E Tool for Image Generation

Below, we set up the **Image Generation Agent** using `DallEAPIWrapper`. 
This allows our pipeline to generate images based on text prompts 
and integrate them into the final report.


```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.tools import tool

dalle = DallEAPIWrapper(
    model="dall-e-3",  # DALL·E model version
    size="1024x1024",  # Image size
    quality="standard",  # Image quality
    n=1,  # Number of images to generate
)


@tool
def dalle_tool(query: str) -> str:
    """Use this tool to generate an image from text"""
    return dalle.run(query)
```

### File Management Tools

Next, we set up file management tools to enable the agent to write, read, 
and list files within a specified directory. This is used to store 
and update the `report.md`, `report-final.md`, and other files.


```python
from langchain_community.agent_toolkits import FileManagementToolkit

working_directory = "tmp"

file_tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["write_file", "read_file", "list_directory"],
).get_tools()
```

### Combining Tools

We now combine all tools (Web Searching, RAG, DALL·E, File Management) 
into a single list.

```python
tools = file_tools + [
    retriever_tool,  # PDF search (RAG)
    search,  # Web search
    dalle_tool,  # Image generation
]
tools
```




<pre class="custom">[WriteFileTool(root_dir='tmp'),
     ReadFileTool(root_dir='tmp'),
     ListDirectoryTool(root_dir='tmp'),
     Tool(name='pdf_search', description='use this tool to search for information in Tesla PDF file', args_schema=<class 'langchain_core.tools.retriever.RetrieverInput'>, func=functools.partial(<function _get_relevant_documents at 0x0000024611FB99E0>, retriever=VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x000002462904FC50>, search_kwargs={}), document_prompt=PromptTemplate(input_variables=['page', 'page_content', 'source'], input_types={}, partial_variables={}, template='<document><content>{page_content}</content><page>{page}</page><filename>{source}</filename></document>'), document_separator='\n\n'), coroutine=functools.partial(<function _aget_relevant_documents at 0x00000246122F5120>, retriever=VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x000002462904FC50>, search_kwargs={}), document_prompt=PromptTemplate(input_variables=['page', 'page_content', 'source'], input_types={}, partial_variables={}, template='<document><content>{page_content}</content><page>{page}</page><filename>{source}</filename></document>'), document_separator='\n\n')),
     TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr('**********'))),
     StructuredTool(name='dalle_tool', description='Use this tool to generate an image from text', args_schema=<class 'langchain_core.utils.pydantic.dalle_tool'>, func=<function dalle_tool at 0x000002467EE4ADE0>)]</pre>



### Creating a Prompt and Agent

Here, we create a `ChatPromptTemplate` and a `LangChain` agent to handle 
LLM calls and tool usage. We store each session’s chat history in a dictionary 
to maintain context across multiple steps.



```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

store = {}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "You are a professional researcher. "
            "You can use the pdf_search tool to search for information in the Tesla PDF file. "
            "You can find further information by using search tool. "
            "You can use image generation tool to generate image from text. "
            "Finally, you can use file management tool to save your research result into files.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)


def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

### Synchronous Parsing

Below, we define callbacks to monitor and process the agent’s steps in real time. 
They capture **tool calls**, **observations**, and **final results**.


```python
from langchain_core.agents import AgentAction, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction

from typing import Any, Dict, List, Callable
from dataclasses import dataclass


# Callback for tool calls
def tool_callback(tool) -> None:
    print("[Tool Called]")
    print(f"Tool: {tool.get('tool')}")
    if tool_input := tool.get("tool_input"):
        for k, v in tool_input.items():
            print(f"{k}: {v}")
    print(f"Log: {tool.get('log')}")


# Callback for observations
def observation_callback(observation) -> None:
    print("[Observation]")
    print(f"Observation: {observation.get('observation')}")


# Callback for final result
def result_callback(result: str) -> None:
    print("[Final Answer]")
    print(result)


@dataclass
class AgentCallbacks:
    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    A class to parse and handle streaming output from the agent.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """Initialize the AgentStreamParser with custom callbacks."""
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """Process each step in the agent's output."""
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """Handle each tool action detected."""
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """Handle logic for a single tool call."""
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """Handle the observations produced by the agent."""
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """Handle the final result from the agent."""
        self.callbacks.result_callback(result)
        self.output = result


# Create the stream parser
agent_stream_parser = AgentStreamParser()
```

## Using the Agent in a Streaming Workflow

Now, we will demonstrate how to use all three agents (**Web Searching**, **RAG**, and **Image Generation**) 
in a **streaming** fashion. Each step is processed by our custom parser, 
allowing us to see **tool calls**, **observations**, and **final answers** in real time.


### Step 1: Summarize PDF Content and Save to `report.md`

First, we instruct the agent to summarize key aspects of the Tesla PDF and save the summary to `report.md`.

```python
# This cell asks the agent to summarize certain aspects of the Tesla PDF,
# then writes them to 'report.md' and reads the file contents.

query_1 = (
    "Please find key information about Tesla's financial status and revenue outlook in the Tesla PDF. "
    "Write the summary in English. "
    "Then create a new file `report.md` and save the summary there.\n\n"
    "#Instructions:\n"
    "1. Write an appropriate header (Markdown heading level 2).\n"
    "2. Include the PDF page numbers and file name in parentheses (e.g., page 10, filename.pdf).\n"
    "3. Provide organized bullet points.\n"
    "4. After writing, save the file as `report.md`.\n"
    "5. Finally, read the `report.md` file and display its content.\n"
)

result_1 = agent_with_chat_history.stream(
    {"input": query_1}, config={"configurable": {"session_id": "tesla_session_1"}}
)

print("=== 1) Agent Execution Result ===")
for step in result_1:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">=== 1) Agent Execution Result ===
    [Tool Called]
    Tool: pdf_search
    query: Tesla financial status and revenue outlook
    Log: 
    Invoking: `pdf_search` with `{'query': 'Tesla financial status and revenue outlook'}`
    
    
    
    [Observation]
    Observation: <document><content>Tesla's revenue forecast base on business model and financial 
    statement analysis 
    Chenhao Fang   
    Intelligent Accounting Management Institute, Guangdong University of Finance and Economics, 510200 Guangzhou, China 
    Abstract. With the rapid development of the electric vehicle industry, Tesla, as the benchmark of this industry, 
    has been studied by many companies in the same industry and investors outside the industry. Based on Tesla's 
    financial reports in recent years and news about Tesla, Tesla's assets, liabilities, and owner's equity are analyzed 
    to analyze Tesla's basic financial situation, from this, it can be concluded that Tesla's asset-liability ratio and 
    other financial trends have changed in recent years. Then analyze Tesla's business model from the aspects of 
    corporate positioning, business type, and profit model, so, in recent years, Tesla's main profit models, income</content><page>0</page><filename>data/shsconf_icdeba2023_02022.pdf</filename></document>
    
    <document><content>Corresponding author: 1910741120@mail.sit.edu.cn 
    year. With a penetration rate of 2.1%, the global sales 
    volume of automobiles in 2018 exceeded 2 million. China, 
    the USA, Germany, France, Sweden, Japan, and other 
    nations account for a large portion of the world's new energy 
    market [3]. If you want to predict the future development of 
    the electric vehicle industry, you can speculate by predicting 
    Tesla's future financial situation. This study hopes to 
    provide some insight into the future forecast of the electric 
    vehicle industry by predicting Tesla's future financial 
    situation. In this research, common-size financial statements 
    and ratio analysis are used. Start with Tesla's basic financial 
    situation, then analyze Tesla's business model, and finally 
    predict Tesla's future financial situation. 
    2 An analysis of Tesla's underlying 
    financial position 
    2.1 Assets 
     
     
     
    SHS Web of Conferences 181, 02022 (2024)	
    https://doi.org/10.1051/shsconf/202418102022
    ICDEBA 2023</content><page>0</page><filename>data/shsconf_icdeba2023_02022.pdf</filename></document>
    
    <document><content>strategy of the company's development. Similarly, gross 
    margin will fluctuate with changes in capacity, sales volume, 
    scale effect, and amortized production costs.. 
    In 2020, Tesla produced a total of 1,369,600 vehicles 
    worldwide and delivered a total of 1,313,900 vehicles, +40% 
    year-on-year, slightly below the 50% growth target set at the 
    beginning of 22 years. 
    Number of vehicle sales are expected to reach 1.8 
    million units in 23 years, approximately +35% y/y. 
    Combined with a forecast for the number of Tesla electric 
    vehicle deliveries, we can predict Tesla's future revenue. 
    6 Conclusion 
    Fig. 1. Revenue forecast. 
    Figure 1 shows Tesla's future revenue. So, Tesla's forecast 
    profit is $71.4 billion in 2023, $84.6 billion in 2024, and 
    $131.2 billion in 2025. Based on the average price of Tesla 
    in 23 years of $47,000, the revenue of the electric vehicle 
    business in 23 years is estimated to be 84.6 billion US 
    dollars, +18% year-on-year. Assuming a vehicle price of</content><page>4</page><filename>data/shsconf_icdeba2023_02022.pdf</filename></document>
    
    <document><content>dollars, +18% year-on-year. Assuming a vehicle price of 
    $45,000 and $43,000 in 24 and '25, by 2025, the revenue 
    from electric vehicle sales will be about $130 billion, +80% 
    from '22. All of this is based on an analysis of Tesla's 
    existing assets and liabilities and business model. This study 
    predicts Tesla's income in the next few years. It is certain 
    that Tesla's income will continue to grow in the next few 
    years, which may help some investors related to Tesla in 
    terms of investment. However, since Tesla discloses less 
    economic information to the outside world, and the research 
    method of this paper is relatively simple, the information 
    that can be provided is limited. In future studies, researchers 
    can use more diverse and accurate methods to predict 
    Tesla's income. 
    References 
    1. Y. Liu, Jiangsu Comm Forum 369, 7 (2015) 
    2. F. Han, Beijing Uni Posts Telec, (2018) 
    3. S. Yan, Modern Busi 589, 36 (2020) 
    4. M. Wang, Shandong Uni, (2021)</content><page>4</page><filename>data/shsconf_icdeba2023_02022.pdf</filename></document>
    [Tool Called]
    Tool: write_file
    file_path: report.md
    text: ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    Log: 
    Invoking: `write_file` with `{'file_path': 'report.md', 'text': "## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)\n\n- Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.\n- Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.\n- The business model is analyzed through corporate positioning, business type, and profit model.\n\n## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)\n\n- In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.\n- Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.\n- Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.\n- Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.\n- The income growth predictions are based on Tesla's existing assets, liabilities, and business model.\n- Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions."}`
    
    
    
    [Observation]
    Observation: File written successfully to report.md.
    [Tool Called]
    Tool: read_file
    file_path: report.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report.md'}`
    
    
    
    [Observation]
    Observation: ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    [Final Answer]
    Here is the summary of Tesla's financial status and revenue outlook, saved in `report.md`:
    
    ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
</pre>

When you check the contents of the generated report file (`report.md`), it will display as follows.  
![](./assets/09-makereport-using-rag-websearching-imagegeneration-report-using-rag.png)

### Step 2: Perform Web Search and Append to report.md

Next, we perform a web search about Tesla's revenue outlook, append the findings to `report.md`, 
and then read the updated file content.

```python
# This cell performs a web search about Tesla's revenue outlook,
# appends the findings to 'report.md', and then reads the file content again.

query_2 = (
    "Now, please search the web about Tesla's revenue outlook, summarize the results, and write them in English. "
    "Then open `report.md` to read its existing content. "
    "Append the new web findings to the end, following the same format.\n\n"
    "#Instructions:\n"
    "1. Use a Markdown heading level 2 for your new section.\n"
    "2. Provide the source (URL) where you found the information.\n"
    "3. Include organized bullet points of the web search results.\n"
    "4. After writing, save the file as `report.md`.\n"
    "5. Finally, read `report.md` and display its content.\n"
)

result_2 = agent_with_chat_history.stream(
    {"input": query_2}, config={"configurable": {"session_id": "tesla_session_1"}}
)

print("=== 2) Agent Execution Result ===")
for step in result_2:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">=== 2) Agent Execution Result ===
    [Tool Called]
    Tool: tavily_search_results_json
    query: Tesla revenue outlook 2023
    Log: 
    Invoking: `tavily_search_results_json` with `{'query': 'Tesla revenue outlook 2023'}`
    
    
    
    [Observation]
    Observation: [{'url': 'https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html', 'content': "S&P Futures\nDow Futures\nNasdaq Futures\nRussell 2000 Futures\nCrude Oil\nGold\nSilver\nEUR/USD\n10-Yr Bond\nVix\ndólar/libra\nUSD/JPY\nBitcoin USD\nCMC Crypto 200\nFTSE 100\nNikkei 225\nTesla Inc (TSLA) Posts Record Vehicle Deliveries and Strong Profitability in 2023 Earnings\nRevenue: Tesla's diluted GAAP EPS for Q4 was $2.27, and for the full year, it was $4.30, marking a 19% increase YoY.\nFounded in 2003 and headquartered in Palo Alto, California, Tesla has established itself as a leader in the electric vehicle and clean energy industry. Tesla's strong free cash flow of $4.4 billion for the year, despite its highest capital expenditures and R&D expenses in company history, underscores its financial discipline and commitment to growth.\n The company's mission to accelerate the world's transition to sustainable energy is supported by its integrated approach, which includes the production and sale of electric vehicles, solar energy generation, and energy storage products. Fiscal Summary and Company Overview\nFor the year 2023, Tesla reported a total revenue of $96.8 billion, a 19% increase from the previous year."}, {'url': 'https://finance.yahoo.com/news/tesla-full-2023-earnings-eps-114607342.html', 'content': 'S&P Futures\nDow Futures\nNasdaq Futures\nRussell 2000 Futures\nCrude Oil\nGold\nSilver\nEUR/USD\n10-Yr Bond\nVix\ndólar/libra\nUSD/JPY\nBitcoin USD\nCMC Crypto 200\nFTSE 100\nNikkei 225\nTesla Full Year 2023 Earnings: EPS Beats Expectations\nTesla (NASDAQ:TSLA) on average during the next 3 years, compared to a 15% growth forecast for the Auto industry in the US.\nPerformance of the American Auto industry.\n All figures shown in the chart above are for the trailing 12 month (TTM) period\nTesla EPS Beats Expectations\nRevenue was in line with analyst estimates. Full Year 2023 Results\nKey Financial Results\nRevenue: US$96.8b (up 19% from FY 2022).\n Risk Analysis\nBefore you take the next step you should know about the 1 warning sign for Tesla that we have uncovered.\n'}, {'url': 'https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/', 'content': "Meanwhile in 2023, analysts expect Tesla earnings per share to grow a cooler 40% to $5.66, with sales up 42% to $118.2 billion. That's a strong pace, but the era of triple-digit growth and idea of"}, {'url': 'https://www.counterpointresearch.com/insights/tesla-q4-2023-results/', 'content': "The company's overall gross profit for the year 2023 declined by 15% YoY as it reduced the prices of its vehicles multiple times and recorded lower-than-expected FSD revenue. Outlook: Tesla did not specify any target for total vehicle sales in 2024 but has indicated a lower growth compared to 2023."}, {'url': 'https://ir.tesla.com/press-release/tesla-releases-fourth-quarter-and-full-year-2023-financial-results', 'content': 'Tesla Releases Fourth Quarter and Full Year 2023 Financial Results | Tesla Investor Relations Skip to main content  Tesla Releases Fourth Quarter and Full Year 2023 Financial Results AUSTIN, Texas, January 24, 2024 – Tesla has released its financial results for the fourth quarter and full year ended December 31, 2023 by posting an update on its Investor Relations website. Please visit https://ir.tesla.com to view the update. As previously announced, Tesla management will host a live question and answer (Q&A) webcast at 4:30 p.m. Central Time (5:30 p.m. Eastern Time) to discuss the results and outlook. What:\xa0Tesla Fourth Quarter and Full Year 2023 Financial Results Q&A Webcast Q4 & FY 23 Update: https://ir.tesla.com Webcast: https://ir.tesla.com (live and replay) ir@tesla.com'}]
    [Tool Called]
    Tool: read_file
    file_path: report.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report.md'}`
    
    
    
    [Observation]
    Observation: ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    [Tool Called]
    Tool: write_file
    file_path: report.md
    text: 
    
    ## Tesla's Revenue Outlook from Web Sources
    
    - **Source:** [Yahoo Finance](https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html)
      - Tesla reported a total revenue of $96.8 billion for the year 2023, representing a 19% increase from the previous year.
      - Strong free cash flow of $4.4 billion for 2023, despite high capital expenditures and R&D expenses.
    
    - **Source:** [Investors.com](https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/)
      - Analysts expect Tesla's earnings per share to grow by 40% in 2023.
      - Projected sales growth of 42% leading to $118.2 billion in revenue for 2023.
    
    - **Source:** [Counterpoint Research](https://www.counterpointresearch.com/insights/tesla-q4-2023-results/)
      - Gross profit declined by 15% year-over-year due to multiple price reductions and lower-than-expected FSD revenue.
      - No specific sales target for 2024, but anticipated lower growth compared to 2023.
    append: True
    Log: 
    Invoking: `write_file` with `{'file_path': 'report.md', 'text': "\n\n## Tesla's Revenue Outlook from Web Sources\n\n- **Source:** [Yahoo Finance](https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html)\n  - Tesla reported a total revenue of $96.8 billion for the year 2023, representing a 19% increase from the previous year.\n  - Strong free cash flow of $4.4 billion for 2023, despite high capital expenditures and R&D expenses.\n\n- **Source:** [Investors.com](https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/)\n  - Analysts expect Tesla's earnings per share to grow by 40% in 2023.\n  - Projected sales growth of 42% leading to $118.2 billion in revenue for 2023.\n\n- **Source:** [Counterpoint Research](https://www.counterpointresearch.com/insights/tesla-q4-2023-results/)\n  - Gross profit declined by 15% year-over-year due to multiple price reductions and lower-than-expected FSD revenue.\n  - No specific sales target for 2024, but anticipated lower growth compared to 2023.", 'append': True}`
    
    
    
    [Observation]
    Observation: File written successfully to report.md.
    [Tool Called]
    Tool: read_file
    file_path: report.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report.md'}`
    
    
    
    [Observation]
    Observation: ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    
    ## Tesla's Revenue Outlook from Web Sources
    
    - **Source:** [Yahoo Finance](https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html)
      - Tesla reported a total revenue of $96.8 billion for the year 2023, representing a 19% increase from the previous year.
      - Strong free cash flow of $4.4 billion for 2023, despite high capital expenditures and R&D expenses.
    
    - **Source:** [Investors.com](https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/)
      - Analysts expect Tesla's earnings per share to grow by 40% in 2023.
      - Projected sales growth of 42% leading to $118.2 billion in revenue for 2023.
    
    - **Source:** [Counterpoint Research](https://www.counterpointresearch.com/insights/tesla-q4-2023-results/)
      - Gross profit declined by 15% year-over-year due to multiple price reductions and lower-than-expected FSD revenue.
      - No specific sales target for 2024, but anticipated lower growth compared to 2023.
    [Final Answer]
    Here is the updated content of `report.md`, which includes both the information from the Tesla PDF and recent web findings about Tesla's revenue outlook:
    
    ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    
    ## Tesla's Revenue Outlook from Web Sources
    
    - **Source:** [Yahoo Finance](https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html)
      - Tesla reported a total revenue of $96.8 billion for the year 2023, representing a 19% increase from the previous year.
      - Strong free cash flow of $4.4 billion for 2023, despite high capital expenditures and R&D expenses.
    
    - **Source:** [Investors.com](https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/)
      - Analysts expect Tesla's earnings per share to grow by 40% in 2023.
      - Projected sales growth of 42% leading to $118.2 billion in revenue for 2023.
    
    - **Source:** [Counterpoint Research](https://www.counterpointresearch.com/insights/tesla-q4-2023-results/)
      - Gross profit declined by 15% year-over-year due to multiple price reductions and lower-than-expected FSD revenue.
      - No specific sales target for 2024, but anticipated lower growth compared to 2023.
</pre>

When you check the contents of the updated report file (`report.md`), it will display as follows.  
![](./assets/09-makereport-using-rag-websearching-imagegeneration-report-using-websearching.png)

### Step 3: Create a Professional Report and Save to `report-final.md`

Then, we instruct the agent to create a more professional report based on `report.md`, 
add a table, and save it as `report-final.md`. Finally, we read and display the final report.

```python
# This cell instructs the agent to create a more professional report based on 'report.md',
# add a table, and save it as 'report-final.md'. Then it reads and displays the final report.

query_3 = (
    "Please open `report.md` and display its content. "
    "Using that information, create a professional-level report. "
    "The report should have exactly 3 sections:\n"
    "1. Overview: an abstract of approximately 300 characters.\n"
    "2. Key Points: the core content of the report, including a Markdown table.\n"
    "3. Conclusion: the final conclusion of the report, with references (file name, URL, etc.). "
    "After writing, save as `report-final.md` and then read it to show its content.\n"
)

result_3 = agent_with_chat_history.stream(
    {"input": query_3}, config={"configurable": {"session_id": "tesla_session_1"}}
)

print("=== 3) Agent Execution Result ===")
for step in result_3:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">=== 3) Agent Execution Result ===
    [Tool Called]
    Tool: read_file
    file_path: report.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report.md'}`
    
    
    
    [Observation]
    Observation: ## Tesla's Financial Status and Revenue Outlook (page 0, data/shsconf_icdeba2023_02022.pdf)
    
    - Tesla's financial reports and news have been used to analyze its assets, liabilities, and owner's equity.
    - Recent trends show changes in Tesla's asset-liability ratio and other financial indicators.
    - The business model is analyzed through corporate positioning, business type, and profit model.
    
    ## Revenue Forecast (page 4, data/shsconf_icdeba2023_02022.pdf)
    
    - In 2020, Tesla produced 1,369,600 vehicles and delivered 1,313,900 vehicles, a 40% increase year-on-year.
    - Vehicle sales are expected to reach 1.8 million units in 2023, approximately a 35% year-on-year increase.
    - Forecasted revenue is $71.4 billion in 2023, $84.6 billion in 2024, and $131.2 billion in 2025.
    - Revenue from electric vehicle sales is expected to rise by 80% from 2022 to 2025.
    - The income growth predictions are based on Tesla's existing assets, liabilities, and business model.
    - Tesla's financial disclosures are limited, affecting the comprehensiveness of these predictions.
    
    ## Tesla's Revenue Outlook from Web Sources
    
    - **Source:** [Yahoo Finance](https://finance.yahoo.com/news/tesla-inc-tsla-posts-record-225430663.html)
      - Tesla reported a total revenue of $96.8 billion for the year 2023, representing a 19% increase from the previous year.
      - Strong free cash flow of $4.4 billion for 2023, despite high capital expenditures and R&D expenses.
    
    - **Source:** [Investors.com](https://www.investors.com/news/tesla-stock-2023-forecast-reveals-a-tale-of-two-challenging-ev-megamarkets/)
      - Analysts expect Tesla's earnings per share to grow by 40% in 2023.
      - Projected sales growth of 42% leading to $118.2 billion in revenue for 2023.
    
    - **Source:** [Counterpoint Research](https://www.counterpointresearch.com/insights/tesla-q4-2023-results/)
      - Gross profit declined by 15% year-over-year due to multiple price reductions and lower-than-expected FSD revenue.
      - No specific sales target for 2024, but anticipated lower growth compared to 2023.
    [Tool Called]
    Tool: write_file
    file_path: report-final.md
    text: # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
    append: False
    Log: 
    Invoking: `write_file` with `{'file_path': 'report-final.md', 'text': '# Tesla Financial and Revenue Analysis Report\n\n## Overview\nTesla\'s financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.\n\n## Key Points\n\n### Financial Status and Revenue Outlook (PDF)\n| Aspect                              | Details                                                                 |\n|-------------------------------------|-------------------------------------------------------------------------|\n| Financial Reports                   | Analyze assets, liabilities, and owner\'s equity                         |\n| Trends                              | Changes in asset-liability ratio and other indicators                   |\n| Business Model                      | Corporate positioning, business type, profit model                      |\n| Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |\n| Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |\n| Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |\n\n### Web-Based Revenue Outlook\n| Source               | Key Information                                                                                       |\n|----------------------|-------------------------------------------------------------------------------------------------------|\n| Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |\n| Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |\n| Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |\n\n## Conclusion\nTesla\'s financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.', 'append': False}`
    
    
    
    [Observation]
    Observation: File written successfully to report-final.md.
    [Tool Called]
    Tool: read_file
    file_path: report-final.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report-final.md'}`
    
    
    
    [Observation]
    Observation: # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
    [Final Answer]
    Here is the professional-level report based on the gathered information:
    
    # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
</pre>

When you check the contents of the newly created report file (`report-final.md`), it will display as follows.  

![](./assets/09-makereport-using-rag-websearching-imagegeneration-report-summary.png)

### Step 4: Generate and Embed an Image into `report-final.md`

Finally, we generate an image symbolizing Tesla’s future using the **Image Generation Agent**, 
and prepend the image URL to `report-final.md`.


```python
# This cell generates an image symbolizing Tesla’s future using the dalle_tool,
# and then prepends the image URL to 'report-final.md'.

query_4 = (
    "Please open `report-final.md` and display its content. "
    "Then generate an image representing Tesla's future using dalle_tool. "
    "Add the image URL to the top of the final report in Markdown format. "
    "Finally, save the updated report in `report-final.md` again."
)

result_4 = agent_with_chat_history.stream(
    {"input": query_4}, config={"configurable": {"session_id": "tesla_session_1"}}
)

print("=== 4) Agent Execution Result ===")
for step in result_4:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">=== 4) Agent Execution Result ===
    [Tool Called]
    Tool: read_file
    file_path: report-final.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report-final.md'}`
    
    
    
    [Observation]
    Observation: # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
    [Tool Called]
    Tool: dalle_tool
    query: Tesla future vision, electric cars, futuristic cityscape, innovation
    Log: 
    Invoking: `dalle_tool` with `{'query': 'Tesla future vision, electric cars, futuristic cityscape, innovation'}`
    
    
    
    [Observation]
    Observation: https://oaidalleapiprodscus.blob.core.windows.net/private/org-oG25CgRXgwdFS9EGu24bU4El/user-xfPeE0u0duR9G7bVjOLozrgQ/img-BmVs7TFcBWrxiqIxxXV4v8G3.png?st=2025-01-15T01%3A58%3A14Z&se=2025-01-15T03%3A58%3A14Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A26%3A08Z&ske=2025-01-16T00%3A26%3A08Z&sks=b&skv=2024-08-04&sig=il7vroBZYS2LfOcuhSpTMp/beSRQZtfxzv1F7227yYU%3D
    [Tool Called]
    Tool: write_file
    file_path: report-final.md
    text: ![Tesla Future Vision](https://oaidalleapiprodscus.blob.core.windows.net/private/org-oG25CgRXgwdFS9EGu24bU4El/user-xfPeE0u0duR9G7bVjOLozrgQ/img-BmVs7TFcBWrxiqIxxXV4v8G3.png?st=2025-01-15T01%3A58%3A14Z&se=2025-01-15T03%3A58%3A14Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A26%3A08Z&ske=2025-01-16T00%3A26%3A08Z&sks=b&skv=2024-08-04&sig=il7vroBZYS2LfOcuhSpTMp/beSRQZtfxzv1F7227yYU%3D)
    
    # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
    append: False
    Log: 
    Invoking: `write_file` with `{'file_path': 'report-final.md', 'text': '![Tesla Future Vision](https://oaidalleapiprodscus.blob.core.windows.net/private/org-oG25CgRXgwdFS9EGu24bU4El/user-xfPeE0u0duR9G7bVjOLozrgQ/img-BmVs7TFcBWrxiqIxxXV4v8G3.png?st=2025-01-15T01%3A58%3A14Z&se=2025-01-15T03%3A58%3A14Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A26%3A08Z&ske=2025-01-16T00%3A26%3A08Z&sks=b&skv=2024-08-04&sig=il7vroBZYS2LfOcuhSpTMp/beSRQZtfxzv1F7227yYU%3D)\n\n# Tesla Financial and Revenue Analysis Report\n\n## Overview\nTesla\'s financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.\n\n## Key Points\n\n### Financial Status and Revenue Outlook (PDF)\n| Aspect                              | Details                                                                 |\n|-------------------------------------|-------------------------------------------------------------------------|\n| Financial Reports                   | Analyze assets, liabilities, and owner\'s equity                         |\n| Trends                              | Changes in asset-liability ratio and other indicators                   |\n| Business Model                      | Corporate positioning, business type, profit model                      |\n| Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |\n| Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |\n| Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |\n\n### Web-Based Revenue Outlook\n| Source               | Key Information                                                                                       |\n|----------------------|-------------------------------------------------------------------------------------------------------|\n| Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |\n| Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |\n| Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |\n\n## Conclusion\nTesla\'s financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.', 'append': False}`
    
    
    
    [Observation]
    Observation: File written successfully to report-final.md.
    [Tool Called]
    Tool: read_file
    file_path: report-final.md
    Log: 
    Invoking: `read_file` with `{'file_path': 'report-final.md'}`
    
    
    
    [Observation]
    Observation: ![Tesla Future Vision](https://oaidalleapiprodscus.blob.core.windows.net/private/org-oG25CgRXgwdFS9EGu24bU4El/user-xfPeE0u0duR9G7bVjOLozrgQ/img-BmVs7TFcBWrxiqIxxXV4v8G3.png?st=2025-01-15T01%3A58%3A14Z&se=2025-01-15T03%3A58%3A14Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A26%3A08Z&ske=2025-01-16T00%3A26%3A08Z&sks=b&skv=2024-08-04&sig=il7vroBZYS2LfOcuhSpTMp/beSRQZtfxzv1F7227yYU%3D)
    
    # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
    [Final Answer]
    Here is the final content of `report-final.md`, now including an image depicting Tesla's future vision:
    
    ![Tesla Future Vision](https://oaidalleapiprodscus.blob.core.windows.net/private/org-oG25CgRXgwdFS9EGu24bU4El/user-xfPeE0u0duR9G7bVjOLozrgQ/img-BmVs7TFcBWrxiqIxxXV4v8G3.png?st=2025-01-15T01%3A58%3A14Z&se=2025-01-15T03%3A58%3A14Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A26%3A08Z&ske=2025-01-16T00%3A26%3A08Z&sks=b&skv=2024-08-04&sig=il7vroBZYS2LfOcuhSpTMp/beSRQZtfxzv1F7227yYU%3D)
    
    # Tesla Financial and Revenue Analysis Report
    
    ## Overview
    Tesla's financial trajectory shows robust growth, with significant increases in vehicle production and sales. Revenue forecasts from both internal and external sources indicate continued expansion, although challenges remain due to market dynamics and financial disclosures.
    
    ## Key Points
    
    ### Financial Status and Revenue Outlook (PDF)
    | Aspect                              | Details                                                                 |
    |-------------------------------------|-------------------------------------------------------------------------|
    | Financial Reports                   | Analyze assets, liabilities, and owner's equity                         |
    | Trends                              | Changes in asset-liability ratio and other indicators                   |
    | Business Model                      | Corporate positioning, business type, profit model                      |
    | Vehicle Production (2020)           | 1,369,600 vehicles produced, 1,313,900 delivered, 40% increase Y-o-Y    |
    | Revenue Forecast (2023-2025)        | $71.4B in 2023, $84.6B in 2024, $131.2B in 2025                         |
    | Revenue Growth                      | 80% rise in electric vehicle revenue from 2022 to 2025                  |
    
    ### Web-Based Revenue Outlook
    | Source               | Key Information                                                                                       |
    |----------------------|-------------------------------------------------------------------------------------------------------|
    | Yahoo Finance        | $96.8B revenue in 2023, 19% increase from 2022, $4.4B free cash flow                                   |
    | Investors.com        | 40% EPS growth, 42% sales growth, $118.2B projected revenue for 2023                                   |
    | Counterpoint Research| 15% decline in gross profit Y-o-Y, anticipated lower growth in 2024                                    |
    
    ## Conclusion
    Tesla's financial performance and revenue outlook reflect a dynamic growth trajectory, driven by increased production and strategic business models. However, the variability in forecasts and financial disclosures highlight the need for cautious optimism. For further details, refer to "data/shsconf_icdeba2023_02022.pdf" and online sources from Yahoo Finance, Investors.com, and Counterpoint Research.
</pre>

Finally, when you check a portion of the most recently generated report file (`report-final.md`), it will display as follows.  

![](./assets/09-makereport-using-rag-websearching-imagegeneration-report-add-image.png)



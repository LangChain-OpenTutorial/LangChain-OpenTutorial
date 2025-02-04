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

# Tools


- Author: [Jaeho Kim](https://github.com/Jae-hoya)
- Design: []()
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## OverView

A tool is an interface that allows agents, chains, or LLMs to interact with the external world.

LangChain provides built-in tools that are easy to use, and it also enables users to easily build custom tools.

**You can find the list of tools integrated into LangChain at the link below.**

- [List of Tools Integrated with LangChain](https://python.langchain.com/docs/integrations/tools/)

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Built-in tools](#built-in-tools)
- [Python REPL Tool](#python-repl-tool)
- [Search API Tool(Tavily)](#search-api-tooltavily)
- [Image Generation Tool (DALL-E)](#image-generation-tool-dall-e)
- [Custom Tools](#custom-tools)
### References

- [LangChain: List of Integrated Tools](https://python.langchain.com/docs/integrations/tools/)
- [LangChain Tools/Toolkits](https://python.langchain.com/docs/integrations/tools/)
- [LangChain: PythonREPLTool](https://python.langchain.com/docs/integrations/tools/python/)
- [LangChain: Tavily Search](https://python.langchain.com/docs/integrations/tools/tavily_search/)
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
        "langchain_core",
        "langchain_openai",
        "langchain_experimental",
        "tavily-python",
        "feedparser",
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
        "TAVILY_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "01-Tools",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Environment variables have been set successfully.
You can alternatively set API keys, such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



**Note**

When using tools, warning messages may be displayed.

For example, there are security warnings in the **REPL** environment.
`PythonREPLTool` is a tool for executing Python code and can potentially execute commands (e.g., file system access, network requests) that may pose security risks.
In such cases, LangChain or Python itself may output warning messages.

To ignore these warning messages, you can use the following code:

```python
import warnings

# Ignore warning messages.
warnings.filterwarnings("ignore")
```

## Built-in tools

You can use pre-defined tools and toolkits provided by LangChain.

A tool refers to a single utility, while a toolkit combines multiple tools into a single unit for use.

You can find the relevant tools at the link below.

**Note**
- [LangChain Tools/Toolkits](https://python.langchain.com/docs/integrations/tools/)

## Python REPL Tool

This tool provides a class for executing Python code in a **REPL (Read-Eval-Print Loop)** environment.
- [PythonREPLTool](https://python.langchain.com/docs/integrations/tools/python/)

**Description**

- Provides a Python shell environment.
- Executes valid Python commands as input.
- Use the `print(...)` function to view results.

**Key Features**

- sanitize_input: Option to sanitize input (default: True)
- python_repl: Instance of **PythonREPL** (default: executed in the global scope)

**Usage**

- Create an instance of `PythonREPLTool `.
- Execute Python code using the `run` , `arun` , or `invoke` methods.

**Input Sanitization**

- Removes unnecessary spaces, backticks, the keyword "python," and other extraneous elements from the input string.

```python
from langchain_experimental.tools import PythonREPLTool

# Creates a tool for executing Python code.
python_tool = PythonREPLTool()
```

```python
# Executes Python code and returns the results.
print(python_tool.invoke("print(100 + 200)"))
```

<pre class="custom">Python REPL can execute arbitrary code. Use with caution.
</pre>

    300
    
    

Below is an example of requesting an LLM to write Python code and returning the results.

**Workflow Summary**
1. Request the LLM model to write Python code for a specific task.
2. Execute the generated code to obtain the results.
3. Output the results.

**Note**

I recommend using a model equivalent to or higher than GPT-4.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# A function that executes Python code, outputs intermediate steps, and returns the tool execution results.
def print_and_execute(code, debug=True):
    if debug:
        print("CODE:")
        print(code)
    return python_tool.invoke(code)


# A prompt requesting Python code to be written.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Raymond Hetting, an expert python programmer, well versed in meta-programming and elegant, concise and short but well documented code. You follow the PEP8 style guide. "
            "Return only the code, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the code.",
        ),
        ("human", "{input}"),
    ]
)
# Create LLM model.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create a chain using the prompt and the LLM model.
chain = prompt | llm | StrOutputParser() | RunnableLambda(print_and_execute)
```

```python
# Outputting the results.
print(chain.invoke("Write code to generate Powerball numbers."))
```

<pre class="custom">CODE:
    import random
    
    def generate_powerball():
        main_numbers = sorted(random.sample(range(1, 70), 5))
        powerball = random.randint(1, 26)
        return main_numbers, powerball
    
    # Example usage
    main_numbers, powerball = generate_powerball()
    print("Main numbers:", main_numbers)
    print("Powerball:", powerball)
    Main numbers: [7, 17, 18, 51, 52]
    Powerball: 16
    
</pre>

## Search API Tool(Tavily)

This is a tool that implements a search function using the `Tavily` Search API. It provides two main classes: `TavilySearchResults` and `TavilyAnswer` .

**API Key Issuance URL**
- https://app.tavily.com/

Set the issued API key as an environment variable.

For example, configure the .env file as follows:

```
TAVILY_API_KEY=tvly-abcdefghijklmnopqrstuvwxyz
```

### TavilySearchResults

**Description**
- Queries the Tavily Search API and returns results in JSON format.
- A search engine optimized for comprehensive, accurate, and reliable results.
- Useful for answering questions about current events.

**Key Parameters**
- `max_results` (int): Maximum number of search results to return (default: 5).
- `search_depth` (str): Search depth ("basic" or "advanced").
- `include_domains` (List[str]): List of domains to include in search results.
- `exclude_domains` (List[str]): List of domains to exclude from search results.
- `include_answer` (bool): Whether to include a short answer to the original query.
- `include_raw_content` (bool): Whether to include refined HTML content from each site.
- `include_images` (bool): Whether to include a list of images related to the query.

**Return Value**
- A JSON-formatted string containing the search results (url, content).

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Create tool.
tool = TavilySearchResults(
    max_results=6,
    include_answer=True,
    include_raw_content=True,
    # include_images=True,
    # search_depth="advanced", # or "basic"
    include_domains=["github.io"],
    # exclude_domains = []
)
```

```python
# Execute tool.
tool.invoke({"query": "What is Langchain Tools?"})
```




<pre class="custom">[{'url': 'https://stonefishy.github.io/2024/11/12/introduction-to-langchain-make-ai-smarter-and-easy-to-use/',
      'content': 'LangChain makes it easier by offering ready-made building blocks to connect these models to other tools, data, and even databases. Think of LangChain like a set of Lego blocks that you can use to build cool things with AI.'},
     {'url': 'https://kirenz.github.io/lab-langchain-functions/slides/05_tools_routing.html',
      'content': 'Langchain Functions - Tools and Routing from langchain.agents import tool from langchain.tools.render import format_tool_to_openai_function format_tool_to_openai_function(get_current_temperature) {‘name’: ‘get_current_temperature’, ‘description’: ‘get_current_temperature(latitude: float, longitude: float) -> dict - Fetch current temperature for given coordinates.’, ‘parameters’: {‘title’: ‘OpenMeteoInput’, ‘type’: ‘object’, ‘properties’: {‘latitude’: {‘title’: ‘Latitude’, ‘description’: ‘Latitude of the location to fetch weather data for’, ‘type’: ‘number’}, ‘longitude’: {‘title’: ‘Longitude’, ‘description’: ‘Longitude of the location to fetch weather data for’, ‘type’: ‘number’}}, ‘required’: [‘latitude’, ‘longitude’]}} {‘name’: ‘search_wikipedia’, ‘description’: ‘search_wikipedia(query: str) -> str - Run Wikipedia search and get page summaries.’, ‘parameters’: {‘title’: ‘search_wikipediaSchemaSchema’, ‘type’: ‘object’, ‘properties’: {‘query’: {‘title’: ‘Query’, ‘type’: ‘string’}}, ‘required’: [‘query’]}} format_tool_to_openai_function(search_wikipedia) search_wikipedia, get_current_temperature result = chain.invoke({"input": "what is the weather in stuttgart right now"}) result.tool_input get_current_temperature(result.tool_input) result = chain.invoke({"input": "What is the weather in stuttgart right now?"}) result = chain.invoke({"input": "What is langchain?"})'},
     {'url': 'https://j3ffyang.github.io/langchain_project_book/fundamentals/index.html',
      'content': 'The fundamental ideas and elements of LangChain, a framework for creating language-model-powered applications, are covered in this chapter. LangChain aims to develop data-aware and agentic applications that enable language models to communicate with their surroundings and establish connections with other data sources, rather than only calling out to a language model via an API. Moreover, LangChain is context-aware, allowing applications to make decisions depending on the context that is supplied by linking a language model to context-giving sources. LangChain is an open-source framework designed to help build applications powered by LLMs, like ChatGPT, and create more advanced use cases around LLMs by chaining together different components from several modules.'},
     {'url': 'https://aws-samples.github.io/amazon-bedrock-samples/agents-and-function-calling/function-calling/tool_binding/tool_bindings/',
      'content': "Tool binding with Langchain Langchain's bind_tools function takes a list of Langchain Tool, Pydantic classes or JSON schemas. We set our tools through Python functions and use the a weather agent example. With this agent, a requester can get up-to-date weather information based on a given location. Tool definition We define ToolsList to include get_lat_long, which gets a set of coordinates for"},
     {'url': 'https://langchain-ai.github.io/langgraph/how-tos/tool-calling/',
      'content': "How to call tools using ToolNode This guide covers how to use LangGraph's prebuilt ToolNode for tool calling. ToolNode is a LangChain Runnable that takes graph state (with a list of messages) as input and outputs state update with the result of tool calls. It is designed to work well out-of-box with LangGraph's prebuilt ReAct agent, but can also work with any StateGraph as long as its state"},
     {'url': 'https://langchain-ai.github.io/langgraph/',
      'content': 'from typing import Annotated, Literal, TypedDict from langchain_core.messages import HumanMessage from langchain_anthropic import ChatAnthropic from langchain_core.tools import tool from langgraph.checkpoint.memory import MemorySaver from langgraph.graph import END, START, StateGraph, MessagesState from langgraph.prebuilt import ToolNode'}]</pre>



## Image Generation Tool (DALL-E)

- `DallEAPIWrapper Class` : A wrapper for OpenAI's DALL-E image generator.

This tool allows easy integration of the DALL-E API to implement text-based image generation functionality. With various configuration options, it can be utilized as a flexible and powerful image generation tool.

**Key Properties**

- `model` : The name of the DALL-E model to use ( **dall-e-2**, **dall-e-3** ).

- `n` : Number of images to generate (default: 1).

- `size` : Size of the generated image:
  - "dall-e-2": "1024x1024", "512x512", "256x256"
  - "dall-e-3": "1024x1024", "1792x1024", "1024x1792"

- `style` : Style of the generated image ( **natural** , **vivid** ).

- `quality` : Quality of the generated image ( **standard**, **hd** ).

- `max_retries` : Maximum number of retries for generation.

**Key Features**
- Generates images based on text descriptions using the DALL-E API.

**Workflow Summary**

The following is an example of generating images using the `DALL-E` Image Generator.

This time, we will use the `DallEAPIWrapper` to generate images.

The input prompt will request the LLM model to write a prompt for generating images.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=1000)

# Define a prompt template for DALL-E image generation
prompt = PromptTemplate.from_template(
    "Generate a detailed IMAGE GENERATION prompt for DALL-E based on the following description. "
    "Return only the prompt, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the prompt"
    "Output should be less than 1000 characters. Write in English only."
    "Image Description: \n{image_desc}",
)

# Create a chain connecting the prompt, LLM, and output parser
chain = prompt | llm | StrOutputParser()

# Execute the chain
image_prompt = chain.invoke(
    {"image_desc": "A Neo-Classicism painting satirizing people looking at their smartphones."}
)

# Output the image prompt
print(image_prompt)
```

<pre class="custom">A Neo-Classicism painting depicting a grand, opulent setting reminiscent of classical art, featuring elegantly dressed figures in flowing robes and togas, gathered in a lush, sunlit garden. Each figure is engrossed in their smartphones, completely absorbed in their screens, creating a stark contrast between their traditional attire and modern technology. The lush greenery and marble statues serve as a backdrop, highlighting the absurdity of their distraction. Subtle expressions of amusement and bewilderment are visible on their faces as they interact with their devices, oblivious to the beauty around them. Incorporate classical architectural elements like columns and arches in the background to emphasize the Neo-Classical style. The color palette should be rich and vibrant, with soft, diffused lighting to enhance the dreamlike quality of the scene while maintaining a satirical tone.
</pre>

Let’s use the previously generated image prompt as input to the `DallEAPIWrapper` to generate an image.

```python
# Importing the DALL-E API Wrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from IPython.display import Image

dalle = DallEAPIWrapper(
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    n=1,
)

# query
query = "A Neo-Classicism painting satirizing people looking at their smartphones."


# Generate image and retrieve URL
# Use chain.invoke() to convert the image description into a DALL-E prompt
# Use dalle.run() to generate the actual image
image_url = dalle.run(chain.invoke({"image_desc": query}))

# Display the generated image.
Image(url=image_url, width=500)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-cklzgJgdr1X4aNqRAAPddNfR/user-UJN3VEkv67JiO9Mm1aeNBkBJ/img-FmgzP2ThfH8hiDFaVTZNN8w9.png?st=2025-01-15T14%3A16%3A31Z&se=2025-01-15T16%3A16%3A31Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-15T00%3A58%3A21Z&ske=2025-01-16T00%3A58%3A21Z&sks=b&skv=2024-08-04&sig=bkBfHzzoIVbYLsYSWS4PNiV8Y2YwLWmsowmJjjgL/ZU%3D" width="500"/>



The image below was generated by DALL-E.

![dall-e_image.png](./img/01-tools-dall-e.png)

## Custom Tools

In addition to the built-in tools provided by LangChain, you can define and use your own custom tools.

To do this, use the `@tool` decorator provided by the `langchain.tools` module to convert a function into a tool.

### @tool Decorator

This decorator allows you to transform a function into a tool. It provides various options to customize the behavior of the tool.

**How to Use**
1. Apply the `@tool` decorator above the function.
2. Set the decorator parameters as needed.

Using this decorator, you can easily convert regular Python functions into powerful tools, enabling automated documentation and flexible interface creation.

```python
from langchain.tools import tool


# Convert a function into a tool using a decorator.
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
```

```python
# Execute tool.
add_numbers.invoke({"a": 3, "b": 4})
```




<pre class="custom">7</pre>



```python
# Execute tool.
multiply_numbers.invoke({"a": 3, "b": 4})
```




<pre class="custom">12</pre>



### Tavily Custom Tool: Enhancing Tool Control through Custom Tool Configuration


By using `@tool Decorator` , you can create a tool with improved control by leveraging the TavilyClient provided by the Tavily package.


Below are the key parameters used in the Tavily.

**Basic Search Configuration**

- query (str): The keyword or phrase to search for.
- search_depth (str): The level of detail for the search. Choose between **basic**  or **advanced** . (default: basic)
- topic (str): The topic area of the search. Choose between **general**  or **news** . (default: general)
- days (int): The recency of search results. Only results within the specified number of days will be returned. (default: 3)
- max_results (int): The maximum number of search results to retrieve. (default: 5)

**Domain Filtering**

- include_domains (list): A list of domains that must be included in the search results.
- exclude_domains (list): A list of domains to exclude from the search results.

**Detailed Result Settings**

- include_answer (bool): Whether to include answers generated by the API.
- include_raw_content (bool): Whether to include the original HTML content of the webpage.
- include_images (bool): Whether to include related image information.
- format_output (bool): Whether to apply formatting to the search results.

**Miscellaneous**

- kwargs : Additional keyword arguments. These may be used for future API updates or special features.

```python
# %pip install tavily-python
```

```python
# Import the TavilyClient class from the Tavily package.
from tavily import TavilyClient

tavily_tool = TavilyClient()

# Example of a search using various parameters
result1 = tavily_tool.search(
    query="Tell me about LangChain",  # Search query
    search_depth="advanced",  # Advanced search depth
    topic="general",  # General topic
    days=7,  # Results from the last 7 days
    max_results=10,  # Maximum of 10 results
    include_answer=True,  # Include answers
    include_raw_content=True,  # Include raw content
    include_images=True,  # Include images
    format_output=True,  # Format the output
)

# Print the results
print("Basic search results:", result1)
```

<pre class="custom">Basic search results: {'query': 'Tell me about LangChain', 'follow_up_questions': None, 'answer': "LangChain is an open-source framework designed to simplify the development of applications using large language models (LLMs) like OpenAI or Hugging Face. It enables the creation of dynamic, data-responsive applications that leverage the latest advancements in natural language processing. LangChain streamlines the development lifecycle of LLM applications, offering components and integrations for building stateful agents with streaming and human-in-the-loop support. It can be used for various applications such as chatbots, Generative Question-Answering (GQA), summarization, and more. LangChain's modular structure allows for chaining together different components to create advanced use cases around LLMs.", 'images': ['https://blog.enterprisedna.co/wp-content/uploads/2023/05/b0de1ae8-5b77-42c8-9be9-9daede50b378.png', 'https://substackcdn.com/image/fetch/w_1200,h_600,c_fill,f_jpg,q_auto:good,fl_progressive:steep,g_auto/https://substack-post-media.s3.amazonaws.com/public/images/ad2da2e9-47ff-46df-87f4-59385508c935_1164x1316.png', 'https://blog.streamlit.io/content/images/2023/05/schematic-1.jpeg', 'https://daxg39y63pxwu.cloudfront.net/images/blog/langchain/LangChain.webp', 'https://miro.medium.com/v2/resize:fit:1200/1*-PlFCd_VBcALKReO3ZaOEg.png'], 'results': [{'title': 'What is LangChain? A Beginners Guide With Examples', 'url': 'https://blog.enterprisedna.co/what-is-langchain-a-beginners-guide-with-examples/', 'content': 'LangChain is an intuitive open-source framework created to simplify the development of applications using large language models (LLMs), such as OpenAI or Hugging Face. This allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.', 'score': 0.8812988, 'raw_content': 'What is LangChain? A Beginners Guide With Examples – Master Data Skills + AI\n\nSign In\nGet Started\n\nTutorials\nArticles\nGuides\nPodcasts\nCourses\nData Mentor\n\nEDNA Chat\n\n\nTutorials\n\nArticles\nGuides\nPodcasts\nCourses\nData Mentor\nEDNA Chat\n\nWhat is LangChain? A Beginners Guide With Examples\nby Sam McKay, CFA | AI\n\nNowadays, apps need to be super smart when it comes to understanding language, and that’s where LangChain comes in! It makes it easy to connect AI models with all kinds of different data sources so you can get your hands on totally customized natural language processing (NLP) solutions.\nLangChain is an intuitive open-source framework created to simplify the development of applications using large language models (LLMs), such as OpenAI or Hugging Face. This allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.\n\nIn this article, we will cover the key features of LangChain, including its AI capabilities, the types of data sources it can connect with, and the range of NLP solutions it can offer.\n\nWe’ll also dive into some potential use cases for LangChain, from sentiment analysis to chatbots and beyond.\nWhether you’re a developer, a data scientist, or simply curious about the latest developments in NLP technology, this article is for you.\nSo, if you want to learn more about LangChain and how it can help you unlock the power of language in your business or organization, keep reading!\n(Running low on time? Scroll down to watch the video version of this article)\nTable of Contents\nToggle\n\nWhat is LangChain?\nWhat are the Core Components of LangChain?\n1. Large Language Models and APIs\n2. Framework and Libraries\n3. Documentation and Modules\n\n\nInstallation and Setup of LangChain\nStep 1: Obtain an API key from the OpenAI platform.\nStep 2: Set up the OpenAI API key as an environment variable in your project to ensures secure access without hardcoding the key in your code. For example, in a .env file, add the following line:\nStep 3: In your Python script, import the necessary libraries and load the environment variable:\nStep 4: Now, you can use LangChain to interact with the OpenAI API. For example, to generate a text response using GPT-3:\n\n\nHow to Develop Applications with LangChain\n3 Application Examples of LangChain\n1. Text Summarization (Data Augmented Generation)\n2. Question Answering\n3. Chatbots (Language Model)\n\n\nWhat are Integrations in LangChain?\nAdvanced Features and Customization of LangChain\nResources and Support for LangChain\nFinal Thoughts\n\nWhat is LangChain?\nLangChain is a powerful, open-source framework designed to help you develop applications powered by a language model, particularly a large language model (LLM).\nIt goes beyond standard API calls by being data-aware and agentic, enabling connections with various data sources for richer, personalized experiences. It can also empower a language model to interact dynamically with its environment.\n\nLangChain streamlines the development of diverse applications, such as chatbots, Generative Question-Answering (GQA), and summarization. By “chaining” components from multiple modules, it allows for the creation of unique applications built around an LLM.\nNow that you understand what LangChain is and why it is important, let’s explore the core components of LangChain in the next section.\nWhat are the Core Components of LangChain?\nTo be able to fully interpret the workings of LangChain, it is important to understand it’s core components.\nThis section covers the primary aspects of LangChain: language models and APIs, framework and libraries, and documentation and modules.\nLet’s get into it!\n\n1. Large Language Models and APIs\nLangChain supports language models, including those from prominent AI platforms like OpenAI, which is the company behind the revolutionary chatbot ChatGPT. These models are the foundation for creating powerful, language-driven applications.\nLangChain provides an application programming interface (APIs) to access and interact with them and facilitate seamless integration, allowing you to harness the full potential of LLMs for various use cases.\n\nFor example, you can create a chatbot that generates personalized travel itineraries based on user’s interests and past experiences.\n2. Framework and Libraries\nThe LangChain framework consists of an array of tools, components, and interfaces that simplify the development process for language model-powered applications.\nIt offers Python libraries to help streamline rich, data-driven interactions with the AI models by chaining different components together.\n\nSome advantages of the LangChain framework include the following:\n\n\nEfficient integration with popular AI platforms such as OpenAI and Hugging Face\n\n\nAccess to language-driven data-aware applications by connecting the models to other data sources\n\n\nAgility through enabling a language model to interact dynamically with their environment\n\n\n3. Documentation and Modules\nTo make it easier for you to develop applications using LangChain, the framework has extensive documentation.\n\nThis guide covers different aspects of development, such as:\n\n\nSetting up your development environment\n\n\nIntegrating preferred AI models\n\n\nCreating advanced use cases supported by LangChain\n\n\nIn addition, modular construction facilitates high levels of customization for your applications. You can choose and combine modules according to your needs, further enhancing LangChain’s versatility.\nInstallation and Setup of LangChain\nTo start using LangChain in your project, first ensure Python is installed on your system. LangChain can be effortlessly installed with pip, Python’s default package manager.\n\nSimply open your terminal or command prompt and enter:\npython\npip install langchain\nCode Explainer!\nCopy\nThis command installs LangChain and its required dependencies in your Python environment. Now, you’re ready to harness the power of LangChain for language model-driven applications.\nThroughout your project, you may need to connect LangChain with various model providers, data stores, and APIs. For instance, to utilize OpenAI’s APIs, install their SDK:\npython\npip install openai\nCode Explainer!\nCopy\nAfter installing the OpenAI SDK, you can connect it with LangChain by following these steps:\nStep 1: Obtain an API key from the OpenAI platform.\nSign up or log in to your account on the OpenAI website, then navigate to the API Keys section.\n\nClick on Create New Secret Key.\n\nGive your key a unique name and click Create New Key.\n\nNow you can copy your newly generated secret key and use it in your applications.\n\nStep 2: Set up the OpenAI API key as an environment variable in your project to ensures secure access without hardcoding the key in your code. For example, in a .env file, add the following line:\npython\nOPENAI_API_KEY=your_api_key_here\nCode Explainer!\nCopy\nStep 3: In your Python script, import the necessary libraries and load the environment variable:\n```python\nimport os\nfrom dotenv import load_dotenv\nimport openai\nimport langchain\nload_dotenv()\nopenai.api_key = os.getenv("OPENAI_API_KEY")\n```\nCode Explainer!\nCopy\n\nStep 4: Now, you can use LangChain to interact with the OpenAI API. For example, to generate a text response using GPT-3:\npython\nresponse = langchain.generate_text(prompt="What are the benefits of using LangChain?", model="openai/gpt-3")\nprint(response)\nCode Explainer!\nCopy\n\nNow that you know how to set up your development environment using the OpenAI API key, we take at how you can develop apps using LangChain in the next section.\nHow to Develop Applications with LangChain\nLangChain is an open-source framework designed for developing applications powered by a language model.\nYou can utilize its capabilities to build powerful applications that make use of AI models like ChatGPT while integrating with external sources such as Google Drive, Notion, and Wikipedia.\n\nDeveloping applications with LangChain is a straightforward process that involves the following steps.\n\n\nDefine your use case: First, you need to define your use case and requirements, which will help you select the appropriate components and LLMs to use.\n\n\nBuild the logic: Next, you can use LangChain’s flexible prompts and chains to build the logic of your application. You can also use LangChain’s code to create custom functionality tailored to your use case. Once you have built the application’s logic, you can use LangChain’s components to add specific features, such as data extraction or language translation.\n\n\nSet and manipulate context: Finally, you can set and manipulate context to guide your application’s behavior and improve its performance. With LangChain, you have the power to create highly customized and feature-rich applications using LLMs with ease.\n\n\nThe above should give you a basic understanding of how to develop applications using LangChain. In the next section, we’ll explore the different applications that find extensive use cases for LangChain.\n3 Application Examples of LangChain\nLangChain allows you to build advanced applications using a large language model (LLM). With its flexibility, customization options, and powerful components, LangChain can be used to create a wide variety of applications across different industries.\n\nThe following are some of the examples where LangChain is extensively used:\n1. Text Summarization (Data Augmented Generation)\nWith LangChain, you can develop applications that handle text summarization tasks efficiently.\nBy leveraging powerful language models like ChatGPT, your application will be able to generate accurate and concise summaries of large texts, allowing your users to quickly grasp the main points of complex documents.\n2. Question Answering\nAnother use case for LangChain is building applications that provide question answering capabilities.\nBy integrating with a large language model, your application can receive user-inputted text data and extract relevant answers from a variety of sources, such as Wikipedia, Notion, or Apify Actors. This functionality can be beneficial for users seeking quick and reliable information on different topics.\n3. Chatbots (Language Model)\nLangChain is a valuable tool for creating chatbots powered by language models. By taking advantage of LangChain’s framework and components, your chatbot applications can provide a more natural and engaging user experience.\n\nUsers can interact with your chatbots for general conversation, support inquiries, or other specific purposes, and the language model will generate context-aware responses.\nThese application examples are just a few ways in which you can utilize LangChain to build powerful and versatile applications. By understanding the strengths of language models, you can create innovative solutions that cater to the needs of your users.\nWhat are Integrations in LangChain?\nLangChain provides end-to-end chains integration to make working with various programming languages, platforms, and data sources easier for you.\nThis ensures that you can seamlessly build applications utilizing a language model in the environment of your choice.\n\nIn terms of programming languages, LangChain provides support for both JavaScript and TypeScript, you can leverage the power of LangChain in web-based and Node.js applications and take advantage of the robust type-safety features TypeScript provides.\nHere’s a short list of key integrations LangChain has to offer:\n\n\nLarge Language Models (LLMs): OpenAI, Hugging Face, Anthropic, and more.\n\n\nCloud Platforms: Azure, Amazon, Google Cloud, and other popular cloud providers\n\n\nData Sources: Google Drive, Notion, Wikipedia, Apify Actors, and more.\n\n\nBy utilizing these integrations, you can create more advanced and versatile applications centered around a language model.\nThis will help you develop and deploy your projects quickly and efficiently, leveraging the right tools and resources for your needs.\nAdvanced Features and Customization of LangChain\nLangChain offers advanced features and customization options for creating powerful applications using LLMs.\nYou can tailor your application’s behavior and build sophisticated use cases such as Generative Question-Answering (GQA) or chatbots.\n\nThe following are some of the key features of LangChain:\n\n\nCustomizable prompts to suit your needs\n\n\nBuilding chain link components for advanced use cases\n\n\nCode customization for developing unique applications\n\n\nModel integration for data augmented generation and accessing high-quality language model application like text-davinci-003\n\n\nFlexible components to mix and match components for specific requirements\n\n\nContext manipulation to set and guide context for improved accuracy and user experience\n\n\nWith LangChain, you can create feature-rich applications that stand out from the crowd, thanks to its advanced customization options.\nTo help you take full advantage of LangChain’s features, let’s take a look at some valuable resources you could use in the next section!\nResources and Support for LangChain\nLangChain comes with various resources and support to help you develop powerful language model-powered applications.\n\nFollowing are some of the key resources that you can use when working with LangChain:\n\n\nAI Libraries such as OpenAI and Hugging Face for AI models\n\n\nExternal sources such as Notion, Wikipedia, and Google Drive for targeted data\n\n\nLangChain documentation for guides on connecting and chaining components\n\n\nOfficial Documentation\n\n\nGitHub Repository\n\n\nPyPI Package Repository\n\n\n\n\nData augmentation to improve context-aware results through external data sources, indexing, and vector representations\n\n\nLastly, engaging with the LangChain community and dedicated support slack channel can be beneficial if you encounter challenges or want to learn from others’ experiences. From forums to online groups, connecting with fellow developers will enrich your journey with LangChain.\nFinal Thoughts\nLangChain offers a comprehensive approach to developing applications powered by generative models and LLMs. By integrating core concepts from data science, developers can leverage multiple components, prompt templates, and vector databases to create innovative solutions beyond traditional metrics.\nAs technology evolves, agents involve more sophisticated elements, including chat interfaces, offering more comprehensive support in various use cases.\nWhether you’re developing chatbots, sentiment analysis tools, or any other NLP application, LangChain can help you unlock the full potential of your data. As NLP technology continues to evolve and grow in importance, platforms like LangChain will only become more valuable.\nSo, if you’re looking to stay ahead of the curve in the world of NLP, be sure to check out LangChain and see what it can do for you!\n\nRelated Posts\n\nMastering Cursor AI: From Basics to Advanced Applications\nAI\nExplore the world of Cursor AI and learn how to leverage its capabilities for automation and enhanced user experience.\n\nMastering Large Language Models\nAI\nA step-by-step guide to understanding and working with Large Language Models (LLMs).\n\n5 Machine Learning Algorithms You Should Know\nAI\nMachine learning algorithms are like the superheroes of the tech world. They swoop in, armed with data...\n\nHow can I get started with neural networks and deep learning?\nAI\nYou may have heard about the power of neural networks and deep learning and wondered how you could get...\n\nWhat is Midjourney and How to Use It\nAI\nRemember the not-so-long-ago days when we had to rely solely on artists to create stunning visuals?...\n\nText to Code Generator: Create Code in any Language\nAI\nHave you ever wondered if there was an easier way to write code? Well, there is; it\'s quick, it\'s easy,...\n\nWhat’s The Difference Between Neural Networks and Decision Trees\nAI\nIn the world of machine learning, two popular techniques stand out: Neural Networks and Decision Trees....\n\nWhat is Caktus AI? A Detailed Overview\nAI\nDo you need to write an essay on the fall of the roman empire with accurate citations but have no time...\n\nHow To Use AI To Supercharge Your Productivity\nAI\nThe rise of Artificial Intelligence (AI) has been a game-changer in many fields, but one of its most...\n\nMachine Learning Engineer Vs Software Engineer\nAI\nIf you\'re planning to pursue a career in the tech industry, it\'s crucial to comprehend the different...\n\nWhat Does A Neural Network Even Mean\nAI\nEver since its inception in the 1940s, the concept of artificial neural networks has captured the...\n\nTop 8 Web3 Ideas 2023: Welcome to The Future\nAI, ChatGPT\nWe\'ve all heard of cryptocurrency, NFTs, and the metaverse. And we\'ve all heard about the...\n« Older Entries\n\nAbout Contact Us Advertise With Us Terms And ConditionsPrivacy Policy\nAll Rights Reserved© 2023\n\nFacebook\nYouTube\nTwitter\nLinkedIn\n'}, {'title': 'Introduction | ️ LangChain', 'url': 'https://python.langchain.com/docs/introduction/', 'content': "Introduction. LangChain is a framework for developing applications powered by large language models (LLMs).. LangChain simplifies every stage of the LLM application lifecycle: Development: Build your applications using LangChain's open-source components and third-party integrations.Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.", 'score': 0.8585411, 'raw_content': "Introduction\nLangChain is a framework for developing applications powered by large language models (LLMs).\nLangChain simplifies every stage of the LLM application lifecycle:\nLangChain implements a standard interface for large language models and related\ntechnologies, such as embedding models and vector stores, and integrates with\nhundreds of providers. See the integrations page for\nmore.\nThese docs focus on the Python LangChain library. Head here for docs on the JavaScript LangChain library.\nArchitecture\u200b\nThe LangChain framework consists of multiple open-source libraries. Read more in the\nArchitecture page.\nGuides\u200b\nTutorials\u200b\nIf you're looking to build something specific or are more of a hands-on learner, check out our tutorials section.\nThis is the best place to get started.\nThese are the best ones to get started with:\nExplore the full list of LangChain tutorials here, and check out other LangGraph tutorials here. To learn more about LangGraph, check out our first LangChain Academy course, Introduction to LangGraph, available here.\nHow-to guides\u200b\nHere you’ll find short answers to “How do I….?” types of questions.\nThese how-to guides don’t cover topics in depth – you’ll find that material in the Tutorials and the API Reference.\nHowever, these guides will help you quickly accomplish common tasks using chat models,\nvector stores, and other common LangChain components.\nCheck out LangGraph-specific how-tos here.\nConceptual guide\u200b\nIntroductions to all the key parts of LangChain you’ll need to know! Here you'll find high level explanations of all LangChain concepts.\nFor a deeper dive into LangGraph concepts, check out this page.\nIntegrations\u200b\nLangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it.\nIf you're looking to get up and running quickly with chat models, vector stores,\nor other LangChain components from a specific provider, check out our growing list of integrations.\nAPI reference\u200b\nHead to the reference section for full documentation of all classes and methods in the LangChain Python packages.\nEcosystem\u200b\n🦜🛠️ LangSmith\u200b\nTrace and evaluate your language model applications and intelligent agents to help you move from prototype to production.\n🦜🕸️ LangGraph\u200b\nBuild stateful, multi-actor applications with LLMs. Integrates smoothly with LangChain, but can be used without it.\nAdditional resources\u200b\nVersions\u200b\nSee what changed in v0.3, learn how to migrate legacy code, read up on our versioning policies, and more.\nSecurity\u200b\nRead up on security best practices to make sure you're developing safely with LangChain.\nContributing\u200b\nCheck out the developer's guide for guidelines on contributing and help getting your dev environment set up.\n"}, {'title': 'LangChain: Introduction and Getting Started - Pinecone', 'url': 'https://www.pinecone.io/learn/series/langchain/langchain-intro/', 'content': 'LangChain. At its core, LangChain is a framework built around LLMs. We can use it for chatbots, Generative Question-Answering (GQA), summarization, and much more. The core idea of the library is that we can "chain" together different components to create more advanced use cases around LLMs. Chains may consist of multiple components from several modules:', 'score': 0.82391036, 'raw_content': 'LangChain: Introduction and Getting Started\nLarge Language Models (LLMs) entered the world stage with the release of OpenAI’s GPT-3 in 2020 [1]. Since then, they’ve enjoyed a steady growth in popularity.\nThat is until late 2022. Interest in LLMs and the broader discipline of generative AI has skyrocketed. The reasons for this are likely the continuous upward momentum of significant advances in LLMs.\nWe saw the dramatic news about Google’s “sentient” LaMDA chatbot. The first high-performance and open-source LLM called BLOOM was released. OpenAI released their next-generation text embedding model and the next generation of “GPT-3.5” models.\nAfter all these giant leaps forward in the LLM space, OpenAI released ChatGPT — thrusting LLMs into the spotlight.\nLangChain appeared around the same time. Its creator, Harrison Chase, made the first commit in late October 2022. Leaving a short couple of months of development before getting caught in the LLM wave.\nDespite being early days for the library, it is already packed full of incredible features for building amazing tools around the core of LLMs. In this article, we’ll introduce the library and start with the most straightforward component offered by LangChain — LLMs.\nLangChain\nAt its core, LangChain is a framework built around LLMs. We can use it for chatbots, Generative\xa0Question-Answering (GQA), summarization, and much more.\nThe core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. Chains may consist of multiple components from several modules:\nWe will dive into each of these in much more detail in upcoming chapters of the LangChain handbook.\nFor now, we’ll start with the basics behind prompt templates and LLMs. We’ll also explore two LLM options available from the library, using models from Hugging Face Hub or OpenAI.\n\nOur First Prompt Templates\nPrompts being input to LLMs are often structured in different ways so that we can get different results. For Q&A, we could take a user’s question and reformat it for different Q&A styles, like conventional Q&A, a bullet list of answers, or even a summary of problems relevant to the given question.\nCreating Prompts in LangChain\nLet’s put together a simple question-answering prompt template. We first need to install the langchain library.\n!pip install langchain\nFollow along with the code via\xa0the walkthrough!\nFrom here, we import the PromptTemplate class and initialize a template like so:\nWhen using these prompt template with the given question we will get:\nQuestion: Which NFL team won the Super Bowl in the 2010 season? Answer:\nFor now, that’s all we need. We’ll use the same prompt template across both Hugging Face Hub and OpenAI LLM generations.\nHugging Face Hub LLM\nThe Hugging Face Hub endpoint in LangChain connects to the Hugging Face Hub and runs the models via their free inference endpoints. We need a Hugging Face account and API key to use these endpoints.\nOnce you have an API key, we add it to the HUGGINGFACEHUB_API_TOKEN environment variable. We can do this with Python like so:\nNext, we must install the huggingface_hub library via Pip.\n!pip install huggingface_hub\nNow we can generate text using a Hub model. We’ll use google/flan-t5-x1.\nThe default Hugging Face Hub inference APIs do not use specialized hardware and, therefore, can be slow. They are also not suitable for running larger models like\xa0bigscience/bloom-560m\xa0or\xa0google/flan-t5-xxl\xa0(note\xa0xxl\xa0vs.\xa0xl).\nFor this question, we get the correct answer of "green bay packers".\nAsking Multiple Questions\nIf we’d like to ask multiple questions, we can try two approaches:\nStarting with option (1), let’s see how to use the generate method:\nHere we get bad results except for the first question. This is simply a limitation of the LLM being used.\nIf the model cannot answer individual questions accurately, grouping all queries into a single prompt is unlikely to work. However, for the sake of experimentation, let’s try it.\nAs expected, the results are not helpful. We’ll see later that more powerful LLMs can do this.\nOpenAI LLMs\nThe OpenAI endpoints in LangChain connect to OpenAI directly or via Azure. We need an OpenAI account and API key to use these endpoints.\nOnce you have an API key, we add it to the OPENAI_API_TOKEN environment variable. We can do this with Python like so:\nNext, we must install the openai library via Pip.\n!pip install openai\nNow we can generate text using OpenAI’s GPT-3 generation (or completion) models. We’ll use text-davinci-003.\nAlternatively, if you’re using OpenAI via Azure, you can do:\nWe’ll use the same simple question-answer prompt template as before with the Hugging Face example. The only change is that we now pass our OpenAI LLM davinci:\nAs expected, we’re getting the correct answer. We can do the same for multiple questions using generate:\nMost of our results are correct or have a degree of truth. The model undoubtedly functions better than the google/flan-t5-xl model. As before, let’s try feeding all questions into the model at once.\nAs we keep rerunning the query, the model will occasionally make errors, but at other times manage to get all answers correct.\nThat’s it for our introduction to LangChain — a library that allows us to build more advanced apps around LLMs like OpenAI’s GPT-3 models or the open-source alternatives available via Hugging Face.\nAs mentioned, LangChain can do much more than we’ve demonstrated here. We’ll be covering these other features in upcoming articles.\nReferences\n[1] GPT-3 Archived Repo (2020), OpenAI GitHub\nChapters\n© Pinecone Systems, Inc. | San Francisco, CA\nPinecone is a registered trademark of Pinecone Systems, Inc.\n'}, {'title': 'Introduction to LangChain - GeeksforGeeks', 'url': 'https://www.geeksforgeeks.org/introduction-to-langchain/', 'content': 'Applications of LangChain. LangChain is a powerful tool that can be used to build a wide range of LLM-powered applications. It is simple to use and has a large user and contributor community. Document analysis and summarization; Chatbots: LangChain can be used to build chatbots that interact with users naturally.', 'score': 0.79883134, 'raw_content': 'Introduction to LangChain\nLangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications. It allows AI developers to develop applications based on the combined Large Language Models (LLMs) such as GPT-4 with external sources of computation and data. This framework comes with a package for both Python and JavaScript.\nLangChain follows a general pipeline where a user asks a question to the language model where the vector representation of the question is used to do a similarity search in the vector database and the relevant information is fetched from the vector database and the response is later fed to the language model. further, the language model generates an answer or takes an action.\nApplications of LangChain\nLangChain is a powerful tool that can be used to build a wide range of LLM-powered applications. It is simple to use and has a large user and contributor community.\nLangChain Key Concepts:\nThe main properties of LangChain Framework are :\nSetting up the environment\nInstallation of langchain is very simple and similar as you install other libraries using the pip command.\nThere are various LLMs that you can use with LangChain. In this article, I will be using OpenAI. Let us install Openai using the following command:\nI am also installing the dotenv library to store the API key in an environmental variable. Install it using the command:\nYou can generate your own API key by signing up to the openai platform. Next, we create a .env file and store our API key in it as follows:\nNow, I am creating a new file named ‘lang.py’ where I will be using the LangChain framework to generate responses. Let us start by importing the required libraries as follows:\nThat was the initial setup required to use the LangChain framework with OpenAI LLM. \nBuilding an Application\nAs this is an introductory article, let us start by generating a simple answer for a simple question such as “Suggest me a skill that is in demand?”. \nWe start by importing lang-chain and initializing an LLM as follows:\nWe are initializing it with a high temperature which means that the results will be random and less accurate. For it to be more accurate you can give a temperature as 0.4 or lesser. We are then assigning openai_api_key as api_key which we have loaded previously from .env file.\nThe next step would be to predict by passing in the text as follows:\nThat is it! The response generated is as follows:\nOne skill in demand right now is software/web development, which includes everything from coding to content management systems to web design. Other skills in demand include cloud computing, machine learning and artificial intelligence, digital marketing, cybersecurity, data analysis, and project management.\nConclusion:\nThat was the basic introduction to langchain framework. I hope you have understood the usage and there are a lot more concepts such as prompt templates, chains and agents to learn. The LangChain framework is a great interface to develop interesting AI-powered applications and from personal assistants to prompt management as well as automating tasks. So, Keep learning and keep developing powerful applications.\nGet IBM Certification and a 90% fee refund on completing 90% course in 90 days! Take the Three 90 Challenge today.\nMaster Machine Learning, Data Science & AI with this complete program and also get a 90% refund. What more motivation do you need? Start the challenge right away!\nN\nSimilar Reads\n\nWhat kind of Experience do you want to share?\n'}, {'title': 'What is LangChain and how to use it: A guide - TechTarget', 'url': 'https://www.techtarget.com/searchEnterpriseAI/definition/LangChain', 'content': "Everything you need to know\nWhat are the features of LangChain?\nLangChain is made up of the following modules that ensure the multiple components needed to make an effective NLP app can run smoothly:\nWhat are the integrations of LangChain?\nLangChain typically builds applications using integrations with LLM providers and external sources where data can be found and stored. What is synthetic data?\nExamples and use cases for LangChain\nThe LLM-based applications LangChain is capable of building can be applied to multiple advanced use cases within various industries and vertical markets, such as the following:\nReaping the benefits of NLP is a key of why LangChain is important. As the airline giant moves more of its data workloads to the cloud, tools from Intel's Granulate are making platforms such as ...\nThe vendor's new platform, now in beta testing, combines its existing lakehouse with AI to better enable users to manage and ...\n The following steps are required to use this:\nIn this scenario, the language model would be expected to take the two input variables -- the adjective and the content -- and produce a fascinating fact about zebras as its output.\n The goal of LangChain is to link powerful LLMs, such as OpenAI's GPT-3.5 and GPT-4, to an array of external data sources to create and reap the benefits of natural language processing (NLP) applications.\n", 'score': 0.75355774, 'raw_content': "The potential of AI technology has been percolating in the background for years. But when ChatGPT, the AI chatbot, began grabbing headlines in early 2023, it put generative AI in the spotlight.\nThis guide is your go-to manual for generative AI, covering its benefits, limits, use cases, prospects and much more.\nYou forgot to provide an Email Address.\nThis email address doesn’t appear to be valid.\nThis email address is already registered. Please log in.\nYou have exceeded the maximum character limit.\nPlease provide a Corporate Email Address.\nPlease check the box if you want to proceed.\nPlease check the box if you want to proceed.\nBy submitting my Email address I confirm that I have read and accepted the Terms of Use and Declaration of Consent.\nLangChain\nWhat is LangChain?\nLangChain is an open source framework that lets software developers working with artificial intelligence (AI) and its machine learning subset combine large language models with other external components to develop LLM-powered applications. The goal of LangChain is to link powerful LLMs, such as OpenAI's GPT-3.5 and GPT-4, to an array of external data sources to create and reap the benefits of natural language processing (NLP) applications.\nDevelopers, software engineers and data scientists with experience in the Python, JavaScript or TypeScript programming languages can make use of LangChain's packages offered in those languages. LangChain was launched as an open source project by co-founders Harrison Chase and Ankush Gola in 2022; the initial version was released that same year.\nWhy is LangChain important?\nLangChain is a framework that simplifies the process of creating generative AI application interfaces. Developers working on these types of interfaces use various tools to create advanced NLP apps; LangChain streamlines this process. For example, LLMs have to access large volumes of big data, so LangChain organizes these large quantities of data so that they can be accessed with ease.\nIn addition, GPT (Generative Pre-trained Transformer) models are generally trained on data up to their release to the public. For instance, ChatGPT was released to the public near the end of 2022, but its knowledge base was limited to data from 2021 and before. LangChain can connect AI models to data sources to give them knowledge of recent data without limitations.\nThis article is part of\nWhat is generative AI? Everything you need to know\nWhat are the features of LangChain?\nLangChain is made up of the following modules that ensure the multiple components needed to make an effective NLP app can run smoothly:\nWhat are the integrations of LangChain?\nLangChain typically builds applications using integrations with LLM providers and external sources where data can be found and stored. For example, LangChain can build chatbots or question-answering systems by integrating an LLM -- such as those from Hugging Face, Cohere and OpenAI -- with data sources or stores such as Apify Actors, Google Search and Wikipedia. This enables an app to take user-input text, process it and retrieve the best answers from any of these sources. In this sense, LangChain integrations make use of the most up-to-date NLP technology to build effective apps.\nOther potential integrations include cloud storage platforms, such as Amazon Web Services, Google Cloud and Microsoft Azure, as well as vector databases. A vector database can store large volumes of high-dimensional data -- such as videos, images and long-form text -- as mathematical representations that make it easier for an application to query and search for those data elements. Pinecone is an example vector database that can be integrated with LangChain.\nHow to create prompts in LangChain\nPrompts serve as input to the LLM that instructs it to return a response, which is often an answer to a query. This response is also referred to as an output. A prompt must be designed and executed correctly to increase the likelihood of a well-written and accurate response from a language model. That is why prompt engineering is an emerging science that has received more attention in recent years.\nPrompts can be generated easily in LangChain implementations using a prompt template, which will be used as instructions for the underlying LLM. Prompt templates can vary in specificity. They can be designed to pose simple questions to a language model. They can also be used to provide a set of explicit instructions to a language model with enough detail and examples to retrieve a high-quality response.\nWith Python programming, LangChain has a premade prompt template that takes the form of structured text. The following steps are required to use this:\nIn this scenario, the language model would be expected to take the two input variables -- the adjective and the content -- and produce a fascinating fact about zebras as its output.\nHow to develop applications in LangChain\nLangChain is built to develop apps powered by language model functionality. There are different ways to do this, but the process typically entails some key steps:\nFor more information on generative AI-related terms, read the following articles:\nWhat is an AI prompt engineer?\nWhat is prompt engineering?\nWhat is synthetic data?\nExamples and use cases for LangChain\nThe LLM-based applications LangChain is capable of building can be applied to multiple advanced use cases within various industries and vertical markets, such as the following:\nReaping the benefits of NLP is a key of why LangChain is important. For a more in-depth understanding of NLP, there are two important subtopics to start with: natural language understanding (NLU) and natural language generation (NLG). The goal of NLU is to process a user's intended meaning, while the goal of NLG is to explain an AI system's structured data in human-readable languages for humans to comprehend.\nGenerative AI isn't going to replace data analysts. It can help analysts be more effective, but it lacks human insights and ...\nThe suite unites Power BI, Azure Synapse Analytics and Data Factory in an integrated environment to better enable data management...\nThe data cloud vendor introduced more than a dozen features currently in various stages of development designed to help users ...\nShould ousted OpenAI CEO Sam Altman be involved in any federal investigations going forward, he now has the legal backing of ...\nBPO and BPM both aim to improve business processes. One is a management approach to optimizing end-to-end processes. The other is...\nGOP presidential candidate Asa Hutchinson is using an AI chatbot to provide insight on his policy stances, a development that ...\nAs the airline giant moves more of its data workloads to the cloud, tools from Intel's Granulate are making platforms such as ...\nThe vendor's new platform, now in beta testing, combines its existing lakehouse with AI to better enable users to manage and ...\nThe data management and BI specialist has released Data Intelligence, an AI-driven suite that includes data quality and data ...\nSupply chain professionals want more technologies such as AI for visibility and traceability in 2024; however, sustainability and...\nWhile WMS streamlines warehouse processes, OMS can help improve order management and customer data tracking. Discover more about ...\nAimed at improving frontline worker performance, Copilot GenAI capabilities will be integrated into Microsoft Dynamics 365 Field ...\nAll Rights Reserved,\nCopyright 2018 - 2023, TechTarget\nPrivacy Policy\nCookie Preferences\nCookie Preferences\nDo Not Sell or Share My Personal Information"}, {'title': 'LangChain', 'url': 'https://www.langchain.com/langchain', 'content': 'Augment the power\nof\xa0LLMs with your data\nLangChain helps connect LLMs to your company’s private sources\nof data and APIs to create context-aware, reasoning applications.\n Our Methods\nReady to start shipping\nreliable GenAI apps faster?\nLangChain and LangSmith are critical parts of the reference\narchitecture to get you from prototype to production. The largest community building the future of LLM apps\nLangChain’s flexible abstractions and AI-first toolkit make it\xa0the\xa0#1\xa0choice for developers when building with GenAI.\n Why choose LangChain?\nLangChain is easy to get started with and\xa0gives\xa0you choice, flexibility, and power as\xa0you scale.\n Get customizability and control with a durable runtime baked in\nLangChain Expression Language (LCEL) lets you build your app in a truly composable way, allowing you to customize it as you see fit.', 'score': 0.7471924, 'raw_content': "The largest community building the future of LLM apps\nLangChain’s flexible abstractions and AI-first toolkit make it\xa0the\xa0#1\xa0choice for developers when building with GenAI.\nLangChain keeps pace with\nthe cutting edge, so you can too. Join 100k+ builders who standardize development in LangChain's Python and TypeScript frameworks.\nAugment the power\nof\xa0LLMs with your data\nLangChain helps connect LLMs to your company’s private sources\nof data and APIs to create context-aware, reasoning applications.\nA complete set of interoperable and interchangeable building blocks\nLeverage our comprehensive library of components that together make up sophisticated, end-to-end applications. Want to change your model? Future-proof your application by making vendor optionality part of your LLM infrastructure design.\nGet customizability and control with a durable runtime baked in\nLangChain Expression Language (LCEL) lets you build your app in a truly composable way, allowing you to customize it as you see fit. The protocol supports parallelization, fallbacks, batch, streaming, and async all out-of-the-box, freeing you to focus on what matters.\nSmart connections to any source of data or knowledge\nNeed turnkey observability?\nLangSmith shines a light into application behavior and performance. Get prompt-level visibility coupled with tools to debug, test, evaluate, deploy, and monitor your applications with your team.\nWhy choose LangChain?\nLangChain is easy to get started with and\xa0gives\xa0you choice, flexibility, and power as\xa0you scale.\nOne framework.\nInfinite use cases.\nOur Methods\nReady to start shipping\nreliable GenAI apps faster?\nLangChain and LangSmith are critical parts of the reference\narchitecture to get you from prototype to production."}, {'title': 'Everything you need to know to get started with LangChain', 'url': 'https://www.gettingstarted.ai/everything-you-need-to-know-when-getting-started-with-langchain/', 'content': 'LangChain Chains. A chain in LangChain consists of multiple individual components executed in a specific order, allowing you to combine different language model calls and actions automatically. LangChain Prompts. A prompt in LangChain is a specific input that is provided to the large language model which is used to generate a response.', 'score': 0.7445268, 'raw_content': 'Just getting started with LangChain? Here\'s everything you need to know\nAre you just getting started with LangChain? You\'ve come to the right place! This post covers everything you need to know to get started quickly.\nOh LangChain, some love it and some hate it. Whether you\'re considering adding AI capabilities to your app or working on a new project you must\'ve run into LangChain, the popular LLM framework for Python and Node.js.\nWhat is LangChain?\nLangChain is a framework that simplifies integrating LLM capabilities into your application. It supports GPT-4, Gemini, and many other LLMs straight out of the box. LangChain uses chains that are linked together to perform a series of tasks.\nLangChain comes with built-in support for various data loaders that can retrieve, organize, and create embeddings for use with LLMs. Data loaders are tools that enable you to extract data into chunks from various sources like PDF files and more.\nDo you need LangChain?\nSo, do you need to use LangChain? Ask yourself the following:\nIf you answer yes to one or more of the questions above, you\'d want to read on because LangChain could be a suitable option for you. \nLet\'s find out!\nHey, there!\nJoin a community of people learning about Artificial Intelligence for zero dollars because why not?\nNo spam. Unsubscribe anytime.\nLangChain alternatives\nOf course, LangChain is not the only data framework out there. There are other popular options. Below I\'ll highlight why you may want to go with LangChan over its alternatives.\nWhy use LangChain instead of Haystack?\nLangChain has more capabilities and supports a broader range of use cases in natural language processing tasks compared to Haystack. It also has greater community support due to its wider developer adoption.\xa0\nWhy use LangChain instead of LlamaIndex?\nLangChain is a broader data framework than LlamaIndex and lets you do more by default. It can also integrate with LlamaIndex for search and retrieval capabilities.\nWhy use LangChain instead of Semantic Kernel?\nSemantic Kernel is a great option if you\'re a C# developer or using the .NET framework. However, LangChain comes with more features out of the box, gets more updates, and you\'ll find many more resources online if you run into problems due to its wider adoption.\nHow does LangChain work?\nA quick overview of LLMs\nLarge language models (LLMs) perform language tasks. LLMs can’t generate images or a song but they can complete a sentence or any other request that deals with natural language.\nThey are stateless, meaning they know nothing about their environment, they have no memory of past conversations and they are trained to generate the next word based on all previous ones. - That\'s it. \nBut if LLMs know nothing about the exterior environment how can AI apps like ChatGPT answer questions about your documents? That\'s where custom knowledge comes in!\nCustom knowledge via RAG\nRetrieval-augmented generation (or RAG for short) enhances large language models by feeding them extra information they weren\'t originally trained on. This is especially useful if you\'re creating an app to interact with specific content, like a PDF or a website.\nLangChain helps you augment LLM knowledge and build RAG apps.\nVector Embeddings and Databases\nIf you have a large document and need specific information from it, you wouldn\'t paste the whole document and ask the LLM about it. Instead, you only use the parts that are relevant to your question. \nThis is where vector embeddings and databases come in. Essentially, you transform a document into vectors, then, using fancy mathematical formulas, these vectors are compared with your query for similarity, and only the relevant parts are sent to the LLM.\nFor example, if we converted your CV into Vector Embeddings and we ran a similarity search using the vector database for the prompt "Where did the candidate study?", just the education section of your CV would be returned. \nPrompt engineering\nThe output of the LLM is as good as the prompt itself. It is very important that you write descriptive and clear prompts that include essential information for the LLM to generate a useful and accurate response.\nThe choice of words, information, and structure directly impacts the answer generated by the model.\nHow LangChain talks to LLMs\nNow, LangChain comes in and orchestrates the communication between all of these components. Using chains we can index and retrieve data from our vector database. Using another chain, we then send the relevant information  from the data to a LLM. You can link many chains together!\nLangChain building blocks\nA chain in LangChain consists of multiple individual components executed in a specific order, allowing you to combine different language model calls and actions automatically.\nA prompt in LangChain is a specific input that is provided to the large language model which is used to generate a response.\nA document loader in LangChain is an add-on that lets you load documents from different sources, such as PDFs and Word.\nAn agent in LangChain is a type of chain that is capable of choosing which action and tools to use to complete a user input. They are similar to Planners in Semantic Kernel.\nExample code using LangChain\nThe best way to learn and understand how all of this works is by writing code. I\'ve compiled the list below from my previous tutorials so you can get started in no time!\nConclusion\nLots of developers have already built apps using LangChain and many more will. Its wide adoption means great community support and frequent updates.\nBefore taking the plunge, consider if you need all the features of LangChain or if you\'re better off building what you need yourself without dependencies. This may require more effort on your part, however, the benefits are increased stability and lower maintenance.\nLangChain is a great framework that simplifies the integration of Large Language Models into your application if you\'re comfortable using Python or Node.js. You can get started quickly thanks to its ability to support a wide range of data loaders, custom knowledge, and more!\nThanks for reading. Please let me know in the comments below if you found the content useful or if you have any questions!\nFurther readings\nMore from the Web\nMore from Getting Started with AI\nRead more\nWhat\'s going on with AutoGen and AG2?\nAutoGen has split into two paths: AG2 and Microsoft\'s version. Learn what this means for your AI projects and which path might be right for you.\nLet\'s Play Pictionary with the OpenAI Vision API\nThe best way to learn is by doing. In this tutorial, you\'ll see how EASY it is to set up image recognition using the OpenAI Python SDK in your app.\nChatGPT Search vs Perplexity: Goodbye Google?\nHere\'s a comprehensive comparison of the newly released ChatGPT Search and Perplexity AI. Through practical demonstrations and direct comparisons, we\'ll see which AI search engine delivers better results and whether the premium features justify the cost.\nGetting Started with OpenAI .NET SDK: Add ChatGPT to Your C# App\nLearn how to integrate ChatGPT into your C# applications using the official OpenAI .NET SDK. This step-by-step guide shows you how to set up the SDK, create a chat client, and build an interactive AI chat application. Includes code examples and best practices.\n\n                        Join other humans, learning about machines.\n                    \n\n                        Gain knowledge, pay nothing.\n                    \n'}, {'title': 'What Is LangChain? - IBM', 'url': 'https://www.ibm.com/think/topics/langchain', 'content': 'LangChain is not limited to out-of-the-box foundation models: the CustomLLM class\xa0(link resides outside ibm.com) allows for custom LLM wrappers. Likewise, you can use the IBM watsonx APIs and Python SDK, which includes a LangChain integration, to build applications in LangChain with models that you’ve already trained or fine-tuned for your specific needs using the WatsonxLLM class (and that model’s specific project ID). Chatbots: Chatbots are among the most intuitive uses of LLMs. LangChain can be used to provide proper context for the specific use of a chatbot, and to integrate chatbots into existing communication channels and workflows with their own APIs. Summarization: Language models can be tasked with summarizing many types of text, from breaking down complex academic articles and transcripts to providing a digest of incoming emails.', 'score': 0.7325349, 'raw_content': "What Is LangChain? | IBM\nWhat is LangChain?\nArtificial Intelligence\n31 October 2023\nLink copied\nWhat is LangChain?\nLangChain is an open source orchestration framework for the development of applications using large language models (LLMs). Available in both Python- and Javascript-based libraries, LangChain’s tools and APIs simplify the process of building LLM-driven applications like chatbots and virtual agents.\nLangChain serves as a generic interface for nearly any LLM, providing a centralized development environment to build LLM applications and integrate them with external data sources and software workflows. LangChain’s module-based approach allows developers and data scientists to dynamically compare different prompts and even different foundation models with minimal need to rewrite code. This modular environment also allows for programs that use multiple LLMs: for example, an application that uses one LLM to interpret user queries and another LLM to author a response.\nLaunched by Harrison Chase in October 2022, LangChain enjoyed a meteoric rise to prominence: as of June 2023, it was the single fastest-growing open source project on Github.1 Coinciding with the momentous launch of OpenAI’s ChatGPT the following month, LangChain has played a significant role in making generative AI more accessible to enthusiasts in the wake of its widespread popularity.\nLangChain can facilitate most use cases for LLMs and natural language processing (NLP), like chatbots, intelligent search, question-answering, summarization services or even virtual agents capable of robotic process automation.\nIntegrations with LLMs\nLLMs are not standalone applications: they are pre-trained statistical models that must be paired with an application (and, in some cases, specific data sources) in order to meet their purpose.\nFor example, Chat-GPT is not an LLM: it is a chatbot application that, depending on the version you’ve chosen, uses the GPT-3.5 or GPT-4 language model. While it’s the GPT model that interprets the user’s input and composes a natural language response, it’s the application that (among other things) provides an interface for the user to type and read and a UX design that governs the chatbot experience. Even at the enterprise level, Chat-GPT is not the only application using the GPT model: Microsoft uses GPT-4 to power Bing Chat.\nFurthermore, though foundation models (like those powering LLMs) are pre-trained on massive datasets, they are not omniscient. If a particular task requires access to specific contextual information, like internal documentation or domain expertise, LLMs must be connected to those external data sources. Even if you simply want your model to reflect real-time awareness of current events, it requires external information: a model’s internal data is only up-to-date through the time period during which it was pre-trained.\nLikewise, if a given generative AI task requires access to external software workflows—for example, if you wanted your virtual agent to integrate with Slack—then you will need a way to integrate the LLM with the API for that software.\nWhile these integrations can generally be achieved with fully manual code, orchestration frameworks like LangChain and the IBM watsonx platform greatly simplify the process. They also make it much easier to experiment with different LLMs to compare results, as different models can be swapped in and out with minimal changes to code.\n      ![Image 1: 3D design of balls rolling on a track](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/pm/ul/g/5a/6e/trailv2_1200x1200.component.think-ad-xl.ts=1730807361767.jpeg/content/experience-fragments/adobe-cms/us/en/site-v2/think-hub/article/_8_column_general_ad/blueprint---think-ad---xf---do-not-modify/artificialintelligence-article-newsletter/_jcr_content/root/think_ad_copy/image)\n\nThe latest AI News + Insights \u2028 Expertly curated insights and news on AI, cloud and more in the weekly Think Newsletter.\xa0\nSubscribe today\nHow does LangChain work?\nAt LangChain’s core is a development environment that streamlines the programming of LLM applications through the use of\xa0abstraction: the simplification of code by representing one or more complex processes as a named component that encapsulates all of its constituent steps.\nAbstractions are a common element of everyday life and language. For example, “π” allows us to represent the ratio of the length of a circle’s circumference to that of its diameter without having to write out its infinite digits. Similarly, a thermostat allows us to control the temperature in our home without needing to understand the complex circuitry this entails—we only need to know how different thermostat settings translate to different temperatures.\nLangChain is essentially a library of abstractions for Python and Javascript, representing common steps and concepts necessary to work with language models. These modular components—like functions and object classes—serve as the building blocks of generative AI programs. They can be “chained” together to create applications, minimizing the amount of code and fine understanding required to execute complex NLP tasks. Though LangChain’s abstracted approach may limit the extent to which an expert programmer can finely customize an application, it empowers specialists and newcomers alike to quickly experiment and prototype.\nImporting language models\nNearly any LLM can be used in LangChain. Importing language models into LangChain is easy, provided you have an API key. The LLM class is designed to provide a standard interface for all models.\nMost LLM providers will require you to create an account in order to receive an API key. Some of these APIs—particularly those for proprietary closed-source models, like those offered by OpenAI or Anthropic—may have associated costs.\nMany open source models, like BigScience’s BLOOM, Meta AI’s LLaMa and Google’s Flan-T5, can be accessed through Hugging Face (link resides outside ibm.com). IBM watsonx, through its partnership with Hugging Face, also offers a curated suite of open source models. Creating an account with either service will allow you to generate an API key for any of the models offered by that provider.\nLangChain is not limited to out-of-the-box foundation models: the CustomLLM class\xa0(link resides outside ibm.com) allows for custom LLM wrappers. Likewise, you can use the IBM watsonx APIs and Python SDK, which includes a LangChain integration, to build applications in LangChain with models that you’ve already trained or fine-tuned for your specific needs using the WatsonxLLM class (and that model’s specific project ID).\nPrompt templates\nPrompts are the instructions given to an LLM. The “art” of composing prompts that effectively provide the context necessary for the LLM to interpret input and structure output in the way most useful to you is often called prompt engineering.\nThe PromptTemplate class in LangChain formalizes the composition of prompts without the need to manually hard code context and queries. Important elements of a prompt are likewise entered as formal classes, like input_variables. A prompt template can thus contain and reproduce context, instructions (like “do not use technical terms”), a set of examples to guide its responses (in what is called “few-shot prompting”), a specified output format or a standardized question to be answered.\u202fYou can save and name an effectively structured prompt template and easily reuse it as needed.\nThough these elements can all be manually coded, PromptTemplate modules empower smooth integration with other LangChain features, like the eponymous chains.\nChains\nAs its name implies, chains are the core of LangChain’s workflows. They combine LLMs with other components, creating applications by executing a sequence of functions.\nThe most basic chain is LLMChain. It simply calls a model and prompt template for that model. For example, imagine you saved a prompt as “ExamplePrompt” and wanted to run it against Flan-T5. You can import LLMChain from langchain.chains, then define chain_example = LLMChain(llm = flan-t5, prompt = ExamplePrompt). To run the chain for a given input, you simply call chain_example.run(“input”).\nTo use the output of one function as the input for the next function, you can use SimpleSequentialChain. Each function could utilize different prompts, different tools, different parameters or even different models, depending on your specific needs.\nIndexes\nTo achieve certain tasks, LLMs will need access to specific external data sources not included in its training dataset, such as internal documents, emails or datasets. LangChain collectively refers to such external documentation as “indexes”.\nDocument loaders\nLangChain offers\xa0a wide variety of document loaders for third party applications\xa0(link resides outside ibm.com). This allows for easy importation of data from sources like file storage services (like Dropbox, Google Drive and Microsoft OneDrive), web content (like YouTube, PubMed or specific URLs), collaboration tools (like Airtable, Trello, Figma and Notion), databases (like Pandas, MongoDB and Microsoft), among many others.\nVector databases\nUnlike “traditional” structured databases,\xa0vector databases\xa0represent data points by converting them into\xa0vector embeddings: numerical representations in the form of vectors with a fixed number of dimensions, often clustering related data points using\xa0unsupervised learning methods. This enables low latency queries, even for massive datasets, which greatly increases efficiency. Vector embeddings also store each vector’s metadata, further enhancing search possibilities.\nLangChain provides integrations for over 25 different embedding methods, as well as for over 50 different vector stores (both cloud-hosted and local).\nText splitters\xa0\nTo increase speed and reduce computational demands, it’s often wise to split large text documents into smaller pieces. LangChain’s\xa0TextSplitters\xa0split text up into small, semantically meaningful chunks that can then be combined using methods and parameters of your choosing.\nRetrieval\nOnce external sources of knowledge have been connected, the model must be able to quickly retrieve and integrate relevant information as needed. Like watsonx, LangChain offers\xa0retrieval augmented generation (RAG):\xa0its\xa0retriever\xa0modules accept a string query as an input and return a list of\xa0Document’s as output.\nMemory\nLLMs, by default, do not have any long-term memory of prior conversations (unless that chat history is used as input for a query). LangChain solves this problem with simple utilities for adding memory to a system, with options ranging from retaining the entirety of all conversations to retaining a summarization of the conversation thus far to retaining the n\xa0most recent exchanges.\nAgents\nLangChain agents can use a given language model as a “reasoning engine” to determine which actions to take. When building a chain for an agent, inputs include:\n\na list of available tools to be leveraged.\nuser input (like prompts and queries).\nany relevant previously executed steps.\n\nTools\nDespite their heralded power and versatility, LLMs have important limitations: namely, a lack of up-to-date information, a lack of domain-specific expertise and a general difficulty with math.\nLangChain tools\xa0(link resides outside ibm.com) are a set of functions that empower LangChain agents to interact with real-world information in order to expand or improve the services it can provide. Examples of prominent LangChain tools include:\n\n\nWolfram Alpha: provides access to powerful computational and data visualization functions, enabling sophisticated mathematical capabilities.\n\n\nGoogle Search: provides access to Google Search, equipping applications and agents with real-time information.\n\n\nOpenWeatherMap: fetches weather information.\n\n\nWikipedia: provides efficient access to information from Wikipedia articles.\n\n\nAI Academy\nWhy foundation models are a paradigm shift for AI\nLearn about a new class of flexible, reusable AI models that can unlock new revenue, reduce costs and increase productivity, then use our guidebook to dive deeper.\nGo to episode\nLangSmith\nReleased in the fall of 2023, LangSmith aims to bridge the gap between the accessible prototyping capabilities that brought LangChain to prominence and building production-quality LLM applications.\nLangSmith provides tools to monitor, evaluate and debug applications, including the ability to automatically trace all model calls to spot errors and test performance under different model configurations. This visibility aims to empower more robust, cost-efficient applications.\nGetting started with LangChain\nLangChain is open source and free to use: source code is\xa0available for download on Github\xa0(link resides outside ibm.com).\nLangChain can also be installed on Python with a simple pip command:\xa0pip install langchain.\u202fTo install all LangChain dependencies (rather than only those you find necessary), you can run the command\xa0pip install langchain[all].\nMany step-by-step tutorials are available from both the greater LangChain community ecosystem and the official documentation at\xa0docs.langchain.com\xa0(link resides outside ibm.com).\nLangChain use cases\nApplications made with LangChain provide great utility for a variety of use cases, from straightforward question-answering and text generation tasks to more complex solutions that use an LLM as a “reasoning engine.”\n\nChatbots: Chatbots are among the most intuitive uses of LLMs. LangChain can be used to provide proper context for the specific use of a chatbot, and to integrate chatbots into existing communication channels and workflows with their own APIs.\nSummarization: Language models can be tasked with summarizing many types of text, from breaking down complex academic articles and transcripts to providing a digest of incoming emails.\nQuestion answering: Using specific documents or specialized knowledge bases (like Wolfram, arXiv or PubMed), LLMs can retrieve relevant information from storage and articulate helpful answers). If fine-tuned or properly prompted, some LLMs can answer many questions even without external information.\nData augmentation: LLMs can be used to generate synthetic data for use in machine learning. For example, an LLM can be trained to generate additional data samples that closely resemble the data points in a training dataset.\nVirtual agents: Integrated with the right workflows, LangChain’s Agent modules can use an LLM to autonomously determine next steps and take action using robotic process automation (RPA).\n\nFootnotes\n1\xa0The fastest-growing open-source startups in Q2 2023\xa0(link resides outside ibm.com), Runa Capital, 2023\nEbook How to choose the right foundation modelLearn how to choose the right approach in preparing datasets and employing foundation models.\nRead the ebook\nRelated solutions Foundation models\nExplore the IBM library of foundation models on the watsonx platform to scale generative AI for your business with confidence.\nDiscover watsonx.ai Artificial intelligence solutions\nPut AI to work in your business with IBM's industry-leading AI expertise and portfolio of solutions at your side.\nExplore AI solutions AI consulting and services\nReinvent critical workflows and operations by adding AI to maximize experiences, real-time decision-making and business value.\nExplore AI services\nResources\nAI models Explore IBM GraniteIBM® Granite™ is our family of open, performant and trusted AI models, tailored for business and optimized to scale your AI applications. Explore language, code, time series and guardrail options.\nMeet Granite Ebook How to choose the right foundation modelLearn how to select the most suitable AI foundation model for your use case.\nRead the ebook Article Discover the power of LLMsDive into IBM Developer articles, blogs and tutorials to deepen your knowledge of LLMs.\nExplore the articles Guide The CEO's guide to model optimizationLearn how to continually push teams to improve model performance and outpace the competition by using the latest AI techniques and infrastructure.\nRead the guide Report A differentiated approach to AI foundation modelsExplore the value of enterprise-grade foundation models that provide trust, performance and cost-effective benefits to all industries.\nRead the report Ebook Unlock the Power of Generative AI + MLLearn how to incorporate generative AI, machine learning and foundation models into your business operations for improved performance.\nRead the ebook Report AI in Action 2024We surveyed 2,000 organizations about their AI initiatives to discover what's working, what's not and how you can get ahead.\nRead the report\nTake the next step\nExplore the IBM library of foundation models on the IBM watsonx platform to scale generative AI for your business with confidence.\nExplore watsonx.ai Explore AI solutions"}, {'title': 'About - LangChain', 'url': 'https://www.langchain.com/about', 'content': 'Ready to start shipping\nreliable GenAI apps faster?\nLangChain and LangSmith are critical parts of the reference\narchitecture to get you from prototype to production. Announcing the General Availability of LangSmith\nToday, we’re thrilled to announce the general availability of LangSmith — our solution for LLM application development, monitoring, and testing.\n We build products that enable developers to go from an idea to working code in an afternoon and in the hands of users in days or weeks. The biggest developer community in \xa0GenAI\nLearn alongside the 100K+ practitioners\nwho are pushing the industry forward.\n And we built LangSmith to support all stages of the AI engineering lifecycle, to get applications into production faster.\n', 'score': 0.73207545, 'raw_content': "We help developers make the\xa0impossible, possible.\nLangChain is the platform developers and enterprises choose to build AI apps\xa0from prototype to production.\nMission\nWe're on a mission to make it easy to build the LLM apps of tomorrow, today. We build products that enable developers to go from an idea to working code in an afternoon and in the hands of users in days or weeks. We’re humbled to support over 50k companies who choose to build with LangChain. And we built LangSmith to support all stages of the AI engineering lifecycle, to get applications into production faster.\nOur small but mighty team based in San Francisco.\nThe biggest developer community in \xa0GenAI\nLearn alongside the 100K+ practitioners\nwho are pushing the industry forward.\nCome join us\nWe’re hiring across many teams. Explore our open positions or read more about Careers at LangChain.\nRead more about what we’re\xa0working on lately.\nAnnouncing the General Availability of LangSmith\nToday, we’re thrilled to announce the general availability of LangSmith — our solution for LLM application development, monitoring, and testing.\nLangChain v0.1.0\nToday we’re excited to announce the release of langchain 0.1.0, our first stable version. It is fully backwards compatible, comes in both Python and JavaScript, and offers improved focus through both functionality and documentation.\nLangChain State of AI 2023\nIn 2023 we saw an explosion of interest in Generative AI upon the heels of ChatGPT. All companies - from startups to enterprises - were (and still are) trying to figure out their GenAI strategy.\nReady to start shipping\nreliable GenAI apps faster?\nLangChain and LangSmith are critical parts of the reference\narchitecture to get you from prototype to production."}, {'title': 'LangChain Explained: Your First Steps Toward Building Intelligent ...', 'url': 'https://www.cohorte.co/blog/langchain-explained-your-first-steps-toward-building-intelligent-applications-with-llms', 'content': "Whether you're building chatbots, intelligent data retrieval systems, or more complex generative applications, LangChain provides a cohesive environment for combining LLMs with different modules to create powerful workflows. LangGraph: Designed for building stateful multi-actor applications, LangGraph uses graph modeling to create sophisticated chains and agents. In LangChain, Prompt Templates help convert user input and context into properly formatted prompts that guide the model. LangServe makes it easy to deploy LangChain applications as REST APIs. This is particularly useful for developers looking to deploy LLM applications in a production environment without needing to manually manage server infrastructure. Now, we'll use LCEL to chain the prompt, model, and parser together. chain = prompt_template | model | parser LangChain offers a powerful, flexible framework to build applications powered by language models.", 'score': 0.7192461, 'raw_content': 'LangChain Explained: Your First Steps Toward Building Intelligent Applications with LLMs - Cohorte Projects\n\nBlogLog inGet Started\nFR\nFR\n\nGet Started\n\nBlogLog in\nFR\nGet Started\nGet Started\nLangChain Explained: Your First Steps Toward Building Intelligent Applications with LLMs\n\nBuilding with large language models can be complex. LangChain makes it simpler. This open-source framework brings together LLMs, data modules, and workflow tools—all in one place—to power up your next AI project.\nLangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). Whether you\'re building chatbots, intelligent data retrieval systems, or more complex generative applications, LangChain provides a cohesive environment for combining LLMs with different modules to create powerful workflows. Below, we provide an overview of the important concepts, building blocks, and integrations available within LangChain.\nKey Components and Building Blocks of LangChain\nLangChain is built around several core packages that serve different purposes:\n\nlangchain-core: This package contains the base abstractions and interfaces for all the components of LangChain. It defines the structure for core concepts like LLMs, vector stores, retrievers, etc. Importantly, no third-party integrations are included here, ensuring lightweight dependencies.\nlangchain: The main package that contains chains, agents, and retrieval strategies. These components form the "cognitive architecture" for building applications, and are generic across different integrations.\nlangchain-community: This package contains community-maintained third-party integrations, covering LLMs, vector stores, and retrievers.\nPartner Packages: Popular integrations, like those for OpenAI and Anthropic, are separated into distinct packages (e.g., langchain-openai) for better support.\n\nAdditionally, there are specialized extensions such as:\n\nLangGraph: Designed for building stateful multi-actor applications, LangGraph uses graph modeling to create sophisticated chains and agents.\nlangserve: A package that helps you deploy LangChain applications as REST APIs for production use.\nLangSmith: A developer platform that supports debugging, testing, evaluating, and monitoring LLM-based applications.\n\nCore Concepts in LangChain\n1. Models: LLMs and Chat Models\n\nLangChain provides integration with multiple LLMs and chat models. These models are used to generate responses based on input prompts. LangChain does not host any models directly but instead integrates with different third-party providers, including:\n\nOpenAI (e.g., GPT-3.5, GPT-4)\nAnthropic (e.g., Claude)\nAzure OpenAI Service\nGoogle Gemini\nCohere\nNVIDIA\nFireworksAI\nGroq\nMistralAI\nTogetherAI\n\nThe chat models accept sequences of messages as input, which allows for more dynamic conversational interactions, distinguishing between roles like user, assistant, and system messages.\n2. Prompt Templates\nPrompts are the way users communicate instructions to language models. In LangChain, Prompt Templates help convert user input and context into properly formatted prompts that guide the model. Prompt templates can include variables, making it easy to create flexible prompts based on different user inputs.\nThere are two main types of prompt templates:\n\nString Prompt Templates: Used for simpler tasks where the prompt is a single string.\nChat Prompt Templates: These are used to format more complex prompts involving multiple messages (e.g., system, user, assistant).\n\nExample:\n```javascript\nfrom langchain_core.prompts import ChatPromptTemplate\nprompt_template = ChatPromptTemplate.from_messages([\n    ("system", "You are a helpful assistant"),\n    ("user", "Tell me a joke about {topic}")\n])\n```\n3. Chains\nChains are sequences of calls that take user input, process it through models and other tools, and return the result. LangChain provides multiple types of chains:\n\nLLMChain: This is the simplest type, consisting of a prompt fed into an LLM.\nConversationalRetrievalChain: A more complex chain used for building conversational applications that need context retrieval from past conversations.\n\n4. Agents\nAgents are dynamic systems that use an LLM to decide which actions to take next. They form the decision-making backbone of applications that need to interact with tools or APIs based on user inputs.\n\nReAct Agents: These agents use reasoning and acting steps iteratively to complete tasks. For example, they might call a search tool, analyze the results, and decide on the next action.\nLangGraph Agents: These are more advanced agents aimed at highly controllable and customizable use cases. LangGraph provides the flexibility to compose custom flows using graph-based modeling.\n\n5. Chat History\n\nLangChain provides Chat History functionality, which is crucial for conversational applications. It enables the system to refer back to previous messages, thus maintaining context throughout the conversation.\n6. Output Parsers\nOutput Parsers are used to convert the raw text output from models into structured formats. LangChain supports a variety of parsers, such as:\n\nJSON Output Parser: Converts output into a JSON object based on a specified schema.\nCSV Output Parser: Returns data as a list of comma-separated values.\nPandas DataFrame Parser: Converts output into a Pandas DataFrame for easy data manipulation.\n\nOutput parsers are especially useful when working with structured data or integrating LLMs with downstream applications.\n7. Retrievers and Vector Stores\nRetrievers are used to fetch documents based on a query. A Vector Store is a common implementation where documents are embedded into vector representations and then searched using similarity metrics.\n\nPopular Vector Stores: LangChain integrates with vector databases like Pinecone, Weaviate, and FAISS, making it easy to set up retrieval-augmented generation (RAG) systems.\nRetrievers from Vector Stores: You can use vector stores to create retrievers that perform similarity searches and return relevant documents.\n\n8. Tools and Toolkits\n\nTools are utility functions that an LLM can call to execute specific tasks, such as making an API call or querying a database.\n\nToolkits: A collection of tools designed for specific tasks. For instance, a toolkit might include tools for querying a database, sending an email, or summarizing a document.\n\nLangChain\'s tools have a name, a description, and a defined schema for inputs, making it easy for the LLM to determine which tool to use in a given context.\nLangChain Integrations\nLangChain supports many integrations to enhance its capabilities:\n\nLLM Integrations: As mentioned earlier, LangChain can integrate with various LLM providers like OpenAI, Anthropic, and Cohere.\nDocument Loaders: These are used to bring data into LangChain from sources like Google Drive, Notion, Slack, and databases.\nText Splitters: Text splitters help divide larger documents into smaller, semantically meaningful chunks, making them suitable for LLM processing. For instance, you can split HTML using HTMLHeaderTextSplitter or Markdown with MarkdownHeaderTextSplitter.\nKey-Value Stores: LangChain also offers key-value stores for use cases like retrieval caching and embeddings management.\n\nLangChain Expression Language (LCEL)\nLCEL is a declarative way to chain components together. It provides features like:\n\nStreaming: LCEL allows streaming of tokens from an LLM to output parsers, offering fast, real-time user experiences.\nAsync Support: Chains defined with LCEL can be run asynchronously, enabling concurrency and better performance in production environments.\nRetries and Fallbacks: LCEL supports robust error handling, such as retrying failed requests and configuring fallbacks for different scenarios.\n\nExample LCEL usage:\n```javascript\nfrom langchain_core.prompts import ChatPromptTemplate\nfrom langchain_anthropic import ChatAnthropic\nprompt = ChatPromptTemplate.from_template("What\'s the weather like in {location}?")\nmodel = ChatAnthropic(model="claude-3")\nchain = prompt | model\n```\nLangChain Packages for Specialized Use Cases\n1. LangGraph\nLangGraph is aimed at building applications with robust state management. It extends LangChain to enable complex, stateful interactions by modeling the workflow as a graph of nodes and edges. This helps in designing reliable, multi-step agents and defining how data flows between components.\n2. LangServe\nLangServe makes it easy to deploy LangChain applications as REST APIs. This is particularly useful for developers looking to deploy LLM applications in a production environment without needing to manually manage server infrastructure.\n3. LangSmith\nLangSmith is a platform for testing, debugging, and monitoring LLM applications. It provides powerful tools for tracking the performance of models, understanding the logic behind their responses, and visualizing how different parts of your chain contribute to the final result.\nPutting It All Together\nTo create a complete LangChain application, you need to:\n\nChoose the right models (e.g., OpenAI\'s GPT-4 or Anthropic\'s Claude).\nDesign a chain or agent that defines how different components (e.g., LLMs, tools, retrievers) interact to achieve your goal.\nDefine prompts and output parsers to guide the model’s output into the appropriate form.\nUse LangServe to deploy your application and LangSmith to monitor and test it.\n\nExample: Building a Simple LLM Application with LCEL\nIn this quickstart example, we\'ll show you how to build a simple LLM application that translates text from English into another language. This is a relatively simple LLM application—just a single LLM call plus some prompting. Still, it\'s a great way to get started with LangChain, as many features can be built with just some prompting and an LLM call!\nSetup\nTo follow along, you\'ll need to have LangChain installed. You can install it via pip:\njavascript\npip install langchain\nYou\'ll also need an API key for the LLM provider of your choice, such as OpenAI.\nUsing Language Models\nFirst, let\'s initialize a language model. In this example, we\'ll use OpenAI\'s GPT-4 model.\n```javascript\nimport os\nfrom langchain_openai import ChatOpenAI\nSet your OpenAI API key\nos.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"\nInitialize the model\nmodel = ChatOpenAI(model="gpt-4")\n```\nPrompt Templates and Output Parsers\nNext, let\'s define a prompt template and an output parser.\n```javascript\nfrom langchain_core.prompts import ChatPromptTemplate\nfrom langchain_core.output_parsers import StrOutputParser\nDefine the prompt template\nprompt_template = ChatPromptTemplate.from_messages([\n    ("system", "Translate the following into {language}:"),\n    ("user", "{text}")\n])\nDefine the output parser\nparser = StrOutputParser()\n```\nChaining Components Together with LCEL\nNow, we\'ll use LCEL to chain the prompt, model, and parser together.\n```javascript\nCreate the chain\nchain = prompt_template | model | parser\nInvoke the chain\nresult = chain.invoke({"language": "Italian", "text": "Hello, how are you?"})\nprint(result)  # Output: \'Ciao, come stai?\'\n```\nDeploying with LangServe\nTo deploy this chain as a REST API, you can use LangServe.\n```javascript\nfrom fastapi import FastAPI\nfrom langserve import add_routes\nDefine the FastAPI app\napp = FastAPI(title="Translation API", version="1.0")\nAdd the chain route\nadd_routes(app, chain, path="/translate")\nif name == "main":\n    import uvicorn\n    uvicorn.run(app, host="localhost", port=8000)\n```\nYou can now run this script to serve your chain at http://localhost:8000/translate.\nFinal Thoughts\nLangChain offers a powerful, flexible framework to build applications powered by language models. With support for different integrations, complex workflows, and robust monitoring tools, it provides all the tools needed to build sophisticated LLM applications. Our simple example shows how you can start building your own applications by chaining components together and deploying them with ease.\n\u200d\nCohorte Team\nNovember 5, 2024\n\nWant to see what AI can do for you?\nBook a free workshop with me.\nGet Started\nGet a free actionable hack every week in your inbox\nJoin my newsletter\n\n© 2024 Cohorte\nPrivacy PolicyTerms of Service\n'}], 'response_time': 6.85}
</pre>

```python
# Example of a news search.
result2 = tavily_tool.search(
    query="Latest AI technology trends",  # Search query
    search_depth="basic",  # Basic search depth
    topic="news",  # News topic
    days=3,  # Results from the last 3 days
    max_results=5,  # Maximum of 5 results
    include_answer=False,  # Exclude answers
    include_raw_content=False,  # Exclude raw content
    include_images=False,  # Exclude images
    format_output=True,  # Format the output
)

print("News search results:", result2)
```

<pre class="custom">News search results: {'query': 'Latest AI technology trends', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://techxplore.com/news/2025-01-lithium-sodium-ion-batteries-breakthroughs.html', 'title': 'Losing to lithium: Research shows sodium-ion batteries need breakthroughs to compete - Tech Xplore', 'score': 0.58976424, 'published_date': 'Mon, 13 Jan 2025 10:00:01 GMT', 'content': '##### Bias and discrimination in AI: Why sociolinguistics holds the key to better LLMs and a fairer world 5 hours ago ##### Sustainable cement: An electrochemical process to help neutralize cement industry CO₂ emissions Jan 11, 2025 ##### Smart glasses enter new era with sleeker designs, lower prices Jan 11, 2025 ##### Robots set to move beyond factory as AI advances Jan 10, 2025 ##### Light, flexible and radiation-resistant: Organic solar cells for space Jan 10, 2025 ##### Exploring quinone-based carbon capture: A promising path to safer CO₂ removal Jan 10, 2025 ##### Microsoft introduces rStar-Math, an SLM for math reasoning and problem solving Jan 10, 2025 ##### AI-enabled technology is 98% accurate at spotting illegal contraband Jan 10, 2025 ##### A Minecraft-based benchmark to train and test multi-modal multi-agent systems Jan 10, 2025 ##### Sustainable building components use passive dehumidification to create a good indoor climate Jan 10, 2025'}, {'url': 'https://www.roboticstomorrow.com/story/2025/01/ieee-reveals-predictions-for-top-technology-trends-of-2025/23881/', 'title': 'IEEE Reveals Predictions for Top Technology Trends of 2025 - Robotics Tomorrow', 'score': 0.52086246, 'published_date': 'Wed, 15 Jan 2025 13:56:44 GMT', 'content': 'Home News Industrial Robotics Mobile Robots Factory Automation Other Topics Site Services In addition to these top technology developments, the Committee also anticipates the following technologies will experience significant growth over the next year: IT/energy convergence; augmented AI; autonomous driving; SmartAg; functional safety/autonomous vehicles; AI-assisted drug discovery; sustainable computing; mis/disinformation; AI-based medical diagnosis, AI-optimized green high-performance computing; next-gen cyberwarfare; new battery chemistries; data feudalism; nuclear-powered data centers; tools and policies for AI regulation; brain-computer interfaces (ones that enhance interfaces between humans and computers, particularly for those with disabilities); and space computing. Beyond outlining computer science and engineering trends, the 2025 Technology Predictions Committee offers insights into how industry, government, academia, and professional organizations can support and advance these developments.'}, {'url': 'https://www.geeky-gadgets.com/self-improving-ai-models/', 'title': 'Researchers Stunned as AI Improves Itself Towards Superintelligence - Geeky Gadgets', 'score': 0.40768766, 'published_date': 'Mon, 13 Jan 2025 09:15:10 GMT', 'content': 'Self-Improving AI Models: The Future of Cost-Effective Intelligence - Geeky Gadgets Researchers at Microsoft have unveiled a new AI model, RStar-Math, that’s rewriting the rules of how machines learn and improve. Unlike traditional models that rely on massive datasets or guidance from larger systems, RStar-Math takes a bold new approach: it teaches itself, paving the way for smaller, more efficient AI systems to outperform even the most resource-intensive giants like GPT-4 in specific tasks. RStar-Math Self-Improving AI Model By demonstrating that smaller models can achieve exceptional results, RStar-Math challenges the conventional emphasis on scale and opens new possibilities for resource-efficient AI development. Filed Under: AI, Technology News, Top NewsLatest Geeky Gadgets Deals'}, {'url': 'https://www.mobihealthnews.com/video/upside-ai-technology-2025', 'title': 'The upside of AI technology in 2025 - Mobihealth News', 'score': 0.38630086, 'published_date': 'Tue, 14 Jan 2025 15:42:44 GMT', 'content': 'The upside of AI technology in 2025 | MobiHealthNews Main Menu AI The upside of AI technology in 2025 Healthcare AI technology is additive and complementary, not punitive. More regional news Exclusive: Century Heath, Nira Medical partner to provide AI-curated EHR data Q&A: Hackensack Meridian Health on its AI expenditures in 2025 The upside of AI technology in 2025 Q&A: Qventus announces $105M investment during JPM Healthcare Conference More News Healthcare IT News Healthcare IT News Australia Healthcare Finance News HIMSS25 Global Health Conference & Exhibition Get ready to immerse yourself in the epicenter of healthcare innovation at the 2025 HIMSS Global Health Conference & Exhibition in Las Vegas! HIMSS25 European Health Conference & Exhibition HEALWELL AI buying Orion Health for $115M'}, {'url': 'https://towardsdatascience.com/the-ai-r-evolution-looking-from-2024-into-the-immediate-future-0261a5db7103', 'title': 'The AI (R)Evolution, Looking From 2024 Into the Immediate Future - Towards Data Science', 'score': 0.36683974, 'published_date': 'Wed, 15 Jan 2025 00:01:26 GMT', 'content': 'The AI (R)Evolution, Looking From 2024 Into the Immediate Future | by LucianoSphere (Luciano Abriata, PhD) | Jan, 2025 | Towards Data Science After half to one decade of technical developments that included the transformer architecture for AI systems together with several other computer science breakthroughs, the last 3–4 years have been crazily active in the development of specific applications resulting in (AI-based) software that we didn’t even dream about just 10–20 years ago. We can now reliably use LLMs to summarize texts, look for pieces of information, or even solve simple to mid-complexity problems; we can boost software writing, scripting, data analysis and software utilization with LLMs that possess vast amounts of knowledge and behave like experts available 24/7. Published in Towards Data Science --------------------------------- Your home for data science and AI.'}], 'response_time': 0.61}
</pre>

```python
# Example of a search with specific domain inclusion.
result3 = tavily_tool.search(
    query="Python programming tips",  # Search query
    search_depth="advanced",  # Advanced search depth
    max_results=3,  # Maximum of 3 results
    include_domains=["github.io"]
)

print("Search results with specific domain inclusion:", result3)
```

<pre class="custom">Search results with specific domain inclusion: {'query': 'Python programming tips', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Python Tips - williamkpchan.github.io', 'url': 'https://williamkpchan.github.io/LibDocs/Python+Tips.html', 'content': "In the past, we'd shared a list of Python programming tips for beginners that aimed to optimize code and reduce coding efforts. And our readers still enjoy reading it. So today, we're back with one more set of essential Python tips and tricks. All these tips can help you minify the code and optimize execution.", 'score': 0.7752821, 'raw_content': None}, {'title': "Python Tips | Ming's Blog - byrzhm.github.io", 'url': 'https://byrzhm.github.io/blog/posts/python-tips/', 'content': 'Python Tips. Posted Dec 20, 2024 . By Hongming Zhu. 1 min read. Python Tips. Contents. Python Tips ** Special Usages 1. Unpacking a dictionary into keyword arguments in a function call ... Programming Languages, Python. python. This post is licensed under CC BY 4.0 by the author. Share. Recently Updated. HPC; PyTorch Internals; Python Tips', 'score': 0.756376, 'raw_content': None}, {'title': 'Awesome Python Tips & Tricks | DevRa - rafed.github.io', 'url': 'https://rafed.github.io/devra/sections/awesome-python-tips--tricks/', 'content': 'Awesome Python Tips & Tricks - (Page 1/1) A better dictionary in python. A Python dictionary is a very useful data structure that stores key value pairs. A python dictionary is declared like the following- ... Become a Python Ninja with the below one-liners. programming python. Reduce Too Many if-elif in Python. if-elif statements are one of', 'score': 0.6519982, 'raw_content': None}], 'response_time': 3.21}
</pre>

```python
# Example of a search excluding specific domains.
result4 = tavily_tool.search(
    query="Healthy diet",  # Search query
    search_depth="basic",  # Basic search depth
    days=30,  # Results from the last 30 days
    max_results=7,  # Maximum of 7 results
    exclude_domains=["ads.com", "spam.com"]
)

print("Search results excluding specific domains:", result4)
```

<pre class="custom">Search results excluding specific domains: {'query': 'Healthy diet', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Healthy diet - World Health Organization (WHO)', 'url': 'https://www.who.int/health-topics/healthy-diet', 'content': 'A healthy diet is a foundation for health, well-being, optimal growth and development. It protects against all forms of malnutrition. Unhealthy diet is one of the leading risks for the global burden of disease, mainly for noncommunicable diseases such as cardiovascular diseases, diabetes, and cancer.', 'score': 0.35555163, 'raw_content': None}, {'title': 'Healthy diet - World Health Organization (WHO)', 'url': 'https://www.who.int/initiatives/behealthy/healthy-diet', 'content': 'A healthy diet is essential for good health and nutrition. It protects you against many chronic noncommunicable diseases, such as heart disease, diabetes and cancer. Eating a variety of foods and consuming less salt, sugars and saturated and industrially-produced trans-fats, are essential for healthy diet.', 'score': 0.3384274, 'raw_content': None}, {'title': 'Healthy diet - World Health Organization (WHO)', 'url': 'https://www.who.int/news-room/fact-sheets/detail/healthy-diet', 'content': 'Advice on a healthy diet for infants and children is similar to that for adults, but the following elements are also important:\nPractical advice on maintaining a healthy diet\nFruit and vegetables\nEating at least 400\xa0g, or five portions, of fruit and vegetables per day reduces the risk of NCDs (2) and helps to ensure an adequate daily intake of dietary fibre.\n In 2012, the Health Assembly adopted a “Comprehensive Implementation Plan on Maternal, Infant and Young Child Nutrition” and six global nutrition targets to be achieved by 2025, including the reduction of stunting, wasting and overweight\nin children, the improvement of breastfeeding, and the reduction of anaemia and low birthweight (9).\n Reduction of salt/sodium intake and elimination of industrially-produced trans-fats from\nthe food supply are identified in GPW13 as part of WHO’s priority actions to achieve the aims of ensuring healthy lives and promote well-being for all at all ages. Therefore, promoting a healthy food environment – including food systems that promote a diversified,\nbalanced and healthy diet – requires the involvement of multiple sectors and stakeholders, including government, and the public and private sectors.\n For adults\nA healthy diet includes the following:\nFor infants and young children\nIn the first 2 years of a child’s life, optimal nutrition fosters healthy growth and improves cognitive development.', 'score': 0.32386157, 'raw_content': None}, {'title': 'What are healthy diets? Joint statement by the Food and Agriculture ...', 'url': 'https://www.who.int/publications/i/item/9789240101876', 'content': 'Healthy diets promote health, growth and development, support active lifestyles, prevent nutrient deficiencies and excesses, communicable and noncommunicable diseases, foodborne diseases and promote wellbeing. ... The exact make-up of a diet will vary depending on individual characteristics, preferences and beliefs, cultural context, locally', 'score': 0.19962023, 'raw_content': None}, {'title': '5 principles of a healthy diet - Harvard Health', 'url': 'https://www.health.harvard.edu/healthbeat/5-principles-of-a-healthy-diet', 'content': 'While details may vary from diet to diet, all healthy eating plans have these five principles in common: Lots of plants. Plant foods — vegetables, fruits, legumes, whole grains, nuts, and seeds — offer a wealth of vitamins and minerals, as well as fiber and healthful compounds called phytochemicals (literally "plant chemicals," natural', 'score': 0.111309014, 'raw_content': None}, {'title': 'Healthy Eating 101: Nutrients, Macros, Tips, and More', 'url': 'https://www.healthline.com/nutrition/how-to-eat-healthy-guide', 'content': 'Written By\nJillian Kubala MS, RD\nEdited By\nGabriel Dunsmith\nMedically Reviewed By\nSade Meeks, MS, RD\nCopy Edited By\nChristina Guzik, BA, MBA\nShare this article\nEvidence Based\nThis article is based on scientific evidence, written by experts and fact checked by experts.\n What’s more, if your current diet is high in ultra-processed foods and beverages like fast food, soda, and sugary cereals but low in whole foods like vegetables, nuts, and fish, you’re likely not eating enough of certain nutrients, which may negatively affect your overall health (10).\n Tips for healthy eating in the real world\nHere are some realistic tips for you to get started with healthy eating:\nThese tips can help you move toward a healthier diet.\n For example, breakfast could be a spinach and egg scramble with avocado and berries, lunch a sweet potato stuffed with veggies, beans, and shredded chicken, and dinner a salmon filet or baked tofu with sautéed broccoli and brown rice.\n Current Version\nMar 8, 2023\nWritten By\nJillian Kubala MS, RD\nEdited By\nGabriel Dunsmith\nJun 24, 2021\n', 'score': 0.097984545, 'raw_content': None}, {'title': 'The 10 rules of a heart-healthy diet - Harvard Health', 'url': 'https://www.health.harvard.edu/nutrition/the-10-rules-of-a-heart-healthy-diet', 'content': 'A fresh look at risks for developing young-onset dementia\nPlyometrics: Three explosive exercises even beginners can try\nNew guidelines aim to screen millions more for lung cancer\nCould men with advanced prostate cancer avoid chemotherapy?\nRelated Content\nHeart Health\nPoor sleep linked to next-day episodes of atrial fibrillation\nHeart Health\nAnti-obesity drug lowers heart-related problems\nHeart Health\nThe best anti-clotting drug for afib?\n "\nImage: © CharlieAJA/Getty Images\nAbout the Author\nHeidi Godman,\nExecutive Editor, Harvard Health Letter\nAbout the Reviewer\nAnthony L. Komaroff, MD,\nEditor in Chief, Harvard Health Letter\nDisclaimer:\nAs a service to our readers, Harvard Health Publishing provides access to our library of archived content. The Best Diets for Cognitive Fitness, is yours absolutely FREE when you sign up to receive Health Alerts from Harvard Medical School\nSign up to get tips for living a healthy lifestyle, with ways to\nfight inflammation and improve cognitive health, plus the latest advances in preventative medicine, diet and exercise, pain relief, blood pressure and cholesterol management, and\xa0more.\n Please enable cookies to submit\nFooter\nMy Account\nOrder Now\nMore\n© 2024 Harvard Health Publishing® of The President and Fellows of Harvard\xa0College\nDo not sell my personal information | Privacy Policy and Terms of Use\nThanks for visiting. A fresh look at risks for developing young-onset dementia\nPlyometrics: Three explosive exercises even beginners can try\nNew guidelines aim to screen millions more for lung cancer\nCould men with advanced prostate cancer avoid chemotherapy?\n', 'score': 0.094820276, 'raw_content': None}], 'response_time': 3.45}
</pre>

```python
# Define the tool.
@tool
def search_news(keyword: str) -> str:
    """Collect recent news for the given query. """
    tavily_client = TavilyClient()
    search_results = tavily_client.search(query=keyword, topic="news", days = 30)
    return search_results

print(search_news.invoke("AI Investment"))
```

<pre class="custom">{'query': 'AI Investment', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://www.jomfruland.net/surprising-stock-picks-uncover-hidden-opportunities-in-ai-stocks/', 'title': 'Surprising Stock Picks! Uncover Hidden Opportunities in AI Stocks. - Jomfruland.net', 'score': 0.6396691, 'published_date': 'Wed, 01 Jan 2025 14:02:16 GMT', 'content': 'Uncover Hidden Opportunities in AI Stocks. Uncover Hidden Opportunities in AI Stocks. Although AI stocks have reached soaring valuations, strategic investors can still find opportunities by identifying unexploited prospects in the market. Its strategic investment in OpenAI has integrated revolutionary AI technologies across Microsoft’s product suite, spurring growth and enhancing customer retention. Unleashing Investment Potential in AI: Exploring TSMC and Microsoft’s Strategic Moves Investor interest in AI technology remains strong, with TSMC and Microsoft poised as key players within the broader tech industry. As TSMC pushes the boundaries in semiconductor manufacturing and Microsoft extends its AI influence through cloud services, investors stand to benefit from the ongoing digital transformation. Uncover Hidden Opportunities in AI Stocks.'}, {'url': 'https://markets.businessinsider.com/news/stocks/vectorspace-ai-x-vaix-revolutionizes-ai-driven-investment-insights-with-graph-based-models-1034172082', 'title': 'Vectorspace AI X (VAIX) Revolutionizes AI-Driven Investment Insights with Graph-Based Models - Markets Insider', 'score': 0.6186197, 'published_date': 'Wed, 25 Dec 2024 04:07:35 GMT', 'content': 'Vectorspace AI X (VAIX) Revolutionizes AI-Driven Investment Insights with Graph-Based Models | Markets Insider Markets Stocks Indices Commodities Cryptocurrencies Currencies ETFs News Vectorspace AI X (VAIX) Revolutionizes AI-Driven Investment Insights with Graph-Based Models Vectorspace AI X (VAIX) Revolutionizes AI-Driven Investment Insights with Graph-Based Models SAN DIEGO, Dec. 24, 2024 (GLOBE NEWSWIRE) -- Vectorspace AI X (VAIX), a trailblazer in AI-driven datasets and graph-based models, is proud to announce its innovative technology that uncovers hidden relationships between stocks, cryptocurrencies, and global events. To access the full potential of VAIX, users can now trade the token on ProBit Exchange, a global platform catering to millions of cryptocurrency enthusiasts. To trade VAIX, ensure your ProBit account is funded with cryptocurrency, such as USDT (Tether).'}, {'url': 'https://www.ftadviser.com/adviser-technology/2025/1/14/quarter-of-ifas-plan-to-introduce-ai-next-year/', 'title': 'Quarter of IFAs plan to introduce AI next year - FT Adviser', 'score': 0.5837973, 'published_date': 'Tue, 14 Jan 2025 09:39:49 GMT', 'content': 'Home Pensions Home Investments Investments Home Investment trusts Mortgages Home Protection Home Regulation Home Tax Home Tax-efficient investments Income investing Home Investments Investment trusts Tax-efficient investments Income investing © Tara Winstead/PexelsThe findings suggest that, while some IFAs are cautious, many are open to embracing AI-driven technologies to improve efficiency and client outcomes. Nearly one in four (23 per cent) IFAs are planning on introducing AI tools in their services to clients within the next 12 months, research from Opinium has suggested. Opinium said its findings suggest that, while some IFAs are cautious, many are open to embracing AI-driven technologies to improve efficiency and client outcomes. “Embracing these technologies can help firms stay ahead of the curve, and help advisers deliver a better service to existing and new clients.”'}, {'url': 'https://hitconsultant.net/2025/01/13/qventus-secures-105m-to-advance-ai-assistants-for-optimal-health-system-efficiencies/', 'title': 'Qventus Secures $105M to Advance AI Assistants for Optimal Health System Efficiencies - - HIT Consultant', 'score': 0.559411, 'published_date': 'Mon, 13 Jan 2025 14:18:48 GMT', 'content': '–\xa0\xa0Qventus, a leading provider of AI-based care automation software for health systems, today announced a $105 million investment led by global investment firm KKR, with additional participation from world-renowned investment firm Bessemer Venture Partners, and new strategic investors, including leading health systems Northwestern Medicine, HonorHealth, and Allina Health. This funding accelerates the Company’s ability to provide AI-based automations and AI operational assistants in more care settings, building upon the success of its existing offerings like Qventus’\xa0Surgical Growth\xa0and\xa0Inpatient Capacity\xa0solutions as well as new solutions built on its first-to-market\xa0AI Operational Assistants platform\xa0capability. Qventus CEO and Co-Founder, Mudit Garg, emphasized the company’s commitment to transforming healthcare operations: “This funding validates our decade-long effort to build AI automation solutions that reduce administrative burdens, enabling healthcare teams to deliver reliable patient care.'}, {'url': 'https://www.rcrwireless.com/20250114/analyst-angle/kagan-softbank-ai', 'title': 'Kagan: Masayoshi Son steers Softbank to AI investment in US - RCR Wireless News', 'score': 0.55362654, 'published_date': 'Tue, 14 Jan 2025 15:29:50 GMT', 'content': 'Kagan: Masayoshi Son steers Softbank to AI investment in US Softbank is one of the world’s largest, most important and powerful investment companies which plays in many spaces including wireless, Internet, AI, IoT and more. While I do not expect that Softbank is solely focused on the United States for AI growth, they obviously see us playing a very important role in its development and deployment. Going forward, the Softbank pace of investment will continue, however they will now focus on AI for growth. That is why I believe Son and Softbank are so focused on growth with AI as a centerpiece. This is what I expect Masayoshi Son has been focused on in the last several years as new technologies like AI and Quantum are quickly growing.'}], 'response_time': 0.51}
</pre>

## Creating a Custom Tool for Google News Article Search

Define the `GoogleNews` class, which will be used as a tool to search for Google News articles.

**Note**
- No API key is required (because it uses RSS feeds).

This tool searches for news articles provided by **news.google.com** .

**Description**
- Uses the Google News search API to retrieve the latest news.
- Allows searching for news based on keywords.

**Key Parameters**
- `k` (int): Maximum number of search results to return (default: 5).

```python
# hl: Language, gl: Region, ceid: Region and Language Code
url = f"{self.base_url}?hl=en&gl=US&ceid=US:en" 
```

In the code, you can adjust the search results' language and region by modifying the language (hl), region (gl), and region and language code (ceid).

**Note**

Save the provided code as `google_news.py` , and then you can import it in other files using `from google_news import GoogleNews` .


```python
#%pip install feedparser
```

```python
import feedparser
from urllib.parse import quote
from typing import List, Dict, Optional


class GoogleNews:
    """
    This is a class for searching Google News and returning the results.
    """

    def __init__(self):
        """
        Initializes the GoogleNews class.
        Sets the base_url attribute.
        """
        self.base_url = "https://news.google.com/rss"

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Fetches news from the given URL.

        Args:
            url (str): The URL to fetch the news from.
            k (int): The maximum number of news articles to fetch (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing news titles and links.
        """
        news_data = feedparser.parse(url)
        return [
            {"title": entry.title, "link": entry.link}
            for entry in news_data.entries[:k]
        ]

    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Formats and returns the list of news articles.

        Args:
            news_list (List[Dict[str, str]]): A list of dictionaries containing news information.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        if not news_list:
            print("No news available for the given keyword.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_latest(self, k: int = 3) -> List[Dict[str, str]]:
        """
        Searches for the latest news.

        Args:
            k (int): The maximum number of news articles to search for (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        #url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        url = f"{self.base_url}?hl=en&gl=US&ceid=US:en" # hl: 언어, gl: 지역, ceid: 지역 및 언어 코드
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

    def search_by_keyword(
        self, keyword: Optional[str] = None, k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Searches for news using a keyword.  

        Args:
            keyword (Optional[str]): The keyword to search for (default: None).
            k (int): The maximum number of news articles to search for (default: 3).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing URLs and content.
        """
        if keyword:
            encoded_keyword = quote(keyword)
            url = f"{self.base_url}/search?q={encoded_keyword}&hl=en&gl=US&ceid=US:en"
        else:
            url = f"{self.base_url}?hl=en&gl=US&ceid=US:en"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

```

```python
google_tool = GoogleNews()
```

```python
google_tool.search_by_keyword("AI Investment")
```




<pre class="custom">[{'url': 'https://news.google.com/rss/articles/CBMimAFBVV95cUxNaThyMnBjNjJpQjlQQTcwQjdoTEZPc1plbDdYU29YWlNVakE1NzVkV2ZNZHRyYlFSSjg1VDRNX2ZPT1NZdkM0ckVxMmxwaW1PbTdpay1EX2lUbFNqblIxOXdUZ3VDTVZoVmRlWVZXLTBUNG5SSUJZa3RieEFST0VjQ1hSRkxWNzRXSlBYb3k3QzEzUGdtNHo2dg?oc=5',
      'content': "A Once-in-a-Decade Investment Opportunity: 1 Artificial Intelligence (AI) Semiconductor Stock to Buy Hand Over Fist and Hold for the Next 10 Years (Hint: It's Not Nvidia) - The Motley Fool"},
     {'url': 'https://news.google.com/rss/articles/CBMipAFBVV95cUxNendoZ0U1eXVkVk5pYnljLThfWVotZENTbUJYZTVYRUhNOEFZNnlaYUZmT29FemhnQmJKRFptMml1cl9oQWdpM0NCQ3FhVkppcWlyeTNpUjdXOF9lNXQwR201eXM1UGxmazIzLTVLYVF2OEE1TVdBeHVwYk1uX0F3aHM0Q2p2R2R2S0p4eV8wZnFYVU9mdWJfd1VwS2I5RjVqTDZVQQ?oc=5',
      'content': 'AI Avatar Startup Synthesia Valued at $2.1 Billion - PYMNTS.com'},
     {'url': 'https://news.google.com/rss/articles/CBMie0FVX3lxTE9NbWttZW12S1BPLXJtM3Fhb1Z3NndndEZKRHo4Tk5sdzV3U0xvTFJHYmxjZ0lGUmNTRDcwLUlpZ3BCX0RrZmVvNDlXcFNVR3g4bEFlaTBjS2UwR016U1pSOFB2NTQzdGRuS2FwVEFDQVBnR3pHYUJaTmY2Yw?oc=5',
      'content': 'EU Asks for Risk Assessments of Chip, AI, Quantum Investments - Yahoo! Voices'}]</pre>



```python
from langchain.tools import tool
from typing import List, Dict


# Create a tool for searching news by keyword
@tool
def search_keyword(query: str) -> List[Dict[str, str]]:
    """Look up news by keyword"""
    print(query)
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)
```

```python
# Execution Results
search_keyword.invoke({"query": "LangChain AI"})
```

<pre class="custom">LangChain AI
</pre>




    [{'url': 'https://news.google.com/rss/articles/CBMid0FVX3lxTFBUVGIzM04zT1RYOC03QUxJenZLU3F4NVpoRmhvbFJRSnNZQVJIeUNyV2ktdERiSlplb2xQcXgyQWpHYS1DbEVRbk40alVBTzNQREFaQW40czNyNkdJcWxlQVgzdjVyZXI3UTN1aVVvYkdIS0tTU2lv?oc=5',
      'content': 'LangChain Unveils Innovative Ambient Agents for AI Interaction - Blockchain News'},
     {'url': 'https://news.google.com/rss/articles/CBMib0FVX3lxTE9qeUFQd08yN3pXenFYeWRTM1dHN3ZtTGVvaDAzREVNRU02ZVhNRFZCWmFpRm9obTN5QXF1TlpaYUI3d2VJYTBNeHNyUGJ3MXdVQWNDVW5MTVk4UDlfazFfX1hRRWpUeHBfUGRfS0pBdw?oc=5',
      'content': 'Vulnerabilities in LangChain Gen AI - Unit 42'},
     {'url': 'https://news.google.com/rss/articles/CBMixgFBVV95cUxNUzhFZEtsS2FrNTZ1ejVDejBQRThtSDA5TUtMbWRHOUxfUEtBVl8teERpZXp5U2wzbktZeWdTWnhKc2V4RnM2SXc0TzBsNVlQMktqUXRwNDliQW9yaTVnT3lzZ0lxeThobTliUEJRd05VM3V1anJ3bUh4cGJRbk44a1V0MVMzSEZfc2c4am04QnJBM2ZhZThTQ1NvYlhQUVBZbmhJWVZQdzlWWjE0UVdQRjYtWU1iSWl6WFBBTXdsQ2ROSHUxWmc?oc=5',
      'content': 'LangChain Meets Home Assistant: Unlock the Power of Generative AI in Your Smart Home - Towards Data Science'},
     {'url': 'https://news.google.com/rss/articles/CBMi6wFBVV95cUxPNEtmajI2MWxEb2FRSUNLWkhBQmV5VThPUVllNkRzbGEwOW1mRDZVZEhxYnI3X3BxZUZ1WThGbmF6WUV6MnVSaDJtN0ZUd2VZdGt5NV9CNHJRVXllVElnZE5DaVJDdFBSZHU5aGI3XzM3RTR1YVdyU19pYjJER2NTM080dlYyVjd6MDNFOTFudm5WRG1zdk9WNUxJTVpoakxEWlZxR3RaSmZiQ2FOdVA4YnFjUXBjeDdaRzNxNzVzbEU1ak9RSkY1bVVIRGFrdFBfNEhkcVhJNS0xdFFDY1IyMXBNSEUxZlJQeGlF?oc=5',
      'content': 'Create a next generation chat assistant with Amazon Bedrock, Amazon Connect, Amazon Lex, LangChain, and WhatsApp - AWS Blog'},
     {'url': 'https://news.google.com/rss/articles/CBMitgFBVV95cUxNSmpOTkdfRFl4WlBqWTdHLWRGajllZVdMZklQaE5mRVNfc1lYLU9vZHNHWlZ0SFJET1ljVzcxVWRDQWRxMnQ3MnBlc0w3RnFlajIwdnk1a0tya05tQTRhOVN2UExLUy1YNEE1U3Jwa3F0YU1hT1kxRHlqYmU0SWpyNW94WXFtSC1zNVh3WkxZWkdWZ0tKX3ZwdTdqeVZrVDBTMG5KQUhia2hXZkY3NmxfMmk5RDlZdw?oc=5',
      'content': "Google Case Study: Five Sigma's Clive Redefines Claims Management with LangChain and Vertex AI - Coverager"}]



```python

```

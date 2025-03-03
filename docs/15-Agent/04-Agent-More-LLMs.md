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

# Tool Calling Agent with More LLM Models

- Author: [JoonHo Kim](https://github.com/jhboyo)
- Design: []()
- Peer Review :
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSV-Loader.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSV-Loader.ipynb)


## Overview
LangChain is not limited to `OpenAI` models. It also supports implementations from diverse LLM providers such as `Anthropic`, `Google Gemini`, `Together.ai`, `Ollama`, and `Mistral`. This flexibility allows developers to leverage the unique characteristics and strengths of each model to create agents optimized specific requirements for their applications.

**Key Topics**

In this chapter, we will delve into the process of creating and executing tool-calling agents using various `LLMs`. Here are the key topics covered, we'll explore:

- Tool Selection: How agents choose the most suitable tools for specific tasks.
- `LLM` Integration: Integrating `LLMs` from `OpenAI` and other providers into LangChain to enable agent functionality.
- Agent Creation: Creating agents using LangChain's agent classes.
- Agent Execution: Executing agents to perform tasks.

Objectives
By the end of this chapter, you will be able to:

- How to create and execute tool-calling agents using various `LLMs`.
- Create automated workflows that call various tools using LangChain's agent classes.
- Combine multiple `LLMs` to implement agents with optimized performance.

Now, let’s explore how to maximize productivity using LangChain’s flexible agent framework. 🚀

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [List of LLMs Supporting Tool Calling](#list-of-llms-supporting-tool-calling)
- [Working with Multiple LLM Integrations in LangChain](#working-with-multiple-llm-integrations-in-langchain)
- [Creating tools](#creating-tools)
- [Generating Prompts for Agents](#generating-prompts-for-agents)
- [Generating an AgentExecutor, run it and review the results](#generating-an-agentexecutor-run-it-and-review-the-results)

### References

- [Tool Calling Agent](https://blog.langchain.dev/tool-calling-with-langchain/)
- [LangChain ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/)
- [LangChain ChatAnthropic](https://python.langchain.com/docs/integrations/chat/anthropic/)
- [LangChain ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/providers/google/)
- [LangChain ChatOllama](https://python.langchain.com/docs/integrations/providers/ollama/)
- [LangChain ChatTogether](https://python.langchain.com/docs/integrations/providers/together/)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.
- `langchain-ollama` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 

```python
%%capture --no-stderr
%pip install -qU langchain-opentutorial
%pip install -qU langchain-ollama==0.1.3
%pip install -qU feedparser
```

<pre class="custom">WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1737867095.902142  824462 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
    I0000 00:00:1737867098.495320  824462 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
    I0000 00:00:1737867101.236374  824462 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
</pre>

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
#if not load_dotenv():
set_env(
    {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "OLLAMA_API_KEY": "",
        "TOGETHER_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Tool Calling Agent with More LLM Models",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_ollama",
        "langchain_community",
        "langchain_core"
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">I0000 00:00:1737867103.792724  824462 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
</pre>


## List of LLMs Supporting Tool Calling

Before we proceed with the hands-on tutorial, you'll need to complete the following setup steps for each LLM you want to use:

1. Obtain an API Kye: Follow the provided link to requiest an API key for each `LLM` call.
2. Add the issued key to the `.env` file.


**Anthropic**

- [Anthropic API Key Issuance](https://console.anthropic.com/settings/keys)
- Add the issued key `ANTHROPIC_API_KEY` to `.env` file.


**Gemini**

- [Gemini API Key Issuance](https://aistudio.google.com/app/apikey?hl=ko)
- Add the issued key `GOOGLE_API_KEY` to `.env` file.


**Ollama**
- [List of Ollama Tool Calling Supported Models](https://ollama.com/search?c=tools)
- Ollama uses a different approach. Instead of API keys, you'll need to install Ollama itself. Follow the instructions here to install Ollama: [Ollama installation](https://ollama.com)
- This tutorial will use the [lama3.1 model](https://ollama.com/library/llama3.1)
- After installing Ollama, you can download it using the following commands: `ollama pull llama3.1`
- You can also download the `qwen2.5` model using the following command: `ollama pull qwen2.5`


**Together AI**
- [Together API Key Issuance](https://api.together.ai/)
- Add the issued key `TOGETHER_API_KEY` to `.env` file.


## Working with Multiple LLM Integrations in LangChain
This section guides you through integrating and configuring various `LLMs` in LangChain, allowing you to do experiments with different models from providers like `OpenAI`, `Anthropic`, `Google`, and others.

```python
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

# GPT-4o-mini
gpt = ChatOpenAI(model="gpt-4o-mini")

# Claude-3-5-sonnet
claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Gemini-1.5-pro-latest
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Llama-3.1-70B-Instruct-Turbo
llama = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
)

# Llama-3.1
ollama = ChatOllama(model="llama3.1", temperature=0)

# Qwen2.5 7B 
qwen = ChatOllama(
    model="qwen2.5:latest",
)
```

## Creating tools

Before creating tools, we will build some functions to fetch news from websites based on user's input keywords.

`_fetch_news(url: str, k: int = 3) -> List[Dict[str, str]]`: This funtion takes a URL as input and retrieves news articles from that source. The function returns a list of dictionaries.
 * Args: `url: str` is for fetching news articles. The `k: int = 3` (default: 3) is a number of news to fetch.
 * Return: `List[Dict[str, str]]` is a list of dictionaries that contains news title and link.

`_collect_news(news_list: List[Dict[str, str]] -> List[Dict[str, str]]`: This function return a sorted list of the same news items.
 * Args: `news_list: List[Dict[str, str]]` is a list of dictionaries that contains news information.
 * Return: `List[Dict[str, str]]` is a list of dictionaries containing the URL and the full contents.

`search_by_keyword(keyword: str, k: int = 3) -> List[Dict[str, str]]`: This funtion is the main entry point for searching news. It accepts a keyword and returns a list of dictionaries.
 * Args: `keyword: str` is a keyword to search. `k: int = 3`(default: 3) is a number of news to fetch.
 * Return: `List[Dict[str, str]]` is a list of dictionaries that contains the URL and contents.


```python
from typing import List, Dict, Optional
from urllib.parse import quote
import feedparser

class GoogleNews:

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        news_data = feedparser.parse(url)
        return [
            {"title": entry.title, "link": entry.link}
            for entry in news_data.entries[:k]
        ]
    
    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not news_list:
            print("There is no news for the keyword.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_by_keyword(
            self, keyword: Optional[str] = None, k: int = 3
        ) -> List[Dict[str, str]]:
            
            if keyword:
                encoded_keyword = quote(keyword)
                url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=en&gl=US&ceid=US:en"
            else:
                url = f"https://news.google.com/rss?hl=en&gl=US&ceid=US:en"

            news_list = self._fetch_news(url, k)
            return self._collect_news(news_list)
```

This set of functions enables a tool that can fetch relevant news from Google News Website based on user-provided input keywords.

Let's create tools.

```python
from langchain.tools import tool
from typing import List, Dict

@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)

print(f"Tool Name: {search_news.name}")
print(f"Tool Description: {search_news.description}")
```

<pre class="custom">Tool Name: search_news
    Tool Description: Search Google News by input keyword
</pre>




```python
tools = [search_news]
```

## Generating Prompts for Agents
A prompt is text that describes the task the model will perform whose input is the tool name and its role.


- `chat_history`: A variable that stores previous conversation history (can be omitted if multi-turn support is not required).
- `agent_scratchpad`: A variable for temporary storage used by the agent.
- `input`: The user's input.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant."
            "Make sure to use the `search_news` tool for searching keyword related news.",        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

Let's generate agents per each `LLM` basis.

```python
gpt_agent = create_tool_calling_agent(gpt, tools, prompt)
claude_agent = create_tool_calling_agent(claude, tools, prompt)
gemini_agent = create_tool_calling_agent(gemini, tools, prompt)
llama_agent = create_tool_calling_agent(llama, tools, prompt)
ollama_agent = create_tool_calling_agent(ollama, tools, prompt)
qwen_agent = create_tool_calling_agent(qwen, tools, prompt)
```

## Generating an AgentExecutor

Now, let's import `AgentExecutor`, run agents, and review the outputs.

```python
from langchain.agents import AgentExecutor

# execute gpt_agent
agent_executor = AgentExecutor(
    agent=gemini_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

result = agent_executor.invoke({"input": "Search the latest AI Investment news"})

print("Results of Agent Execution:")
print(result["output"])
```

<pre class="custom">
    
    > Entering new AgentExecutor chain...
    
    Invoking: `search_news` with `{'query': 'AI Investment'}`
    
    
    [{'url': 'https://news.google.com/rss/articles/CBMiqwFBVV95cUxOcU9jRHlEdXlNQnc1RmxFc3B5S2tqZmNSbjY0LVc3d3RSODlLUnpfcGc1T2RhcVJncC1oVUNFQ0xFbF8tNlZXQkRWSS1iVGJsc281OTlGRjEyLTAyaDg3dE5hV1o0TkJjVWxOWnZwb0JzLXZqcXpKUklJclVYMTE4dk9iQ0xyd0ZHOEtuSWVTWFNvN0VFZFpVWGRaaDN2MmxuZnRCLWJaSnJtNGs?oc=5', 'content': 'A $500bn investment plan says a lot about Trump’s AI priorities - The Economist'}, {'url': 'https://news.google.com/rss/articles/CBMihwFBVV95cUxQSDduczRjNDJkMFQwQk05aXMzQ21EY2hqZkxEdlpEZ1plX3NxalpGaEpwckdQM0tQa2ZqbkJGMkxReFFzeDFnRzY4U3dKMmdTUm1xeXpod1VoN1YzTzA4VDlHQVhzeXgxLTc3UjczVlMzdGhQWHZsMG5RdGZjY2pJRDBaUF9pWmc?oc=5', 'content': 'Tech titans bicker over $500bn AI investment announced by Trump - The Guardian'}, {'url': 'https://news.google.com/rss/articles/CBMilAFBVV95cUxOeGExNlIteFo5R09vclNhcHVKbjlnY010TGxmVWhsOW1LcWJidzFpSDNWUzNPbUlpM0l1bGd2a1FjRnFlMHBnbC1iR1FWVEU2dFRRczdpU3NYR0J1Y1lXNzNHeHgzMWxnSklPUTB6dTAzYzBDdDVfT3JyVnZkdVUydmg2c3RHWFI5MFFjNEl4cDVsZW5T?oc=5', 'content': 'Meta to invest up to $65 billion this year to power AI goals, Zuckerberg says - USA TODAY'}, {'url': 'https://news.google.com/rss/articles/CBMiugFBVV95cUxQSDc5RFFWN204N3J3X3lEdUF0RXVWOW1FNk9KVVZBNFQ4RndZTzVybFhURFVWVFRGLWJ1VGVsUGtTXzhLZVdRQ21CQ1VHUFoyWkhSZUJ6cFU4bUQ1dUlWX1p1c05TVkdnWk83UmJ6amQwbzlQTlNVSHl0alJfZmRKSTVJNUVfdmliSmo3anQ3cXlGUjdZUEwyY25pOGxmU3luRG5heVlsWHo0SDJBTWxGT3hjM0JhQ05xS0E?oc=5', 'content': 'Trump’s AI Push: Understanding The $500 Billion Stargate Initiative - Forbes'}, {'url': 'https://news.google.com/rss/articles/CBMingFBVV95cUxPRkhzc0UwM2l4N0pXSDFibkFvTnhOeVppSC15aTRRNW02VzhoWEhpVHp3d3gtb1p5NXNZLUd4TDJudjRSNFlnOE9BZUJtUm14UUlsMF9FNTBaZE1pNnZiaUxwdUI4RHBWS3FXMVFlVGxGQjJSbHRlcVNjV3pMWjRsQkhFV0FiWXNsRDVzOTBuTmZwUmJFcGE5QXIza3JVZw?oc=5', 'content': "'What Is Great For The Country Isn't Always What's Optimal For Your Companies': Sam Altman Lashes Out At Elon Musk Amid Stargate Squabble - Investor's Business Daily"}]Here are some of the latest news about AI investment:
    
    * A $500bn investment plan says a lot about Trump’s AI priorities - The Economist
    * Tech titans bicker over $500bn AI investment announced by Trump - The Guardian
    * Meta to invest up to $65 billion this year to power AI goals, Zuckerberg says - USA TODAY
    * Trump’s AI Push: Understanding The $500 Billion Stargate Initiative - Forbes
    * 'What Is Great For The Country Isn't Always What's Optimal For Your Companies': Sam Altman Lashes Out At Elon Musk Amid Stargate Squabble - Investor's Business Daily
    
    
    > Finished chain.
    Results of Agent Execution:
    Here are some of the latest news about AI investment:
    
    * A $500bn investment plan says a lot about Trump’s AI priorities - The Economist
    * Tech titans bicker over $500bn AI investment announced by Trump - The Guardian
    * Meta to invest up to $65 billion this year to power AI goals, Zuckerberg says - USA TODAY
    * Trump’s AI Push: Understanding The $500 Billion Stargate Initiative - Forbes
    * 'What Is Great For The Country Isn't Always What's Optimal For Your Companies': Sam Altman Lashes Out At Elon Musk Amid Stargate Squabble - Investor's Business Daily
    
</pre>

The following function generates and runs an `agent` using the provided `LLM` and outputs the results.

```python
def execute_agent(llm, tools, input_text, label):
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    result = executor.invoke({"input": input_text})
    print(f"Results from [{label}] ")

    if isinstance(result["output"], list) and len(result["output"]) > 0:
        for item in result["output"]:
            if "text" in item:
                print(item["text"])
    elif isinstance(result["output"], str):
        print(result["output"])
    else:
        print(result["output"])
```

Generate and run agents for each LLM and outputs the results.

```python
query = (
    "Search for news related to AI investment and write the results in the format of an Instagram post."
)
```

```python
# gpt
execute_agent(gpt, tools, query, "gpt")
```

<pre class="custom">Results from [gpt] 
    🌟 **AI Investment News Update!** 🌟
    
    🚀 Big moves in the AI world! Here’s what you need to know:
    
    1️⃣ **Trump’s $500bn AI Investment Plan** - A massive financial commitment that's shaping the future of AI! 💰🔍 [Read more](https://news.google.com/rss/articles/CBMiqwFBVV95cUxOcU9jRHlEdXlNQnc1RmxFc3B5S2tqZmNSbjY0LVc3d3RSODlLUnpfcGc1T2RhcVJncC1oVUNFQ0xFbF8tNlZXQkRWSS1iVGJsc281OTlGRjEyLTAyaDg3dE5hV1o0TkJjVWxOWnZwb0JzLXZqcXpKUklJclVYMTE4dk9iQ0xyd0ZHOEtuSWVTWFNvN0VFZFpVWGRaaDN2MmxuZnRCLWJaSnJtNGs?oc=5)
    
    2️⃣ **Tech Titans Clash** - The tech giants are bickering over the $500 billion investment announced by Trump! Who will come out on top? 🤔💥 [Check it out](https://news.google.com/rss/articles/CBMihwFBVV95cUxQSDduczRjNDJkMFQwQk05aXMzQ21EY2hqZkxEdlpEZ1plX3NxalpGaEpwckdQM0tQa2ZqbkJGMkxReFFzeDFnRzY4U3dKMmdTUm1xeXpod1VoN1YzTzA4VDlHQVhzeXgxLTc3UjczVlMzdGhQWHZsMG5RdGZjY2pJRDBaUF9pWmc?oc=5)
    
    3️⃣ **Meta's Bold Move** - Zuckerberg announces a staggering $65 billion investment this year to supercharge AI initiatives! 🔥🌐 [Learn more](https://news.google.com/rss/articles/CBMilAFBVV95cUxOeGExNlIteFo5R09vclNhcHVKbjlnY010TGxmVWhsOW1LcWJidzFpSDNWUzNPbUlpM0l1bGd2a1FjRnFlMHBnbC1iR1FWVEU2dFRRczdpU3NYR0J1Y1lXNzNHeHgzMWxnSklPUTB6dTAzYzBDdDVfT3JyVnZkdVUydmg2c3RHWFI5MFFjNEl4cDVsZW5T?oc=5)
    
    4️⃣ **Understanding the Stargate Initiative** - Dive into Trump’s ambitious $500 billion Stargate Initiative and its implications for AI! ✨🚀 [Discover here](https://news.google.com/rss/articles/CBMiugFBVV95cUxQSDc5RFFWN204N3J3X3lEdUF0RXVWOW1FNk9KVVZBNFQ4RndZTzVybFhURFVWVFRGLWJ1VGVsUGtTXzhLZVdRQ21CQ1VHUFoyWkhSZUJ6cFU4bUQ1dUlWX1p1c05TVkdnWk83UmJ6amQwbzlQTlNVSHl0alJfZmRKSTVJNUVfdmliSmo3anQ3cXlGUjdZUEwyY25pOGxmU3luRG5heVlsWHo0SDJBTWxGT3hjM0JhQ05xS0E?oc=5)
    
    5️⃣ **Altman vs. Musk** - A dramatic exchange between Sam Altman and Elon Musk over the implications of the Stargate Initiative! 🔥💬 [Read the details](https://news.google.com/rss/articles/CBMingFBVV95cUxPRkhzc0UwM2l4N0pXSDFibkFvTnhOeVppSC15aTRRNW02VzhoWEhpVHp3d3gtb1p5NXNZLUd4TDJudjRSNFlnOE9BZUJtUm14UUlsMF9FNTBaZE1pNnZiaUxwdUI4RHBWS3FXMVFlVGxGQjJSbHRlcVNjV3pMWjRsQkhFV0FiWXNsRDVzOTBuTmZwUmJFcGE5QXIza3JVZw?oc=5)
    
    Stay tuned for more updates on the world of AI! 🔍✨ #AIInvestment #TechNews #Innovation
</pre>

```python
# claude
execute_agent(claude, tools, query, "claude")
```

<pre class="custom">Results from [claude] 
    
    
    Now that I have the search results, I'll create an Instagram post format for you based on the AI investment news:
    
    📢 Breaking News: AI Investment Boom! 💰🤖
    
    🔥 Hot off the press: The AI world is buzzing with massive investment plans and tech titans are making big moves! Here's what you need to know:
    
    1️⃣ Trump's $500 Billion "Stargate Initiative" 🚀
    Former President Trump has announced a jaw-dropping $500 billion AI investment plan, sparking debates and setting priorities for the future of AI in the US.
    
    2️⃣ Tech Giants Clash 🥊
    The announcement has ignited a fierce debate among tech leaders. Elon Musk and Sam Altman are at odds over the initiative's impact on their companies vs. the country's interests.
    
    3️⃣ Meta's AI Power Play 💪
    Mark Zuckerberg isn't holding back! Meta plans to invest up to $65 billion this year alone to fuel their AI ambitions. Talk about going all-in! 🎰
    
    4️⃣ Global Impact 🌍
    The Economist weighs in on how Trump's massive AI investment plan could reshape priorities in the AI landscape worldwide.
    
    5️⃣ What's Next? 🔮
    As investments pour in and tech leaders take sides, the AI race is heating up like never before. Will this usher in a new era of innovation or widen the gap between tech giants?
    
    💡 What do you think about these massive AI investments? Are they a step in the right direction or cause for concern? Share your thoughts below! 👇
    
    #AIInvestment #TechNews #FutureOfAI #StargatePlan #MetaAI #TechGiants #InnovationRace
    
    📸: [Insert an relevant image here, perhaps a futuristic AI concept or a collage of tech leaders mentioned]
    
    Remember to stay informed and think critically about the impact of AI on our future! 🤔💡
</pre>

```python
# gemini
execute_agent(gemini, tools, query, "gemini")
```

<pre class="custom">Results from [gemini] 
    ⚡️ **AI Investment is Booming!** ⚡️
    
    The world of tech is buzzing with news about massive investments in Artificial Intelligence.  From Trump's proposed \$500 billion Stargate Initiative to Meta's \$65 billion commitment, the race to dominate the AI landscape is heating up.
    
    Want to stay ahead of the curve?  Check out these headlines:
    
    * **A $500bn investment plan says a lot about Trump’s AI priorities:** [Link to Economist article]
    * **Tech titans bicker over $500bn AI investment announced by Trump:** [Link to Guardian article]
    * **Meta to invest up to $65 billion this year to power AI goals, Zuckerberg says:** [Link to USA Today article]
    * **Trump’s AI Push: Understanding The $500 Billion Stargate Initiative:** [Link to Forbes article]
    * **'What Is Great For The Country Isn\'t Always What\'s Optimal For Your Companies\': Sam Altman Lashes Out At Elon Musk Amid Stargate Squabble:** [Link to Investor's Business Daily article]
    
    
    #AI #Investment #TechNews #FutureTech #Innovation #StargateInitiative
    
</pre>

```python
# llama3.1 70B (Together.ai)
execute_agent(
    llama,
    tools,
    "Search AI related news and write it in Instagram post format",
    "llama3.1 70B",
)
```

<pre class="custom">Results from [llama3.1 70B] 
    Here's a possible Instagram post based on the search results:
    
    "Hey everyone! 
    
    Want to stay up-to-date on the latest AI news? Here are some of the top stories from around the web:
    
    * How Chinese AI Startup DeepSeek Made a Model that Rivals OpenAI (via WIRED)
    * AI can now replicate itself — a milestone that has experts terrified (via Livescience.com)
    * Meta to Increase Spending to $65 Billion This Year in A.I. Push (via The New York Times)
    * 2 Artificial Intelligence (AI) Stocks That Could Make You a Millionaire (via The Motley Fool)
    * Trump Signs Executive Actions Related to Cryptocurrency, AI (via Bloomberg)
    
    Stay informed and ahead of the curve with these latest developments in the world of AI! #AI #ArtificialIntelligence #Technology #Innovation"
    
     Sources:
    https://news.google.com/rss/articles/CBMiYkFVX3lxTE9KdENSTTNfNzF1OU0xaXY5U1hhcTJqcmlPYUViZGlNV3E3TVJ6bXVqQW8yWG14TVZWaTVIM0FSV3BxdTAtWjA5U2JpNzRFQ0p5a3FPd1F4a0ZxUWpPQzN1aW5R?oc=5
    https://news.google.com/rss/articles/CBMiyAFBVV95cUxQcUxGdk5MRFFOT0hoN3RRX1drTmpja2NRYmhwRGZVNUpQRFllcGw0dHlMeDEtSS1SSHgxQ1dSNFV2NWZ5S2gxRmxXVFptWTE0bU9SMHpwenRla1F4NWFlVWY3NVZkZ2xjQmEwc2NBbzMtdU5zdHBmdDBKQ2NaTEhpN3pjNHpqVkJxc18wQzd2TUJCSVR4VjNEcTh2Ti15enpRUGRwSVVsSllRM1JjeGVzLWxERmstVzBnQS1JS1hwbGFOel9aOHJyOQ?oc=5
    https://news.google.com/rss/articles/CBMidkFVX3lxTE1DOHIyamJTVm95dDVNRGladzF5cVNLYnlBcVJXTTgzXzJULXFWODlNVG5jS0hDazJHdjRVMFd5YWJGSTZ0VU9QOUplSmpoWGxsQm5UUjg1c3hfTnVtUjF4NmZvb09FVDh1RFRzTTZ0QnpDU2MtYUE?oc=5
    https://news.google.com/rss/articles/CBMimAFBVV95cUxQLTNUNXFiNVdRS2x0MHRvZnJfQ1NCdjdwSExsVzFYYy1meWpCUVMzbXlBYm84d0paRklGUkJIaVdSc1Q0ZGZ3UnBiZk9TTFVQSDUyRmMzZXl4cF9uUTQ1Y281MEF1S2RqaTFUREt1X3ZIYnFmOXprcjFBLV9NdE10dWZUZEROU0ZhaXE0RzUyQndwZnhQWWVDVw?oc=5
    https://news.google.com/rss/articles/CBMisAFBVV95cUxPWWJCSWhMb0ctdDNxcWY0N3N1RDhDM2VyZ2xIOVRhUGR2MWFnSmNmaVZZNHBVTkowRVNlVXJVNzRmRGhSZUpRY1pKS2VIbk04ZEsxRUM5THJrSDJfdFRKYXpKYi1oTkVKalZ3WVkzU1cyeHQ1ZjJ6NTlBMXhXd2RfeG9OZUc0b3ZBYTJYMEdKZ0U1VWtuWTRyZm9TZnphWEMyR2ZGVTZlTkFkUFNtTEtKZw?oc=5
</pre>

```python
# llama3.1 8B (ollama)
execute_agent(ollama, tools, query, "llama3.1(Ollama)")
```

<pre class="custom">Results from [llama3.1(Ollama)] 
    "Breaking News!
    
    The future of AI is looking bright! 
    
    Did you know that a $500 billion investment plan has been announced by Trump, prioritizing AI development? This massive investment aims to propel the US ahead in the global AI race.
    
    But what does this mean for tech titans like Meta and Google? They're already investing big in AI, with plans to spend up to $65 billion this year!
    
    And it's not just about the money - it's also about the vision. Trump's Stargate initiative aims to make the US a leader in AI innovation.
    
    But not everyone is on board. Elon Musk and Sam Altman are at odds over the plan, with some questioning its feasibility.
    
    Stay tuned for more updates on this exciting development!
    
    #AI #Investment #FutureOfTech"
</pre>

```python
# qwen2.5 7B (ollama)
execute_agent(qwen, tools, query, "qwen2.5(Ollama)")
```

<pre class="custom">Results from [qwen2.5(Ollama)] 
    🌟 [AI Investment News] 🌟
    
    A $500 billion investment plan says a lot about Trump’s AI priorities. #AI #Investment #TechNews 
    
    ✨ Tech titans bicker over the $500 billion AI investment announced by Trump, as per The Guardian. #AIInvestment #TechTitans
    
    🚀 Meta is planning to invest up to $65 billion this year to power its AI goals, according to USA TODAY. #Meta #AI #BusinessNews
    
    💡 Understanding the $500 billion Stargate initiative: Trump’s push for AI, as explained by Forbes. #StargateInitiative #AIInvestment
    
    📣 Sam Altman of Y Combinator lashes out at Elon Musk amid the Stargate squabble. Investor's Business Daily shares his thoughts on this tech controversy. #SamAltman #ElonMusk #TechControversy
    
    💡 These are some interesting insights and updates in AI investment news! Stay tuned for more developments. #AIUpdate #TechnologyTrends
</pre>

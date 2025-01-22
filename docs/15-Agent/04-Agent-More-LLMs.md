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
It is not limited to `OpenAI` models but also supports implementations from diverse LLM providers such as `Anthropic`, `Google Gemini`, `Together.ai`, `Ollama`, and `Mistral`. This allows developers to leverage the unique characteristics and strengths of each model to create agents optimized for the specific requirements of their applications.

Key Topics
In this chapter, we will delve into the process of creating and executing tool-calling agents using various `LLMs`. The key topics covered include:

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
- [Using Multiple LLM Integrations with LangChain](#using-multiple-llm-integrations-with-langchain)
- [Define the tool](#define-the-tool)
- [Generate Prompts for Agents](#generate-prompts-for-agents)
- [Generate an AgentExecutor, run it and review the results](#generate-an-agentexecutor-run-it-and-review-the-results)

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

```python
# Set environment variables
from langchain_opentutorial import set_env

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


## List of LLMs Supporting Tool Calling

To proceed with the hands-on tutorial, the following setup steps are required

1. Issue API keys for each `LLM` call.
2. Add the issued key to the `.env` file.


**Anthropic**

- [Anthropic API Key Issuance](https://console.anthropic.com/settings/keys)
- Add the issued key to the `.env` file as `ANTHROPIC_API_KEY`.


**Gemini**

- [Gemini API Key Issuance](https://aistudio.google.com/app/apikey?hl=ko)
- Add the issued key to the `.env` file as `GOOGLE_API_KEY`.


**Ollama**
- [List of Ollama Tool Calling Supported Models](https://ollama.com/search?c=tools)
- Instead of issuing an API Key, [Ollama installation](https://ollama.com) is required to proceed with the ollama tutorial.
- We will use [lama3.1 Model](https://ollama.com/library/llama3.1) for This Tutorial for this tutorial.
- After installing Ollama, run the following commands in the terminal to download the models `ollama pull llama3.1` and `ollama pull qwen2.5`.


**Together AI**
- [Together API Key Issuance](https://api.together.ai/)
- Add the issued key to the `.env` file as `TOGETHER_API_KEY`.

## Using Multiple LLM Integrations with LangChain
This walks you through integrating and configuring various `LLMs` using LangChain, allowing you to experiment with different models from providers like `OpenAI`, `Anthropic`, `Google`, and others.

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

## Define the tool

Before defining the tool, we will build some functions to collect and fetch news from Website for keyword that user input.

 `_fetch_news` funtion gets news from given url and return the result as a list of dictionaries.
 * `Args:` url (str) is a url for fetching news. k (int) is a number of news to fetch.(default: 3)
 * `Return:` List[Dict[str, str]] is a list of dictionaries that contains news title and link.


 `_collect_news` return a sorted list of news items.
 * `Args:` news_list (List[Dict[str, str]]) is a list of dictionaries that contains news information.
 * `Return:` List[Dict[str, str]] is a list of dictionaries that contains url and contents.

`search_by_keyword` funtion searches news using keyword and return the result as a list of dictionaries.
 * `Args:` keyword (str) is a keyword to search. k (int) is a number of news to fetch.(default: 3)
 * `Return:` List[Dict[str, str]] is a list of dictionaries that contains url and contents.


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

Fetch News for keyword that user input from Google News Website.

```python
from langchain.tools import tool
from typing import List, Dict

# Define the tool
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


Define Tools

```python
tools = [search_news]
```

## Generate Prompts for Agents
Prompt is a text that describes the task the model will perform.(Input the tool name and role)


- `chat_history` : A variable that stores previous conversation history (can be omitted if multi-turn support is not required).
- `agent_scratchpad` : A variable for temporary storage used by the agent.
- `input` : The user's input.

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

Generate an `Agent` based on an `LLM` (Large Language Model)

```python
gpt_agent = create_tool_calling_agent(gpt, tools, prompt)
claude_agent = create_tool_calling_agent(claude, tools, prompt)
gemini_agent = create_tool_calling_agent(gemini, tools, prompt)
llama_agent = create_tool_calling_agent(llama, tools, prompt)
ollama_agent = create_tool_calling_agent(ollama, tools, prompt)
qwen_agent = create_tool_calling_agent(qwen, tools, prompt)
```

## Generate an AgentExecutor, run it and review the results


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
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `search_news` with `{'query': 'AI Investment'}`
    
    
    [0m[36;1m[1;3m[{'url': 'https://news.google.com/rss/articles/CBMikgFBVV95cUxOakdlR2ltVnh5WjRsa0UzaUJ1YkVpR1Z0Z0tpVWRKbXJ4RkZpVkdKTHJGcUxMblpsUDA4UVk5engxWktDY2J1UG1YQzVTQm9fcnp1d0hyV083YVZkejZUVXJMem9tZ1ZWUzllNDNpUXBfSElhU01sQm9RUW1jZE9vZGlpZEVDZ1lnU2hnS192V0Y3dw?oc=5', 'content': 'The Best AI Stocks to Invest $500 in This Year - The Motley Fool'}, {'url': 'https://news.google.com/rss/articles/CBMiswFBVV95cUxQTGZVQnZlaFBBUVpCTVVqQzhvM3AyNzR5QXRIbUczM1FZQzMwTmpIZUxIelB2TUlyeGxGTVhmMFJFa3V4MXA0TklYZEpLcXZDVlNoQmI4RWZibkFka0JudTREZ2s2VlduTUp3OExkcjA3Z01tX0hCS0JuQkpoUlp6Nm1IRnByR2FnZEtlcUNDZFdKUWtKZGR5aTZYWEp5SnNEZ19nUi1zN1RhTFdxUFNESk5RMA?oc=5', 'content': 'Health AI investments are off to a roaring start in 2025 - STAT'}, {'url': 'https://news.google.com/rss/articles/CBMijAFBVV95cUxQZ0FnbS1MOWJYeFhtWE1FSGNrQjgwZ3hqbnpLNXpnOEpaR192LW5FV1NVOTBQUUlNVEhTRHlfd3VoRnJFRkl6M0pndWJwemlMUFdPa25PRWt6LWh1Uk4ta2RVQV9lb0Vjb2ZGVlNJWXQxVlNtWF9uTEFmZnFlemxfT2Z3bEYzcnJkRl9CNQ?oc=5', 'content': 'Microsoft’s Stock Revival Hinges on Showing Growth From AI Binge - Yahoo Finance'}, {'url': 'https://news.google.com/rss/articles/CBMiqwFBVV95cUxNWE0wMHdXSDN3aTlMal9aTGpkaUdkOEVmRHhxajFWRXJiOVNweXV0M2RHSWFyWDdwSWYwSmp5UVlva1hFTFRyOXRZY050X25JbWlDcDgtTHlya1Zha2EtMGlvVFEwcmEzblUtLUZhby1uMks1eDlCdGY4ZkV0dm5ES1BYTlM3cXhYeG8wTDd6NlZNWDFrNm9fNkp0bHJkRm1IRXRzbXNwRW5CZTg?oc=5', 'content': 'Palantir in Talks to Invest in Drone Startup at $5 Billion Valuation - The Information'}, {'url': 'https://news.google.com/rss/articles/CBMiiAFBVV95cUxNWjFlOHRHa3N3TVpadWlSTjlKeFNaX3g3MVhyMzlHNzNMbXEzb2tlNV9fRXUwUTFVWWxYZm9NVFhoMlFYdkExS1FEVEVXdWdlNHR5NFJTMkFNcVR2TkxBTjR2UzBTeG9XUGhLd2RFa1VPMUNsOHBiWWtQWWsxRkVKNmd3cXd3MDBs?oc=5', 'content': 'Best AI Stocks to Invest in Now - Morningstar'}][0m[32;1m[1;3mHere are some of the latest news about AI investment:
    
    * The Best AI Stocks to Invest $500 in This Year - The Motley Fool
    * Health AI investments are off to a roaring start in 2025 - STAT
    * Microsoft’s Stock Revival Hinges on Showing Growth From AI Binge - Yahoo Finance
    * Palantir in Talks to Invest in Drone Startup at $5 Billion Valuation - The Information
    * Best AI Stocks to Invest in Now - Morningstar
    [0m
    
    [1m> Finished chain.[0m
    Results of Agent Execution:
    Here are some of the latest news about AI investment:
    
    * The Best AI Stocks to Invest $500 in This Year - The Motley Fool
    * Health AI investments are off to a roaring start in 2025 - STAT
    * Microsoft’s Stock Revival Hinges on Showing Growth From AI Binge - Yahoo Finance
    * Palantir in Talks to Invest in Drone Startup at $5 Billion Valuation - The Information
    * Best AI Stocks to Invest in Now - Morningstar
    
</pre>

Run the agent using various `LLMs` (Large Language Models).

The following is a function that generates and runs an Agent using the provided `LLM` and outputs the results.

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
    🌟 **AI Investment News Update** 🌟
    
    🚀 **The Best AI Stocks to Invest $500 in This Year** 📈  
    Check out the latest insights from The Motley Fool on which AI stocks are worth your investment! [Read more](https://news.google.com/rss/articles/CBMikgFBVV95cUxOakdlR2ltVnh5WjRsa0UzaUJ1YkVpR1Z0Z0tpVWRKbXJ4RkZpVkdKTHJGcUxMblpsUDA4UVk5engxWktDY2J1UG1YQzVTQm9fcnp1d0hyV083YVZkejZUVXJMem9tZ1ZWUzllNDNpUXBfSElhU01sQm9RUW1jZE9vZGlpZEVDZ1lnU2hnS192V0Y3dw?oc=5)
    
    💡 **Health AI investments are off to a roaring start in 2025** 🏥  
    Discover how health tech is booming with AI investments this year! [Read full article](https://news.google.com/rss/articles/CBMiswFBVV95cUxQTGZVQnZlaFBBUVpCTVVqQzhvM3AyNzR5QXRIbUczM1FZQzMwTmpIZUxIelB2TUlyeGxGTVhmMFJFa3V4MXA0TklYZEpLcXZDVlNoQmI4RWZibkFka0JudTREZ2s2VlduTUp3OExkcjA3Z01tX0hCS0JuQkpoUlp6Nm1IRnByR2FnZEtlcUNDZFdKUWtKZGR5aTZYWEp5SnNEZ19nUi1zN1RhTFdxUFNESk5RMA?oc=5)
    
    📊 **Microsoft’s Stock Revival Hinges on Showing Growth From AI Binge** 💻  
    Learn how Microsoft plans to leverage AI for its stock revival! [More info here](https://news.google.com/rss/articles/CBMijAFBVV95cUxQZ0FnbS1MOWJYeFhtWE1FSGNrQjgwZ3hqbnpLNXpnOEpaR192LW5FV1NVOTBQUUlNVEhTRHlfd3VoRnJFRkl6M0pndWJwemlMUFdPa25PRWt6LWh1Uk4ta2RVQV9lb0Vjb2ZGVlNJWXQxVlNtWF9uTEFmZnFlemxfT2Z3bEYzcnJkRl9CNQ?oc=5)
    
    🛩️ **Palantir in Talks to Invest in Drone Startup at $5 Billion Valuation** 🚁  
    Exciting developments in the drone industry as Palantir eyes a significant investment! [Read more](https://news.google.com/rss/articles/CBMiqwFBVV95cUxNWE0wMHdXSDN3aTlMal9aTGpkaUdkOEVmRHhxajFWRXJiOVNweXV0M2RHSWFyWDdwSWYwSmp5UVlva1hFTFRyOXRZY050X25JbWlDcDgtTHlya1Zha2EtMGlvVFEwcmEzblUtLUZhby1uMks1eDlCdGY4ZkV0dm5ES1BYTlM3cXhYeG8wTDd6NlZNWDFrNm9fNkp0bHJkRm1IRXRzbXNwRW5CZTg?oc=5)
    
    📊 **Best AI Stocks to Invest in Now** 💸  
    Find out which AI stocks are making waves right now! [Read more](https://news.google.com/rss/articles/CBMiiAFBVV95cUxNWjFlOHRHa3N3TVpadWlSTjlKeFNaX3g3MVhyMzlHNzNMbXEzb2tlNV9fRXUwUTFVWWxYZm9NVFhoMlFYdkExS1FEVEVXdWdlNHR5NFJTMkFNcVR2TkxBTjR2UzBTeG9XUGhLd2RFa1VPMUNsOHBiWWtQWWsxRkVKNmd3cXd3MDBs?oc=5)
    
    #AIInvestment #StockMarket #TechNews #InvestSmart #FutureOfAI
</pre>

```python
# claude
execute_agent(claude, tools, query, "claude")
```

<pre class="custom">Results from [claude] 
    
    
    Great! Now that I have the latest news about AI investment, I'll create an Instagram post format for you based on this information.
    
    📱 Instagram Post: AI Investment Buzz 🚀💰
    
    🔥 Hot off the press: AI investments are making waves in 2025! 🌊
    
    Here's what's trending in the world of AI stocks and investments:
    
    1. 💼 The Motley Fool suggests the best AI stocks to invest $500 in this year. Smart money moves! 💡
    
    2. 🏥 Health AI investments are off to a roaring start in 2025. The future of healthcare is here! 🩺
    
    3. 🖥️ Microsoft's stock revival hinges on showing growth from their AI binge. Will their bet pay off? 🤔
    
    4. 🚁 Palantir in talks to invest in a drone startup valued at $5 billion! AI takes flight! ✈️
    
    5. 💎 Morningstar reveals the best AI stocks to invest in now. Time to update your portfolio? 📊
    
    🗣️ What's your take on these AI investment trends? Are you ready to ride the AI wave? 🏄‍♂️
    
    #AIInvestment #TechStocks #FutureOfFinance #InvestInAI #TechTrends2025
    
    👉 Swipe for more details and let us know your thoughts in the comments below! 💬
    
    ---
    
    Remember, always do your own research and consult with financial advisors before making investment decisions. Happy investing, tech enthusiasts! 🤖💼
</pre>

```python
# gemini
execute_agent(gemini, tools, query, "gemini")
```

<pre class="custom">Results from [gemini] 
    ⚡️ **AI Investment is Booming!** ⚡️
    
    The world of finance is buzzing with the potential of Artificial Intelligence.  From healthcare to tech giants, AI is attracting serious investment. Here's a quick rundown:
    
    * **Motley Fool** highlights the best AI stocks for investing $500 this year.
    * **STAT** reports that health AI investments are off to a roaring start in 2025.
    * **Yahoo Finance** discusses how Microsoft's stock revival hinges on demonstrating growth from its AI investments.
    * **The Information** reveals Palantir is in talks to invest in a drone startup at a $5 billion valuation.
    * **Morningstar** shares its picks for the best AI stocks to invest in now.
    
    #AI #Investment #Tech #Future #Finance #Stocks #Innovation
    
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
    "Hey everyone! Did you know that AI caused layoffs in NY? Employers may have to disclose this information. Additionally, student and teacher AI use continued to climb in the 2023-24 school year. Microsoft has also introduced a new paradigm of materials design with generative AI called MatterGen. Furthermore, Connecticut College has launched an AI initiative called AI@Conn. And if you're looking to invest in AI stocks, here are the top 4 no-brainer AI stocks to buy for 2025. #AI #News #Tech"
</pre>

```python
# llama3.1 8B (ollama)
execute_agent(ollama, tools, query, "llama3.1(Ollama)")
```

<pre class="custom">Results from [llama3.1(Ollama)] 
    "Breaking News!
    
    Investing in AI is on the rise! Here are some top picks:
    
     Microsoft's stock revival hinges on showing growth from AI binge (Yahoo Finance)
     Health AI investments are off to a roaring start in 2025 (STAT)
     Palantir in talks to invest in drone startup at $5 billion valuation (The Information)
     Best AI stocks to invest $500 in this year (The Motley Fool)
    
    Don't miss out on the opportunity to invest in the future of technology! #AIinvestment #stockmarket #futureoftech"
</pre>

```python
# qwen2.5 7B (ollama)
execute_agent(qwen, tools, query, "qwen2.5(Ollama)")
```

<pre class="custom">Results from [qwen2.5(Ollama)] 
    🌟 [IG Post] 🌟
    
    🚀 *AI Investment Update* 🚀
    
    Here's the latest buzz in the world of artificial intelligence and investments! Dive into these must-read articles:
    
    1. **The Best AI Stocks to Invest $500 in This Year** - The Motley Fool
       - Read about which companies are leading the pack in this rapidly evolving sector.
       
    2. **Health AI Investments Are Off to a Roaring Start in 2025** - STAT
       - Discover how AI is transforming healthcare and driving growth in the health industry.
    
    3. **Microsoft’s Stock Revival Hinges on Showing Growth From AI Binge** - Yahoo Finance
       - Explore Microsoft's ambitious plans as they integrate AI into their business strategy.
    
    4. **Palantir in Talks to Invest in Drone Startup at $5 Billion Valuation** - The Information
       - Keep an eye on new partnerships and investments that could shape the future of technology.
    
    5. **Best AI Stocks to Invest in Now** - Morningstar
       - Get tips from experts on picking the right stocks for your portfolio.
    
    💡 *Stay ahead of the curve with these insights!*
    
    #AI #Investment #TechTrends
    
    ---
    
    *Let us know which article you found most interesting!*
</pre>

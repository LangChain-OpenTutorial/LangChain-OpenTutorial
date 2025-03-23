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

# Building a Agent API with LangServe: Integrating Currency Exchange and Trip Planning

- Author: [Hwayoung Cha](https://github.com/forwardyoung)
- Design: []()
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

This tutorial guides you through creating a Agent API using `LangServe`, enabling you to build intelligent and dynamic applications. You'll learn how to leverage LangChain agents and deploy them as production-ready APIs with ease. Discover how to define tools, orchestrate agent workflows, and expose them via a simple and scalable REST interface.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [LangServe](#langserve)
- [Implementing a Travel Planning Agent](#implementing-a-travel-planning-agent)
- [Implementing a Currency exchange agent](#implementing-a-currency-exchange-agent)
- [Testing in the LangServe Playground](#testing-in-the-langserve-playground)



## References

- [LangServe](https://python.langchain.com/docs/langserve/)
- [FreecurrencyAPI](https://freecurrencyapi.com/docs/)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial sse_starlette uvicorn
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [   "langchain_openai",
        "langserve",
        "sse_starlette",
        "uvicorn"
    ],
    verbose=False,
    upgrade=False,
)
```

You can alternatively set API keys in .env file and load it.

[Note] This is not necessary if you've already set API keys in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "FREECURRENCY_API_KEY": ""
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## LangServe

LangServe is a tool that allows you to easily deploy LangChain runnables and chains as REST APIs. It integrates with FastAPI and uses Pydantic for data validation.

## Implementing a Travel Planning Agent

This section demonstrates how to implement a travel planning agent. This agent suggests customized travel plans based on the user's travel requirements.

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langserve import add_routes
from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel, Field

# Define input/output models
class TravelPlanRequest(BaseModel):
    """Travel planning request structure"""
    destination: str = Field(..., description="City or country to visit")
    duration: int = Field(..., description="Number of days for the trip")
    interests: List[str] = Field(
        default_factory=list,
        description="List of interests (e.g., ['food', 'culture', 'history'])"
    )

class TravelPlanResponse(BaseModel):
    """Travel planning response structure"""
    itinerary: List[str]
    recommendations: List[str]
    estimated_budget: str

@tool
def get_travel_suggestions(destination: str, duration: int, interests: str) -> str:
    """Generates travel suggestions based on the destination, duration, and interests."""
    # In a real implementation, you might use a travel API or database
    return f"Here's a {duration}-day itinerary for {destination} focusing on {interests}..."

llm = ChatOpenAI(model="gpt-4.0")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel planning assistant."),
    ("human", "Plan a trip to {destination} for {duration} days with interests in {interests}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
tools = [get_travel_suggestions]

agent = create_openai_functions_agent(llm, tools, prompt)
travel_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

app = FastAPI()
add_routes(
    app,
    travel_executor,
    path="/travel-planner",
    input_type=TravelPlanRequest,
    output_type=TravelPlanResponse
)
```

## Implementing a Currency exchange agent

This section shows how to implement a currency exchange agent. This agent performs currency conversions using real-time exchange rate information.

```python
import os
import requests
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class CurrencyExchangeRequest(BaseModel):
    """Currency exchange request structure"""
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency code (e.g., USD)")
    to_currency: str = Field(..., description="Target currency code (e.g., EUR)")

    @field_validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

    @field_validator('from_currency', 'to_currency')
    def currency_must_be_valid(cls, v):
        if len(v) != 3:
            raise ValueError('Currency code must be 3 characters')
        return v.upper()

class CurrencyExchangeResponse(BaseModel):
    """Currency exchange response structure"""
    converted_amount: float
    exchange_rate: float
    timestamp: str
    from_currency: str
    to_currency: str

API_KEY = os.getenv("FREECURRENCY_API_KEY")

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Gets the current exchange rate between two currencies."""
    url = f"https://api.freecurrencyapi.com/v1/latest"
    params = {
        "apikey": API_KEY,
        "base_currency": from_currency,
        "currencies": to_currency
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data['data'][to_currency]

llm = ChatOpenAI(model="gpt-4.0")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful currency exchange assistant."),
    ("human", "Convert {amount} {from_currency} to {to_currency}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
tools = [get_exchange_rate]

agent = create_openai_functions_agent(llm, tools, prompt)
currency_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

add_routes(
    app,
    currency_executor,
    path="/currency-exchange",
    input_type=CurrencyExchangeRequest,
    output_type=CurrencyExchangeResponse
)
```

## Testing in the LangServe Playground

LangServe provides a playground for easily testing the implemented agents. This allows you to directly verify and debug the API's behavior.

```python
import nest_asyncio
import uvicorn

nest_asyncio.apply()

uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [25888]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
</pre>

    
         __          ___      .__   __.   _______      _______. _______ .______     ____    ____  _______
        |  |        /   \     |  \ |  |  /  _____|    /       ||   ____||   _  \    \   \  /   / |   ____|
        |  |       /  ^  \    |   \|  | |  |  __     |   (----`|  |__   |  |_)  |    \   \/   /  |  |__
        |  |      /  /_\  \   |  . `  | |  | |_ |     \   \    |   __|  |      /      \      /   |   __|
        |  `----./  _____  \  |  |\   | |  |__| | .----)   |   |  |____ |  |\  \----.  \    /    |  |____
        |_______/__/     \__\ |__| \__|  \______| |_______/    |_______|| _| `._____|   \__/     |_______|
        
    LANGSERVE: Playground for chain "/currency-exchange/" is live at:
    LANGSERVE:  │
    LANGSERVE:  └──> /currency-exchange/playground/
    LANGSERVE:
    LANGSERVE: Playground for chain "/travel-planner/" is live at:
    LANGSERVE:  │
    LANGSERVE:  └──> /travel-planner/playground/
    LANGSERVE:
    LANGSERVE: See all available routes at /docs/
    

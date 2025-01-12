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

# ConversationSummaryMemory

- Author: [Jinu Cho](https://github.com/jinucho)
- Design: []()
- Peer Review : [Secludor](https://github.com/Secludor), [Shinar12](https://github.com/Shinar12)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb)

## Overview

This tutorial covers how to summarize and manage conversation history using `LangChain`.  

`ConversationSummaryMemory` optimizes memory usage by summarizing conversation content, allowing efficient management of long conversation histories.  

In this tutorial, we will demonstrate how to implement conversation summarization functionality using LangChain's `ConversationSummaryMemory`.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Conversation Summary Memory](#conversation-summary-memory)
- [Conversation Summary Buffer Memory](#conversation-summary-buffer-memory)

### References

- [LangChain ConversationSummaryMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary.ConversationSummaryMemory.html)
- [LangChain ConversationSummaryBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html)
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
        "langchain_community",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "06-ConversationSummaryMemory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Load environment variables
# Reload any variables that need to be overwritten from the previous cell

from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Conversation Summary Memory

Now, let's explore how to use a more complex memory type: `ConversationSummaryMemory`.

This type of memory generates **summaries of conversations over time**, which can be useful for compressing conversational information as it progresses.

The conversation summary memory summarizes the conversation as it continues and **saves the current summary in memory** .

This memory can then be used to insert the summarized conversation history into prompts or chains.

It is most useful for long conversations where retaining the entire message history in the prompt would consume too many tokens.

Let's `create a ConversationSummaryMemory`.

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0), return_messages=True)
```

<pre class="custom">/var/folders/c4/0f7nfvt16ln8630csjtkk_1w0000gn/T/ipykernel_1536/3606106198.py:4: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory = ConversationSummaryMemory(
</pre>

It allows saving multiple conversations.

```python
memory.save_context(
    inputs={"human": "What is the price of the Europe travel package?"},
    outputs={
        "ai": "The base price for the 14-night, 15-day Europe package is €3,500. This price includes airfare, hotel accommodations, and admission fees to designated tourist attractions. Additional costs may vary depending on optional tours or personal expenses."
    },
)
memory.save_context(
    inputs={"human": "What are the main tourist attractions included in the trip?"},
    outputs={
        "ai": "This trip includes visits to famous European landmarks such as the Eiffel Tower in Paris, the Colosseum in Rome, the Brandenburg Gate in Berlin, and Rhine Falls in Zurich. You'll comprehensively experience iconic sites in each city."
    },
)
memory.save_context(
    inputs={"human": "Is travel insurance included?"},
    outputs={
        "ai": "Yes, basic travel insurance is provided for all travelers. This insurance includes medical expense coverage and support in emergency situations. Enhanced coverage is available upon request."
    },
)
memory.save_context(
    inputs={
        "human": "Can I upgrade my flight seat to business class? How much does it cost?"
    },
    outputs={
        "ai": "Upgrading your flight seat to business class is possible. The upgrade cost is approximately €1,200 round-trip. Business class offers benefits such as wider seats, premium in-flight meals, and additional baggage allowance."
    },
)
memory.save_context(
    inputs={"human": "What is the hotel rating included in the package?"},
    outputs={
        "ai": "The package includes accommodation in 4-star hotels. Each hotel offers comfort and convenience, with central locations providing easy access to tourist sites. All hotels are equipped with excellent services and amenities."
    },
)
memory.save_context(
    inputs={"human": "Can you provide more details about the meal options?"},
    outputs={
        "ai": "This travel package includes daily breakfast served at the hotel. Lunch and dinner are not included, giving travelers the flexibility to explore and enjoy a variety of local cuisines. Additionally, a list of recommended restaurants in each city is provided to help maximize your culinary experience."
    },
)
memory.save_context(
    inputs={"human": "How much is the deposit for booking the package? What is the cancellation policy?"},
    outputs={
        "ai": "A deposit of €500 is required when booking the package. The cancellation policy allows a full refund if canceled at least 30 days before the booking date. After that, the deposit becomes non-refundable. If canceled within 14 days of the travel start date, 50% of the total cost will be charged, and after that, the full cost will be non-refundable."
    },
)
```

You can check the history of the saved memory.  

It provides a concise summary of all previous conversations.

```python
# Check saved memory.
print(memory.load_memory_variables({})["history"])
```

<pre class="custom">[SystemMessage(content='The human asks about the price of the Europe travel package. The AI responds that the base price for the 14-night, 15-day Europe package is €3,500, which includes airfare, hotel accommodations, and admission fees to designated tourist attractions. Additional costs may vary depending on optional tours or personal expenses. The trip includes visits to famous European landmarks such as the Eiffel Tower in Paris, the Colosseum in Rome, the Brandenburg Gate in Berlin, and Rhine Falls in Zurich, offering a comprehensive experience of iconic sites in each city. Basic travel insurance is included, covering medical expenses and emergency support, with enhanced coverage available upon request. The human inquires about upgrading the flight seat to business class, and the AI informs that it is possible at an additional cost of approximately €1,200 round-trip, offering benefits like wider seats, premium in-flight meals, and extra baggage allowance. The human then asks about the hotel rating included in the package, and the AI states that the package includes accommodation in 4-star hotels, which offer comfort, convenience, and central locations with excellent services and amenities. The human asks for more details about the meal options, and the AI explains that the package includes daily breakfast at the hotel, while lunch and dinner are not included, allowing travelers to explore local cuisines. A list of recommended restaurants in each city is provided to enhance the culinary experience. The human inquires about the deposit and cancellation policy, and the AI explains that a deposit of €500 is required when booking the package. The cancellation policy allows a full refund if canceled at least 30 days before the booking date. After that, the deposit becomes non-refundable, and if canceled within 14 days of the travel start date, 50% of the total cost will be charged, with the full cost being non-refundable after that.', additional_kwargs={}, response_metadata={})]
</pre>

## Conversation Summary Buffer Memory

`ConversationSummaryBufferMemory` combines two key ideas:

It retains a buffer of recent conversation history in memory while compiling older interactions into a summary without fully flushing them.

Instead of using the number of interactions, it determines when to flush the conversation based on the **token length**.

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

llm = ChatOpenAI()

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,  # Set the token length threshold for summarization.
    return_messages=True,
)
```

<pre class="custom">/var/folders/c4/0f7nfvt16ln8630csjtkk_1w0000gn/T/ipykernel_1536/2100373999.py:6: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory = ConversationSummaryBufferMemory(
</pre>

First, let's save a single conversation and then check the memory.

```python
memory.save_context(
    inputs={"human": "What is the price of the Europe travel package?"},
    outputs={
        "ai": "The base price for the 14-night, 15-day Europe package is €3,500. This price includes airfare, hotel accommodations, and admission fees to designated tourist attractions. Additional costs may vary depending on optional tours or personal expenses."
    },
)
```

Check the conversation saved in memory.

At this point, the conversation is not yet summarized because it hasn't reached the **200-token** threshold.

```python
# Check the saved conversation history in memory
memory.load_memory_variables({})["history"]
```




<pre class="custom">[HumanMessage(content='What is the price of the Europe travel package?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='The base price for the 14-night, 15-day Europe package is €3,500. This price includes airfare, hotel accommodations, and admission fees to designated tourist attractions. Additional costs may vary depending on optional tours or personal expenses.', additional_kwargs={}, response_metadata={})]</pre>



Let's add more conversations to exceed the 200-token limit.

```python
memory.save_context(
    inputs={"human": "What are the main tourist attractions included in the trip?"},
    outputs={
        "ai": "This trip includes visits to famous European landmarks such as the Eiffel Tower in Paris, the Colosseum in Rome, the Brandenburg Gate in Berlin, and Rhine Falls in Zurich. You'll comprehensively experience iconic sites in each city."
    },
)
memory.save_context(
    inputs={"human": "Is travel insurance included?"},
    outputs={
        "ai": "Yes, basic travel insurance is provided for all travelers. This insurance includes medical expense coverage and support in emergency situations. Enhanced coverage is available upon request."
    },
)
memory.save_context(
    inputs={
        "human": "Can I upgrade my flight seat to business class? How much does it cost?"
    },
    outputs={
        "ai": "Upgrading your flight seat to business class is possible. The upgrade cost is approximately €1,200 round-trip. Business class offers benefits such as wider seats, premium in-flight meals, and additional baggage allowance."
    },
)
memory.save_context(
    inputs={"human": "What is the hotel rating included in the package?"},
    outputs={
        "ai": "The package includes accommodation in 4-star hotels. Each hotel offers comfort and convenience, with central locations providing easy access to tourist sites. All hotels are equipped with excellent services and amenities."
    },
)
```

Check the stored conversation history.  

The most recent conversation remains unsummarized, while the previous conversations are stored as a summary.

```python
# Check the saved conversation history in memory
memory.load_memory_variables({})["history"]
```




<pre class="custom">[SystemMessage(content="The human asks for the price of the Europe travel package. The AI responds that the base price for the 14-night, 15-day Europe package is €3,500, which includes airfare, hotel accommodations, and admission fees to tourist attractions. Optional tours or personal expenses may incur additional costs. The trip includes visits to famous European landmarks such as the Eiffel Tower in Paris, the Colosseum in Rome, the Brandenburg Gate in Berlin, and Rhine Falls in Zurich. You'll comprehensively experience iconic sites in each city.", additional_kwargs={}, response_metadata={}),
     HumanMessage(content='Is travel insurance included?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='Yes, basic travel insurance is provided for all travelers. This insurance includes medical expense coverage and support in emergency situations. Enhanced coverage is available upon request.', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='Can I upgrade my flight seat to business class? How much does it cost?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='Upgrading your flight seat to business class is possible. The upgrade cost is approximately €1,200 round-trip. Business class offers benefits such as wider seats, premium in-flight meals, and additional baggage allowance.', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='What is the hotel rating included in the package?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='The package includes accommodation in 4-star hotels. Each hotel offers comfort and convenience, with central locations providing easy access to tourist sites. All hotels are equipped with excellent services and amenities.', additional_kwargs={}, response_metadata={})]</pre>



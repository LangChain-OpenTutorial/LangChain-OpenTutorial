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

# New Employee Onboarding Chatbot

- Author: [Mark](https://github.com/obov)
- Design:
- Peer Review :
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/07-Agent/19-NewEmployeeOnboardingChatbot.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/07-Agent/19-NewEmployeeOnboardingChatbot.ipynb)

## Overview

This tutorial demonstrates how to build an Onboarding Helper using `LangChain`, designed to centralize and leverage Notion-based documentation for new employees. By integrating structured data from Notion pages, databases, and wikis into a Retrieval-Augmented Generation (RAG) system, this solution enables seamless access to company protocols, role-specific guides, and FAQs. New hires can query this unified knowledge base in natural language, helping them quickly adapt to their roles without sifting through scattered documents.

### Why This Matters

Traditional onboarding often involves digging through wikis, PDFs, and email threads to find relevant information. This approach is time-consuming and inefficient, especially as companies scale. With a Notion-powered RAG system, employees can simply ask questions in natural language and get precise, contextual answers in seconds.

Beyond efficiency, this approach also reduces friction in communication. New employees often hesitate to repeatedly ask their managers or senior colleagues the same questions, worrying that they might be bothering them. With a chatbot, however, this concern disappears—they can ask the same question multiple times, verify small details, and gradually build confidence in their tasks, leading to a faster adaptation process.

Additionally, this is not just about onboarding—it’s about knowledge retention and accessibility. In any workplace, documenting work is essential, whether it’s through small personal notes or structured data within a digital workspace. If a team already uses Notion as their primary documentation tool, integrating this RAG system means that records naturally become part of the chatbot’s knowledge base. As team members work and document processes, the chatbot continuously updates its resource pool, making the information available for future queries.

This means that once the system is in place, it doesn’t just serve new employees—it benefits the entire team. What starts as an onboarding assistant evolves into a company-wide knowledge hub, reducing redundant questions, ensuring information consistency, and making expertise accessible to everyone. Instead of spending time searching or asking around, employees can simply ask the chatbot, allowing them to focus on doing their work more effectively.

### Applying This to Other Use Cases

Even if your organization doesn’t use Notion, the concepts covered here can be easily adapted to other structured knowledge bases such as Confluence, Google Drive, SharePoint, or internal databases. If your company already maintains a central knowledge repository, you can apply the same techniques to build a similar retrieval-based AI assistant tailored to your needs.

This tutorial will provide both the foundational understanding and practical implementation steps necessary to deploy an AI-powered onboarding assistant, improving information accessibility and reducing the time it takes for new employees to become productive. 🚀

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Notion Database Setup](#notion-database-setup)
- [Langchain Only RAG](#langchain-only-rag)
- [Apply Langgraph Basic](#apply-langgraph-basic)
- [Apply Langgraph Advanced](#apply-langgraph-advanced)
- [Wrap Up](#wrap-up)

### References

- [LangGraph Compiled State Graph](https://github.com/langchain-ai/langgraph/blob/a6b8098548ec2c28ca58307782845e91465328e3/libs/langgraph/langgraph/graph/state.py#L598-L1039)
- [LangGraph Compiled Graph](https://github.com/langchain-ai/langgraph/blob/a6b8098548ec2c28ca58307782845e91465328e3/libs/langgraph/langgraph/graph/graph.py#L467-L639)
- [LangGraph Pregel](https://github.com/langchain-ai/langgraph/blob/a6b8098548ec2c28ca58307782845e91465328e3/libs/langgraph/langgraph/pregel/__init__.py#L199-L2139)
- [LangGraph Pregel Protocol](https://github.com/langchain-ai/langgraph/blob/a6b8098548ec2c28ca58307782845e91465328e3/libs/langgraph/langgraph/pregel/protocol.py#L18-L130)
- [Pregel: A System for Large-Scale Graph Processing](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/)


All the data used in this tutorial is synthetic. Company names, personal names, business emails, contact information, and all other details are entirely fictitious and have been generated using LLM models from ChatGPT and DeepSeek.

---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

[Note]

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
        "langchain-community",
        "langchain-openai",
        "langchain-chroma",
        "langchain-core",
        "langgraph",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0
    [notice] To update, run: pip install --upgrade pip
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "07-Agent/19-NewEmployeeOnboardingChatbot",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Notion Database Setup

We use Notion as our central hub for team wikis, documentation, and task management, making it an essential tool for organizing and sharing knowledge. Notion provides a highly flexible digital workspace that combines note-taking, structured databases, and real-time collaboration tools, allowing teams to centralize their knowledge in a way that is both intuitive and scalable.

Unlike traditional documentation tools, Notion allows for seamless integration of structured and unstructured data, meaning that everything from meeting notes and company policies to project roadmaps and task assignments can coexist within a single, well-organized system. With its database-driven approach, information can be filtered, sorted, and interconnected, making retrieval faster and more efficient.

By leveraging Notion as the foundation for our RAG system, we can transform passive documentation into an active knowledge base. This means that instead of manually searching through scattered files and folders, employees can interact with a chatbot that intelligently retrieves relevant information based on natural language queries. The result is a dynamic, constantly evolving repository of company knowledge, improving both onboarding experiences and day-to-day productivity.

Key Concepts of Notion:

- Pages: Individual documents (like this guide) for text, images, or embedded content.
- Databases: Structured tables that organize information (e.g., tasks, project trackers, SOPs) with filter/sort capabilities.

### Example Database

You can view with the exact Notion database used in this tutorial here: [Tutorial Example Database](https://shrouded-lantana-c42.notion.site/1870d31b38698044b3f2fdd3c2c15e4c?v=1870d31b38698086a4dd000cd1ddd37a&pvs=4)

There is a list of documents for Retrieval Augmented Generation (RAG). Every document is augmented for this tutorial. Names and contents are all virtual data.

### Setup Notion Integration

To use Notion as a knowledge base, you need to create an integration in Notion.

#### 1. Get API Key

1. Go to Notion Developers:  
   Log in to [Notion Developers](https://developers.notion.com) → Click "View my integrations".
2. Create a New Integration:
   - Click "New integration".
   - Name it (e.g., MyApp Integration).
   - Select your workspace.
   - Set permissions:
     - Read content
     - Update content (if needed)
3. Copy the API Key:  
   After creation, copy the Internal Integration Token
4. More Information:
   - [Notion API Documentation](https://developers.notion.com/reference/intro)
   - [Notion API Key](https://developers.notion.com/docs/create-a-notion-integration)

#### 2. Find Database ID

1. Open Notion Database:
   Go to the database you or your team want to use → Click "Share" → "Copy link".

2. Extract the ID:  
   The URL looks like:  
   https://www.notion.so/your-workspace/{DATABASE_ID}?v=...  
   Copy the 32-character string between / and ? (e.g., 1870d31b38698044b3f2fdd3c2c15e4c).


```python
from langchain_community.document_loaders import NotionDBLoader

# Use this token and database ID to load the data from Notion for this tutorial
NOTION_TOKEN = "ntn" + "_L3541776489aPP4RRULRr1dAfxDeeeBoJUufhX8ON0y4tM"
DATABASE_ID = "1870d31b38698044b3f2fdd3c2c15e4c"

loader = NotionDBLoader(
    integration_token=NOTION_TOKEN,
    database_id=DATABASE_ID,
)

data = loader.load()

# If you can see list of documents, it means you successfully loaded the data from Notion.
print(len(data))
```

<pre class="custom">31
</pre>

```python
print(data[0].metadata["title"])
print(data[0].metadata["tags"])
print(data[0].page_content[:800])
```

<pre class="custom">On Boarding 
    ['Task']
    New Hire Onboarding Seminar
    : Held on the first Monday of each month at 2 PM in the 10F auditorium.  
    Team Wikis/Notion
    : Check department-specific wikis for detailed work manuals and FAQs.  
    Company Notices
    : [notice.xyzshop.com](https://notice.xyzshop.com/) updates daily with announcements.
</pre>

All the data used in this tutorial is synthetic. Company names, personal names, business emails, contact information, and all other details are entirely fictitious and have been generated using LLM models from ChatGPT and DeepSeek.


## Langchain Only RAG


In this section, we will implement RAG using only `LangChain`. Since the data prepared for this tutorial is not very long, we have skipped the chunking process during the data preprocessing stage of RAG. To enhance RAG performance, we have enabled similarity search based on the titles of the documents.

If you are already familiar with `LangChain`, this should be a very straightforward example. The core concepts—retrieving relevant documents and passing them to an LLM—are fundamental to `LangChain`’s functionality, making this implementation relatively simple.

However, if you are new to `LangChain` or unfamiliar with key concepts such as vector stores, RAG (Retrieval-Augmented Generation), and similarity search, some parts of this section might feel a bit challenging. These components are essential for building powerful AI-driven retrieval systems, so taking the time to understand them will be highly beneficial.

If you find certain steps difficult to follow, consider revisiting the basics of how vector stores index and retrieve information or how similarity search helps match queries with relevant documents. Once you gain a solid grasp of these foundational ideas, integrating them into a LangChain-based RAG system will become much more intuitive.


```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

data_processed = [
    *[
        Document(
            page_content=item.page_content,
            metadata={
                # Attributes
                "title": item.metadata["title"],
                "use_title_as_page_content": False,
            },
        )
        for item in data
    ],
    *[
        Document(
            # Use title as page content for similarity search
            # If you want some documents to be retrieved more frequently, you can use this method
            page_content=item.metadata["title"],
            metadata={
                # Attributes
                "page_content": item.page_content,
                "title": item.metadata["title"],
                "use_title_as_page_content": True,
            },
        )
        for item in data
    ],
]


vector_store = Chroma.from_documents(
    documents=data_processed,
    embedding=OpenAIEmbeddings(),
)

retriever_from_notion = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
    }
)
```

```python
from langchain_core.runnables import chain
from typing import List


@chain
def context_parser(docs: List[Document]) -> str:
    # Retrieved documents turn into string
    return "\n\n".join(
        [
            f"# {doc.metadata['title']}\n"
            f"{doc.metadata['page_content'] if doc.metadata['use_title_as_page_content'] else doc.page_content}"
            for doc in docs
        ]
    )
```

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant for onboarding new employees. \n"
            "Please answer the question based on the following documents. \n"
            "Documents: \n"
            "{context}",
        ),
        ("human", "{question}"),
    ]
)
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

```python
## Use LangChain Only

from langchain_core.runnables import RunnablePassthrough

langchain_only_rag_chatbot = (
    {
        "question": RunnablePassthrough(),
        "context": retriever_from_notion | context_parser,
    }
    | prompt
    | llm
)

result = langchain_only_rag_chatbot.invoke("how to use conference room?")

print(result.content)
```

<pre class="custom">To use a conference room at XYZ Shopping Mall, follow these steps:
    
    1. **Login:**
       - Access the internal portal at [meet.xyzshop.com](https://meet.xyzshop.com/).
       - Log in using your `@xyzshop.com` account through Single Sign-On (SSO).
       - Approve the “Meeting Room Booking” permissions when prompted.
    
    2. **View Meeting Rooms & Schedule:**
       - Navigate through the available rooms per floor (e.g., “10F-Alpha Room,” “10F-Beta Room,” “11F-Gamma Room,” etc.).
       - Check the availability of the rooms in a calendar view.
    
    3. **Booking Procedure:**
       - Click on an open time slot to proceed with booking.
       - Fill out the request form with the following details:
         - Meeting Title
         - Number of Attendees
         - Required Equipment (e.g., projector, video conference tools).
       - Once booked, a Google Calendar invite will be automatically sent to all participants.
    
    4. **Cancellation/Changes:**
       - If you need to cancel or change the reservation, click “Cancel” on the reservation detail page.
       - The event will be removed from the calendar, and cancellation notices will be sent to all invitees.
       - The same process applies for making changes to the reservation.
    
    By following these steps, you can successfully reserve and use a conference room at XYZ Shopping Mall.
</pre>

The responses generated by the basic RAG implementation above are generally acceptable. Similarity search using a vector store is efficient and widely used for initial retrieval. However, this method has a fundamental limitation: the retrieved documents may appear similar to the query but are not always semantically relevant to answering the user’s question. This can lead to inaccurate or misleading responses, ultimately reducing the reliability of the system.


## Apply Langgraph Basic


To address the issue I mentioned above, in this section, we will implement a simple filtering agent using `LangGraph` to verify whether the retrieved documents are truly useful for generating an answer. Instead of blindly relying on similarity scores, this agent will act as an additional validation layer, ensuring that only relevant information is passed to the response generation stage.

By integrating this additional verification step, we expect to see more accurate and contextually appropriate responses while reducing the inclusion of unnecessary or misleading information. Importantly, this modification requires only a small adjustment to the retriever component of our original RAG implementation, keeping the rest of the process unchanged.


```python
# Apply LangGraph to retriever_from_notion

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from functools import reduce

prompt_relevance_check = ChatPromptTemplate(
    [
        "Please determine whether the following question is relevant to the retrieved document.\n"
        "If it is relevant, output 'yes'; otherwise, output 'no' only.\n"
        "Question: {question}\n"
        "Retrieved Document:\n"
        "{context}"
    ]
)


class RetrievalState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    relevant_docs: List[Document]


# retriever_from_notion
def retrieve_node(state: RetrievalState) -> RetrievalState:
    question = state["question"]
    return {
        "question": question,
        "retrieved_docs": retriever_from_notion.invoke(question),
        "relevant_docs": [],
    }


def filter_relevant_docs_node(state: RetrievalState) -> RetrievalState:
    question = state["question"]
    docs = state["retrieved_docs"]
    if not docs:
        return {
            "question": question,
            "retrieved_docs": docs,
            "relevant_docs": [],
        }
    idxed_docs = reduce(
        lambda acc, item: {**acc, item[0]: item[1]},
        enumerate(docs),
        {},
    )

    is_each_docs_relevant_chain = RunnableParallel(
        # Dynamically create a chain as documents retrieved
        {
            str(idx): {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(
                    lambda _, doc=doc: context_parser.invoke([doc])
                ),
            }
            | prompt_relevance_check
            | llm
            | StrOutputParser()
            for idx, doc in idxed_docs.items()
        }
    ) | RunnableLambda(lambda result: list(result.values()))

    relevance_response = is_each_docs_relevant_chain.invoke(question)
    print(relevance_response)  # ['yes', 'yes', 'no']

    return {
        "question": question,
        "retrieved_docs": docs,
        "relevant_docs": [
            doc for doc, flag in zip(docs, relevance_response) if flag == "yes"
        ],
    }


graph = StateGraph(state_schema=RetrievalState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("filter_relevant_docs", filter_relevant_docs_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "filter_relevant_docs")

langgraph_retriever = graph.compile()
```

`LangGraph` is fundamentally implemented by inheriting Runnable from `LangChain`, allowing it to leverage various built-in functionalities such as structured execution, dependency management, and asynchronous processing. This enables seamless integration with LangChain’s existing components while providing enhanced control over workflow orchestration.

Therefore, it can be used as follows:


```python
langgraph_retriever_result = langgraph_retriever.invoke(
    {"question": "how to use conference room?"}
)
print(langgraph_retriever_result["question"])
print(len(langgraph_retriever_result["retrieved_docs"]))
print(len(langgraph_retriever_result["relevant_docs"]))
```

<pre class="custom">['yes', 'yes', 'yes', 'yes', 'yes']
    how to use conference room?
    5
    5
</pre>

The most important aspect is that`LangGraph` can be seamlessly integrated into an existing response chain by utilizing LCEL (LangChain Expression Language) syntax. This means that rather than introducing a completely separate process, it can be directly embedded as a natural extension of the existing pipeline.

By leveraging LCEL, it not only enhances modularity but also improves flexibility, making it easier to modify or expand the workflow without disrupting the overall system. This ability to integrate smoothly while maintaining the structured execution of `LangChain` makes it a highly effective tool for optimizing retrieval-augmented generation (RAG) pipelines.


```python
langgraph_applied_rag = (
    {
        "question": RunnablePassthrough(),
        # LangGraph applied to retriever_from_notion
        # before: "context": retriever_from_notion | context_parser,
        "context": {
            "question": RunnablePassthrough(),
        }
        | langgraph_retriever
        | RunnableLambda(lambda result: result["relevant_docs"])
        | context_parser,
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
result = langgraph_applied_rag.invoke("how to use conference room?")

print(result)
```

<pre class="custom">['yes', 'yes', 'yes', 'yes', 'yes']
    To use a conference room at XYZ Shopping Mall, follow these steps:
    
    1. **Login**: Access the internal portal at [meet.xyzshop.com](https://meet.xyzshop.com/) using Single Sign-On (SSO) with your `@xyzshop.com` account. Approve any prompts for “Meeting Room Booking” permissions.
    
    2. **View Meeting Rooms & Schedule**: Browse the available conference rooms, which are listed per floor (e.g., “10F-Alpha Room,” “10F-Beta Room,” “11F-Gamma Room,” etc.). Check the availability of rooms in the calendar view and click on an open time slot to start the booking process.
    
    3. **Booking Procedure**:
       - Fill out the booking request form with the following details:
         - Meeting Title
         - Number of Attendees
         - Required Equipment (e.g., projector, video conference tools)
       - After successfully booking the room, a Google Calendar invite will automatically be sent to all participants.
    
    4. **Handling Conflicts**: If there is a scheduling conflict, the system will suggest alternative rooms or times for your meeting.
    
    5. **Cancellation/Changes**:
       - If you need to cancel your reservation, click “Cancel” on the reservation detail page. This action will remove the event from the calendar and send cancellation notices to all invitees.
       - For any changes to the booking, follow the same procedure, ensuring all participants are kept updated.
    
    By following these steps, you can effectively book and use a conference room for your meetings.
</pre>

In fact, the previous example was more about optimization rather than an overall process improvement. It was designed as a lightweight example for users who may not be familiar with `LangGraph`, providing an easy introduction to its capabilities.

Now that we’ve warmed up, we can move on to the next stage. Although we’re using the term "advanced", don’t be intimidated—it’s not as complex as it might sound. The following concepts build upon what we’ve already covered, making the transition smooth and intuitive.


## Apply Langgraph Advanced


In this section, we will add an agent that further refines and segments the user's query.

Since chat interfaces are so common in everyday life, it's easy to overlook a crucial aspect of user experience: typing out a detailed question word by word is not an ideal UX for users. Just consider how we use Google—most people don’t carefully format their search queries into neatly structured sentences. Instead, they type short phrases, incomplete thoughts, or even just keywords, expecting the system to interpret their intent correctly.

This is an important reminder that we should not design a chatbot assuming that users will always phrase their questions in a clear, well-formatted manner. If we want to build a truly high-level chatbot, it must be capable of handling fragmented, unstructured, and even ambiguous queries. This section will serve as a first step toward achieving that goal.

To address this, the new agent we introduce will break down the user's question into more specific sub-questions. Each sub-question will then be processed in parallel using our existing retrieval and response generation chain. Once all sub-questions have been answered, the responses will be aggregated and structured into a coherent final answer.

By implementing this, we can significantly improve the chatbot's ability to handle vague or complex queries, ensuring that users receive more accurate, detailed, and structured responses without needing to carefully format their input.


```python
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

prompt_split_question = ChatPromptTemplate(
    [
        "You are an assistant that helps refine and decompose complex questions.\n"
        "Your task is to split the given question into a few concise sub-questions only if necessary.\n"
        "Do not introduce any new topics or unrelated details.\n"
        "Keep the sub-questions directly relevant to the original question.\n"
        "If the question is already specific, return it as is.\n"
        "Ensure that no extra interpretations or additional information beyond the provided question are included.\n"
        "\n"
        "Original Question: {question}\n"
        "Output (one or more refined sub-questions, separated by newlines):"
    ]
)


class QuestionState(TypedDict):
    question: str
    sub_questions: List[str]


# Node to split question
def split_question_node(state: QuestionState) -> QuestionState:
    question = state["question"]
    response = (
        prompt_split_question
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda result: result.replace("\n\n", "\n"))
    ).invoke({"question": question})

    # Convert response to list
    sub_questions = response.split("\n") if "\n" in response else [response]
    print("===== sub_questions =====")
    print(sub_questions)
    return {
        "question": question,
        "sub_questions": sub_questions,
    }


graph = StateGraph(state_schema=QuestionState)

graph.add_node("split_question", split_question_node)
graph.set_entry_point("split_question")

langgraph_question_splitter = graph.compile()

# Example executions
question = "I need to check the current inventory levels for an upcoming product launch. How can I request an inventory status report from the Operations Management Team, and what key details should I include in my request?"
result = langgraph_question_splitter.invoke({"question": question})

question = "how to use conference room?"
result = langgraph_question_splitter.invoke({"question": question})
```

<pre class="custom">===== sub_questions =====
    ['How can I request an inventory status report from the Operations Management Team?  ', 'What key details should I include in my request for the inventory status report?']
    ===== sub_questions =====
    ['- What are the steps to book a conference room?', '- What equipment is available in the conference room?', '- What are the rules or guidelines for using the conference room?']
</pre>

```python
def list_to_dict(l):
    return {str(i): v for i, v in enumerate(l)}
```

```python
def dict_to_dynamic_runnable(runnable):
    # Convert dictionary to RunnableParallel Dynamically
    @chain
    def _dic_to_runnable(d):
        return RunnableParallel(
            {k: (RunnableLambda(lambda x, key=k: x[key]) | runnable) for k in d.keys()}
        ).invoke(d)

    return _dic_to_runnable
```

```python
prompt_summarize_sub_answers = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant summarizing multiple responses for better readability.\n"
            "Please consolidate the following sub answers into a clear and concise response.\n"
            "Ensure the final answer is not too long while maintaining the key points.\n"
            "Sub Answers: {sub_answers}",
        ),
        ("human", "My question was {question}. Summarize the key points clearly."),
    ]
)
```

```python
sub_answers_chain = (
    {
        "question": RunnablePassthrough(),
    }
    | langgraph_question_splitter
    | RunnableLambda(lambda result: list_to_dict(result["sub_questions"]))
    | dict_to_dynamic_runnable(langgraph_applied_rag)
    | RunnableLambda(lambda result: list(result.values()))
)
```

```python
chat_bot = (
    {
        "question": RunnablePassthrough(),
        "sub_answers": sub_answers_chain,
    }
    | prompt_summarize_sub_answers
    | llm
)
```

```python
# De-abstraction of the chatbot
chat_bot = (
    {
        "question": RunnablePassthrough(),
        "sub_answers": {  # sub_answers_chain
            "question": RunnablePassthrough(),
        }
        | langgraph_question_splitter  # agent to split question
        | RunnableLambda(lambda result: list_to_dict(result["sub_questions"]))
        | dict_to_dynamic_runnable(
            {  # langgraph_applied_rag
                "question": RunnablePassthrough(),
                "context": {
                    "question": RunnablePassthrough(),
                }
                | langgraph_retriever  # agent to retrieve relevant docs
                | RunnableLambda(lambda result: result["relevant_docs"])
                | context_parser,
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        | RunnableLambda(lambda result: list(result.values())),
    }
    | prompt_summarize_sub_answers
    | llm
)
```

```python
response = chat_bot.invoke(
    "I need to check the current inventory levels for an upcoming product launch.\n"
    "How can I request an inventory status report from the Operations Management Team,\n"
    "and what key details should I include in my request?"
)
print(response.content)
```

<pre class="custom">===== sub_questions =====
    ['How can I request an inventory status report from the Operations Management Team?  ', 'What key details should I include in my request for the inventory status report?']
    ['no', 'yes', 'yes', 'no', 'no']
    ['no', 'no', 'no', 'yes', 'yes']
    To request an inventory status report for your upcoming product launch, follow these steps:
    
    1. **Identify the Contact:** Reach out to Jiwoo Shin (Assistant Manager, jiwoo.shin@xyzshop.com), with Hyeonseo Kim (Junior Staff, hyeonseo.kim@xyzshop.com) as a backup.
    
    2. **Compose Your Email:** Include the following key details:
       - Your name and position.
       - Purpose of the request (e.g., checking inventory levels for a product launch).
       - Specific timeframe for the inventory needed (e.g., current levels).
       - Preferred data format (e.g., Excel, PDF).
       - Key metrics to include (e.g., total inventory levels, discrepancies).
       - Deadline for the report (e.g., "Please send by Friday 5:00 PM").
       - Your contact information for follow-up.
    
    3. **Provide Context:** If relevant, mention any specific projects or issues related to your request.
    
    4. **Follow Up:** If you don't receive a response within a few days, send a polite follow-up email.
    
    Here’s a sample email template:
    
    ---
    Subject: Request for Inventory Status Report
    
    Dear Jiwoo,
    
    I hope this message finds you well. I am writing to request an inventory status report for our upcoming product launch. Specifically, I need the current inventory levels.
    
    If possible, I would appreciate receiving this report in [preferred format] by [insert deadline]. 
    
    Thank you for your assistance!
    
    Best regards,  
    [Your Name]  
    [Your Position]  
    [Your Contact Information]  
    ---
    
    Sending your email during business hours can help ensure a timely response.
</pre>

## Wrap up

Throughout this tutorial, we explored various ways to enhance a basic RAG system using `LangChain` and `LangGraph`. We started with a simple similarity search-based RAG implementation, then introduced an agent to filter retrieved documents, ensuring they contribute meaningfully to the final response. Finally, we refined the user query handling by segmenting and processing sub-questions in parallel, creating a more structured and intelligent response system.

One point I want to highlight—though it may seem functionally less critical—is the tight integration between `LangChain` and `LangGraph`. Rather than thinking of them as separate choices, it's more effective to use them flexibly depending on the situation.

`LangGraph` builds on LangChain's Runnable-based architecture, meaning you don’t have to choose one over the other. Instead, you can seamlessly invoke `LangGraph` workflows within a standard `LangChain` chain, or even integrate LCEL (LangChain Expression Language) to construct more modular and expressive logic.

Ultimately, the key takeaway is that `LangChain` and `LangGraph` complement each other—leveraging both increases adaptability, whether you're optimizing retrieval, structuring workflows, or improving response generation. The best approach isn't about choosing one, but about knowing when and how to use each effectively.


---


All the data used in this tutorial is synthetic. Company names, personal names, business emails, contact information, and all other details are entirely fictitious and have been generated using LLM models from ChatGPT and DeepSeek.


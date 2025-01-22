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

# Agentic RAG

- Author: [Harheem Kim](https://github.com/harheem)
- Design:
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/15-Agent/06-Agentic-RAG.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Agent/06-Agentic-RAG.ipynb)

## Overview

**Agentic RAG** extends traditional RAG (Retrieval-Augmented Generation) systems by incorporating an agent-based approach for more sophisticated information retrieval and response generation. This system goes beyond simple document retrieval and response generation by enabling agents to utilize various tools for more intelligent information processing. These tools include `Tavily Search` for accessing up-to-date information, `Python` code execution capabilities, and custom function implementations, all integrated within the `LangChain` framework to provide a comprehensive solution for information processing and generation tasks.

This tutorial demonstrates how to build a document retrieval system using `FAISS DB` for effective PDF document processing and searching. Using the AI Brief from the Software Policy Research Institute as an example document, we'll explore how to integrate web-based document loaders, text splitters, vector stores, and `OpenAI` embeddings to create a practical **Agentic RAG** system. The implementation showcases how the `Retriever` tool can be effectively combined with various `LangChain` components to create a robust document search and response generation pipeline.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Configuring Tools](#configuring-tools)
- [Building the Agent](#building-the-agent)
- [Implementing Chat History](#implementing-chat-history)
- [Running Examples](#running-examples)

### References

- [LangChain Docs - Build an Agent with AgentExecutor (Legacy)](https://python.langchain.com/docs/how_to/agent_executor/)
- [LangChain Docs - How to use a vectorstore as a retriever](https://python.langchain.com/docs/how_to/vectorstore_retriever/)
- [LangCHain Docs - How to add chat history](https://python.langchain.com/docs/how_to/qa_chat_history_how_to/)
- [Tavily](https://tavily.com/)
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
        "langchain_community",
        "langchain_openai",
        "langchain_core",
        "faiss-cpu",
        "pypdf",
    ],
    verbose=False,
    upgrade=False,
)
```

`LangChain` provides built-in tools that make it easy to use the `Tavily` search engine as a tool in your applications.

To use `Tavily Search`, you'll need to obtain an API key.

Click [here](https://app.tavily.com/sign-in) to sign up on the `Tavily` website and get your `Tavily Search` API key.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "TAVILY_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "06-Agentic-RAG",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys in a `.env` file and load it.

[Note] This is not necessary if you've already set API keys in previous steps.

```python
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



##  Configuring Tools

The foundational stage of setting up tools for the agent to use. We implement a web search tool using the `Tavily Search API` and a PDF document retrieval tool. These tools enable the agent to effectively search and utilize information from various sources. By combining these tools, the agent can select and use the appropriate tool based on the context.

### Implementing Web Search

The web search tool utilizes the `Tavily Search API` to retrieve real-time information from the web. It returns up to 6 results ranked by relevance, with each result containing a URL and content snippet.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Create a search tool instance that returns up to 6 results
search = TavilySearchResults(k=6)
```

```python
# Example usage
result = search.invoke("When was the movie A.I. released and who is the director?")
print(result)
```

<pre class="custom">[{'url': 'https://spielberg.fandom.com/wiki/A.I._Artificial_Intelligence', 'content': "A.I. Artificial Intelligence | Spielberg Wiki | Fandom Spielberg Wiki All Pages Recent Blog Posts Blogs Wikis Explore Wikis Community Central Spielberg Wiki All Pages Recent Blog Posts Blogs Sign In Register Spielberg Wiki pages All Pages Recent Blog Posts Blogs A.I. Artificial Intelligence[1] (or simply A.I.) is a 2001 science fiction film directed by Steven Spielberg. In 1995, Kubrick handed A.I. to Spielberg, but the film did not gain momentum until Kubrick died in 1999. Spielberg remained close to Watson's treatment for the screenplay, and dedicated the film to Kubrick. In a 2016 BBC poll of 177 critics around the world, A.I. Artificial Intelligence was voted the eighty-third greatest film since 2000. Spielberg Wiki is a FANDOM Movies Community."}, {'url': 'https://movies.fandom.com/wiki/A.I._Artificial_Intelligence', 'content': 'films A.I. Artificial Intelligence, also known as A.I., is a 2001 American science fiction drama film written, directed, and produced by Steven Spielberg, and based on Brian Aldiss\'s short story "Super-Toys Last All Summer Long". The film languished in development hell for years, partly because Kubrick felt computer-generated imagery was not advanced enough to create the David character, whom he believed no child actor would believably portray. Having received and comprehended his memories, the advanced Mecha use them to reconstruct the Swinton home and explain to David via an interactive image of the Blue Fairy (Streep) that it is impossible to make him human. 2001 films films'}, {'url': 'https://simple.wikipedia.org/wiki/A.I._Artificial_Intelligence', 'content': 'Show any page Page A.I. Artificial Intelligence, or A.I., is a 2001 American science fiction drama movie directed by Steven Spielberg. The movie was produced by Kathleen Kennedy, Spielberg and Bonnie Curtis. It stars Haley Joel Osment, Jude Law, Frances O\'Connor, Brendan Gleeson and William Hurt. The couple acquires David, a robot capable of loving, to replace Martin. Martin tries to get David to do dangerous things that may hurt others. David gets captured by a "Flesh Fair" but gets out, because the people believe he is really human. Haley Joel Osment as David. English-language movies 2001 drama movies Robot movies Movies directed by Steven Spielberg Movies composed by John Williams movies Movies about artificial intelligence'}, {'url': 'https://www.allmovie.com/movie/ai-artificial-intelligence-am6620', 'content': 'A.I. Artificial Intelligence (2001) - Steven Spielberg | Synopsis, Movie Info, Moods, Themes and Related | AllMovie Genres › Themes › Collections › Comedy Drama A.I. Artificial Intelligence (2001) Genres - Action-Adventure, Drama, Science Fiction\xa0\xa0|\xa0\xa0 Sub-Genres - Post-Apocalyptic Film\xa0\xa0|\xa0\xa0 Release Date - Jun 29, 2001\xa0\xa0|\xa0\xa0 Run Time - 146 min. AllMovie Rating User Reviews ↓ Streams ↓ Related ↓ A.I. Artificial Intelligence is a 2001 American science fiction film directed by Steven Spielberg. Android, Artificial Intelligence, Coming Of Age/Teen, Dark, Death/Extinction, Dystopia, Fairy Tale, Film Based On Literature, Hologram, Ice Age, Mind-Bending/Experimental, Myths/Legends/Magic, Robot, Technology Subject: android, artificial intelligence Inteligencia artificial A.I. Artificial Intelligence A.I. Inteligencia Artificial A.I. inteligencia artificial Artificial Intelligence: AI Inteligencia artificial Inteligencia artificial'}, {'url': 'https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence', 'content': 'But it\'s a brilliant piece of film and of course it\'s a phenomenon because it contains the energies and talents of two brilliant filmmakers".[50] Richard Corliss heavily praised Spielberg\'s direction, as well as the cast and visual effects.[51]\nRoger Ebert gave the film three stars out of a possible four, saying that it is "wonderful and maddening".[52] Ebert later gave the film a full four stars and added it to his "Great Movies" list in 2011.[53] Leonard Maltin, on the other hand, gives the film two stars out of four in his Movie Guide, writing: "[The] intriguing story draws us in, thanks in part to Osment\'s exceptional performance, but takes several wrong turns; ultimately, it just doesn\'t work. Plus, quite a few critics in America misunderstood the film, thinking for instance that the Giacometti-style beings in the final 20 minutes were aliens (whereas they were robots of the future who had evolved themselves from the robots in the earlier part of the film) and also thinking that the final 20 minutes were a sentimental addition by Spielberg, whereas those scenes were exactly what I wrote for Stanley and exactly what he wanted, filmed faithfully by Spielberg. However, Spielberg asked Angel to be on the set every day to make line alterations wherever he felt necessary.[32] Social robotics expert Cynthia Breazeal served as technical consultant during production.[21][33] Costume designer Bob Ringwood studied pedestrians on the Las Vegas Strip for his influence on the Rouge City extras.[34] Additional visual effects such as removing the visible rods controlling Teddy and removing Haley Joel Osment\'s breath, were provided in-house by PDI/DreamWorks.[35]\nCasting[edit]\nJulianne Moore and Gwyneth Paltrow were considered for the role of Monica Swinton before Frances O\'Connor was cast and Jerry Seinfeld was originally considered to voice and play the Comedian Robot before Chris Rock was cast.[36]\nSoundtrack[edit]\n To avoid audiences mistaking A.I. for a family film, no action figures were created, although Hasbro released a talking Teddy following the film\'s release in June 2001.[21]\nA.I. premiered at the Venice Film Festival in 2001.[38]\nHome media[edit]\nA.I. Artificial Intelligence was released on VHS and DVD in the United States by DreamWorks Home Entertainment on March 5, 2002[39][40] in widescreen and full-screen 2-disc special editions featuring an extensive sixteen-part documentary detailing the film\'s development, production, music and visual effects. After the release of Spielberg\'s Jurassic Park, with its innovative computer-generated imagery, it was announced in November 1993 that production of A.I. would begin in 1994.[16] Dennis Muren and Ned Gorman, who worked on Jurassic Park, became visual effects supervisors,[13] but Kubrick was displeased with their previsualization, and with the expense of hiring Industrial Light & Magic.[17]\n"Stanley [Kubrick] showed Steven [Spielberg] 650 drawings which he had, and the script and the story, everything.'}]
</pre>

### Implementing PDF Search

This tutorial demonstrates how to build a PDF search tool that leverages vector databases for efficient document retrieval. The system divides PDF documents into manageable chunks and utilizes `OpenAI` embeddings for text vectorization alongside `FAISS` for fast similarity searching.

For this tutorial, we'll work with a sample document from the academic text "*An Introduction to Ethics in Robotics and AI*" (2021). This comprehensive book explores fundamental concepts including AI definitions, machine learning principles, robotics fundamentals, and the current limitations of AI technology.

- Title: What Is AI?
- Authors:
    - Christoph Bartneck (University of Canterbury)
    - Christoph Lütge (Technical University of Munich)
- Link: https://www.researchgate.net/publication/343611353_What_Is_AI
- File: What_Is_AI.pdf

To begin, please place the PDF file in your data directory.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool

# Load and process the PDF
loader = PyPDFLoader("data/What_Is_AI.pdf")

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Split the document
split_docs = loader.load_and_split(text_splitter)

# Create vector store
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Create retriever
retriever = vector.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search information from the PDF document",
)
```

```python
retriever.invoke("What are the main limitations of AI discussed in the text?")
```




<pre class="custom">[Document(id='21264c71-a716-4b90-8909-2eece1374346', metadata={'source': 'What_Is_AI.pdf', 'page': 9}, page_content='14 2 What Is AI?\nalgorithms and optimisation software can handle everything from airline reservation\nsystems to the management of nuclear power plants. But they only take well-deﬁned\nactions within strictly deﬁned limits. In this section, we focus on some of the major\nchallenges that make AI so difﬁcult. The limitations of sensors and the resulting lack\nof perception have already been highlighted.\nAI systems are rarely capable of generalising across learned concepts. Although\na classiﬁer may be trained on very related problems, typically classiﬁer performance\ndrops substantially when the data is generated from other sources or in other ways.\nFor example, face recognition classiﬁers may obtain excellent results when faces are\nviewed straight on, but performance drops quickly as the view of the face changes\nto, say proﬁle. Considered another way, AI systems lack robustness when dealing\nwith a changing, dynamic, and unpredictable world. As mentioned, AI systems lack'),
     Document(id='14bc2537-2213-4466-9012-e905f1b5462f', metadata={'source': 'What_Is_AI.pdf', 'page': 11}, page_content='16 2 What Is AI?\nDiscussion Questions:\n\x81 Explain the difference between weak and strong AI. Give examples from\nscience ﬁction describing machines that could be categorised as displaying\nstrong and weak AI.\n\x81 Given the description of supervised machine learning above, how might\na classiﬁer come to include societal biases? How might the removal of\nsuch biases impact classiﬁer performance? Describe a situation in which\nstakeholders must balance the tradeoff between bias and performance.\n\x81 Consider the sense-plan-act paradigm described above. How might errors\nat one step of this process impact the other steps? Draw an informal graph\nof robot performance versus time.\nFurther Reading:\n\x81 Stuart J. Russell and Peter Norvig. Artiﬁcial intelligence: a modern\napproach. Prentice Hall, Upper Saddle River, N.J, 3rd edition, 2010. ISBN\n9780132071482. URL http://www.worldcat.org/oclc/688385283\n\x81 Ryszard S Michalski, Jaime G Carbonell, and Tom M Mitchell. Machine'),
     Document(id='1daa3a43-c635-4a92-8963-affe541c0593', metadata={'source': 'What_Is_AI.pdf', 'page': 0}, page_content='Chapter 2\nWhat Is AI?\nIn this chapter we discuss the different deﬁnitions of Artiﬁcial Intelligence\n(AI). We then discuss how machines learn and how a robot works in general.\nFinally we discuss the limitations of AI and the inﬂuence the media has on our\npreconceptions of AI.\nchris: Siri, should I lie about my weight on my dating proﬁle?\nsiri: I can’t answer that, Chris.\nSiri is not the only virtual assistant that will struggle to answer this question\n(see Fig. 2.1). Toma et al. ( 2008) showed that almost two thirds of people provide\ninaccurate information about their weight on dating proﬁles. Ignoring, for a moment,\nwhat motivates people to lie about their dating proﬁles, why is it so difﬁcult, if not\nimpossible, for digital assistants to answer this question?\nTo better understand this challenge it is necessary to look behind the scene and\nto see how this question is processed by Siri. First, the phone’s microphone needs'),
     Document(id='16987f25-7991-4ad9-8dd2-d204b9ce62f2', metadata={'source': 'What_Is_AI.pdf', 'page': 6}, page_content='2.1 Introduction to AI 11\nAI currently works best in constrained environments, but has trouble with open\nworlds, poorly deﬁned problems, and abstractions. Constrained environments include\nsimulated environments and environments in which prior data accurately reﬂects\nfuture challenges. The real world, however, is open in the sense that new challenges\narise constantly. Humans use solutions to prior related problems to solve new prob-\nlems. AI systems have limited ability to reason analogically from one situation to\nanother and thus tend to have to learn new solutions even for closely related prob-\nlems. In general, they lack the ability to reason abstractly about problems and to use\ncommon sense to generate solutions to poorly deﬁned problems.\n2.2 What Is Machine Learning?\nMachine learning is a sub-ﬁeld of AI focused on the creation of algorithms that use\nexperience with respect to a class of tasks and feedback in the form of a performance')]</pre>



### Combining Tools

We combine multiple tools into a single list, allowing the agent to select and use the appropriate tool based on the context. This enables flexible switching between web search and document retrieval.

```python
# Combine tools into a single list for the agent to use
tools = [search, retriever_tool]
```

## Building the Agent

The core stage of building an agent. We initialize a Large Language Model (LLM) and set up prompt templates that enable the agent to effectively utilize tools. The agent is configured to combine PDF search and web search capabilities, allowing it to find answers from various information sources. Specifically, we use `create_tool_calling_agent` to create an agent with tool-using capabilities and explain how to set up the execution environment using `AgentExecutor`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `pdf_search` tool for searching information from the PDF document. "
            "If you can't find the information from the PDF document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

> *Note*: We set `verbose=False` to suppress intermediate step outputs from the agent executor.


## Implementing Chat History

The essential implementation stage for managing conversation history. We implement a session-based chat history store that allows the agent to remember and reference previous conversations. Using `ChatMessageHistory`, we maintain independent conversation histories for each session, and through `RunnableWithMessageHistory`, we enable the agent to understand conversation context and maintain natural dialogue flow. This allows users to ask follow-up questions naturally based on previous interactions.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a store for session histories
store = {}


def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]


# Create agent with chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

## Running Examples

Introduction to running the implemented agent and examining its results. Using streaming output, we can observe the agent's thought process and results in real-time. Through various examples, we showcase the agent's core functionalities including PDF document search, web search, independent session management across conversations, and response restructuring. The `process_response` function helps structure the agent's responses, clearly showing tool usage and results in an organized manner.

```python
def process_response(response):
    """
    Process and display streaming response from the agent.

    Args:
        response: Agent's streaming response iterator
    """
    for chunk in response:
        if chunk.get("output"):
            print(chunk["output"])
        elif chunk.get("actions"):
            for action in chunk["actions"]:
                print(f"\nTool Used: {action.tool}")
                print(f"Tool Input: {action.tool_input}")
                if action.log:
                    print(f"Tool Log: {action.log}")
```

```python
# Example 1: Searching in PDF
response = agent_with_chat_history.stream(
    {
        "input": "What information can you find about Samsung's AI model in the document?"
    },
    config={"configurable": {"session_id": "tutorial_session_1"}},
)
process_response(response)
```

<pre class="custom">
    Tool Used: pdf_search
    Tool Input: {'query': 'Samsung AI model'}
    Tool Log: 
    Invoking: `pdf_search` with `{'query': 'Samsung AI model'}`
    
    
    
    The document does not contain specific information about Samsung's AI model. If you need information about Samsung's AI model, I can search the web for you. Would you like me to do that?
</pre>

```python
# Example 2: Following up with web search (same session)
response = agent_with_chat_history.stream(
    {
        "input": "Yes, please search the web for information about Samsung's latest AI model"
    },
    config={"configurable": {"session_id": "tutorial_session_1"}},
)
process_response(response)
```

<pre class="custom">
    Tool Used: tavily_search_results_json
    Tool Input: {'query': 'Samsung latest AI model 2023'}
    Tool Log: 
    Invoking: `tavily_search_results_json` with `{'query': 'Samsung latest AI model 2023'}`
    
    
    
    Samsung has recently unveiled its new generative AI model called Samsung Gauss. This model was introduced at the Samsung AI Forum 2023. Named after the mathematician Carl Friedrich Gauss, the model signifies the infinite possibilities of generative AI that Samsung aims to realize. Samsung Gauss is designed to improve performance and efficiency, and it will be used to enhance Galaxy AI features. It is also capable of generating text, code, and images, positioning it as a potential alternative to models like ChatGPT. The development of Samsung Gauss was led by Samsung Research.
</pre>

```python
# Example 3: New session with different topic (Session 2)
response = agent_with_chat_history.stream(
    {"input": "What can you tell me about Stroing and Weak AI from the PDF document?"},
    config={"configurable": {"session_id": "tutorial_session_2"}},
)
process_response(response)
```

<pre class="custom">
    Tool Used: pdf_search
    Tool Input: {'query': 'Strong and Weak AI'}
    Tool Log: 
    Invoking: `pdf_search` with `{'query': 'Strong and Weak AI'}`
    
    
    
    The PDF document discusses the concepts of Strong and Weak AI as follows:
    
    - **Weak AI**: This type of AI is limited to a single, narrowly defined task. Most modern AI systems fall into this category. They are developed to handle a specific problem, task, or issue and generally cannot solve other problems, even if they are related. Examples of weak AI include systems that can beat a grandmaster in chess or Go, or experienced players in Poker.
    
    - **Strong AI**: In contrast, Strong AI is defined by John Searle as an AI that, when appropriately programmed with the right inputs and outputs, would have a mind in exactly the same sense human beings have minds. This implies that Strong AI would have the ability to understand, reason, and have consciousness similar to humans. As of the document's writing, no AI system has achieved Strong AI.
    
    The document also mentions that while some software systems have claimed to pass the Turing test, these claims are disputed, and no AI system has yet achieved Strong AI.
</pre>

```python
# Example 4: Request to summarize previous response in a table (Session 2)
response = agent_with_chat_history.stream(
    {"input": "Can you organize your previous response into a table format?"},
    config={"configurable": {"session_id": "tutorial_session_2"}},
)
process_response(response)
```

<pre class="custom">Certainly! Here's the information organized into a table format:
    
    | Type of AI  | Description                                                                                                                                         | Examples                                                                                   |
    |-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
    | Weak AI     | Limited to a single, narrowly defined task. Most modern AI systems fall into this category and cannot solve unrelated problems.                     | Systems that can beat a grandmaster in chess or Go, or experienced players in Poker.       |
    | Strong AI   | Defined by John Searle as an AI that would have a mind in the same sense human beings have minds, with the ability to understand and reason.        | No AI system has achieved Strong AI as of the document's writing.                          |
    
    This table summarizes the key differences between Weak and Strong AI as discussed in the PDF document.
</pre>

```python

```

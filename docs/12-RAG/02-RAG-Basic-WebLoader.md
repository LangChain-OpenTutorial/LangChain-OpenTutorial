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

# RAG Basic WebBaseLoader

- Author: [Sunyoung Park (architectyou)](https://github.com/architectyou)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial will cover the implementation of a news article QA app that can query the content of news articles using web data for RAG practice. This guide builds a RAG pipeline using OpenAI Chat models, Embedding, and ChromaDB vector store, utilizing Forbes News pages and Naver News pages which is the most popular news website in Korea.

### 1. Pre-processing - Steps 1 to 4
![Pre-processing](./img/12-rag-rag-basic-pdf-rag-process-01.png)
![](./assets/12-rag-rag-basic-pdf-rag-graphic-1.png)

The pre-processing stage involves four steps to load, split, embed, and store documents into a Vector DB (database).

- **Step 1: Document Load** : Load the document content.  
- **Step 2: Text Split** : Split the document into chunks based on specific criteria.  
- **Step 3: Embedding** : Generate embeddings for the chunks and prepare them for storage.  
- **Step 4: Vector DB Storage** : Store the embedded chunks in the database.  

### 2. RAG Execution (RunTime) - Steps 5 to 8
![RAG Execution](./img/12-rag-rag-basic-pdf-rag-process-02.png)
![](./assets/12-rag-rag-basic-pdf-rag-graphic-1.png)
- **Step 5: Retriever** : Define a retriever to fetch results from the database based on the input query. Retrievers use search algorithms and are categorized as Dense or Sparse:
  - **Dense** : Similarity-based search.
  - **Sparse** : Keyword-based search.

- **Step 6: Prompt** : Create a prompt for executing RAG. The **context** in the prompt includes content retrieved from the document. Through prompt engineering, you can specify the format of the answer.  

- **Step 7: LLM** : Define the language model (e.g., GPT, Clause, Gemini).  

- **Step 8: Chain** : Create a chain that connects the prompt, LLM, and output.  


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Web News Based QA(Question-Answering) Chatbot](#web-news-based-qa(question-answering)-chatbot)

### References

- [LangChain Tutorial : QA with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [LangChain WebLoader Tutorial](https://python.langchain.com/docs/integrations/document_loaders/web_base/)
- [Naver News](https://n.news.naver.com/)
- [Forbes](https://www.forbes.com/)

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
        "bs4",
        "langsmith",
        "langchain",
        "langchain-text-splitters",
        "langchain-community",
        "langchain-core",
        "langchain-openai",
        "langchain-chroma",
        "faiss-cpu" #if gpu is available, use faiss-gpu
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
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
        "LANGCHAIN_PROJECT": "RAG-Basic-WebLoader",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



If a warning is displayed due to the USER_AGENT not being set when using the WebBaseLoader,

please add USER_AGENT = myagent to the .env file

## Web News Based QA(Question-Answering) Chatbot

In this tutorial we'll learn about the implementation of a news article QA app that can query the content of news articles using web data for RAG practice. This guide builds a RAG pipeline using OpenAI Chat models, Embedding, and FAISS vector store, utilizing Forbes News pages and Naver News pages which is the most popular news website in Korea.

First, through the following process, we can implement a simple indexing pipeline and RAG chain with approximately 20 lines of code.

**[Note]**
- `bs4` is a library for parsing web pages.
- `langchain` is a library that provides various AI-related functionalities. Here, we'll specifically cover text splitting (`RecursiveCharacterTextSplitter`), document loading (`WebBaseLoader`), vector storage (`Chroma`, `FAISS`), output parsing (`StrOutputParser`), and runnable passthrough (`RunnablePassthrough`).
- Through the `langchain_openai` module, we can use OpenAI's chatbot (`ChatOpenAI`) and embedding (`OpenAIEmbeddings`) functionalities.

```python
import bs4
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

We implement a process that loads web page content, splits text into chunks for indexing, and then searches for relevant text snippets to generate new content.

`WebBaseLoader` uses `bs4.SoupStrainer` to parse only the necessary parts from the specified web page.

[Note]

- `bs4.SoupStrainer` allows you to conveniently retrieve desired elements from the web.

(Example)

```python
bs4.SoupStrainer(
    "div",
    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
)
```

```python
# Load news article content, split into chunks, and index them.

url = "https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/"
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["article-body fs-article fs-premium fs-responsive-text current-article font-body color-body bg-base font-accent article-subtype__masthead",
                             "header-content-container masthead-header__container"]},
        )
    ),
)
docs = loader.load()

docs = loader.load()
print(f"Number of documents: {len(docs)}")
docs
```

<pre class="custom">Number of documents: 1
</pre>




    [Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content="ForbesInnovationEditors' PickThe Prompt: Scarlett Johansson Vs OpenAIPlus AI-generated kids draw predators on TikTok and Instagram. \nShare to FacebookShare to TwitterShare to Linkedin“I was shocked, angered and in disbelief,” Scarlett Johansson said about OpenAI's Sky voice for ChatGPT that sounds similar to her own.FilmMagic\nThe Prompt is a weekly rundown of AI’s buzziest startups, biggest breakthroughs, and business deals. To get it in your inbox, subscribe here.\n\n\nWelcome back to The Prompt.\n\nScarlett Johansson’s lawyers have demanded that OpenAI take down a voice for ChatGPT that sounds much like her own after she’d declined to work with the company to create it. The actress said in a statement provided to Forbes that her lawyers have asked the AI company to detail the “exact processes” it used to create the voice, which sounds eerily similar to Johansson’s voiceover work in the sci-fi movie Her. “I was shocked, angered and in disbelief,” she said.\n\nThe actress said in the statement that last September Sam Altman offered to hire her to voice ChatGPT, adding that her voice would be comforting to people. She turned down the offer, citing personal reasons. Two days before OpenAI launched its latest model, GPT-4o, Altman reached out again, asking her to reconsider. But before she could respond, the voice was used in a demo, where it flirted, laughed and sang on stage. (“Oh stop it! You’re making me blush,” the voice said to the employee presenting the demo.)\n\nOn Monday, OpenAI said it would take down the voice, while claiming that it is not “an imitation of Scarlett Johansson” and that it had partnered with professional voice actors to create it. But Altman’s one-word tweet – “Her” – posted after the demo last week only further fueled the connection between the AI’s voice and Johannson’s.\nNow, let’s get into the headlines.\nBIG PLAYSActor and filmmaker Donald Glover tests out Google's new AI video tools.GOOGLE \n\nGoogle made a long string of AI-related announcements at its annual developer conference last week. The biggest one is that AI overviews — AI-generated summaries on any topic that will sit on top of search results — are rolling out to everyone across the U.S. But users were quick to express their frustration with the inaccuracies of these AI-generated snapshots. “90% of the results are pure nonsense or just incorrect,” one person wrote. “I literally might just stop using Google if I can't figure out how to turn off the damn AI overview,” another posted on X.\nConsumers will also be able to use videos recorded with Google Lens to search for answers to questions like “What breed is this dog?” or “How do I fix this?” Plus, a new feature built on Gemini models will let them search their Google Photos gallery. Workspace products are getting an AI uplift as well: Google’s AI model Gemini 1.5 will let paying users find and summarize information in their Google Drive, Docs, Slides, Sheets and Gmail, and help generate content across these apps. Meanwhile, Google hired artists like actor and filmmaker Donald Glover and musician Wyclef Jean to promote Google’s new video and music creation AI tools.\nDeepMind CEO Demis Hassabis touted Project Astra, a “universal assistant” that the company claims can see, hear and speak while understanding its surroundings. In a demo, the multimodel AI agent helps identify and fix pieces of code, create a band name and even find misplaced glasses.\nTALENT RESHUFFLE\nKey safety researchers at OpenAI, including cofounder and Chief Scientist Ilya Sutskever and machine learning researcher Jan Leike, have resigned. The two led the company’s efforts to develop ways to control AI systems that might become smarter than humans and prevent them from going rogue at the company’s superalignment team, which now no longer exists, according to Wired. In a thread on X, Leike wrote: “Over the past few months my team has been sailing against the wind. Sometimes we were struggling for compute and it was getting harder and harder to get this crucial research done. Over the past years, safety culture and processes have taken a backseat to shiny products.”\nThe departure of these researchers also shone a light on OpenAI’s strict and binding nondisclosure agreements and off-boarding documents. Employees who refused to sign them when they left the company risked losing their vested equity in the company, according to Vox. OpenAI CEO Sam Altman responded on X saying “there was a provision about potential equity cancellation in our previous exit docs; although we never clawed anything back, it should never have been something we had in any documents or communication.”\nAI DEALS OF THE WEEKAlexandr Wang was just 19 when he started Scale. His cofounder, Lucy Guo, was 21.Scale AI\nScale AI has raised $1 billion at a $14 billion valuation in a round led by Accel. Amazon, Meta, Intel Capital and AMD Ventures are among the firm’s new investors. The company has hired hundreds of thousands of contractors in countries like Kenya and Venezuela through its in-house agency RemoTasks to complete data labeling tasks for training AI models, Forbes reported last year. In February, Forbes reported that the startup secretly scrapped a deal with TikTok amid national security concerns.\nPlus: Coactive AI, which sorts through and structures a company’s visual data, has raised a $30 million round at a $200 million valuation led by Emerson Collective and Cherryrock Capital. And London-based PolyAI, which sells generative AI voice assistants for customer service and was cofounded by three machine learning PhD students at Cambridge, has raised $50 million at a nearly $500 million valuation.\nDEEP DIVE Images of AI children on TikTok and Instagram are becoming magnets for many with a sexual interest in minors.ILLUSTRATION BY CECILIA RUNXI ZHANG; IMAGE BY ANTAGAIN/GETTY IMAGES\nThe girls in the photos on TikTok and Instagram look like they could be 5 or 6 years old. On the older end, not quite 13. They’re pictured in lace and leather, bikinis and crop tops. They’re dressed suggestively as nurses, superheroes, ballerinas and french maids. Some wear bunny ears or devil horns; others, pigtails and oversized glasses. They’re black, white and Asian, blondes, redheads and brunettes. They were all made with AI, and they’ve become magnets for the attention of a troubling audience on some of the biggest social media apps in the world—older men.\n“AI makes great works of art: I would like to have a pretty little virgin like that in my hands to make it mine,” one TikTok user commented on a recent post of young blonde girls in maid outfits, with bows around their necks and flowers in their hair.\nSimilar remarks flooded photos of AI kids on Instagram. “I would love to take her innocence even if she’s a fake image,” one person wrote on a post of a small, pale child dressed as a bride. On another, of a young girl in short-shorts, the same user commented on “her cute pair of small size [breasts],” depicted as two apple emojis, “and her perfect innocent slice of cherry pie down below.”\nForbes found hundreds of posts and comments like these on images of AI-generated kids on the platforms from 2024 alone. Many were tagged to musical hits—like Beyonce’s “Texas Hold ‘Em,” Taylor Swift’s “Shake It Off” and Tracy Chapman’s “Fast Car”—to help them reach more eyeballs.\nChild predators have prowled most every major social media app—where they can hide behind screens and anonymous usernames—but TikTok and Instagram’s popularity with teens and minors has made them both top destinations. And though platforms’ struggle to crack down on child sexual abuse material (or CSAM) predates today’s AI boom, AI text-to-image generators are making it even easier for predators to find or create exactly what they’re looking for.\nTikTok and Instagram permanently removed the accounts, videos and comments referenced in this story after Forbes asked about them; both companies said they violated platform rules.\nRead the full story in Forbes here.\nYOUR WEEKLY DEMO\nOn Monday, Microsoft introduced a new line of Windows computers that have a suite of AI features built-in. Called “Copilot+ PCs”, the computers come equipped with AI-powered apps deployed locally on the device so you can run them without using an internet connection. The computers can record your screen to help you find anything you may have seen on it, generate images from text-based prompts and translate audio from 40 languages. Sold by brands like Dell, Lenovo and Samsung, the computers are able to do all this without internet access because their Qualcomm Snapdragon chips have a dedicated AI processor. The company claims its new laptops are about 60% faster and have 20% more battery life than Apple’s MacBook Air M3, and the first models will be on sale in mid-June.\nMODEL BEHAVIOR\nIn the past, universities have invited esteemed alumni to deliver commencement speeches at graduation ceremonies. This year, some institutions turned to AI. At D’Youville University in Buffalo, New York, a rather creepy-looking robot named Sophia delivered the commencement speech, doling out generic life lessons to an audience of 2,000 people. At Rensselaer Polytechnic Institute’s bicentennial graduation ceremony, GPT-4 was used to generate a speech from the perspective of Emily Warren Roebling, who helped complete the construction of the Brooklyn Bridge and received a posthumous degree from the university. The speech was read out by actress Liz Wisan.\n")]



You can retrieve the main news from the Forbes page and check its **title** and **content** as follows.

Similarly to the code tutorial above, you can load news articles from **Naver news article pages** using a similar method.

<br/>

```python
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
```

`RecursiveCharacterTextSplitter` splits documents into chunks of specified size.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

splits = text_splitter.split_documents(docs)
len(splits)
```




<pre class="custom">12</pre>



Vector stores like `FAISS` or `Chroma` generate vector representations of documents based on these chunks.

```python
vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
```

```python
# Create a vector store.
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Search for and generate information contained in the news.
retriever = vectorstore.as_retriever()
```

The retriever created through `vectorstore.as_retriever()` generates new content using the prompt fetched with `hub.pull` and the `ChatOpenAI` model.

Finally, `StrOutputParser` parses the generated results into a string.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """You are a friendly AI assistant performing Question-Answering. 
Your mission is to answer the given question based on the provided context.
Please answer the question using the following retrieved context. 
If you cannot find the answer in the given context or if you don't know the answer, please respond with 'The information related to the question cannot be found in the provided information'.
Please answer in English. 
However, keep technical terms and names in their original form without translation.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)
```

**[Note]**
<br/>
If you practice with Naver-News URL, you can download and input the **teddynote/rag-prompt-korean** prompt from hub (which is set in Korean).

In this case, the separate prompt writing process can be skipped.

```python
prompt = hub.pull("teddynote/rag-prompt-korean")
prompt
```

```python
# English rag prompt

prompt = hub.pull("rlm/rag-prompt")
```

```python
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


# Create a chain.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

To use streaming output, use `stream_response`.

```python
stream_response = rag_chain.stream_response(
    {"question": "What is the latest news about AI?"}
)

for chunk in stream_response:
    print(chunk)
```

```python
answer = rag_chain.invoke("What is the latest news about AI?")
print(answer)
```

<pre class="custom">The latest news about AI includes Google's rollout of AI-generated summaries on search results, which have faced criticism for inaccuracies. Scale AI has raised $1 billion at a $14 billion valuation, with new investors like Amazon and Meta. Additionally, Microsoft introduced "Copilot+ PCs" with built-in AI features that operate without an internet connection.
</pre>

```python
answer = rag_chain.invoke("What is the main idea of latest news about?")
print(answer)

```

<pre class="custom">The latest news primarily focuses on advancements in AI technology, including Google's AI-generated summaries for search results and Microsoft's new line of AI-powered Windows computers. Google's AI features have faced criticism for inaccuracies, while Microsoft's "Copilot+ PCs" offer AI capabilities without internet access. Additionally, AI's role in social media platforms like TikTok and Instagram is highlighted in the context of combating child sexual abuse material.
</pre>

```python
answer = rag_chain.invoke("Why did OpenAI and Scarlett Johansson have a conflict?")
print(answer)
```

<pre class="custom">Scarlett Johansson had a conflict with OpenAI because the company used a voice for ChatGPT that sounded similar to hers without her consent. She had previously declined an offer from OpenAI to voice ChatGPT, and her lawyers demanded that OpenAI take down the voice and explain how it was created. OpenAI claimed the voice was not an imitation of Johansson's, but the situation was exacerbated by a tweet from OpenAI's CEO, Sam Altman, referencing her film "Her."
</pre>

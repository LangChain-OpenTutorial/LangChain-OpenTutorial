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

# Exploring RAG in LangChain

- Author: [Jaeho Kim](https://github.com/Jae-hoya)
- Design: []()
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

![rag-1.png](./img/12-rag-rag-basic-pdf-rag-process-01.png)

![rag-2.png](./img/12-rag-rag-basic-pdf-rag-process-02.png)

## OverView

This tutorial explores the entire process of indexing, retrieval, and generation using LangChain's RAG framework. It provides a broad overview of a typical RAG application pipeline and demonstrates how to effectively retrieve and generate responses by using LangChain's key features, such as data loaders, vector databases, embedding, retrievers, and generators, structured in a modular design.

### 1. Question Processing

The question processing stage involves receiving a user's question, handling it, and finding relevant data. The following components are required for this process:

- **Data Source Connection**
To find answers to the question, it is necessary to connect to various text data sources. LangChain helps you easily establish connections to various data sources.
- **Data Indexing and Retrieval**
To efficiently find relevant information from data sources, the data must be indexed. LangChain automates the indexing process and provides tools to retrieve data related to the user's question.


### 2. Answer Generation

Once the relevant data is found, the next step is to generate an answer based on it. The following components are essential for this stage:

- **Answer Generation Model**
LangChain uses advanced natural language processing (NLP) models to generate answers from the retrieved data. These models take the user's question and the retrieved data as input and generate an appropriate answer.


## Architecture

This Tutorial will build a typical RAG application as outlined in the [Q&A Introduction](https://python.langchain.com/docs/tutorials/). This consists of two main components:

- **Indexing** : A pipeline that collects data from the source and indexes it. _This process typically occurs offline._

- **Retrieval and Generation** : The actual RAG chain processes user queries in real-time, retrieves relevant data from the index, and passes it to the model.

The entire workflow from raw data to generating an answer is as follows:

### Indexing

![](https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png)

- Indexing Image Source: https://python.langchain.com/docs/tutorials/rag/

1. **Load** : The first step is to load the data. For this, we will use [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/).

2. **Split** : [Text splitters](https://python.langchain.com/docs/concepts/text_splitters/) divide large `Documents` into smaller chunks.
This is useful for indexing data and passing it to the model, as large chunks can be difficult to retrieve and may not fit within the model's limited context window.
3. **Store** : The split data needs to be stored and indexed in a location for future retrieval. This is often accomplished using [VectorStore](https://python.langchain.com/docs/concepts/vectorstores/) and [Embeddings](https://python.langchain.com/docs/integrations/text_embedding/) Models.

### Retrieval and Generation

![](https://python.langchain.com/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png)

- Retrieval and Generation Image Source: https://python.langchain.com/docs/tutorials/rag/

1. **Retrieval** : When user input is provided, [Retriever](https://python.langchain.com/docs/integrations/retrievers/) is used to retrieve relevant chunks from the data store.
2. **Generation** : [ChatModel](https://python.langchain.com/docs/integrations/chat/) / [LLM](https://python.langchain.com/docs/integrations/llms/) enerates an answer using a prompt that includes the question and the retrieved data.

## Document Used for Practice

A European Approach to Artificial Intelligence - A Policy Perspective

- Author: Digital Enlightenment Forum under the guidance of EIT Digital, supported by contributions from EIT Manufacturing, EIT Urban Mobility, EIT Health, and EIT Climate-KIC
- Link : https://eit.europa.eu/news-events/news/european-approach-artificial-intelligence-policy-perspective
- File Name: **A European Approach to Artificial Intelligence - A Policy Perspective.pdf**

_Please copy the downloaded file into the **data** folder for practice._

### Table of Contents

- [Overview](#overview)
- [Document Used for Practice](#document-used-for-practice)
- [Environment Setup](#environment-setup)
- [Explore Each Module](#explore-each-module)
- [Step 1: Load Document](#step-1:-load-document)
- [Step 2: Split Documents](#step-2:-split-documents)
- [Step 3: Embedding](#step-3:-embedding)
- [Step 4: Create Vectorstore](#step-4-create-vectorstore)
- [Step 5: Create Retriever ](#step-5-create-retriever)
- [Step 6: Create Prompt](#step-6-create-prompt)
- [Step 7: Create LLM](#step-7-create-llm)


### References

- [LangChain: Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [LangChain: Text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [LangChain: Vector Store](https://python.langchain.com/docs/concepts/vectorstores/)
- [LangChain: Embeddings](https://python.langchain.com/docs/integrations/text_embedding/)
- [LangChain: Retriever](https://python.langchain.com/docs/integrations/retrievers/)
- [LangChain: Chat Models](https://python.langchain.com/docs/integrations/chat/)
- [LangChain: LLM](https://python.langchain.com/docs/integrations/llms/)
- [Langchain: Indexing](https://python.langchain.com/docs/tutorials/rag/)
- [Langchain: Retrieval and Generation](https://python.langchain.com/docs/tutorials/rag/)
- [Semantic Similarity Splitter](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)
- [OpenAI API Model List / Pricing](https://openai.com/api/pricing/)
- [HuggingFace LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
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
        "faiss-cpu",
        "pypdf",
        "pypdf2"
        "unstructured",
        "unstructured[pdf]",
        "fastembed",
        "chromadb",
        "rank_bm25",
        "langsmith",
        "langchain",
        "langchain_text_splitters",
        "langchain_community",
        "langchain_core",
        "langchain_openai",
        "langchain_experimental"
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
        "HUGGINGFACEHUB_API_TOKEN": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "03-RAG-Advanced",
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



## Explore Each Module

```python
import bs4
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

Below is an example of using a basic RAG model for handling web pages `(WebBaseLoader)` .

In each step, you can configure various options or apply new techniques.

If a warning is displayed due to the `USER_AGENT` not being set when using the WebBaseLoader,

please add `USER_AGENT = myagent` to the `.env` file.

```python
# Step 1: Load Documents
# Load the contents of news articles, split them into chunks, and index them.
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


# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

splits = text_splitter.split_documents(docs)

# Step 3: Embedding & Create Vectorstore
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

# Step 4: retriever
# Retrieve and generate information contained in the news.
retriever = vectorstore.as_retriever()

# Step 5: Create Prompt
prompt = hub.pull("rlm/rag-prompt")

# Step 6: Create LLM
# Generate the language model (LLM).
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def format_docs(docs):
    # Combine the retrieved document results into a single paragraph.
    return "\n\n".join(doc.page_content for doc in docs)


# Create Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 8: Run Chain
# Input queries about the documents and output answers.
question = "Why did OpenAI and Scarlett Johansson have a conflict?"
response = rag_chain.invoke(question)

# output the results.
print(f"URL: {url}")
print(f"Number of documents: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")
```

<pre class="custom">URL: https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/
    Number of documents: 1
    ============================================================
    [HUMAN]
    Why did OpenAI and Scarlett Johansson have a conflict?
    
    [AI]
    Scarlett Johansson and OpenAI had a conflict over a voice for ChatGPT that sounded similar to her own, which she claimed was created without her consent. After declining an offer to voice the AI, Johansson expressed shock and anger when the voice was used in a demo shortly thereafter. Her lawyers demanded details on the voice's creation and requested its removal, while OpenAI stated it was not an imitation of her voice.
</pre>

```python
print(docs)
```

<pre class="custom">[Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content="ForbesInnovationEditors' PickThe Prompt: Scarlett Johansson Vs OpenAIPlus AI-generated kids draw predators on TikTok and Instagram. \nShare to FacebookShare to TwitterShare to Linkedin“I was shocked, angered and in disbelief,” Scarlett Johansson said about OpenAI's Sky voice for ChatGPT that sounds similar to her own.FilmMagic\nThe Prompt is a weekly rundown of AI’s buzziest startups, biggest breakthroughs, and business deals. To get it in your inbox, subscribe here.\n\n\nWelcome back to The Prompt.\n\nScarlett Johansson’s lawyers have demanded that OpenAI take down a voice for ChatGPT that sounds much like her own after she’d declined to work with the company to create it. The actress said in a statement provided to Forbes that her lawyers have asked the AI company to detail the “exact processes” it used to create the voice, which sounds eerily similar to Johansson’s voiceover work in the sci-fi movie Her. “I was shocked, angered and in disbelief,” she said.\n\nThe actress said in the statement that last September Sam Altman offered to hire her to voice ChatGPT, adding that her voice would be comforting to people. She turned down the offer, citing personal reasons. Two days before OpenAI launched its latest model, GPT-4o, Altman reached out again, asking her to reconsider. But before she could respond, the voice was used in a demo, where it flirted, laughed and sang on stage. (“Oh stop it! You’re making me blush,” the voice said to the employee presenting the demo.)\n\nOn Monday, OpenAI said it would take down the voice, while claiming that it is not “an imitation of Scarlett Johansson” and that it had partnered with professional voice actors to create it. But Altman’s one-word tweet – “Her” – posted after the demo last week only further fueled the connection between the AI’s voice and Johannson’s.\nNow, let’s get into the headlines.\nBIG PLAYSActor and filmmaker Donald Glover tests out Google's new AI video tools.GOOGLE \n\nGoogle made a long string of AI-related announcements at its annual developer conference last week. The biggest one is that AI overviews — AI-generated summaries on any topic that will sit on top of search results — are rolling out to everyone across the U.S. But users were quick to express their frustration with the inaccuracies of these AI-generated snapshots. “90% of the results are pure nonsense or just incorrect,” one person wrote. “I literally might just stop using Google if I can't figure out how to turn off the damn AI overview,” another posted on X.\nConsumers will also be able to use videos recorded with Google Lens to search for answers to questions like “What breed is this dog?” or “How do I fix this?” Plus, a new feature built on Gemini models will let them search their Google Photos gallery. Workspace products are getting an AI uplift as well: Google’s AI model Gemini 1.5 will let paying users find and summarize information in their Google Drive, Docs, Slides, Sheets and Gmail, and help generate content across these apps. Meanwhile, Google hired artists like actor and filmmaker Donald Glover and musician Wyclef Jean to promote Google’s new video and music creation AI tools.\nDeepMind CEO Demis Hassabis touted Project Astra, a “universal assistant” that the company claims can see, hear and speak while understanding its surroundings. In a demo, the multimodel AI agent helps identify and fix pieces of code, create a band name and even find misplaced glasses.\nTALENT RESHUFFLE\nKey safety researchers at OpenAI, including cofounder and Chief Scientist Ilya Sutskever and machine learning researcher Jan Leike, have resigned. The two led the company’s efforts to develop ways to control AI systems that might become smarter than humans and prevent them from going rogue at the company’s superalignment team, which now no longer exists, according to Wired. In a thread on X, Leike wrote: “Over the past few months my team has been sailing against the wind. Sometimes we were struggling for compute and it was getting harder and harder to get this crucial research done. Over the past years, safety culture and processes have taken a backseat to shiny products.”\nThe departure of these researchers also shone a light on OpenAI’s strict and binding nondisclosure agreements and off-boarding documents. Employees who refused to sign them when they left the company risked losing their vested equity in the company, according to Vox. OpenAI CEO Sam Altman responded on X saying “there was a provision about potential equity cancellation in our previous exit docs; although we never clawed anything back, it should never have been something we had in any documents or communication.”\nAI DEALS OF THE WEEKAlexandr Wang was just 19 when he started Scale. His cofounder, Lucy Guo, was 21.Scale AI\nScale AI has raised $1 billion at a $14 billion valuation in a round led by Accel. Amazon, Meta, Intel Capital and AMD Ventures are among the firm’s new investors. The company has hired hundreds of thousands of contractors in countries like Kenya and Venezuela through its in-house agency RemoTasks to complete data labeling tasks for training AI models, Forbes reported last year. In February, Forbes reported that the startup secretly scrapped a deal with TikTok amid national security concerns.\nPlus: Coactive AI, which sorts through and structures a company’s visual data, has raised a $30 million round at a $200 million valuation led by Emerson Collective and Cherryrock Capital. And London-based PolyAI, which sells generative AI voice assistants for customer service and was cofounded by three machine learning PhD students at Cambridge, has raised $50 million at a nearly $500 million valuation.\nDEEP DIVE Images of AI children on TikTok and Instagram are becoming magnets for many with a sexual interest in minors.ILLUSTRATION BY CECILIA RUNXI ZHANG; IMAGE BY ANTAGAIN/GETTY IMAGES\nThe girls in the photos on TikTok and Instagram look like they could be 5 or 6 years old. On the older end, not quite 13. They’re pictured in lace and leather, bikinis and crop tops. They’re dressed suggestively as nurses, superheroes, ballerinas and french maids. Some wear bunny ears or devil horns; others, pigtails and oversized glasses. They’re black, white and Asian, blondes, redheads and brunettes. They were all made with AI, and they’ve become magnets for the attention of a troubling audience on some of the biggest social media apps in the world—older men.\n“AI makes great works of art: I would like to have a pretty little virgin like that in my hands to make it mine,” one TikTok user commented on a recent post of young blonde girls in maid outfits, with bows around their necks and flowers in their hair.\nSimilar remarks flooded photos of AI kids on Instagram. “I would love to take her innocence even if she’s a fake image,” one person wrote on a post of a small, pale child dressed as a bride. On another, of a young girl in short-shorts, the same user commented on “her cute pair of small size [breasts],” depicted as two apple emojis, “and her perfect innocent slice of cherry pie down below.”\nForbes found hundreds of posts and comments like these on images of AI-generated kids on the platforms from 2024 alone. Many were tagged to musical hits—like Beyonce’s “Texas Hold ‘Em,” Taylor Swift’s “Shake It Off” and Tracy Chapman’s “Fast Car”—to help them reach more eyeballs.\nChild predators have prowled most every major social media app—where they can hide behind screens and anonymous usernames—but TikTok and Instagram’s popularity with teens and minors has made them both top destinations. And though platforms’ struggle to crack down on child sexual abuse material (or CSAM) predates today’s AI boom, AI text-to-image generators are making it even easier for predators to find or create exactly what they’re looking for.\nTikTok and Instagram permanently removed the accounts, videos and comments referenced in this story after Forbes asked about them; both companies said they violated platform rules.\nRead the full story in Forbes here.\nYOUR WEEKLY DEMO\nOn Monday, Microsoft introduced a new line of Windows computers that have a suite of AI features built-in. Called “Copilot+ PCs”, the computers come equipped with AI-powered apps deployed locally on the device so you can run them without using an internet connection. The computers can record your screen to help you find anything you may have seen on it, generate images from text-based prompts and translate audio from 40 languages. Sold by brands like Dell, Lenovo and Samsung, the computers are able to do all this without internet access because their Qualcomm Snapdragon chips have a dedicated AI processor. The company claims its new laptops are about 60% faster and have 20% more battery life than Apple’s MacBook Air M3, and the first models will be on sale in mid-June.\nMODEL BEHAVIOR\nIn the past, universities have invited esteemed alumni to deliver commencement speeches at graduation ceremonies. This year, some institutions turned to AI. At D’Youville University in Buffalo, New York, a rather creepy-looking robot named Sophia delivered the commencement speech, doling out generic life lessons to an audience of 2,000 people. At Rensselaer Polytechnic Institute’s bicentennial graduation ceremony, GPT-4 was used to generate a speech from the perspective of Emily Warren Roebling, who helped complete the construction of the Brooklyn Bridge and received a posthumous degree from the university. The speech was read out by actress Liz Wisan.\n")]
</pre>

## Step 1: Load Document

- [Link to official documentation - Document loaders](https://python.langchain.com/docs/integrations/document_loaders/)


### Web Page

`WebBaseLoader` uses `bs4.SoupStrainer` to parse only the necessary parts from a specified web page.

[Note]

- `bs4.SoupStrainer` makes it convenient to extract desired elements from the web

(example)

```python
bs4.SoupStrainer(
    "div",
    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]}, # Input the class name.
)

bs4.SoupStrainer(
    "article",
    attrs={"id": ["dic_area"]}, # Input the class name.
)
```


Here is another example, a BBC news article. Try running it!

```python
# Load the contents of the news article, split it into chunks, and index it.
loader = WebBaseLoader(
    web_paths=("https://www.bbc.com/news/business-68092814",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "main",
            attrs={"id": ["main-content"]},
        )
    ),
)
docs = loader.load()
print(f"Number of documents: {len(docs)}")
docs[0].page_content[:500]
```

<pre class="custom">Number of documents: 1
</pre>




    'Could AI \'trading bots\' transform the world of investing?Getty ImagesIt is hard for both humans and computers to predict stock market movementsSearch for "AI investing" online, and you\'ll be flooded with endless offers to let artificial intelligence manage your money.I recently spent half an hour finding out what so-called AI "trading bots" could apparently do with my investments.Many prominently suggest that they can give me lucrative returns. Yet as every reputable financial firm warns - your '



### PDF


```python
from langchain.document_loaders import PyPDFLoader

# Load PDF file. Enter the file path.
loader = PyPDFLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf")



docs = loader.load()
print(f"Number of documents: {len(docs)}")

# Output the content of the 10th page.
print(f"\n[page_content]\n{docs[9].page_content[:500]}")
print(f"\n[metadata]\n{docs[9].metadata}\n")
```

<pre class="custom">Number of documents: 24
    
    [page_content]
    A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    10
    requirements becomes mandatory in all sectors and create bar -
    riers especially for innovators and SMEs. Public procurement ‘data 
    sovereignty clauses’ induce large players to withdraw from AI for 
    urban ecosystems. Strict liability sanctions block AI in healthcare, 
    while limiting space of self-driving experimentation. The support 
    measures to boost European AI are not sufficient to offset the 
    unintended effect of generic
    
    [metadata]
    {'source': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'page': 9}
    
</pre>

### CSV

CSV retrieves data using row numbers instead of page numbers.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Load CSV file
loader = CSVLoader(file_path="data/titanic.csv")
docs = loader.load()
print(f"Number of documents: {len(docs)}")

# Output the content of the 10th row.
print(f"\n[row_content]\n{docs[9].page_content[:500]}")
print(f"\n[metadata]\n{docs[9].metadata}\n")
```

<pre class="custom">Number of documents: 20
    
    [row_content]
    PassengerId: 10
    Survived: 1
    Pclass: 2
    Name: Nasser, Mrs. Nicholas (Adele Achem)
    Sex: female
    Age: 14
    SibSp: 1
    Parch: 0
    Ticket: 237736
    Fare: 30.0708
    Cabin: 
    Embarked: C
    
    [metadata]
    {'source': 'data/titanic.csv', 'row': 9}
    
</pre>

### TXT file


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/appendix-keywords_eng.txt", encoding="utf-8")
docs = loader.load()
print(f"Number of documents: {len(docs)}")

# Output the content of the 10th page.
print(f"\n[page_content]\n{docs[0].page_content[:500]}")
print(f"\n[metadata]\n{docs[0].metadata}\n")
```

<pre class="custom">Number of documents: 1
    
    [page_content]
    - Semantic Search
    
    Definition: Semantic search is a search method that goes beyond simple keyword matching to understand the meaning of the user’s query and return relevant results.
    Example: When a user searches for "planets in the solar system," it returns information about related planets such as "Jupiter" or "Mars."
    Keywords: Natural Language Processing, Search Algorithm, Data Mining
    
    - Embedding
    
    Definition: Embedding is the process of converting textual data, such as words or sentences, int
    
    [metadata]
    {'source': 'data/appendix-keywords_eng.txt'}
    
</pre>

### Load all files in the folder

Here is an example of loading all `.txt` files in the folder.


```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(".", glob="data/*.txt", show_progress=True)
docs = loader.load()

print(f"Number of documents: {len(docs)}")

# Output the content of the 10th page.
print(f"\n[page_content]\n{docs[0].page_content[:500]}")
print(f"\n[metadata]\n{docs[0].metadata}\n")
print(f"\n[metadata]\n{docs[1].metadata}\n")
```

<pre class="custom">100%|██████████| 2/2 [00:08<00:00,  4.26s/it]</pre>

    Number of documents: 2
    
    [page_content]
    Selecting the “right” amount of information to include in a summary is a difficult task. A good summary should be detailed and entity-centric without being overly dense and hard to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary before iteratively incorporating missing salient entities without increasing the length. Summaries generat
    
    [metadata]
    {'source': 'data/chain-of-density.txt'}
    
    
    [metadata]
    {'source': 'data/appendix-keywords_eng.txt'}
    
    

    
    

The following is an example of loading all `.pdf` files in the folder.

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(".", glob="data/*.pdf")
docs = loader.load()

print(f"page_content: {len(docs)}\n")
print("[metadata]\n")
print(docs[0].metadata)
print("\n========= [Preview] Front Section =========\n")
print(docs[0].page_content[2500:3000])
```

<pre class="custom">page_content: 1
    
    [metadata]
    
    {'source': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf'}
    
    ========= [Preview] Front Section =========
    
    While a clear cut definition of Artificial Intelligence (AI) would be the building block for its regulatory and governance framework, there is not yet a widely accepted definition of what AI is (Buiten, 2019; Scherer, 2016). Definitions focussing on intelligence are often circular in that defining what level of intelligence is nee- ded to qualify as ‘artificial intelligence’ remains subjective and situational1. Pragmatic ostensive definitions simply group under the AI labels a wide array of tech
</pre>

### Python

The following is an example of loading `.py` files.

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader

loader = DirectoryLoader(".", glob="**/*.py", loader_cls=PythonLoader)
docs = loader.load()

print(f"page_content: {len(docs)}\n")
print("[metadata]\n")
print(docs[0].metadata)
print("\n========= [Preview] Front Section =========\n")
print(docs[0].page_content[:500])
```

<pre class="custom">page_content: 1
    
    [metadata]
    
    {'source': 'data/audio_utils.py'}
    
    ========= [Preview] Front Section =========
    
    import re
    import os
    from pytube import YouTube
    from moviepy.editor import AudioFileClip, VideoFileClip
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    
    
    def extract_abr(abr):
        youtube_audio_pattern = re.compile(r"\d+")
        kbps = youtube_audio_pattern.search(abr)
        if kbps:
            kbps = kbps.group()
            return int(kbps)
        else:
            return 0
    
    
    def get_audio_filepath(filename):
        # Create the audio folder if it doesn't exist
        if not os.path.isdir("au
</pre>

---


## Step 2: Split Documents

```python
# Load the content of the news article, split it into chunks, and index it.
loader = WebBaseLoader(
    web_paths=("https://www.bbc.com/news/business-68092814",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "main",
            attrs={"id": ["main-content"]},
        )
    ),
)
docs = loader.load()
print(f"Number of Documents: {len(docs)}")
docs[0].page_content[:500]
```

<pre class="custom">Number of Documents: 1
</pre>




    'Could AI \'trading bots\' transform the world of investing?Getty ImagesIt is hard for both humans and computers to predict stock market movementsSearch for "AI investing" online, and you\'ll be flooded with endless offers to let artificial intelligence manage your money.I recently spent half an hour finding out what so-called AI "trading bots" could apparently do with my investments.Many prominently suggest that they can give me lucrative returns. Yet as every reputable financial firm warns - your '



### CharacterTextSplitter

This is the simplest method. It splits the text based on characters (default: "\n\n") and measures the chunk size by the number of characters.

1. **How the text is split** : By single character units.
2. **How the chunk size is measured** : By the `len` of characters.

Visualization example: https://chunkviz.up.railway.app/


The `CharacterTextSplitter` class provides functionality to split text into chunks of a specified size.

- `separator` parameter specifies the string used to separate chunks, with two newline characters ("\n\n") being used in this case.
- `chunk_size`determines the maximum length of each chunk.
- `chunk_overlap`specifies the number of overlapping characters between adjacent chunks.
- `length_function`defines the function used to calculate the length of a chunk, with the default being the `len` function, which returns the length of the string.
- `is_separator_regex`is a boolean value that determines whether the `separator` is interpreted as a regular expression.


```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
```

This function uses the `create_documents` method of the `text_splitter` object to split the given text (`state_of_the_union`) into multiple documents, storing the results in the `texts` variable. It then outputs the first document from texts. This process can be seen as an initial step for processing and analyzing text data, particularly useful for splitting large text data into manageable chunks.

```python
# Load a portion of the "Chain of Density" paper.
with open("data/chain-of-density.txt", "r", encoding="utf-8") as f:
    text = f.read()[:500]
```

```python
text_splitter = CharacterTextSplitter(
    chunk_size=100, chunk_overlap=10, separator="\n\n"
)
text_splitter.split_text(text)
```




<pre class="custom">['Selecting the “right” amount of information to include in a summary is a difficult task. \nA good summary should be detailed and entity-centric without being overly dense and hard to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary before iteratively incorporating missing salient entities without increasing the length. Summaries genera']</pre>



```python
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="\n")
text_splitter.split_text(text)
```




<pre class="custom">['Selecting the “right” amount of information to include in a summary is a difficult task.',
     'A good summary should be detailed and entity-centric without being overly dense and hard to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary before iteratively incorporating missing salient entities without increasing the length. Summaries genera']</pre>



```python
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator=" ")
text_splitter.split_text(text)
```




<pre class="custom">['Selecting the “right” amount of information to include in a summary is a difficult task. \nA good',
     'A good summary should be detailed and entity-centric without being overly dense and hard to follow.',
     'to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with',
     'with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial',
     'an initial entity-sparse summary before iteratively incorporating missing salient entities without',
     'without increasing the length. Summaries genera']</pre>



```python
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator=" ")
text_splitter.split_text(text)
```




<pre class="custom">['Selecting the “right” amount of information to include in a summary is a difficult task. \nA good',
     'summary should be detailed and entity-centric without being overly dense and hard to follow. To',
     'better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to',
     'as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary',
     'before iteratively incorporating missing salient entities without increasing the length. Summaries',
     'genera']</pre>



```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=" ")
# Split the text file into chunks.
text_splitter.split_text(text)

# Split the document into chunks.
split_docs = text_splitter.split_documents(docs)
len(split_docs)
```




<pre class="custom">8</pre>



```python
split_docs[0]
```




<pre class="custom">Document(metadata={'source': 'https://www.bbc.com/news/business-68092814'}, page_content='Could AI \'trading bots\' transform the world of investing?Getty ImagesIt is hard for both humans and computers to predict stock market movementsSearch for "AI investing" online, and you\'ll be flooded with endless offers to let artificial intelligence manage your money.I recently spent half an hour finding out what so-called AI "trading bots" could apparently do with my investments.Many prominently suggest that they can give me lucrative returns. Yet as every reputable financial firm warns - your capital may be at risk.Or putting it more simply - you could lose your money - whether it is a human or a computer that is making stock market decisions on your behalf.Yet such has been the hype about the ability of AI over the past few years, that almost one in three investors would be happy to let a trading bot make all the decisions for them, according to one 2023 survey in the US.John Allan says investors should be more cautious about using AI. He is head of innovation and operations for the')</pre>



```python
# Load the content of the news article, split it into chunks, and index it.
loader = WebBaseLoader(
    web_paths=("https://www.bbc.com/news/business-68092814",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "main",
            attrs={"id": ["main-content"]},
        )
    ),
)

# Define the splitter.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=" ")

# Split the document while loading it.
split_docs = loader.load_and_split(text_splitter=text_splitter)
print(f"Number of documents: {len(docs)}")
docs[0].page_content[:500]
```

<pre class="custom">Number of documents: 1
</pre>




    'Could AI \'trading bots\' transform the world of investing?Getty ImagesIt is hard for both humans and computers to predict stock market movementsSearch for "AI investing" online, and you\'ll be flooded with endless offers to let artificial intelligence manage your money.I recently spent half an hour finding out what so-called AI "trading bots" could apparently do with my investments.Many prominently suggest that they can give me lucrative returns. Yet as every reputable financial firm warns - your '



### RecursiveTextSplitter
This text splitter is recommended for general text.

1. `How the text is split` : Based on a list of separators.
2. `How the chunk size is measured` : By the len of characters.

The `RecursiveCharacterTextSplitter` class provides functionality to recursively split text. This class takes parameters such as `chunk_size` to specify the size of the chunks to be split, `chunk_overlap` to define the overlap size between adjacent chunks, length_function to calculate the length of the chunks, and `is_separator_regex` to indicate whether the separator is a regular expression.

In the example, the chunk size is set to 100, the overlap size to 20, the length calculation function to `len` , and `is_separator_regex` is set to `False` to indicate that the separator is not a regular expression.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
```

```python
# Load a portion of the "Chain of Density" paper.
with open("data/chain-of-density.txt", "r", encoding="utf-8") as f:
    text = f.read()[:500]
```

```python
character_text_splitter = CharacterTextSplitter(
    chunk_size=100, chunk_overlap=10, separator=" "
)
for sent in character_text_splitter.split_text(text):
    print(sent)
print("===" * 20)
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=10
)
for sent in recursive_text_splitter.split_text(text):
    print(sent)
```

<pre class="custom">Selecting the “right” amount of information to include in a summary is a difficult task. 
    A good
    A good summary should be detailed and entity-centric without being overly dense and hard to follow.
    to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with
    with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial
    an initial entity-sparse summary before iteratively incorporating missing salient entities without
    without increasing the length. Summaries genera
    ============================================================
    Selecting the “right” amount of information to include in a summary is a difficult task.
    A good summary should be detailed and entity-centric without being overly dense and hard to follow.
    follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what
    with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an
    an initial entity-sparse summary before iteratively incorporating missing salient entities without
    without increasing the length. Summaries genera
</pre>

- Attempts to split the given document sequentially using the specified list of separators.
- Attempts splitting in order until the chunks are sufficiently small. The default list is ["\n\n", "\n", " ", ""].
- This generally has the effect of keeping all paragraphs (as well as sentences and words) as long as possible, while appearing to be the most semantically relevant pieces of text.


```python
# Check the default separators specified in recursive_text_splitter.
recursive_text_splitter._separators
```




<pre class="custom">['\n\n', '\n', ' ', '']</pre>



### Semantic Similarity

Text is split based on semantic similarity.

Source: [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)

At a high level, the process involves splitting the text into sentences, grouping them into sets of three, and then merging similar sentences in the embedding space.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Create a SemanticChunker.
semantic_text_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"), add_start_index=True)
```

```python
# Load a portion of the "Chain of Density" paper.
with open("data/chain-of-density.txt", "r", encoding="utf-8") as f:
    text = f.read()

for sent in semantic_text_splitter.split_text(text):
    print(sent)
    print("===" * 20)
```

<pre class="custom">Selecting the “right” amount of information to include in a summary is a difficult task. A good summary should be detailed and entity-centric without being overly dense and hard to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to as a “Chain of Density” (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary before iteratively incorporating missing salient entities without increasing the length. Summaries generated by CoD are more abstractive, exhibit more fusion, and have less of a lead bias than GPT-4 summaries generated by a vanilla prompt. We conduct a human preference study on 100 CNN DailyMail articles and find that that humans prefer GPT-4 summaries that are more dense than those generated by a vanilla prompt and almost as dense as human written summaries. Qualitative analysis supports the notion that there exists a tradeoff between infor-mativeness and readability. 500 annotated CoD summaries, as well as an extra 5,000 unannotated summaries, are freely available on HuggingFace. Introduction
    
    Automatic summarization has come a long way in the past few years, largely due to a paradigm shift away from supervised fine-tuning on labeled datasets to zero-shot prompting with Large Language Models (LLMs), such as GPT-4 (OpenAI, 2023). Without additional training, careful prompting can enable fine-grained control over summary characteristics, such as length (Goyal et al., 2022), topics (Bhaskar et al., 2023), and style (Pu and Demberg, 2023). An overlooked aspect is the information density of an summary. In theory, as a compression of another text, a summary should be denser–containing a higher concentration of information–than the source document. Given the high latency of LLM decoding (Kad-dour et al., 2023), covering more information in fewer words is a worthy goal, especially for real-time applications. Yet, how dense is an open question.
    ============================================================
    A summary is uninformative if it contains insufficient detail. If it contains too much information, however, it can be-come difficult to follow without having to increase the overall length. Conveying more information subject to a fixed token budget requires a combination of abstrac-tion, compression, and fusion. There is a limit to how much space can be made for additional information before becoming illegible or even factually incorrect.
    ============================================================
</pre>

## Step 3: Embedding

- [Link to official documentation - Embedding](https://python.langchain.com/docs/integrations/text_embedding)


### Paid Embeddings (OpenAI)

```python
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

# Step 3: Create Embeddings & Vectorstore
# Generate the vector store.
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
```

Below is a list of Embedding models supported by `OpenAI` :

The default model is `text-embedding-ada-002` .


| MODEL                  | ROUGH PAGES PER DOLLAR | EXAMPLE PERFORMANCE ON MTEB EVAL |
| ---------------------- | ---------------------- | -------------------------------- |
| text-embedding-3-small | 62,500                 | 62.3%                            |
| text-embedding-3-large | 9,615                  | 64.6%                            |
| text-embedding-ada-002 | 12,500                 | 61.0%                            |


```python
vectorstore = FAISS.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
```

### Free Open Source-Based Embeddings
1. HuggingFaceEmbeddings (Default model: sentence-transformers/all-mpnet-base-v2)
2. FastEmbedEmbeddings

**Note**
- When using embeddings, make sure to verify that the language you are using is supported.

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Generate the vector store. (Default model: sentence-transformers/all-mpnet-base-v2)
vectorstore = FAISS.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings()
)
```

<pre class="custom">/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
</pre>

```python
# %pip install fastembed
```

```python
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

vectorstore = FAISS.from_documents(documents=splits, embedding=FastEmbedEmbeddings())
```

## Step 4: Create Vectorstore

```python
from langchain_community.vectorstores import FAISS

# Apply FAISS DB
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
```

```python
from langchain_community.vectorstores import Chroma

# Apply Chroma DB
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
```

## Step 5: Create Retriever

A Retriever is an interface that returns documents when given an unstructured query.

The Retriever does not need to store documents; it only returns (or retrieves) them.

- [Link to official documentation - Retriever](https://python.langchain.com/docs/integrations/retrievers/)

The **Retriever** is created by using the `invoke()` method on the generated VectorStore.


### Similarity Retrieval

- The default setting is `similarity` , which uses cosine similarity.


```python
question = "Why did OpenAI and Scarlett Johansson have a conflict?"

retriever = vectorstore.as_retriever(search_type="similarity")
search_result = retriever.invoke(question)
print(search_result)
```

<pre class="custom">[Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content="ForbesInnovationEditors' PickThe Prompt: Scarlett Johansson Vs OpenAIPlus AI-generated kids draw predators on TikTok and Instagram. \nShare to FacebookShare to TwitterShare to Linkedin“I was shocked, angered and in disbelief,” Scarlett Johansson said about OpenAI's Sky voice for ChatGPT that sounds similar to her own.FilmMagic\nThe Prompt is a weekly rundown of AI’s buzziest startups, biggest breakthroughs, and business deals. To get it in your inbox, subscribe here.\n\n\nWelcome back to The Prompt.\n\nScarlett Johansson’s lawyers have demanded that OpenAI take down a voice for ChatGPT that sounds much like her own after she’d declined to work with the company to create it. The actress said in a statement provided to Forbes that her lawyers have asked the AI company to detail the “exact processes” it used to create the voice, which sounds eerily similar to Johansson’s voiceover work in the sci-fi movie Her. “I was shocked, angered and in disbelief,” she said."), Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content="The actress said in the statement that last September Sam Altman offered to hire her to voice ChatGPT, adding that her voice would be comforting to people. She turned down the offer, citing personal reasons. Two days before OpenAI launched its latest model, GPT-4o, Altman reached out again, asking her to reconsider. But before she could respond, the voice was used in a demo, where it flirted, laughed and sang on stage. (“Oh stop it! You’re making me blush,” the voice said to the employee presenting the demo.)\n\nOn Monday, OpenAI said it would take down the voice, while claiming that it is not “an imitation of Scarlett Johansson” and that it had partnered with professional voice actors to create it. But Altman’s one-word tweet – “Her” – posted after the demo last week only further fueled the connection between the AI’s voice and Johannson’s.\nNow, let’s get into the headlines.\nBIG PLAYSActor and filmmaker Donald Glover tests out Google's new AI video tools.GOOGLE"), Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content='The departure of these researchers also shone a light on OpenAI’s strict and binding nondisclosure agreements and off-boarding documents. Employees who refused to sign them when they left the company risked losing their vested equity in the company, according to Vox. OpenAI CEO Sam Altman responded on X saying “there was a provision about potential equity cancellation in our previous exit docs; although we never clawed anything back, it should never have been something we had in any documents or communication.”\nAI DEALS OF THE WEEKAlexandr Wang was just 19 when he started Scale. His cofounder, Lucy Guo, was 21.Scale AI'), Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content='TALENT RESHUFFLE\nKey safety researchers at OpenAI, including cofounder and Chief Scientist Ilya Sutskever and machine learning researcher Jan Leike, have resigned. The two led the company’s efforts to develop ways to control AI systems that might become smarter than humans and prevent them from going rogue at the company’s superalignment team, which now no longer exists, according to Wired. In a thread on X, Leike wrote: “Over the past few months my team has been sailing against the wind. Sometimes we were struggling for compute and it was getting harder and harder to get this crucial research done. Over the past years, safety culture and processes have taken a backseat to shiny products.”')]
</pre>

The `similarity_score_threshold` returns only the results with a `score_threshold` or higher in similarity-based retrieval.

```python
question = "Why did OpenAI and Scarlett Johansson have a conflict?"

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
)
search_result = retriever.invoke(question)
print(search_result)
```

<pre class="custom">WARNING:langchain_core.vectorstores.base:No relevant docs were retrieved using the relevance score threshold 0.8
</pre>

    []
    

Search using the `maximum marginal search result(mmr)` .


```python
question = "Why did OpenAI and Scarlett Johansson have a conflict?"

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
search_result = retriever.invoke(question)
print(search_result)
```

<pre class="custom">WARNING:chromadb.segment.impl.vector.local_hnsw:Number of requested results 20 is greater than number of elements in index 12, updating n_results = 12
</pre>

    [Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content="ForbesInnovationEditors' PickThe Prompt: Scarlett Johansson Vs OpenAIPlus AI-generated kids draw predators on TikTok and Instagram. \nShare to FacebookShare to TwitterShare to Linkedin“I was shocked, angered and in disbelief,” Scarlett Johansson said about OpenAI's Sky voice for ChatGPT that sounds similar to her own.FilmMagic\nThe Prompt is a weekly rundown of AI’s buzziest startups, biggest breakthroughs, and business deals. To get it in your inbox, subscribe here.\n\n\nWelcome back to The Prompt.\n\nScarlett Johansson’s lawyers have demanded that OpenAI take down a voice for ChatGPT that sounds much like her own after she’d declined to work with the company to create it. The actress said in a statement provided to Forbes that her lawyers have asked the AI company to detail the “exact processes” it used to create the voice, which sounds eerily similar to Johansson’s voiceover work in the sci-fi movie Her. “I was shocked, angered and in disbelief,” she said."), Document(metadata={'source': 'https://www.forbes.com/sites/rashishrivastava/2024/05/21/the-prompt-scarlett-johansson-vs-openai/'}, page_content='TALENT RESHUFFLE\nKey safety researchers at OpenAI, including cofounder and Chief Scientist Ilya Sutskever and machine learning researcher Jan Leike, have resigned. The two led the company’s efforts to develop ways to control AI systems that might become smarter than humans and prevent them from going rogue at the company’s superalignment team, which now no longer exists, according to Wired. In a thread on X, Leike wrote: “Over the past few months my team has been sailing against the wind. Sometimes we were struggling for compute and it was getting harder and harder to get this crucial research done. Over the past years, safety culture and processes have taken a backseat to shiny products.”')]
    

### Create a variety of queries

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

question = "Why did OpenAI and Scarlett Johansson have a conflict?"

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)
```

```python
# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

```python
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)
```

<pre class="custom"><ipython-input-43-1b4fb66403e0>:1: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
      unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    INFO:langchain.retrievers.multi_query:Generated queries: ['What was the nature of the disagreement between OpenAI and Scarlett Johansson?  ', 'Can you explain the reasons behind the conflict involving OpenAI and Scarlett Johansson?  ', 'What led to the tensions between OpenAI and Scarlett Johansson?']
</pre>




    4



### Ensemble Retriever
**BM25 Retriever + Embedding-based Retriever**

- `BM25 retriever` (Keyword Search, Sparse Retriever): Based on TF-IDF, considering term frequency and document length normalization.
- `Embedding-based retriever` (Contextual Search, Dense Retriever): Transforms text into embedding vectors and retrieves documents based on vector similarity (e.g. cosine similarity, dot product). This reflects the semantic similarity of words.
- `Ensemble retriever` : Combines BM25 and embedding-based retrievers to combine the term frequency of keyword searches with the semantic similarity of contextual searches.

**Note**

TF-IDF(Term Frequency - Inverse Document Frequency) : TF-IDF evaluates words that frequently appear in a specific document as highly important, while words that frequently appear across all documents are considered less important.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

```python
doc_list = [
    "We saw a seal swimming in the ocean.",
    "The seal is clapping its flippers.",
    "Make sure the envelope has a proper seal before sending it.",
    "Every official document requires a seal to authenticate it.",
]

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 4

faiss_vectorstore = FAISS.from_texts(doc_list, OpenAIEmbeddings(model="text-embedding-3-small"))
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

```python
def pretty_print(docs):
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}")
```

```python
sample_query = "The seal rested on a rock."
print(f"[Query]\n{sample_query}\n")
relevant_docs = bm25_retriever.invoke(sample_query)
print("[BM25 Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = faiss_retriever.invoke(sample_query)
print("[FAISS Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = ensemble_retriever.invoke(sample_query)
print("[Ensemble Retriever]")
pretty_print(relevant_docs)
```

<pre class="custom">[Query]
    The seal rested on a rock.
    
    [BM25 Retriever]
    [1] The seal is clapping its flippers.
    [2] We saw a seal swimming in the ocean.
    [3] Every official document requires a seal to authenticate it.
    [4] Make sure the envelope has a proper seal before sending it.
    ============================================================
    [FAISS Retriever]
    [1] The seal is clapping its flippers.
    [2] We saw a seal swimming in the ocean.
    [3] Every official document requires a seal to authenticate it.
    [4] Make sure the envelope has a proper seal before sending it.
    ============================================================
    [Ensemble Retriever]
    [1] The seal is clapping its flippers.
    [2] We saw a seal swimming in the ocean.
    [3] Every official document requires a seal to authenticate it.
    [4] Make sure the envelope has a proper seal before sending it.
</pre>

```python
sample_query = "Ensure the package is securely sealed before handing it to the courier."
print(f"[Query]\n{sample_query}\n")
relevant_docs = bm25_retriever.invoke(sample_query)
print("[BM25 Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = faiss_retriever.invoke(sample_query)
print("[FAISS Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = ensemble_retriever.invoke(sample_query)
print("[Ensemble Retriever]")
pretty_print(relevant_docs)
```

<pre class="custom">[Query]
    Ensure the package is securely sealed before handing it to the courier.
    
    [BM25 Retriever]
    [1] The seal is clapping its flippers.
    [2] Every official document requires a seal to authenticate it.
    [3] Make sure the envelope has a proper seal before sending it.
    [4] We saw a seal swimming in the ocean.
    ============================================================
    [FAISS Retriever]
    [1] Make sure the envelope has a proper seal before sending it.
    [2] Every official document requires a seal to authenticate it.
    [3] The seal is clapping its flippers.
    [4] We saw a seal swimming in the ocean.
    ============================================================
    [Ensemble Retriever]
    [1] The seal is clapping its flippers.
    [2] Make sure the envelope has a proper seal before sending it.
    [3] Every official document requires a seal to authenticate it.
    [4] We saw a seal swimming in the ocean.
</pre>

```python
sample_query = "The certificate must bear an official seal to be considered valid."
print(f"[Query]\n{sample_query}\n")
relevant_docs = bm25_retriever.invoke(sample_query)
print("[BM25 Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = faiss_retriever.invoke(sample_query)
print("[FAISS Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = ensemble_retriever.invoke(sample_query)
print("[Ensemble Retriever]")
pretty_print(relevant_docs)
```

<pre class="custom">[Query]
    The certificate must bear an official seal to be considered valid.
    
    [BM25 Retriever]
    [1] Every official document requires a seal to authenticate it.
    [2] The seal is clapping its flippers.
    [3] We saw a seal swimming in the ocean.
    [4] Make sure the envelope has a proper seal before sending it.
    ============================================================
    [FAISS Retriever]
    [1] Every official document requires a seal to authenticate it.
    [2] Make sure the envelope has a proper seal before sending it.
    [3] The seal is clapping its flippers.
    [4] We saw a seal swimming in the ocean.
    ============================================================
    [Ensemble Retriever]
    [1] Every official document requires a seal to authenticate it.
    [2] The seal is clapping its flippers.
    [3] Make sure the envelope has a proper seal before sending it.
    [4] We saw a seal swimming in the ocean.
</pre>

```python
sample_query = "animal"

print(f"[Query]\n{sample_query}\n")
relevant_docs = bm25_retriever.invoke(sample_query)
print("[BM25 Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = faiss_retriever.invoke(sample_query)
print("[FAISS Retriever]")
pretty_print(relevant_docs)
print("===" * 20)
relevant_docs = ensemble_retriever.invoke(sample_query)
print("[Ensemble Retriever]")
pretty_print(relevant_docs)
```

<pre class="custom">[Query]
    animal
    
    [BM25 Retriever]
    [1] Every official document requires a seal to authenticate it.
    [2] Make sure the envelope has a proper seal before sending it.
    [3] The seal is clapping its flippers.
    [4] We saw a seal swimming in the ocean.
    ============================================================
    [FAISS Retriever]
    [1] We saw a seal swimming in the ocean.
    [2] The seal is clapping its flippers.
    [3] Every official document requires a seal to authenticate it.
    [4] Make sure the envelope has a proper seal before sending it.
    ============================================================
    [Ensemble Retriever]
    [1] Every official document requires a seal to authenticate it.
    [2] We saw a seal swimming in the ocean.
    [3] The seal is clapping its flippers.
    [4] Make sure the envelope has a proper seal before sending it.
</pre>

## Step 6: Create Prompt

Prompt engineering plays a crucial role in deriving the desired outputs based on the given data( `context` ) .

[TIP1]

1. If important information is missing from the results provided by the `retriever `, you should modify the `retriever` logic.
2. If the results from the `retriever` contain sufficient information, but the llm fails to extract the key information or doesn't produce the output in the desired format, you should adjust the prompt.

[TIP2]

1. LangSmith's **hub** contains numerous verified prompts.
2. Utilizing or slightly modifying these verified prompts can save both cost and time.

- https://smith.langchain.com/hub/search?q=rag


```python
from langchain import hub
```

```python
prompt = hub.pull("rlm/rag-prompt")
prompt.pretty_print()
```

<pre class="custom">================================[1m Human Message [0m=================================
    
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: [33;1m[1;3m{question}[0m 
    Context: [33;1m[1;3m{context}[0m 
    Answer:
</pre>

## Step 7: Create LLM

Select one of the OpenAI models:

- `gpt-4o` : OpenAI GPT-4o model
- `gpt-4o-mini` : OpenAI GPT-4o-mini model

For detailed pricing information, please refer to the [OpenAI API Model List / Pricing](https://openai.com/api/pricing/)

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
```

You can check token usage in the following way.

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = model.invoke("Where is the capital of South Korea?")
print(cb)
```

<pre class="custom">Tokens Used: 24
    	Prompt Tokens: 15
    		Prompt Tokens Cached: 0
    	Completion Tokens: 9
    		Reasoning Tokens: 0
    Successful Requests: 1
    Total Cost (USD): $7.65e-06
</pre>

### Use Higgingface

You need a Hugging Face token to access LLMs on HuggingFace.

You can easily download and use open-source models available on HuggingFace.

You can also check the open-source leaderboard, which improves performance daily, at the link below:

- [HuggingFace LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

**Note**

Hugging Face's free API has a 10GB size limit.
For example, the microsoft/Phi-3-mini-4k-instruct model is 11GB, making it inaccessible via the free API.

Choose one of the options below:

1. Option: Use Hugging Face Inference Endpoints

Activate Inference Endpoints through a paid plan to perform large-scale model inference.

2. Option: Run the model locally

Use the transformers library to run the microsoft/Phi-3-mini-4k-instruct model in a local environment (GPU recommended).

3. Option: Use a smaller model.

Reduce the model size to one supported by the free API and execute it.


```python
# Creating a HuggingFaceEndpoint object
from langchain_huggingface import HuggingFaceEndpoint

repo_id = "microsoft/Phi-3-mini-4k-instruct"

hugging_face_llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.1,
)

```

```python
hugging_face_llm.invoke("Where is the capital of South Korea?")
```




<pre class="custom">'\n\n# Answer\nThe capital of South Korea is Seoul.'</pre>



## RAG Template Experiment


```python
# Step 1: Load Documents
# Load the documents, split them into chunks, and index them.
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the PDF file. Enter the file path.
file_path = "data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf"
loader = PyPDFLoader(file_path=file_path)

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

split_docs = loader.load_and_split(text_splitter=text_splitter)

# Step 3, 4: Embeding & Create Vectorstore
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=split_docs, embedding=embedding)

# Step 5: Create Retriever
# Search for documents that match the user's query.

# Retrieve the top K documents with the highest similarity.
k = 3

# Initialize the (Sparse) BM25 retriever and (Dense) FAISS retriever.
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = k

faiss_vectorstore = FAISS.from_documents(split_docs, embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# Step 6: Create Prompt

prompt = hub.pull("rlm/rag-prompt")

# Step 7: Create LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def format_docs(docs):
    # Combine the retrieved document results into a single paragraph.
    return "\n\n".join(doc.page_content for doc in docs)


# Step 8: Create Chain
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run Chain: Input a query about the document and output the answer.

question = "Which region's approach to artificial intelligence is the focus of this document?"
response = rag_chain.invoke(question)

# Get Output
print(f"PDF Path: {file_path}")
print(f"Number of documents: {len(split_docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")
```

<pre class="custom">PDF Path: data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf
    Number of documents: 86
    ============================================================
    [HUMAN]
    Which region's approach to artificial intelligence is the focus of this document?
    
    [AI]
    The focus of this document is on the European approach to artificial intelligence. It discusses the strategies and policies implemented by the European Commission and EU Member States to enhance AI development and governance in Europe. The document emphasizes the importance of trust, data governance, and collaboration in fostering AI innovation within the region.
</pre>

Document: A European Approach to Artificial Intelligence - A Policy Perspective.pdf

- LangSmith: https://smith.langchain.com/public/0951c102-de61-482b-b42a-6e7d78f02107/r


```python
question = "Which region's approach to artificial intelligence is the focus of this document?"
response = rag_chain.invoke(question)
print(response)

```

<pre class="custom">The focus of this document is on the European approach to artificial intelligence. It discusses the strategies and policies implemented by the European Commission and EU Member States to enhance AI development and governance in Europe. The document emphasizes the importance of trust, data governance, and collaboration in fostering AI innovation within the region.
</pre>

Document: A European Approach to Artificial Intelligence - A Policy Perspective.pdf

- LangSmith: https://smith.langchain.com/public/c968bf7e-e22e-4eb1-a76a-b226eedc6c51/r

```python
question = "What is the primary principle of the European AI approach?"
response = rag_chain.invoke(question)
print(response)
```

<pre class="custom">The primary principle of the European AI approach is to place people at the center of AI development, often referred to as "human-centric AI." This approach aims to support technological and industrial capacity, prepare for socio-economic changes, and ensure an appropriate ethical and legal framework. It emphasizes the need for AI to comply with the law, fulfill ethical principles, and be robust to achieve "trustworthy AI."
</pre>

Ask a question unrelated to the document.

- LangSmith: https://smith.langchain.com/public/d8a49d52-3a63-4206-9166-58605bd990a6/r

```python
question = "What is the obligation of the United States in AI?"
response = rag_chain.invoke(question)
print(response)
```

<pre class="custom">The obligation of the United States in AI primarily involves ensuring ethical standards, transparency, and accountability in AI development and deployment. This includes addressing concerns related to privacy, data governance, and the societal impacts of AI technologies. Additionally, the U.S. may need to engage in international cooperation to establish norms and regulations that promote responsible AI use.
</pre>

```python

```

```python

```

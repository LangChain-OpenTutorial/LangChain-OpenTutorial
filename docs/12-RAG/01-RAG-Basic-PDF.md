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

# Understanding the basic structure of RAG

- Author: [Sun Hyoung Lee](https://github.com/LEE1026icarus)
- Design: 
- Peer Review: 
- Proofread: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

### 1. Pre-processing - Steps 1 to 4
![rag-1.png](./img/12-rag-rag-basic-pdf-rag-process-01.png)

The pre-processing stage involves four steps to load, split, embed, and store documents into a Vector DB (database).

- **Step 1: Document Load** : Load the document content.  
- **Step 2: Text Split** : Split the document into chunks based on specific criteria.  
- **Step 3: Embedding** : Generate embeddings for the chunks and prepare them for storage.  
- **Step 4: Vector DB Storage** : Store the embedded chunks in the database.  

### 2. RAG Execution (RunTime) - Steps 5 to 8
![rag-2.png](./img/12-rag-rag-basic-pdf-rag-process-02.png)

- **Step 5: Retriever** : Define a retriever to fetch results from the database based on the input query. Retrievers use search algorithms and are categorized as **dense** or **sparse**:
  - **Dense** : Similarity-based search.
  - **Sparse** : Keyword-based search.

- **Step 6: Prompt** : Create a prompt for executing RAG. The `context` in the prompt includes content retrieved from the document. Through prompt engineering, you can specify the format of the answer.  

- **Step 7: LLM** : Define the language model (e.g., GPT-3.5, GPT-4, Claude).  

- **Step 8: Chain** : Create a chain that connects the prompt, LLM, and output.  

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [RAG Basic Pipeline](#rag-basic-pipeline)
- [Complete Code](#complete-code)

### References

- [LangChain How-to guides : Q&A with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
------

Document Used for Practice
A European Approach to Artificial Intelligence - A Policy Perspective

- Author: EIT Digital and 5 EIT KICs (EIT Manufacturing, EIT Urban Mobility, EIT Health, EIT Climate-KIC, EIT Digital)
- Link: https://eit.europa.eu/sites/default/files/eit-digital-artificial-intelligence-report.pdf
- File Name: A European Approach to Artificial Intelligence - A Policy Perspective.pdf

 _Please copy the downloaded file to the data folder for practice._ 

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

 **[Note]** 
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [ `langchain-opentutorial` ](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.


```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

Set the API key.

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [   
        "langchain_community",
        "langsmith"
        "langchain"
        "langchain_text_splitters"
        "langchain_core"
        "langchain_openai"
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {   "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "RAG-Basic-PDF",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Configuration file for managing API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## RAG Basic Pipeline

Below is the skeleton code for understanding the basic structure of RAG (Retrieval Augmented Generation).

The content of each module can be adjusted to fit specific scenarios, allowing for iterative improvement of the structure to suit the documents.

(Different options or new techniques can be applied at each step.)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
```

```python
# Step 1: Load Documents
loader = PyMuPDFLoader("./data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf")
docs = loader.load()
print(f"Number of pages in the document: {len(docs)}")
```

<pre class="custom">Number of pages in the document: 24
</pre>

Print the content of the page.

```python
print(docs[10].page_content)
```

<pre class="custom">A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    11
    GENERIC 
    There are five issues that, though from slightly different angles, 
    are considered strategic and a potential source of barriers and 
    bottlenecks: data, organisation, human capital, trust, markets. The 
    availability and quality of data, as well as data governance are of 
    strategic importance. Strictly technical issues (i.e., inter-operabi-
    lity, standardisation) are mostly being solved, whereas internal and 
    external data governance still restrain the full potential of AI Inno-
    vation. Organisational resources and, also, cognitive and cultural 
    routines are a challenge to cope with for full deployment. On the 
    one hand, there is the issue of the needed investments when evi-
    dence on return is not yet consolidated. On the other hand, equally 
    important, are cultural conservatism and misalignment between 
    analytical and business objectives. Skills shortages are a main 
    bottleneck in all the four sectors considered in this report where 
    upskilling, reskilling, and new skills creation are considered crucial. 
    For many organisations data scientists are either too expensive or 
    difficult to recruit and retain. There is still a need to build trust on 
    AI, amongst both the final users (consumers, patients, etc.) and 
    intermediate / professional users (i.e., healthcare professionals). 
    This is a matter of privacy and personal data protection, of building 
    a positive institutional narrative backed by mitigation strategies, 
    and of cumulating evidence showing that benefits outweigh costs 
    and risks. As demand for AI innovations is still limited (in many 
    sectors a ‘wait and see’ approach is prevalent) this does not fa-
    vour the emergence of a competitive supply side. Few start-ups 
    manage to scale up, and many are subsequently bought by a few 
    large dominant players. As a result of the fact that these issues 
    have not yet been solved on a large scale, using a 5 levels scale 
    GENERIC AND CONTEXT DEPENDING 
    OPPORTUNITIES AND POLICY LEVERS
    of deployment maturity (1= not started; 2= experimentation; 3= 
    practitioner use; 4= professional use; and 5= AI driven companies), 
    it seems that, in all four vertical domains considered, adoption re-
    mains at level 2 (experimentation) or 3 (practitioner use), with only 
    few advanced exceptions mostly in Manufacturing and Health-
    care. In Urban Mobility, as phrased by interviewed experts, only 
    lightweight AI applications are widely adopted, whereas in the Cli-
    mate domain we are just at the level of early predictive models. 
    Considering the different areas of AI applications, regardless of the 
    domains, the most adopted ones include predictive maintenance, 
    chatbots, voice/text recognition, NPL, imagining, computer vision 
    and predictive analytics.
    MANUFACTURING 
    The manufacturing sector is one of the leaders in application of 
    AI technologies; from significant cuts in unplanned downtime to 
    better designed products, manufacturers are applying AI-powe-
    red analytics to data to improve efficiency, product quality and 
    the safety of employees. The key application of AI is certainly in 
    predictive maintenance. Yet, the more radical transformation of 
    manufacturing will occur when manufacturers will move to ‘ser-
    vice-based’ managing of the full lifecycle from consumers pre-
    ferences to production and delivery (i.e., the Industry 4.0 vision). 
    Manufacturing companies are investing into this vision and are 
    keen to protect their intellectual property generated from such in-
    vestments. So, there is a concern that a potential new legislative 
    action by the European Commission, which would follow the prin-
    ciples of the GDPR and the requirements of the White Paper, may 
    
</pre>

Check the metadata.

```python
docs[10].__dict__
```




<pre class="custom">{'id': None,
     'metadata': {'source': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
      'file_path': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
      'page': 10,
      'total_pages': 24,
      'format': 'PDF 1.4',
      'title': '',
      'author': '',
      'subject': '',
      'keywords': '',
      'creator': 'Adobe InDesign 17.3 (Macintosh)',
      'producer': 'Adobe PDF Library 16.0.7',
      'creationDate': "D:20220823105611+02'00'",
      'modDate': "D:20220823105617+02'00'",
      'trapped': ''},
     'page_content': 'A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE\n11\nGENERIC \nThere are five issues that, though from slightly different angles, \nare considered strategic and a potential source of barriers and \nbottlenecks: data, organisation, human capital, trust, markets. The \navailability and quality of data, as well as data governance are of \nstrategic importance. Strictly technical issues (i.e., inter-operabi-\nlity, standardisation) are mostly being solved, whereas internal and \nexternal data governance still restrain the full potential of AI Inno-\nvation. Organisational resources and, also, cognitive and cultural \nroutines are a challenge to cope with for full deployment. On the \none hand, there is the issue of the needed investments when evi-\ndence on return is not yet consolidated. On the other hand, equally \nimportant, are cultural conservatism and misalignment between \nanalytical and business objectives. Skills shortages are a main \nbottleneck in all the four sectors considered in this report where \nupskilling, reskilling, and new skills creation are considered crucial. \nFor many organisations data scientists are either too expensive or \ndifficult to recruit and retain. There is still a need to build trust on \nAI, amongst both the final users (consumers, patients, etc.) and \nintermediate / professional users (i.e., healthcare professionals). \nThis is a matter of privacy and personal data protection, of building \na positive institutional narrative backed by mitigation strategies, \nand of cumulating evidence showing that benefits outweigh costs \nand risks. As demand for AI innovations is still limited (in many \nsectors a ‘wait and see’ approach is prevalent) this does not fa-\nvour the emergence of a competitive supply side. Few start-ups \nmanage to scale up, and many are subsequently bought by a few \nlarge dominant players. As a result of the fact that these issues \nhave not yet been solved on a large scale, using a 5 levels scale \nGENERIC AND CONTEXT DEPENDING \nOPPORTUNITIES AND POLICY LEVERS\nof deployment maturity (1= not started; 2= experimentation; 3= \npractitioner use; 4= professional use; and 5= AI driven companies), \nit seems that, in all four vertical domains considered, adoption re-\nmains at level 2 (experimentation) or 3 (practitioner use), with only \nfew advanced exceptions mostly in Manufacturing and Health-\ncare. In Urban Mobility, as phrased by interviewed experts, only \nlightweight AI applications are widely adopted, whereas in the Cli-\nmate domain we are just at the level of early predictive models. \nConsidering the different areas of AI applications, regardless of the \ndomains, the most adopted ones include predictive maintenance, \nchatbots, voice/text recognition, NPL, imagining, computer vision \nand predictive analytics.\nMANUFACTURING \nThe manufacturing sector is one of the leaders in application of \nAI technologies; from significant cuts in unplanned downtime to \nbetter designed products, manufacturers are applying AI-powe-\nred analytics to data to improve efficiency, product quality and \nthe safety of employees. The key application of AI is certainly in \npredictive maintenance. Yet, the more radical transformation of \nmanufacturing will occur when manufacturers will move to ‘ser-\nvice-based’ managing of the full lifecycle from consumers pre-\nferences to production and delivery (i.e., the Industry 4.0 vision). \nManufacturing companies are investing into this vision and are \nkeen to protect their intellectual property generated from such in-\nvestments. So, there is a concern that a potential new legislative \naction by the European Commission, which would follow the prin-\nciples of the GDPR and the requirements of the White Paper, may \n',
     'type': 'Document'}</pre>



```python
# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"Number of split chunks: {len(split_documents)}")
```

<pre class="custom">Number of split chunks: 163
</pre>

```python
# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings()
```

```python
# Step 4: Create and Save the Database
# Create a vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
```

```python
for doc in vectorstore.similarity_search("URBAN MOBILITY"):
    print(doc.page_content)
```

<pre class="custom">A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    14
    Table 3: Urban Mobility: concerns, opportunities and policy levers.
    URBAN MOBILITY
    The adoption of AI in the management of urban mobility systems 
    brings different sets of benefits for private stakeholders (citizens, 
    private companies) and public stakeholders (municipalities, trans-
    portation service providers). So far only light-weight task specific 
    AI applications have been deployed (i.e., intelligent routing, sharing
    A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    15
    One of the most interesting development close to scale up is the 
    creation of platforms, which are fed by all different data sources 
    of transport services (both private and public) and provide the ci-
    tizens a targeted recommendation on the best way to travel, also 
    based on personal preferences and characteristics. 
    Urban Mobility should focus on what is already potentially avai-
    apps, predictive models based on citizens’ location and personal 
    data). On the other hand, the most advanced and transformative 
    AI applications, such as autonomous vehicles are lagging behind, 
    especially if compared to the US or China. The key challenge for AI 
    deployment in Urban Mobility sector is the need to find a common 
    win-win business model across a diversity of public and private 
    sector players with different organisational objectives, cultures,
    care. In Urban Mobility, as phrased by interviewed experts, only 
    lightweight AI applications are widely adopted, whereas in the Cli-
    mate domain we are just at the level of early predictive models. 
    Considering the different areas of AI applications, regardless of the 
    domains, the most adopted ones include predictive maintenance, 
    chatbots, voice/text recognition, NPL, imagining, computer vision 
    and predictive analytics.
    MANUFACTURING
</pre>

```python
# Step 5: Create Retriever
# Search and retrieve information contained in the documents.
retriever = vectorstore.as_retriever()
```

Send a query to the retriever and check the resulting chunks.

```python
retriever.invoke("What is the phased implementation timeline for the EU AI Act?")
```




<pre class="custom">[Document(id='fdfb5187-141a-4693-b5d0-e1066b0ef27f', metadata={'source': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'file_path': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'page': 9, 'total_pages': 24, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'Adobe InDesign 17.3 (Macintosh)', 'producer': 'Adobe PDF Library 16.0.7', 'creationDate': "D:20220823105611+02'00'", 'modDate': "D:20220823105617+02'00'", 'trapped': ''}, page_content='A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE\n10\nrequirements becomes mandatory in all sectors and create bar-\nriers especially for innovators and SMEs. Public procurement ‘data \nsovereignty clauses’ induce large players to withdraw from AI for \nurban ecosystems. Strict liability sanctions block AI in healthcare, \nwhile limiting space of self-driving experimentation. The support \nmeasures to boost European AI are not sufficient to offset the'),
     Document(id='5aada0ed-9a07-4c9b-a290-d24856d64494', metadata={'source': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'file_path': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'page': 22, 'total_pages': 24, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'Adobe InDesign 17.3 (Macintosh)', 'producer': 'Adobe PDF Library 16.0.7', 'creationDate': "D:20220823105611+02'00'", 'modDate': "D:20220823105617+02'00'", 'trapped': ''}, page_content='A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE\n23\nACKNOWLEDGEMENTS\nIn the context of their strategic innovation activities for Europe, five EIT Knowledge and Innovation Communities (EIT Manufacturing, EIT Ur-\nban Mobility, EIT Health, EIT Climate-KIC, and EIT Digital as coordinator) decided to launch a study that identifies general and sector specific \nconcerns and opportunities for the deployment of AI in Europe.'),
     Document(id='37657411-894d-4e9c-975b-d1a99ef0e20a', metadata={'source': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'file_path': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'page': 21, 'total_pages': 24, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'Adobe InDesign 17.3 (Macintosh)', 'producer': 'Adobe PDF Library 16.0.7', 'creationDate': "D:20220823105611+02'00'", 'modDate': "D:20220823105617+02'00'", 'trapped': ''}, page_content='sion/presscorner/detail/en/IP_18_6689.\nEuropean Commission. (2020a). White Paper on Artificial Intelligence. A European Ap-\nproach to Excellence and Trust. COM(2020) 65 final, Brussels: European Commission. \nEuropean Commission. (2020b). A European Strategy to Data. COM(2020) 66 final, Brus-\nsels: European Commission. \nEuropean Parliament. (2020). Digital sovereignty for Europe. Brussels: European Parliament \n(retrieved from: https://www.europarl.europa.eu/RegData/etudes/BRIE/2020/651992/'),
     Document(id='1aa90862-fe35-4797-ad6a-225f9da47824', metadata={'source': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'file_path': './data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf', 'page': 5, 'total_pages': 24, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'Adobe InDesign 17.3 (Macintosh)', 'producer': 'Adobe PDF Library 16.0.7', 'creationDate': "D:20220823105611+02'00'", 'modDate': "D:20220823105617+02'00'", 'trapped': ''}, page_content='ries and is the result of a combined effort from five EIT KICs (EIT \nManufacturing, EIT Urban Mobility, EIT Health, EIT Climate-KIC, \nand EIT Digital as coordinator). It identifies both general and sec-\ntor specific concerns and opportunities for the further deployment \nof AI in Europe. Starting from the background and policy context \noutlined in this introduction, some critical aspects of AI are fur-\nther discussed in Section 2. Next, in Section 3 four scenarios')]</pre>



```python
# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)
```

```python
# Step 7: Setup LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
```

```python
# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Input a query (question) into the created chain and execute it.

```python
# Run Chain
# Input a query about the document and print the response.
question = "Where has the application of AI in healthcare been confined to so far?"
response = chain.invoke(question)
print(response)
```

<pre class="custom">The application of AI in healthcare has so far been confined to administrative tasks, such as Natural Language Processing to extract information from clinical notes or predictive scheduling of visits, and diagnostic tasks, including machine and deep learning applied to imaging in radiology, pathology, and dermatology.
</pre>

## Complete Code
This is a combined code that integrates steps 1 through 8.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Step 1: Load Documents
loader = PyMuPDFLoader("./data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf")
docs = loader.load()

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# Step 3: Generate Embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Create and Save the Database
# Create a vector store.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Create Retriever
# Search and retrieve information contained in the documents.
retriever = vectorstore.as_retriever()

# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# Step 7: Load LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
# Run Chain
# Input a query about the document and print the response.
question = "Where has the application of AI in healthcare been confined to so far?"
response = chain.invoke(question)
print(response)
```

<pre class="custom">The application of AI in healthcare has so far been confined to administrative tasks, such as Natural Language Processing to extract information from clinical notes or predictive scheduling of visits, and diagnostic tasks, including machine and deep learning applied to imaging in radiology, pathology, and dermatology.
</pre>

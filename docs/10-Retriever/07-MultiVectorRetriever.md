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

# MultiVectorRetriever

- Author: [YooKyung Jeon](https://github.com/sirena1)
- Peer Review: [choincnp](https://github.com/choincnp), [Hye-yoonJeong](https://github.com/Hye-yoonJeong)
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/07-MultiVectorRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/07-MultiVectorRetriever.ipynb)
## Overview

```MultiVectorRetriever``` enables efficient querying of documents in various contexts. 
It allows documents to be stored and managed with **multiple vectors**, significantly enhancing the accuracy and efficiency of information retrieval.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Methods to Create Multiple Vectors per Document](#methods-to-create-multiple-vectors-per-document)
- [Creating Smaller Chunks](#creating-smaller-chunks)
- [Storing Summary Embeddings](#storing-summary-embeddings)
- [Utilizing Hypothetical Questions](#utilizing-hypothetical-questions)

### References

- [Retriever](https://python.langchain.com/docs/integrations/retrievers)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain",
        "langchain_chroma",
        "langchain_openai",
        "langchain_core",
        "langchain_text_splitters",
        "pymupdf"
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
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "07-MultiVectorRetriever",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Alternatively, environment variables can also be set using a ```.env``` file.

**[Note]**

- This is not necessary if you've already set the environment variables in the previous step.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Methods to Create Multiple Vectors per Document

There are several approaches to creating multiple vectors for a given document.
Some of them include:

1. **Creating Smaller Chunks**: Split the document into smaller chunks and embed them. This method enables a more granular focus on specific parts of the document. It can be implemented using the ```ParentDocumentRetriever```, making it easier to explore detailed information.

2. **Storing Summary Embeddings**: Create a summary for each document and embed it along with the original document. Summary embeddings are particularly useful for quickly grasping the core content of a document. By focusing only on the summary instead of analyzing the entire document, efficiency can be significantly improved.

3. **Utilizing Hypothetical Questions**: Create relevant hypothetical questions for each document and embed them along with the original document. This approach is helpful when deeper exploration of specific topics or content is needed. Hypothetical questions enable a broader perspective on the document's content, facilitating a more comprehensive understanding.

4. **Manual Addition**: Users can manually add specific questions or queries that should be considered during document retrieval. This method provides users with more control over the search process, allowing for customized searches tailored to their specific needs.

Let's first preprocess the document data by loading data from a text file, and splitting the loaded documents into specified sizes.

The split documents can later be used for tasks such as vectorization and retrieval.

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf")
docs = loader.load()
```

The original documents loaded from the data are stored in the ```docs``` variable.

```python
print(docs[5].page_content[:500])
```

<pre class="custom">A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    6
    data for innovators, particularly in the business-to-business (B2B) 
    or government-to-citizens (G2C) domains: e.g. by open access to 
    government data in sectors such as transportation and health-
    care (Burghin et al., 2019), privacy-preserving data marketplaces 
    for companies to share data (de Streel et al., 2019). The genuine 
    concern for innovators access to data is shown by the city of Bar-
    celona where ‘data sovereignty’
</pre>

## Creating Smaller Chunks

When searching through large volumes of information, embedding data into smaller chunks can be highly beneficial.

With ```MultiVectorRetriever```, documents can be stored and managed as multiple vectors.

- The original documents are stored in the ```docstore```.
- The embedded documents are stored in the ```vectorstore```.

This allows for splitting documents into smaller units, enabling more accurate searches. Additionally, the contents of the original document can be accessed when needed.


```python
import uuid
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Vector store for indexing child chunks
vectorstore = Chroma(
    collection_name="small_bigger_chunks",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# Storage layer for parent documents
store = InMemoryStore()

id_key = "doc_id"

# Retriever (initially empty)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# Generate document IDs
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Check two of the generated IDs
print(doc_ids[:2])
```

<pre class="custom">['46aba7dd-39cd-4852-beed-e8e0560e7a98', 'dc741e0e-89a0-41b5-8090-688ec75748b8']
</pre>

Here we define a ```parent_text_splitter``` for splitting into larger chunks and a ```child_text_splitter``` for splitting into smaller chunks.

```python
# Create a RecursiveCharacterTextSplitter object for larger chunks
parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=600)

# Splitter to be used for generating smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
```

Create **parent documents** , which are larger chunks.

```python
parent_docs = []

for i, doc in enumerate(docs):
    # Retrieve the ID of the current document
    _id = doc_ids[i]
    # Split the current document into smaller parent documents
    parent_doc = parent_text_splitter.split_documents([doc])

    for _doc in parent_doc:
        # Set the document ID in the metadata
        _doc.metadata[id_key] = _id
    parent_docs.extend(parent_doc)
```

Verify the ```doc_id``` assigned to ```parent_docs```


```python
# Check the metadata of the generated parent documents.
parent_docs[0].metadata
```




<pre class="custom">{'source': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
     'file_path': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
     'page': 0,
     'total_pages': 24,
     'format': 'PDF 1.4',
     'title': '',
     'author': '',
     'subject': '',
     'keywords': '',
     'creator': 'Adobe InDesign 15.1 (Macintosh)',
     'producer': 'Adobe PDF Library 15.0',
     'creationDate': "D:20200922223534+02'00'",
     'modDate': "D:20200922223544+02'00'",
     'trapped': '',
     'doc_id': '46aba7dd-39cd-4852-beed-e8e0560e7a98'}</pre>



Create **child documents** , which are relatively smaller chunks.

```python
child_docs = []
for i, doc in enumerate(docs):
    # Retrieve the ID of the current document
    _id = doc_ids[i]
    # Split the current document into child documents
    child_doc = child_text_splitter.split_documents([doc])
    for _doc in child_doc:
        # Set the document ID in the metadata
        _doc.metadata[id_key] = _id
    child_docs.extend(child_doc)
```

Verify the ```doc_id``` assigned to ```child_docs```.

```python
# Check the metadata of the generated child documents.
child_docs[0].metadata
```




<pre class="custom">{'source': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
     'file_path': 'data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf',
     'page': 0,
     'total_pages': 24,
     'format': 'PDF 1.4',
     'title': '',
     'author': '',
     'subject': '',
     'keywords': '',
     'creator': 'Adobe InDesign 15.1 (Macintosh)',
     'producer': 'Adobe PDF Library 15.0',
     'creationDate': "D:20200922223534+02'00'",
     'modDate': "D:20200922223544+02'00'",
     'trapped': '',
     'doc_id': '46aba7dd-39cd-4852-beed-e8e0560e7a98'}</pre>



Check the number of chunks for each split document.

```python
print(f"Number of split parent_docs: {len(parent_docs)}")
print(f"Number of split child_docs: {len(child_docs)}")
```

<pre class="custom">Number of split parent_docs: 177
    Number of split child_docs: 950
</pre>

Add the newly created smaller child document set to the vector store.

Then, map the parent documents to the generated UUIDs and add them to the ```docstore```.

- Use the ```mset``` method to store document IDs and their content as key-value pairs in the document store.

```python
# Add both parent and child documents to the vector store
retriever.vectorstore.add_documents(parent_docs)
retriever.vectorstore.add_documents(child_docs)

# Store the original documents in the docstore
retriever.docstore.mset(list(zip(doc_ids, docs)))
```

Perform similarity search, and display the most similar document chunks.

Use the ```retriever.vectorstore.similarity_search``` method to search within the child and parent document chunks.

The first document chunk with the highest similarity will be displayed.

```python
# Perform similarity search on the vectorstore
relevant_chunks = retriever.vectorstore.similarity_search(
    "What is the phased implementation timeline for the EU AI Act?"
)
print(f"Number of retrieved documents: {len(relevant_chunks)}")
```

<pre class="custom">Number of retrieved documents: 4
</pre>

```python
for chunk in relevant_chunks:
    print(chunk.page_content, end="\n\n")
    print(">" * 100, end="\n\n")
```

<pre class="custom">peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
</pre>

Execute a query using the ```retriever.invoke``` method.

The ```retriever.invoke``` method performs a search across the full content of the original documents.

```python
relevant_docs = retriever.invoke(
    "What is the phased implementation timeline for the EU AI Act?"
)
print(f"Number of retrieved documents: {len(relevant_docs)}", end="\n\n")
print("=" * 100, end="\n\n")
print(relevant_docs[0].page_content)
```

<pre class="custom">Number of retrieved documents: 1
    
    ====================================================================================================
    
    A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    5
    laws and regulation. Some negative examples have been given 
    wide attention in the media: a fatal accident involving an autono-
    mous vehicle2; Microsoft’s chatting bot Tay being shut down after 
    16 hours because it became racist, sexist, and denied the Holo-
    caust3; racially biased decisions with credit checks and recidivism 
    (Teich & Tirias Research, 2018). Such examples are fuelling a va-
    riety of concerns about accountability, fairness, bias, autonomy, 
    and due process of AI systems (Pasquale, 2015; Ziewitz, 2015). 
    Beyond these anecdotal instances, AI presents several challenges 
    (Dwivedi et al., 2019), which are economic (need of funds, impact 
    on employment and performances) and organizational (changing 
    working practices, cultural barriers, need of new skills, data inte-
    gration, etc.) issues to be tackled. At societal level AI may challenge 
    cultural norms and face resistance (Hu et al, 2019). In Europe there 
    is an ongoing discussion on the legal and ethical challenges posed 
    by a greater use of AI. One key point is transparency, or lack the-
    reof, of algorithms on which AI applications rely. There is a need 
    to study and understand where algorithms may go wrong as to 
    adopt adequate and proportional remedial and mitigation mea-
    sures. Algorithmic rules may imply moral judgements, such as for 
    driverless cars deciding which lives to save in the event of a se-
    rious accident (Nyholm, & Smids, 2016). 
    The European Commission has launched a series of policy initia-
    tives with the aim to boost the development of sustainable AI 
    in Europe, including the communication ‘Artificial Intelligence for 
    Europe’ (European Commission, 2018a), the declaration of coo-
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to 
    place people at the centre of the development of AI, what has been 
    called ‘human-centric AI’. It is a three-pronged approach to support 
    the EU’s technological and industrial capacity and AI uptake across 
    the economy, prepare for socio-economic changes, and ensure an 
    appropriate ethical and legal framework. The Commission has set 
    up a High-Level Expert Group on AI representing a wide range of 
    stakeholders and has tasked it with drafting AI ethics guidelines as 
    well as preparing a set of recommendations for broader AI policy. 
    The Group drafted AI Ethical Guidelines4, which postulate that in 
    order to achieve ‘trustworthy AI’, three components are necessary: 
    (1) it should comply with the law, (2) it should fulfil ethical prin-
    ciples and (3) it should be robust. Based on these three compo-
    nents and the European values, the guidelines identify seven key 
    requirements that AI applications should respect to be considered 
    trustworthy5. These policies culminated in the White Paper on AI 
    – A European Approach to Excellence and Trust (European Com-
    mission, 2020a) and a Communication on ‘A European Strategy 
    for Data’ (European Commission, 2020b). The strategy set out in 
    the Paper is built on two main blocks. On the one hand, it aims to 
    create an ‘ecosystem of excellence’, by boosting the development 
    of AI, partnering with private sector, focusing on R&D, skills and 
    SMEs in particular. On the other hand, it aims to create an ‘ecosys-
    tem of trust’ within an EU regulatory framework. The strategy set 
    out in the White Paper is to build and retain trust in AI. This needs 
    a multi-layered approach that includes critical engagement of civil 
    society to discuss the values guiding and being embedded into AI; 
    public debates to translate these values into strategies and guide-
    lines; and responsible design practices that encode these values 
    and guidelines into AI systems making these ‘ethical by design’. 
    In line with this we have the European data strategy, adopted in 
    February 2020, aiming to establish a path for the creation of Euro-
    pean data spaces whereby more data becomes available for use in 
    the economy and society but under firm control of European com-
    panies and individuals. As noted in a recent parliamentary brief 
    (European Parliament, 2020), the objective of creating European 
    data spaces is related to the ongoing discourse on Europe digital 
    sovereignty (EPSC, 20196)  and the concern that, while Europe is at 
    the frontier in terms of research and on a par with its global com-
    petitors, it nonetheless lags behind the US and China when it co-
    mes to private investment (European Commission, 2018a). The le-
    vel of adoption of AI technologies by companies and by the general 
    public appears comparatively low compared to the US (Probst et 
    al., 2018). This leads to the concern that citizens, businesses and 
    Member States of the EU are gradually losing control over their 
    data, their capacity for innovation, and their ability to shape and 
    enforce legislation in the digital environment. To address these 
    concerns the data strategy proposes the construction of an EU 
    data framework that would favour and support the sharing of 
    
</pre>

The default search type performed by the retriever in the vector database is similarity search.

LangChain's ```VectorStore``` also support searching using [Max Marginal Relevance](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.max_marginal_relevance_search). 

If you want to use this method instead, you can configure the ```search_type``` property as follows.

- Set the ```search_type``` property of the ```retriever``` object to ```SearchType.mmr```.
  - This specifies that the MMR (Maximal Marginal Relevance) algorithm should be used during the search.

```python
from langchain.retrievers.multi_vector import SearchType

# Set the search type to Maximal Marginal Relevance (MMR)
retriever.search_type = SearchType.mmr

# Search all related documents
print(
    retriever.invoke(
        "What is the phased implementation timeline for the EU AI Act?"
    )[0].page_content
)
```

<pre class="custom">A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    5
    laws and regulation. Some negative examples have been given 
    wide attention in the media: a fatal accident involving an autono-
    mous vehicle2; Microsoft’s chatting bot Tay being shut down after 
    16 hours because it became racist, sexist, and denied the Holo-
    caust3; racially biased decisions with credit checks and recidivism 
    (Teich & Tirias Research, 2018). Such examples are fuelling a va-
    riety of concerns about accountability, fairness, bias, autonomy, 
    and due process of AI systems (Pasquale, 2015; Ziewitz, 2015). 
    Beyond these anecdotal instances, AI presents several challenges 
    (Dwivedi et al., 2019), which are economic (need of funds, impact 
    on employment and performances) and organizational (changing 
    working practices, cultural barriers, need of new skills, data inte-
    gration, etc.) issues to be tackled. At societal level AI may challenge 
    cultural norms and face resistance (Hu et al, 2019). In Europe there 
    is an ongoing discussion on the legal and ethical challenges posed 
    by a greater use of AI. One key point is transparency, or lack the-
    reof, of algorithms on which AI applications rely. There is a need 
    to study and understand where algorithms may go wrong as to 
    adopt adequate and proportional remedial and mitigation mea-
    sures. Algorithmic rules may imply moral judgements, such as for 
    driverless cars deciding which lives to save in the event of a se-
    rious accident (Nyholm, & Smids, 2016). 
    The European Commission has launched a series of policy initia-
    tives with the aim to boost the development of sustainable AI 
    in Europe, including the communication ‘Artificial Intelligence for 
    Europe’ (European Commission, 2018a), the declaration of coo-
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to 
    place people at the centre of the development of AI, what has been 
    called ‘human-centric AI’. It is a three-pronged approach to support 
    the EU’s technological and industrial capacity and AI uptake across 
    the economy, prepare for socio-economic changes, and ensure an 
    appropriate ethical and legal framework. The Commission has set 
    up a High-Level Expert Group on AI representing a wide range of 
    stakeholders and has tasked it with drafting AI ethics guidelines as 
    well as preparing a set of recommendations for broader AI policy. 
    The Group drafted AI Ethical Guidelines4, which postulate that in 
    order to achieve ‘trustworthy AI’, three components are necessary: 
    (1) it should comply with the law, (2) it should fulfil ethical prin-
    ciples and (3) it should be robust. Based on these three compo-
    nents and the European values, the guidelines identify seven key 
    requirements that AI applications should respect to be considered 
    trustworthy5. These policies culminated in the White Paper on AI 
    – A European Approach to Excellence and Trust (European Com-
    mission, 2020a) and a Communication on ‘A European Strategy 
    for Data’ (European Commission, 2020b). The strategy set out in 
    the Paper is built on two main blocks. On the one hand, it aims to 
    create an ‘ecosystem of excellence’, by boosting the development 
    of AI, partnering with private sector, focusing on R&D, skills and 
    SMEs in particular. On the other hand, it aims to create an ‘ecosys-
    tem of trust’ within an EU regulatory framework. The strategy set 
    out in the White Paper is to build and retain trust in AI. This needs 
    a multi-layered approach that includes critical engagement of civil 
    society to discuss the values guiding and being embedded into AI; 
    public debates to translate these values into strategies and guide-
    lines; and responsible design practices that encode these values 
    and guidelines into AI systems making these ‘ethical by design’. 
    In line with this we have the European data strategy, adopted in 
    February 2020, aiming to establish a path for the creation of Euro-
    pean data spaces whereby more data becomes available for use in 
    the economy and society but under firm control of European com-
    panies and individuals. As noted in a recent parliamentary brief 
    (European Parliament, 2020), the objective of creating European 
    data spaces is related to the ongoing discourse on Europe digital 
    sovereignty (EPSC, 20196)  and the concern that, while Europe is at 
    the frontier in terms of research and on a par with its global com-
    petitors, it nonetheless lags behind the US and China when it co-
    mes to private investment (European Commission, 2018a). The le-
    vel of adoption of AI technologies by companies and by the general 
    public appears comparatively low compared to the US (Probst et 
    al., 2018). This leads to the concern that citizens, businesses and 
    Member States of the EU are gradually losing control over their 
    data, their capacity for innovation, and their ability to shape and 
    enforce legislation in the digital environment. To address these 
    concerns the data strategy proposes the construction of an EU 
    data framework that would favour and support the sharing of 
    
</pre>

```python
from langchain.retrievers.multi_vector import SearchType

# Set search type to similarity_score_threshold
retriever.search_type = SearchType.similarity_score_threshold
retriever.search_kwargs = {"score_threshold": 0.3}

# Search all related documents
print(
    retriever.invoke(
        "What is the phased implementation timeline for the EU AI Act?"
    )[0].page_content
)
```

<pre class="custom">A EUROPEAN APPROACH TO ARTIFICIAL INTELLIGENCE - A POLICY PERSPECTIVE
    5
    laws and regulation. Some negative examples have been given 
    wide attention in the media: a fatal accident involving an autono-
    mous vehicle2; Microsoft’s chatting bot Tay being shut down after 
    16 hours because it became racist, sexist, and denied the Holo-
    caust3; racially biased decisions with credit checks and recidivism 
    (Teich & Tirias Research, 2018). Such examples are fuelling a va-
    riety of concerns about accountability, fairness, bias, autonomy, 
    and due process of AI systems (Pasquale, 2015; Ziewitz, 2015). 
    Beyond these anecdotal instances, AI presents several challenges 
    (Dwivedi et al., 2019), which are economic (need of funds, impact 
    on employment and performances) and organizational (changing 
    working practices, cultural barriers, need of new skills, data inte-
    gration, etc.) issues to be tackled. At societal level AI may challenge 
    cultural norms and face resistance (Hu et al, 2019). In Europe there 
    is an ongoing discussion on the legal and ethical challenges posed 
    by a greater use of AI. One key point is transparency, or lack the-
    reof, of algorithms on which AI applications rely. There is a need 
    to study and understand where algorithms may go wrong as to 
    adopt adequate and proportional remedial and mitigation mea-
    sures. Algorithmic rules may imply moral judgements, such as for 
    driverless cars deciding which lives to save in the event of a se-
    rious accident (Nyholm, & Smids, 2016). 
    The European Commission has launched a series of policy initia-
    tives with the aim to boost the development of sustainable AI 
    in Europe, including the communication ‘Artificial Intelligence for 
    Europe’ (European Commission, 2018a), the declaration of coo-
    peration on AI (European Commission, 2018c), and coordinated 
    action plan on the development of AI in the EU (European Com-
    mission, 2018d), among others. The European strategy aims to 
    place people at the centre of the development of AI, what has been 
    called ‘human-centric AI’. It is a three-pronged approach to support 
    the EU’s technological and industrial capacity and AI uptake across 
    the economy, prepare for socio-economic changes, and ensure an 
    appropriate ethical and legal framework. The Commission has set 
    up a High-Level Expert Group on AI representing a wide range of 
    stakeholders and has tasked it with drafting AI ethics guidelines as 
    well as preparing a set of recommendations for broader AI policy. 
    The Group drafted AI Ethical Guidelines4, which postulate that in 
    order to achieve ‘trustworthy AI’, three components are necessary: 
    (1) it should comply with the law, (2) it should fulfil ethical prin-
    ciples and (3) it should be robust. Based on these three compo-
    nents and the European values, the guidelines identify seven key 
    requirements that AI applications should respect to be considered 
    trustworthy5. These policies culminated in the White Paper on AI 
    – A European Approach to Excellence and Trust (European Com-
    mission, 2020a) and a Communication on ‘A European Strategy 
    for Data’ (European Commission, 2020b). The strategy set out in 
    the Paper is built on two main blocks. On the one hand, it aims to 
    create an ‘ecosystem of excellence’, by boosting the development 
    of AI, partnering with private sector, focusing on R&D, skills and 
    SMEs in particular. On the other hand, it aims to create an ‘ecosys-
    tem of trust’ within an EU regulatory framework. The strategy set 
    out in the White Paper is to build and retain trust in AI. This needs 
    a multi-layered approach that includes critical engagement of civil 
    society to discuss the values guiding and being embedded into AI; 
    public debates to translate these values into strategies and guide-
    lines; and responsible design practices that encode these values 
    and guidelines into AI systems making these ‘ethical by design’. 
    In line with this we have the European data strategy, adopted in 
    February 2020, aiming to establish a path for the creation of Euro-
    pean data spaces whereby more data becomes available for use in 
    the economy and society but under firm control of European com-
    panies and individuals. As noted in a recent parliamentary brief 
    (European Parliament, 2020), the objective of creating European 
    data spaces is related to the ongoing discourse on Europe digital 
    sovereignty (EPSC, 20196)  and the concern that, while Europe is at 
    the frontier in terms of research and on a par with its global com-
    petitors, it nonetheless lags behind the US and China when it co-
    mes to private investment (European Commission, 2018a). The le-
    vel of adoption of AI technologies by companies and by the general 
    public appears comparatively low compared to the US (Probst et 
    al., 2018). This leads to the concern that citizens, businesses and 
    Member States of the EU are gradually losing control over their 
    data, their capacity for innovation, and their ability to shape and 
    enforce legislation in the digital environment. To address these 
    concerns the data strategy proposes the construction of an EU 
    data framework that would favour and support the sharing of 
    
</pre>

```python
from langchain.retrievers.multi_vector import SearchType

# Set search type to similarity and k value to 1
retriever.search_type = SearchType.similarity
retriever.search_kwargs = {"k": 1}

# Search all related documents
print(
    len(
        retriever.invoke(
            "What is the phased implementation timeline for the EU AI Act?"
        )
    )
)
```

<pre class="custom">0
</pre>

## Storing Summary Embeddings

Summaries can often provide a more accurate extraction of the contents of a chunk, which can lead to better search results.

This section describes how to generate summaries, and how to embed them.

```python
# Importing libraries for loading PDF files and splitting text
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the PDF file loader
loader = PyMuPDFLoader("data/A European Approach to Artificial Intelligence - A Policy Perspective.pdf")

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

# Load a PDF file and run Text Split
split_docs = loader.load_and_split(text_splitter)

# Output the number of split documents
print(f"Number of split documents: {len(split_docs)}")
```

<pre class="custom">Number of split documents: 135
</pre>

```python
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


summary_chain = (
    {"doc": lambda x: x.page_content}
    # Create a prompt template for document summaries
    | ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in summarizing documents in English."),
            (
                "user",
                "Summarize the following documents in 3 sentences in bullet points format.\n\n{doc}",
            ),
        ]
    )
    # Using OpenAI's ChatGPT model to generate summaries
    | ChatOpenAI(temperature=0, model="gpt-4o-mini")
    | StrOutputParser()
)
```

Summarize the documents in the ```docs``` list in batch using the ```chain.batch``` method.
- Here, we set the ```max_concurrency``` parameter to 10 to allow up to 10 documents to be processed simultaneously.

```python
# Handling batches of documents
summaries = summary_chain.batch(split_docs, {"max_concurrency": 10})
```

```python
len(summaries)
```




<pre class="custom">135</pre>



Print the summary to see the results.

```python
# Prints the contents of the original document.
print(split_docs[33].page_content, end="\n\n")
# Print a summary.
print("[summary]")
print(summaries[33])
```

<pre class="custom">decision-making process may become less tractable9. The chosen 
    decision model may also turn out to be unsuitable if the real-world 
    environment behaves differently from what was expected. While 
    more and better data be used for training can help improving pre-
    diction, it will never be perfect or include all justifiable outliers. On 
    the other hand, as technology advances more instruments may 
    become available to quantify the degree of influence of input va-
    riables on algorithm outputs (Datta et al., 2016). Research is also 
    underway in pursuit of rendering algorithms more amenable to
    
    [summary]
    - The decision-making process can become complex and less manageable if the chosen model does not align with real-world conditions.  
    - Although improved data can enhance predictions, it will never be flawless or account for all valid outliers.  
    - Advancements in technology may provide new tools to measure the impact of input variables on algorithm outputs, and research is ongoing to make algorithms more adaptable.  
</pre>

Initialize the ```Chroma``` vector store to index the child chunks. Use ```OpenAIEmbeddings``` as the embedding function.

- Use ```“doc_id”``` as the key representing the document ID.


```python
import uuid

# Create a vector store to store the summary information.
summary_vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# Create a repository to store the parent document.
store = InMemoryStore()

# Specify a key name to store the document ID.
id_key = "doc_id"

# Initialize the searcher (empty at startup).
retriever = MultiVectorRetriever(
    vectorstore=summary_vectorstore,  # vector store
    byte_store=store,  # byte store
    id_key=id_key,  # document ID
)
# Create a document ID.
doc_ids = [str(uuid.uuid4()) for _ in split_docs]
```

Save the summarized document and its metadata (here, the ```Document ID``` for the summary you created).


```python
summary_docs = [
    # Create a Document object with the summary as the page content and the document ID as metadata.
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
```

The number of articles in the digest matches the number of original articles.


```python
# Number of documents in the summary
len(summary_docs)
```




<pre class="custom">135</pre>



- Add ```summary_docs``` to the vector store with ```retriever.vectorstore.add_documents(summary_docs)```.
- Map ```doc_ids``` and ```docs``` with ```retriever.docstore.mset(list(zip(doc_ids, docs))))``` to store them in the document store.


```python
retriever.vectorstore.add_documents(
    summary_docs
)  # Add the summarized document to the vector repository.

# Map the document ID to the document and store it in the document store.
retriever.docstore.mset(list(zip(doc_ids, split_docs)))
```

Perform a similarity search using the ```similarity_search``` method of the ```vectorstore``` object.


```python
# Perform a similarity search.
result_docs = summary_vectorstore.similarity_search(
    "What is the phased implementation timeline for the EU AI Act?"
)
```

```python
# Output 1 result document.
print(result_docs[0].page_content)
```

<pre class="custom">- The European Commission and EU member states are collaborating to enhance the development and implementation of artificial intelligence (AI) technologies within Europe.  
    - In 2018, a commitment was made to boost AI "made in Europe," focusing on fostering innovation and ensuring ethical standards.  
    - The 2020 White Paper outlines a European approach to AI, emphasizing the importance of excellence and trust in AI systems.
</pre>

Use the ```invoke``` method of the ```retriever``` object to retrieve documents related to the query.


```python
# Search for and fetch related articles.
retrieved_docs = retriever.invoke(
    "What is the phased implementation timeline for the EU AI Act?"
)
print(retrieved_docs[0].page_content)
```

<pre class="custom">cial Intelligence. Retrieved from https://ec.europa.eu/digital-single-market/en/news/
    eu-member-states-sign-cooperate-artificial-intelligence.
    European Commission. (2018d). Member States and Commission to work together to 
    boost artificial intelligence ‘made in Europe’. Retrieved from https://ec.europa.eu/commis-
    sion/presscorner/detail/en/IP_18_6689.
    European Commission. (2020a). White Paper on Artificial Intelligence. A European Ap-
    proach to Excellence and Trust. COM(2020) 65 final, Brussels: European Commission.
</pre>

## Utilizing Hypothetical Questions

An LLM can also be used to generate a list of questions that can be hypothesized about a particular document.

These generated questions can be embedded to further explore and understand the content of the document.

Generating hypothetical questions can help you identify key topics and concepts in your document, and can encourage readers to ask more questions about the content of your document.


Below is an example of creating a hypothesis question via **function calling** .

```python
functions = [
    {
        "name": "hypothetical_questions",  # Specify a name for the function.
        "description": "Generate hypothetical questions",  # Write a description of the function.
        "parameters": {  # Define the parameters of the function.
            "type": "object",  # Specifies the type of the parameter as an object.
            "properties": {  # Defines the properties of an object.
                "questions": {  # Define the 'questions' attribute.
                    "type": "array",  # Type 'questions' as an array.
                    "items": {
                        "type": "string"
                    },  # Specifies the array's element type as String.
                },
            },
            "required": ["questions"],  # Specify 'questions' as a required parameter.
        },
    }
]
```

Use ```ChatPromptTemplate``` to define a prompt template that generates three hypothetical questions based on the given document.

- Set ```functions``` and ```function_call``` to call the virtual question generation functions.
- Use ```JsonKeyOutputFunctionsParser``` to parse the generated virtual questions and extract the values corresponding to the ```questions``` key.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_openai import ChatOpenAI

hypothetical_query_chain = (
    {"doc": lambda x: x.page_content}
    # We ask you to create exactly 3 hypothetical questions that you can answer using the documentation below. This number can be adjusted.
    | ChatPromptTemplate.from_template(
        "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer. "
        "Potential users are those interested in the AI industry. Create questions that they would be interested in. "
        "Output should be written in English:\n\n{doc}"
    )
    | ChatOpenAI(max_retries=0, model="gpt-4o-mini").bind(
        functions=functions, function_call={"name": "hypothetical_questions"}
    )
    # Extract the value corresponding to the “questions” key from the output.
    | JsonKeyOutputFunctionsParser(key_name="questions")
)
```

Output the answers to the documents.

- The output contains the three hypothetical questions you created.


```python
# Run the chain for the given document.
hypothetical_query_chain.invoke(split_docs[33])
```




<pre class="custom">['How might advancements in technology influence the accuracy of decision-making algorithms in the AI industry?',
     'What could be the consequences if decision models in AI fail to adapt to unexpected changes in real-world environments?',
     'In what ways could the availability of better data impact the training of AI algorithms and their ability to predict outcomes?']</pre>



Use the ```chain.batch``` method to process multiple requests for ```split_docs``` data at the same time.

```python
# Create a batch of hypothetical questions for a list of articles
hypothetical_questions = hypothetical_query_chain.batch(
    split_docs, {"max_concurrency": 10}
)
```

```python
hypothetical_questions[33]
```




<pre class="custom">['What could happen if an AI decision-making model is trained on incomplete or biased data?',
     'How might advancements in technology influence the reliability of AI predictions in unpredictable environments?',
     'What would be the implications for the AI industry if algorithms could be made more transparent and interpretable in their decision-making processes?']</pre>



Below is the process for storing the **hypothetical questions** you created in the **vector store**, the same way we did before.


```python
# Vector store to use for indexing child chunks
hypothetical_vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=OpenAIEmbeddings()
)
# Storage hierarchy for parent documents
store = InMemoryStore()

id_key = "doc_id"
# Retriever (empty on startup)
retriever = MultiVectorRetriever(
    vectorstore=hypothetical_vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in split_docs]  # Create a document ID
```

Add metadata (document IDs) to the ```question_docs``` list.


```python
question_docs = []
# save hypothetical_questions
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        # Create a Document object for each question in the list of questions, and include the document ID for that question in the metadata.
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )
```

Add the ```hypothetical_questions``` to the document, and add the original document to ```docstore```.


```python
# Add the hypothetical_questions document to the vector repository.
retriever.vectorstore.add_documents(question_docs)

# Map the document ID to the document and store it in the document store.
retriever.docstore.mset(list(zip(doc_ids, split_docs)))
```

Perform a similarity search using the ```similarity_search``` method of the ```vectorstore``` object.


```python
# Search the vector repository for similar documents.
result_docs = hypothetical_vectorstore.similarity_search(
    "What is the phased implementation timeline for the EU AI Act?"
)
```

Below are the results of the similarity search.

Here, we've only added the hypothetical questions we created, so it returns the documents with the highest similarity among the stored hypothetical questions.


```python
# Output the results of the similarity search.
for doc in result_docs:
    print(doc.page_content)
    print(doc.metadata)
```

<pre class="custom">What potential socio-economic changes could arise from the implementation of the EU's coordinated action plan on AI?
    {'doc_id': 'accd841f-1474-410f-b600-54e646eac1ec'}
    How might the guidelines set by the Next European Commission impact the regulatory landscape for AI in Europe?
    {'doc_id': '73899cc8-b216-4642-906c-1e73b49bd479'}
    What might be the long-term effects of implementing the operational principles outlined in the EC AI White Paper on the AI industry in Europe?
    {'doc_id': 'dcf69a31-e872-4bef-b4f0-190bb3a8889c'}
    What potential scenarios could arise from the implementation of the proposed AI governance regimes in Europe, and how might they affect the AI industry?
    {'doc_id': 'ba58ee93-5bad-4ba9-958b-4e0af5fc0162'}
</pre>

Use the ```invoke``` method of the ```retriever``` object to retrieve documents related to the query.


```python
# Search for and fetch related articles.
retrieved_docs = retriever.invoke(result_docs[1].page_content)

# Output the documents found.
for doc in retrieved_docs:
    print(doc.page_content)
```

<pre class="custom">Guidelines for the Next European Commission 2019-2024. Retrieved from: https://ec.eu-
    ropa.eu/commission/sites/beta-political/files/political-guidelines-next-commission_en-
    .pdf.
    Wachter, S., Mittelstadt, B., & Floridi, L. (2017). Transparent, explainable, and accountable 
    AI for robotics. Science Robotics, 2(6), eaan6080. doi:10.1126/scirobotics.aan6080.
    Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual Explanations Without Ope-
    ning the Black Box: Automated Decisions and the GDPR. Harvard Journal of Law & Tech-
    nology, 31(2), 841-887.
    reof, of algorithms on which AI applications rely. There is a need 
    to study and understand where algorithms may go wrong as to 
    adopt adequate and proportional remedial and mitigation mea-
    sures. Algorithmic rules may imply moral judgements, such as for 
    driverless cars deciding which lives to save in the event of a se-
    rious accident (Nyholm, & Smids, 2016). 
    The European Commission has launched a series of policy initia-
    tives with the aim to boost the development of sustainable AI 
    in Europe, including the communication ‘Artificial Intelligence for
    ferences to production and delivery (i.e., the Industry 4.0 vision). 
    Manufacturing companies are investing into this vision and are 
    keen to protect their intellectual property generated from such in-
    vestments. So, there is a concern that a potential new legislative 
    action by the European Commission, which would follow the prin-
    ciples of the GDPR and the requirements of the White Paper, may
    (21/11/2019).
    Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic deci-
    sion-making and a ‘right to explanation. AI Magazine, 38(3), 50-57., 38(3), 50-57. 
    Hof, R. (2013, April 23). Deep Learning. The MIT Technology Review. Retrieved from https://
    www.technologyreview.com/s/513696/deep-learning/.
    Jia, R., & Liang, P. (2016). Data Recombination for Neural Semantic Parsing. Paper pre-
    sented at the Annual Meeting of the Association for Computational Linguistics, Berlin.
    Klossa, G. (2019). Towards European Media Sovereignity. An Industrial Media Strategy
</pre>

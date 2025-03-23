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

# Qdrant

- Author: [HyeonJong Moon](https://github.com/hj0302)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)


## Overview

This notebook demonstrates how to utilize the features related to the `Qdrant` vector database.

[`Qdrant`](https://python.langchain.com/docs/integrations/vectorstores/qdrant/) is an open-source vector similarity search engine designed to store, search, and manage high-dimensional vectors with additional payloads. It offers a production-ready service with a user-friendly API, suitable for applications such as semantic search, recommendation systems, and more.

**Qdrant's architecture** is optimized for efficient vector similarity searches, employing advanced indexing techniques like **Hierarchical Navigable Small World (HNSW)** graphs to enable fast and scalable retrieval of relevant data.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Credentials](#credentials)
- [Installation](#installation)
- [Initialization](#initialization)
- [Manage Vector Store](#manage-vector-store)
  - [Create a Collection](#create-a-collection)
  - [List Collections](#list-collections)
  - [Delete a Collection](#delete-a-collection)
  - [Add Items to the Vector Store](#add-items-to-the-vector-store)
  - [Delete Items from the Vector Store](#delete-items-from-the-vector-store)
  - [Upsert Items to Vector Store (Parallel)](#upsert-items-to-vector-store-parallel)
- [Query Vector Store](#query-vector-store)
  - [Query Directly](#query-directly)
  - [Similarity Search with Score](#similarity-search-with-score)
  - [Query by Turning into Retriever](#query-by-turning-into-retriever)
  - [Search with Filtering](#search-with-filtering)
  - [Delete with Filtering](#delete-with-filtering)
  - [Filtering and Updating Records](#filtering-and-updating-records)

### References

- [LangChain Qdrant Reference](https://python.langchain.com/docs/integrations/vectorstores/qdrant/)
- [Qdrant Official Reference](https://qdrant.tech/documentation/frameworks/langchain/)
- [Qdrant Install Reference](https://qdrant.tech/documentation/guides/installation/)
- [Qdrant Cloud Reference](https://cloud.qdrant.io)
- [Qdrant Cloud Quickstart Reference](https://qdrant.tech/documentation/quickstart-cloud/)
----

## Environment Setup

Set up the environment. You may refer to Environment Setup for more details.

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
        "langsmith",
        "langchain_openai",
        "langchain_qdrant",
        "qdrant_client",
        "langchain_core",
        "fastembed",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0.1
    [notice] To update, run: pip install --upgrade pip
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPEN_API_KEY": "",
        "QDRANT_API_KEY": "",
        "QDRANT_URL": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Qdrant",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

**[Note]** If you are using a `.env` file, proceed as follows.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## **Credentials**

Create a new account or sign in to your existing one, and generate an API key for use in this notebook.

1. **Log in to Qdrant Cloud** : Go to the [Qdrant Cloud](https://cloud.qdrant.io) website and log in using your email, Google account, or GitHub account.

2. **Create a Cluster** : After logging in, navigate to the **"Clusters"** section and click the **"Create"** button. Choose your desired configurations and region, then click **"Create"** to start building your cluster. Once the cluster is created, an API key will be generated for you.

3. **Retrieve and Store Your API Key** : When your cluster is created, you will receive an API key. Ensure you save this key in a secure location, as you will need it later. If you lose it, you will have to generate a new one.

4. **Manage API Keys** : To create additional API keys or manage existing ones, go to the **"Access Management"** section in the Qdrant Cloud dashboard and select *"Qdrant Cloud API Keys"* Here, you can create new keys or delete existing ones.

```
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
```

## **Installation**

There are several main options for initializing and using the **Qdrant** vector store:

- **Local Mode** : This mode doesn't require a separate server.
    - **In-memory storage** (data is not persisted)
    - **On-disk storage** (data is saved to your local machine)
- **Docker Deployments** : You can run **Qdrant** using **Docker**.
- **Qdrant Cloud** : Use **Qdrant** as a managed cloud service.

For detailed instructions, see the [installation instructions](https://qdrant.tech/documentation/guides/installation/).

### In-Memory

For simple tests or quick experiments, you might choose to store data directly in memory. This means the data is automatically removed when your client terminates, typically at the end of your script or notebook session.

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings

# Define the collection name for storing documents
collection_name = "demo_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with in-memory storage
db = QdrantDocumentManager(
    location=":memory:",  # Use in-memory database for temporary storage
    collection_name=collection_name,
    embedding=embedding,
)
```

<pre class="custom">Collection 'demo_collection' does not exist or force recreate is enabled. Creating new collection...
    Collection 'demo_collection' created successfully with configuration: {'vectors_config': VectorParams(size=3072, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None)}
</pre>

### On-Disk Storage

With **on-disk storage**, you can store your vectors directly on your hard drive without requiring a **Qdrant server**. This ensures that your data persists even when you restart the program.

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings

# Define the path for Qdrant storage
qdrant_path = "./qdrant_memory"

# Define the collection name for storing documents
collection_name = "demo_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with specified storage path
db = QdrantDocumentManager(
    path=qdrant_path,  # Specify the path for Qdrant storage
    collection_name=collection_name,
    embedding=embedding,
)
```

<pre class="custom">Collection 'demo_collection' does not exist or force recreate is enabled. Creating new collection...
    Collection 'demo_collection' created successfully with configuration: {'vectors_config': VectorParams(size=3072, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None)}
</pre>

### Docker Deployments

You can deploy `Qdrant` in a **production environment** using [`Docker`](https://qdrant.tech/documentation/guides/installation/#docker) and [`Docker Compose`](https://qdrant.tech/documentation/guides/installation/#docker-compose). Refer to the `Docker` and `Docker Compose` setup instructions in the development section for detailed information.

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings

# Define the URL for Qdrant server
url = "http://localhost:6333"

# Define the collection name for storing documents
collection_name = "demo_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with specified storage path
db = QdrantDocumentManager(
    url=url,  # Specify the path for Qdrant storage
    collection_name=collection_name,
    embedding=embedding,
)
```

### Qdrant Cloud

For a **production environment**, you can use [**Qdrant Cloud**](https://cloud.qdrant.io/). It offers fully managed `Qdrant` databases with features such as **horizontal and vertical scaling**, **one-click setup and upgrades**, **monitoring**, **logging**, **backups**, and **disaster recovery**. For more information, refer to the [**Qdrant Cloud documentation**](https://qdrant.tech/documentation/cloud/).


```python
import getpass
import os

# Fetch the Qdrant server URL from environment variables or prompt for input
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = getpass.getpass("Enter your Qdrant Cloud URL key: ")
QDRANT_URL = os.environ.get("QDRANT_URL")

# Fetch the Qdrant API key from environment variables or prompt for input
if not os.getenv("QDRANT_API_KEY"):
    os.environ["QDRANT_API_KEY"] = getpass.getpass("Enter your Qdrant API key: ")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
```

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings

# Define the collection name for storing documents
collection_name = "demo_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with specified storage path
db = QdrantDocumentManager(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
    embedding=embedding,
)
```

## Initialization

Once you've established your **vector store**, you'll likely need to manage the **collections** within it. Here are some common operations you can perform:

- **Create a collection**
- **List collections**
- **Delete a collection**

To proceed with the tutorial, we will use **Qdrant Cloud** for the next steps. This approach ensures that your data is securely stored in the cloud, allowing for seamless access, comprehensive testing, and experimentation across different environments.

### Create a Collection

The `QdrantDocumentManager` class allows you to create a new **collection** in `Qdrant`. It can automatically create a collection if it doesn't exist or if you want to **recreate** it. You can specify configurations for **dense** and **sparse vectors** to meet different search needs. Use the `_ensure_collection_exists` method for **automatic creation** or call `create_collection` directly when needed.

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import Distance

# Define the collection name for storing documents
collection_name = "test_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with specified storage path
db = QdrantDocumentManager(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
    embedding=embedding,
    metric=Distance.COSINE,
)
```

<pre class="custom">Collection 'test_collection' does not exist or force recreate is enabled. Creating new collection...
    Collection 'test_collection' created successfully with configuration: {'vectors_config': VectorParams(size=3072, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None)}
</pre>

### List Collections

The `QdrantDocumentManager` class lets you list all **collections** in your `Qdrant` instance using the `get_collections` method. This retrieves and displays the **names** of all existing collections.


```python
# Retrieve the list of collections from the Qdrant client
collections = db.client.get_collections()

# Iterate over each collection and print its details
for collection in collections.collections:
    print(f"Collection Name: {collection.name}")
```

<pre class="custom">Collection Name: test_collection
    Collection Name: sparse_collection
    Collection Name: dense_collection
    Collection Name: insta_image_search_test
    Collection Name: insta_image_search
    Collection Name: demo_collection
</pre>

### Delete a Collection

The `QdrantDocumentManager` class allows you to delete a **collection** using the `delete_collection` method. This method removes the specified collection from your `Qdrant` instance.

```python
# Define collection name
collection_name = "test_collection"

# Delete the collection
if db.client.delete_collection(collection_name=collection_name):
    print(f"Collection '{collection_name}' has been deleted.")
```

<pre class="custom">Collection 'test_collection' has been deleted.
</pre>

## Manage VectorStore

After you've created your **vector store**, you can interact with it by **adding** or **deleting** items. Here are some common operations:

### Add Items to the Vector Store

The `QdrantDocumentManager` class lets you add items to your **vector store** using the `upsert` method. This method **updates** existing documents with new data if their IDs already exist.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from uuid import uuid4

# Load the text file
loader = TextLoader("./data/the_little_prince.txt")
documents = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, length_function=len
)

split_docs = text_splitter.split_documents(documents)

# Generate unique IDs for documents
uuids = [str(uuid4()) for _ in split_docs[:30]]
page_contents = [doc.page_content for doc in split_docs[:30]]
metadatas = [doc.metadata for doc in split_docs[:30]]
```

```python
from utils.qdrant import QdrantDocumentManager
from langchain_openai import OpenAIEmbeddings

# Define the collection name for storing documents
collection_name = "demo_collection"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an instance of QdrantDocumentManager with specified storage path
db = QdrantDocumentManager(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
    embedding=embedding,
)

db.upsert(texts=page_contents, metadatas=metadatas, ids=uuids)
```




<pre class="custom">['22417c4f-bf11-4e92-978a-6c436dec39ca',
     '28f56a01-34af-46ae-aeb4-ea6e0fcacb62',
     'c6d06501-9595-4272-80b5-f0747cb145fc',
     'b4b901bf-6e83-4658-b5e9-a1d5a80c767d',
     '21b1b98d-0707-4128-a0bd-78c94db6cbf3',
     'c49b5d7c-c330-4d59-9097-25c3c52510b9',
     '36ddc677-4fa9-47ee-b2e0-284bdb9062a1',
     '32fde659-84c6-4679-b4df-d4b1d11e645f',
     'caf0b611-4a38-4a94-84a9-c3a98ac0b2a1',
     '0e655834-9a6c-48a8-8a3b-5d5e2b1d6c2c',
     '493aaa5c-b89d-429b-a425-57f20f3564ed',
     '6f7f0755-d226-4aec-a714-a53d7a705e51',
     '8b68a39b-f990-4ce1-9fbd-675f5103d3ff',
     '73ef217b-9114-48a4-a447-0deb916b3d5a',
     '63b99932-4e84-4cb2-a5ef-1d83fdbc4e6a',
     '45fd3628-ca2f-439d-97ba-cc34da564f36',
     '876f59dd-a9ae-4af7-84e8-5d8fe78cf7d3',
     '5aa82f42-534f-447f-94b5-9ed4f3571091',
     'eb69cc2a-8899-4d9e-ad8f-adebea281ff0',
     '1defc340-16b4-4ee0-94de-0dabc23e5d07',
     '368d5f90-75d2-406c-8dd2-c7d8736b6944',
     '842812f6-ee9f-43ae-8f6d-53015a5e57af',
     '61031399-09ed-4c88-bc93-1018b942df71',
     'a6ac25f2-2dd5-445f-95dd-6a4d9fc4081c',
     '08215031-2393-4d0c-82a2-53a6a90d169f',
     'f41de48c-1e7d-4036-a75e-a10ac579081d',
     'a2d6b6d1-5bbc-4f17-9b95-c917021614f0',
     '3603a2e7-6021-46c9-8f4c-d53056849c1a',
     'e1fb95a1-7c1c-4aed-a628-b39e0907b744',
     '2a42fbb6-9450-4d86-a5f8-65f333c10d4c']</pre>



### Delete Items from the Vector Store

The `QdrantDocumentManager` class allows you to delete items from your **vector store** using the `delete` method. You can specify items to delete by providing **IDs** or **filters**.


```python
delete_ids = [uuids[0]]

db.delete(ids=delete_ids)
```

### Upsert Items to Vector Store (Parallel)

The `QdrantDocumentManager` class supports **parallel upserts** using the `upsert_parallel` method. This efficiently **adds** or **updates** multiple items with unique **IDs**, **data**, and **metadata**.

```python
# Generate unique IDs for documents
uuids = [str(uuid4()) for _ in split_docs[30:60]]
page_contents = [doc.page_content for doc in split_docs[30:60]]
metadatas = [doc.metadata for doc in split_docs[30:60]]

db.upsert_parallel(
    texts=page_contents,
    metadatas=metadatas,
    ids=uuids,
    batch_size=32,
    workers=10,
)
```




<pre class="custom">['286d99ae-019b-41ed-962a-c1a26bf41c4a',
     'e17ce584-3576-45bb-8d82-36cfdd4c89d1',
     'aed142fa-a13a-421f-9e60-ab1af13a8b15',
     '14337336-edb2-4ea1-880c-2f4613f1f999',
     '91d47b16-4a1f-4f1f-ba07-78f9b2db06d8',
     '6b58d2d9-1a4b-4e03-97fd-d584d502b606',
     'e7b6f4b5-27e0-4787-a74c-b8d17a7038ea',
     '01579e1a-9935-443d-a7a5-b9ffdd1e07f9',
     '4d516f16-09cf-4b7e-8d65-455eced738e7',
     '7fd284a3-5f10-407f-a8fe-44a923263748',
     '55fae9b6-046a-4f09-9cf0-08568efde43c',
     'b4386ade-1590-41fa-94e7-cc34d4f4c9da',
     'd27d8f98-349a-4c45-9f82-31e983edfa8c',
     '20537c5d-80d1-4d72-8507-73fd21e3f11a',
     'ae418ede-69f6-4703-9d9d-2e31d59441b2',
     '975d663d-f825-446d-9824-7997058ca24a',
     'c8086e33-6345-4403-a98c-a4cd46375cd1',
     'ec887b4f-eecf-4325-8117-293e6fd8dfd6',
     'c5fa1381-e30d-47d8-aad3-d46cc8520953',
     '1b20e891-e44f-4640-ab24-03d692627265',
     '0d37a3dd-329f-4901-a828-71a704f7a35e',
     '170420dc-b02c-42f3-a36d-c56973784fb7',
     'f11893c3-20c5-43e4-9c0f-905d91c7a668',
     '37327ff1-7f17-43b0-89ca-65ab69c14df6',
     '92a4e2ec-7418-4241-a1e3-3bf2668a9fd6',
     'ea018faa-293f-4329-b8ae-92dc3fcdd909',
     '09c78d94-0b4c-41cc-b530-7504f3d62dc4',
     '907ad8d0-427d-4f29-b801-aea90a6a86aa',
     '86508b0c-4ff7-422f-b13e-1443e47ef5d3',
     'b12e4c37-50a1-4257-80ae-de372a4a77ce']</pre>



## Query VectorStore

Once your **vector store** has been created and the relevant **documents** have been added, you will most likely wish to **query** it during the running of your `chain` or `agent`.

### Query Directly

The `QdrantDocumentManager` class allows direct **querying** using the `search` method. It performs **similarity searches** by converting queries into **vector embeddings** to find similar **documents**.


```python
query = "What is the significance of the rose in The Little Prince?"

response = db.search(
    query=query,
    k=3,
)

for res in response:
    payload = res["payload"]
    print(f"* {payload['page_content'][:200]}\n [{payload['metadata']}]\n\n")
```

<pre class="custom">* for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
    * for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
    * for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
</pre>

### Similarity Search with Score

The `QdrantDocumentManager` class enables **similarity searches** with **scores** using the `search` method. This provides a **relevance score** for each **document** found.


```python
# Define the query to search in the database
query = "What is the significance of the rose in The Little Prince?"

# Perform the search with the specified query and number of results
response = db.search(query=query, k=3)

for res in response:
    payload = res["payload"]
    score = res["score"]
    print(
        f"* [SIM={score:.3f}] {payload['page_content'][:200]}\n [{payload['metadata']}]\n\n"
    )
```

<pre class="custom">* [SIM=0.527] for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
    * [SIM=0.527] for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
    * [SIM=0.527] for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt'}]
    
    
</pre>

### Query by Turning into Retriever

The `QdrantDocumentManager` class can transform the **vector store** into a `retriever`. This allows for easier **integration** into **workflows** or **chains**.


```python
from langchain_qdrant import QdrantVectorStore

# Initialize QdrantVectorStore with the client, collection name, and embedding
vector_store = QdrantVectorStore(
    client=db.client, collection_name=db.collection_name, embedding=db.embedding
)

query = "What is the significance of the rose in The Little Prince?"

# Transform the vector store into a retriever with specific search parameters
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)

results = retriever.invoke(query)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt', '_id': 'c49b5d7c-c330-4d59-9097-25c3c52510b9', '_collection_name': 'demo_collection'}]
    
    
    * for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt', '_id': '9567e6cf-2f89-4c3b-8a41-7167770fbcd3', '_collection_name': 'demo_collection'}]
    
    
    * for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt', '_id': 'e2a0d06a-9ccd-4e9e-8d4a-4e1292b6ccef', '_collection_name': 'demo_collection'}]
    
    
</pre>

### Search with Filtering

The `QdrantDocumentManager` class allows **searching with filters** to retrieve records based on specific **metadata values**. This is done using the `scroll` method with a defined **filter query**.

```python
from qdrant_client import models

# Define a filter query to match documents containing the text "Chapter" in the page content
filter_query = models.Filter(
    must=[
        models.FieldCondition(
            key="page_content",
            match=models.MatchText(text="Chapter"),
        ),
    ]
)

# Retrieve records from the collection that match the filter query
db.scroll(
    scroll_filter=filter_query,
    k=10,
)
```




<pre class="custom">[Record(id='09c78d94-0b4c-41cc-b530-7504f3d62dc4', payload={'page_content': '[ Chapter 7 ]\n- the narrator learns about the secret of the little prince‘s life \nOn the fifth day-- again, as always, it was thanks to the sheep-- the secret of the little prince‘s life was revealed to me. Abruptly, without anything to lead up to it, and as if the question had been born of long and silent meditation on his problem, he demanded: \n"A sheep-- if it eats little bushes, does it eat flowers, too?"\n"A sheep," I answered, "eats anything it finds in its reach."\n"Even flowers that have thorns?"\n"Yes, even flowers that have thorns." \n"Then the thorns-- what use are they?"', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='0e655834-9a6c-48a8-8a3b-5d5e2b1d6c2c', payload={'page_content': '[ Chapter 1 ]\n- we are introduced to the narrator, a pilot, and his ideas about grown-ups\nOnce when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest. It was a picture of a boa constrictor in the act of swallowing an animal. Here is a copy of the drawing. \n(picture)\nIn the book it said: "Boa constrictors swallow their prey whole, without chewing it. After that they are not able to move, and they sleep through the six months that they need for digestion."', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='286d99ae-019b-41ed-962a-c1a26bf41c4a', payload={'page_content': '[ Chapter 4 ]\n- the narrator speculates as to which asteroid from which the little prince came\u3000\u3000\nI had thus learned a second fact of great importance: this was that the planet the little prince came from was scarcely any larger than a house!', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='45fd3628-ca2f-439d-97ba-cc34da564f36', payload={'page_content': '[ Chapter 2 ]\n- the narrator crashes in the desert and makes the acquaintance of the little prince\nSo I lived my life alone, without anyone that I could really talk to, until I had an accident with my plane in the Desert of Sahara, six years ago. Something was broken in my engine. And as I had with me neither a mechanic nor any passengers, I set myself to attempt the difficult repairs all alone. It was a question of life or death for me: I had scarcely enough drinking water to last a week.', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='d27d8f98-349a-4c45-9f82-31e983edfa8c', payload={'page_content': '[ Chapter 5 ]\n- we are warned as to the dangers of the baobabs\nAs each day passed I would learn, in our talk, something about the little prince‘s planet, his departure from it, his journey. The information would come very slowly, as it might chance to fall from his thoughts. It was in this way that I heard, on the third day, about the catastrophe of the baobabs.\nThis time, once more, I had the sheep to thank for it. For the little prince asked me abruptly-- as if seized by a grave doubt-- "It is true, isn‘t it, that sheep eat little bushes?" \n"Yes, that is true." \n"Ah! I am glad!"', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='f11893c3-20c5-43e4-9c0f-905d91c7a668', payload={'page_content': '[ Chapter 6 ]\n- the little prince and the narrator talk about sunsets\nOh, little prince! Bit by bit I came to understand the secrets of your sad little life... For a long time you had found your only entertainment in the quiet pleasure of looking at the sunset. I learned that new detail on the morning of the fourth day, when you said to me: \n"I am very fond of sunsets. Come, let us go look at a sunset now." \n"But we must wait," I said. \n"Wait? For what?" \n"For the sunset. We must wait until it is time."', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None),
     Record(id='f41de48c-1e7d-4036-a75e-a10ac579081d', payload={'page_content': '[ Chapter 3 ]\n- the narrator learns more about from where the little prince came\nIt took me a long time to learn where he came from. The little prince, who asked me so many questions, never seemed to hear the ones I asked him. It was from words dropped by chance that, little by little, everything was revealed to me. \nThe first time he saw my airplane, for instance (I shall not draw my airplane; that would be much too complicated for me), he asked me: \n"What is that object?"\n"That is not an object. It flies. It is an airplane. It is my airplane."', 'metadata': {'source': './data/the_little_prince.txt'}}, vector=None, shard_key=None, order_value=None)]</pre>



### Delete with Filtering

The `QdrantDocumentManager` class allows you to **delete records** using **filters** based on specific **metadata values**. This is achieved with the `delete` method and a **filter query**.

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchText

# Define a filter query to match documents containing the text "Chapter" in the page content
filter_query = models.Filter(
    must=[
        models.FieldCondition(
            key="page_content",
            match=models.MatchText(text="Chapter"),
        ),
    ]
)

# Delete records from the collection that match the filter query
db.client.delete(collection_name=db.collection_name, points_selector=filter_query)
```




<pre class="custom">UpdateResult(operation_id=31, status=<UpdateStatus.COMPLETED: 'completed'>)</pre>



### Filtering and Updating Records

The `QdrantDocumentManager` class supports **filtering and updating records** based on specific **metadata values**. This is done by **retrieving records** with **filters** and **updating** them as needed.


```python
from qdrant_client import models

# Define a filter query to match documents with a specific metadata source
filter_query = models.Filter(
    must=[
        models.FieldCondition(
            key="metadata.source",
            match=models.MatchValue(value="./data/the_little_prince.txt"),
        ),
    ]
)

# Retrieve records matching the filter query, including their vectors
response = db.scroll(scroll_filter=filter_query, k=10, with_vectors=True)
new_source = "the_little_prince.txt"

# Update the point IDs and set new metadata for the records
for point in response:  # response[0] returns a list of points
    payload = point.payload

    # Check if metadata exists in the payload
    if "metadata" in payload:
        payload["metadata"]["source"] = new_source
    else:
        payload["metadata"] = {
            "source": new_source
        }  # Add new metadata if it doesn't exist

    # Update the point with new metadata
    db.client.upsert(
        collection_name=db.collection_name,
        points=[
            models.PointStruct(
                id=point.id,
                payload=payload,
                vector=point.vector,
            )
        ],
    )
```

### Similarity Search Options

When using `QdrantVectorStore`, you have three options for performing **similarity searches**. You can select the desired search mode using the `retrieval_mode` parameter when you set up the class. The available modes are:

- **Dense Vector Search** (Default)
- **Sparse Vector Search**
- **Hybrid Search**

### Dense Vector Search

To perform a search using only **dense vectors**:

- The `retrieval_mode` parameter must be set to `RetrievalMode.DENSE`. This is also the **default setting**.
- You need to provide a [dense embeddings](https://python.langchain.com/docs/integrations/text_embedding/) value through the `embedding` parameter.


```python
from langchain_qdrant import RetrievalMode
from langchain_openai import OpenAIEmbeddings

query = "What is the significance of the rose in The Little Prince?"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize QdrantVectorStore with documents, embeddings, and configuration
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs[:50],
    embedding=embedding,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="dense_collection",
    retrieval_mode=RetrievalMode.DENSE,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=3,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt', '_id': '3cc041d5-2700-498f-8114-85f3c96e26b9', '_collection_name': 'dense_collection'}]
    
    
    * for decades. In the book, a pilot is stranded in the midst of the Sahara where he meets a tiny prince from another world traveling the universe in order to understand life. In the book, the little pri
     [{'source': './data/the_little_prince.txt', '_id': '24d766ea-3383-40e5-bd0e-051d51de88a3', '_collection_name': 'dense_collection'}]
    
    
    * Indeed, as I learned, there were on the planet where the little prince lived-- as on all planets-- good plants and bad plants. In consequence, there were good seeds from good plants, and bad seeds fro
     [{'source': './data/the_little_prince.txt', '_id': 'd25ba992-e54d-4e8a-9572-438c78d0288b', '_collection_name': 'dense_collection'}]
    
    
</pre>

### Sparse Vector Search

To search with only **sparse vectors**:

- The `retrieval_mode` parameter should be set to `RetrievalMode.SPARSE`.
- An implementation of the [SparseEmbeddings](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any **sparse embeddings provider** has to be provided as a value to the `sparse_embedding` parameter.
- The `langchain-qdrant` package provides a **FastEmbed** based implementation out of the box.

To use it, install the [FastEmbed](https://github.com/qdrant/fastembed) package:

```bash
pip install fastembed
```

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant import RetrievalMode
from langchain_openai import OpenAIEmbeddings

query = "What is the significance of the rose in The Little Prince?"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
# Initialize sparse embeddings using FastEmbedSparse
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Initialize QdrantVectorStore with documents, embeddings, and configuration
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="sparse_collection",
    retrieval_mode=RetrievalMode.SPARSE,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=3,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* [ Chapter 20 ]
    - the little prince discovers a garden of roses
    But it happened that after walking for a long time through sand, and rocks, and snow, the little prince at last came upon a road. And all
     [{'source': './data/the_little_prince.txt', '_id': '30d70339-4233-427b-b839-208c7618ae82', '_collection_name': 'sparse_collection'}]
    
    
    * [ Chapter 20 ]
    - the little prince discovers a garden of roses
    But it happened that after walking for a long time through sand, and rocks, and snow, the little prince at last came upon a road. And all
     [{'source': './data/the_little_prince.txt', '_id': '45ad1b0e-45cd-46f0-b6cd-d8e2b19ea8fa', '_collection_name': 'sparse_collection'}]
    
    
    * And he went back to meet the fox. 
    "Goodbye," he said. 
    "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the heart that one can see rightly; what is essential
     [{'source': './data/the_little_prince.txt', '_id': 'ab098119-c45f-4e33-b105-a6c6e01a918b', '_collection_name': 'sparse_collection'}]
    
    
</pre>

### Hybrid Vector Search

To perform a **hybrid search** using **dense** and **sparse vectors** with **score fusion**:

- The `retrieval_mode` parameter should be set to `RetrievalMode.HYBRID`.
- A [`dense embeddings`](https://python.langchain.com/docs/integrations/text_embedding/) value should be provided to the `embedding` parameter.
- An implementation of the [`SparseEmbeddings`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any **sparse embeddings provider** has to be provided as a value to the `sparse_embedding` parameter.

**Note**: If you've added documents with the `HYBRID` mode, you can switch to any **retrieval mode** when searching, since both the **dense** and **sparse vectors** are available in the **collection**.

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant import RetrievalMode
from langchain_openai import OpenAIEmbeddings

query = "What is the significance of the rose in The Little Prince?"

# Initialize the embedding model with a specific OpenAI model
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
# Initialize sparse embeddings using FastEmbedSparse
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Initialize QdrantVectorStore with documents, embeddings, and configuration
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="hybrid_collection",
    retrieval_mode=RetrievalMode.HYBRID,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=3,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* "Go and look again at the roses. You will understand now that yours is unique in all the world. Then come back to say goodbye to me, and I will make you a present of a secret." 
    The little prince went
     [{'source': './data/the_little_prince.txt', '_id': '447a916c-d8a9-46f2-b035-d0ac4c7ea901', '_collection_name': 'hybrid_collection'}]
    
    
    * [ Chapter 20 ]
    - the little prince discovers a garden of roses
    But it happened that after walking for a long time through sand, and rocks, and snow, the little prince at last came upon a road. And all
     [{'source': './data/the_little_prince.txt', '_id': '894a9222-ef0c-4e28-b736-8a334cbdc83b', '_collection_name': 'hybrid_collection'}]
    
    
    * [ Chapter 8 ]
    - the rose arrives at the little prince‘s planet
     [{'source': './data/the_little_prince.txt', '_id': 'a3729fa0-b734-4316-ad18-83ea16263a2f', '_collection_name': 'hybrid_collection'}]
    
    
</pre>

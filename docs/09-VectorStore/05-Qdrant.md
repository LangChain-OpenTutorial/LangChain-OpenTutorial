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

Qdrant's architecture is optimized for efficient vector similarity searches, employing advanced indexing techniques like Hierarchical Navigable Small World (HNSW) graphs to enable fast and scalable retrieval of relevant data.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Credentials](#credentials)
- [Installation](#installation)
- [Initialization](#initialization)
- [Manage VectorStore](#manage-vectorstore)
- [Query VectorStore](#query-vectorstore)

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



## Credentials

Create a new account or sign in to your existing one, and generate an API key for use in this notebook.

1. **Log in to Qdrant Cloud** : Go to the [Qdrant Cloud](https://cloud.qdrant.io) website and log in using your email, Google account, or GitHub account.

2. **Create a Cluster** : After logging in, navigate to the `"Clusters"` section and click the `"Create"` button. Choose your desired configurations and region, then click `"Create"` to start building your cluster. Once the cluster is created, an API key will be generated for you.

3. **Retrieve and Store Your API Key** : When your cluster is created, you will receive an API key. Ensure you save this key in a secure location, as you will need it later. If you lose it, you will have to generate a new one.

4. **Manage API Keys** : To create additional API keys or manage existing ones, go to the `"Access Management"` section in the Qdrant Cloud dashboard and select `"Qdrant Cloud API Keys"` Here, you can create new keys or delete existing ones.

```
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
```

## Installation

There are several main options for initializing and using the Qdrant vector store:

- **Local Mode** : This mode doesn't require a separate server.
    - **In-memory storage** (data is not persisted)
    - **On-disk storage** (data is saved to your local machine)
- **Docker Deployments** : You can run Qdrant using Docker.
- **Qdrant Cloud** : Use Qdrant as a managed cloud service.

For detailed instructions, see the [installation instructions](https://qdrant.tech/documentation/guides/installation/).

### In-Memory

For simple tests or quick experiments, you might choose to store data directly in memory. This means the data is automatically removed when your client terminates, typically at the end of your script or notebook session.

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

# Step 1: Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 2: Initialize Qdrant client
client = QdrantClient(":memory:")

# Step 3: Create a Qdrant collection
collection_name = "demo_collection"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Step 4: Initialize QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
```

### On-Disk Storage

With on-disk storage, you can store your vectors directly on your hard drive without requiring a Qdrant server. This ensures that your data persists even when you restart the program.

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

# Step 1: Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 2: Initialize Qdrant client
qdrant_path = "./qdrant_memory"
client = QdrantClient(path=qdrant_path)

# Step 3: Create a Qdrant collection
collection_name = "demo_collection"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Step 4: Initialize QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
```

### Docker Deployments

You can deploy `Qdrant` in a production environment using [Docker](https://qdrant.tech/documentation/guides/installation/#docker) and [Docker Compose](https://qdrant.tech/documentation/guides/installation/#docker-compose). Refer to the Docker and Docker Compose setup instructions in the development section for detailed information.

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

# Step 1: Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 2: Initialize Qdrant client
url = "http://localhost:6333"
client = QdrantClient(url=url)

# Step 3: Create a Qdrant collection
collection_name = "demo_collection"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Step 4: Initialize QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
```

### Qdrant Cloud

For a production environment, you can use [Qdrant Cloud](https://cloud.qdrant.io/). It offers fully managed `Qdrant` databases with features such as horizontal and vertical scaling, one-click setup and upgrades, monitoring, logging, backups, and disaster recovery. For more information, refer to the [Qdrant Cloud documentation](https://qdrant.tech/documentation/cloud/).

```python
import getpass
import os

# Fetch the Qdrant server URL from environment variables or prompt for input
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = getpass.getpass("Enter your Qdrant Cloud URL key: ")
url = os.environ.get("QDRANT_URL")

# Fetch the Qdrant API key from environment variables or prompt for input
if not os.getenv("QDRANT_API_KEY"):
    os.environ["QDRANT_API_KEY"] = getpass.getpass("Enter your Qdrant API key: ")
api_key = os.environ.get("QDRANT_API_KEY")
```

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

# Step 1: Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 2: Initialize Qdrant client
client = QdrantClient(
    url=url,
    api_key=api_key,
)

# Step 3: Create a Qdrant collection
collection_name = "demo_collection"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Step 4: Initialize QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)
```

## Initialization

Once you've established your vector store, you'll likely need to manage the collections within it. Here are some common operations you can perform:

- Create a collection
- List collections
- Delete a collection
- Use an existing collection

### Create a Collection

To create a new collection in your Qdrant instance, you can use the `QdrantClient` class from the `qdrant-client` library.

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Step 1: Define collection name
collection_name = "my_new_collection"

# Initialize the Qdrant client
client = QdrantClient(
    url=url,
    api_key=api_key,
)

# Create a new collection in Qdrant
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Print confirmation
print(f"Collection '{collection_name}' created successfully.")
```

<pre class="custom">Collection 'my_new_collection' created successfully.
</pre>

### List Collections

To list all existing collections in your Qdrant instance, you can use the `QdrantClient` class from the `qdrant-client` library.

```python
from qdrant_client import QdrantClient

# Initialize the Qdrant client
client = QdrantClient(
    url=url,
    api_key=api_key,
)

# Retrieve and print collection names
collections_response = client.get_collections()
for collection in collections_response.collections:
    print(f"Collection Name: {collection.name}")
```

<pre class="custom">Collection Name: my_new_collection
    Collection Name: demo_collection
</pre>

### Delete a Collection

To delete a collection in Qdrant using the Python client, you can use the `delete_collection` method of the `QdrantClient` object.

```python
from qdrant_client import QdrantClient

# Define collection name
collection_name = "my_new_collection"

# Initialize the Qdrant client
client = QdrantClient(
    url=url,
    api_key=api_key,
)

# Delete the collection
if client.delete_collection(collection_name=collection_name):
    print(f"Collection '{collection_name}' has been deleted.")
```

<pre class="custom">Collection 'my_new_collection' has been deleted.
</pre>

### Use an Existing Collection

This code snippet demonstrates how to initialize a `QdrantVectorStore` using the `from_existing_collection` method provided by the langchain_qdrant library

```python
from langchain_qdrant import QdrantVectorStore

collection_name = "demo_collection"

# Initialize QdrantVectorStore using from_existing_collection method
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=url,
    api_key=api_key,
    prefer_grpc=False,
)
```

**Direct Initialization** 
- Offers more control by utilizing an existing `QdrantClient` instance, making it suitable for complex applications that require customized client configurations.

**from_existing_collection Method** 
- Provides a simplified and concise way to connect to an existing collection, ideal for quick setups or simpler applications.

## Manage VectorStore

After you've created your vector store, you can interact with it by adding or deleting items. Here are some common operations:

### Add Items to the Vector Store

With `Qdrant`, you can add items to your vector store using the `add_documents` function. If you add a document with an ID that already exists, the existing document will be updated with the new data. This process is called `upsert`.

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
uuids = [str(uuid4()) for _ in split_docs]

# Add documents to the vector store
vector_store.add_documents(
    documents=split_docs,
    ids=uuids,
    batch_size=10,
)
print(
    f"Uploaded {len(split_docs)} documents to Qdrant collection 'little_prince_collection'"
)
```

<pre class="custom">Uploaded 222 documents to Qdrant collection 'little_prince_collection'
</pre>

### Delete Items from the Vector Store

To remove items from your vector store, use the `delete` function. You can specify the items to delete using either IDs or filters.

```python
# Retrieve the last point ID from the list of UUIDs
point_id = uuids[-1]

# Delete the vector point by its point_id
vector_store.delete(ids=[point_id])

# Print confirmation of deletion
print(f"Vector point with ID {point_id} has been deleted.")
```

<pre class="custom">Vector point with ID c824af22-779a-4294-8c7b-6bc9de1ee9ce has been deleted.
</pre>

### Update items from vector store

To update items in your vector store, use the `set_payload` function. This function allows you to modify the content or metadata of existing item

```python
def retrieve_point_payload(vector_store, point_id):
    """
    Retrieve the payload of a point from the Qdrant collection using its ID.

    Args:
        vector_store (QdrantVectorStore): The vector store instance connected to the Qdrant collection.
        point_id (str): The unique identifier of the point to retrieve.

    Returns:
        dict: The payload of the retrieved point.

    Raises:
        ValueError: If the point with the specified ID is not found in the collection.
    """
    # Retrieve the vector point using the client
    response = vector_store.client.retrieve(
        collection_name=vector_store.collection_name,
        ids=[point_id],
    )

    # Check if the response is empty
    if not response:
        raise ValueError(f"Point ID {point_id} not found in the collection.")

    # Extract the payload from the retrieved point
    point = response[0]
    payload = point.payload
    print(f"Payload for point ID {point_id}: \n{payload}\n")

    return payload
```

```python
point_id = uuids[0]

# Retrieve the payload for the specified point ID
payload = retrieve_point_payload(vector_store, point_id)
```

<pre class="custom">Payload for point ID 13d90d2d-2988-4c33-9b55-8449c8525200: 
    {'page_content': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'metadata': {'source': './data/the_little_prince.txt'}}
    
</pre>

```python
def update_point_payload(vector_store, point_id, new_payload):
    """
    Update the payload of a specific point in a Qdrant collection.

    Args:
        vector_store (QdrantVectorStore): The vector store instance connected to the Qdrant collection.
        point_id (str): The unique identifier of the point to update.
        new_payload (dict): A dictionary containing the new payload data to set for the point.

    Returns:
        None

    Raises:
        Exception: If the update operation fails.
    """
    try:
        # Update the payload for the specified point
        vector_store.client.set_payload(
            collection_name=vector_store.collection_name,
            payload=new_payload,
            points=[point_id],
        )
        print(f"Successfully updated payload for point ID {point_id}.")
    except Exception as e:
        print(f"Failed to update payload for point ID {point_id}: {e}")
        raise
```

```python
point_id = uuids[0]
new_payload = {"page_content": "The Little Prince (1943)"}

# Update the point's payload
update_point_payload(vector_store, point_id, new_payload)
```

<pre class="custom">Successfully updated payload for point ID 13d90d2d-2988-4c33-9b55-8449c8525200.
</pre>

### Upsert items to vector store (parallel)

Use the `set_payload` function in parallel to efficiently add or update multiple items in the vector store using unique IDs, data, and metadata.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple


def update_payloads_parallel(
    vector_store, updates: List[Tuple[str, Dict]], num_workers: int
):
    """
    Update the payloads of multiple points in a Qdrant collection in parallel.

    Args:
        updates (List[Tuple[str, Dict]]): A list of tuples containing point IDs and their corresponding new payloads.
        num_workers (int): Number of worker threads to use for parallel execution.

    Returns:
        None
    """
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit update tasks to the executor
        future_to_point_id = {
            executor.submit(
                update_point_payload, vector_store, point_id, new_payload
            ): point_id
            for point_id, new_payload in updates
        }

        # Process completed futures
        for future in as_completed(future_to_point_id):
            point_id = future_to_point_id[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error updating point ID {point_id}: {e}")
```

```python
payload = retrieve_point_payload(vector_store, uuids[2])
```

<pre class="custom">Payload for point ID c0c2356a-5010-4bd6-aaee-990d0ab6fb48: 
    {'page_content': 'Born in 1900 in Lyons, France, young Antoine was filled with a passion for adventure. When he failed an entrance exam for the Naval Academy, his interest in aviation took hold. He joined the French Army Air Force in 1921 where he first learned to fly a plane. Five years later, he would leave the military in order to begin flying air mail between remote settlements in the Sahara desert.', 'metadata': {'source': './data/the_little_prince.txt'}}
    
</pre>

```python
# Update example
updates = [
    (
        uuids[1],
        {
            "page_content": "Antoine de Saint-Exupéry's passion for aviation not only fueled remarkable stories but also reflected the enduring allure of flight, inspiring technological advancements and daring feats that captivated the world over the past century."
        },
    ),
    (
        uuids[2],
        {
            "page_content": "Antoine de Saint-Exupéry, born in 1900 in Lyons, France, had an adventurous spirit from a young age. After failing the Naval Academy entrance exam, his fascination with aviation began to take flight. In 1921, he joined the French Army Air Force and learned to pilot an aircraft. By 1926, he left the military to embark on a career as an airmail pilot, delivering letters to isolated communities in the vast Sahara desert"
        },
    ),
    # Add more (point_id, new_payload) tuples as needed
]

# Update payloads in parallel
num_workers = 4
update_payloads_parallel(vector_store, updates, num_workers)
```

<pre class="custom">Successfully updated payload for point ID e72f942f-8f24-4855-b99e-41fa11e467fc.
    Successfully updated payload for point ID c0c2356a-5010-4bd6-aaee-990d0ab6fb48.
</pre>

## Query VectorStore

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Query directly

The most straightforward use case for the `Qdrant` vector store is performing similarity searches. Internally, your query is converted into a vector embedding, which is then used to identify similar documents within the `Qdrant` collection.

```python
query = "What is the significance of the rose in The Little Prince?"

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=1,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* "Go and look again at the roses. You will understand now that yours is unique in all the world. Then come back to say goodbye to me, and I will make you a present of a secret." 
    The little prince went
     [{'source': './data/the_little_prince.txt', '_id': '634892c2-9fc9-4bb5-9310-531149d1ade1', '_collection_name': 'demo_collection'}]
    
    
</pre>

### Similarity search with score

You can also search with score:

```python
query = "What is the significance of the rose in The Little Prince?"

results = vector_store.similarity_search_with_score(
    query=query,
    k=1,
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content[:200]}\n [{doc.metadata}]\n\n")
```

<pre class="custom">* [SIM=0.584994] "Go and look again at the roses. You will understand now that yours is unique in all the world. Then come back to say goodbye to me, and I will make you a present of a secret." 
    The little prince went
     [{'source': './data/the_little_prince.txt', '_id': '634892c2-9fc9-4bb5-9310-531149d1ade1', '_collection_name': 'demo_collection'}]
    
    
</pre>

### Query by turning into retreiver

You can also transform the vector store into a `retriever` for easier usage in your workflows or chains.

```python
query = "What is the significance of the rose in The Little Prince?"

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

results = retriever.invoke(query)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* "Go and look again at the roses. You will understand now that yours is unique in all the world. Then come back to say goodbye to me, and I will make you a present of a secret." 
    The little prince went
     [{'source': './data/the_little_prince.txt', '_id': '634892c2-9fc9-4bb5-9310-531149d1ade1', '_collection_name': 'demo_collection'}]
    
    
</pre>

### Search with Filtering

This code demonstrates how to search for and retrieve records from a Qdrant vector database based on specific metadata field values.

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchText


def filter_and_retrieve_records(vector_store, filter_condition):
    """
    Retrieve records from a Qdrant vector store based on a given filter condition.

    Args:
        vector_store (QdrantVectorStore): The vector store instance connected to the Qdrant collection.
        filter_condition (Filter): The filter condition to apply for retrieving records.

    Returns:
        list: A list of records matching the filter condition.
    """
    all_records = []
    next_page_offset = None

    while True:
        response, next_page_offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filter_condition,
            limit=10,
            offset=next_page_offset,
            with_payload=True,
        )
        all_records.extend(response)
        if next_page_offset is None:
            break

    return all_records
```

```python
filter_condition = Filter(
    must=[
        FieldCondition(
            key="page_content",  # Ensure this key matches your payload structure
            match=MatchText(text="Academy"),  # Use MatchValue for exact matches
            # key="metadata.source",
            # match=MatchValue(value="./data/the_little_prince.txt")
        )
    ]
)

# Retrieve records based on the filter condition
records = filter_and_retrieve_records(vector_store, filter_condition)

# Print the retrieved records
for record in records[:1]:
    print(f"ID: {record.id}\nPayload: {record.payload}\n")
```

<pre class="custom">ID: c0c2356a-5010-4bd6-aaee-990d0ab6fb48
    Payload: {'page_content': 'Antoine de Saint-Exupéry, born in 1900 in Lyons, France, had an adventurous spirit from a young age. After failing the Naval Academy entrance exam, his fascination with aviation began to take flight. In 1921, he joined the French Army Air Force and learned to pilot an aircraft. By 1926, he left the military to embark on a career as an airmail pilot, delivering letters to isolated communities in the vast Sahara desert', 'metadata': {'source': './data/the_little_prince.txt'}}
    
</pre>

### Delete with Filtering

This code demonstrates how to delete records from a Qdrant vector database based on specific metadata field values.

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Define the filter condition
filter_condition = Filter(
    must=[
        FieldCondition(
            key="page_content",  # Ensure this key matches your payload structure
            match=MatchText(text="Academy"),  # Use MatchValue for exact matches
        )
    ]
)

# Perform the delete operation
client.delete(
    collection_name=vector_store.collection_name,
    points_selector=filter_condition,
    wait=True,
)

print("Delete operation completed.")
```

<pre class="custom">Delete operation completed.
</pre>

### Filtering and Updating Records

This code demonstrates how to retrieve and display records from a Qdrant collection based on a specific metadata field value.

```python
# Define the filter condition
filter_condition = Filter(
    must=[
        FieldCondition(
            key="page_content",  # Ensure this key matches your payload structure
            match=MatchText(text="Chapter"),  # Use MatchValue for exact matches
        )
    ]
)
# Retrieve matching records using the existing function
matching_points = filter_and_retrieve_records(vector_store, filter_condition)

# Prepare updates for matching points
for point in matching_points:
    updated_payload = point.payload.copy()

    # Update the page_content field by replacing "Chapter" with "Chapter -"
    updated_payload["page_content"] = updated_payload["page_content"].replace(
        "Chapter", "Chapter -"
    )

    # Update the payload using the existing function
    update_point_payload(vector_store, point.id, updated_payload)

print("Update operation completed.")
```

<pre class="custom">Successfully updated payload for point ID 071cae6b-5dc8-40ab-aac2-aff8796bff7f.
    Successfully updated payload for point ID 09628d96-3ec1-4914-b849-1e90dbe4dbc0.
    Successfully updated payload for point ID 0fe36061-9a47-4499-a5b5-bba74d7370a5.
    Successfully updated payload for point ID 12325628-09db-4526-8429-31b99c04e0ec.
    Successfully updated payload for point ID 19533ec1-ea7b-4e83-9a37-a71c3bc489f2.
    Successfully updated payload for point ID 2416e48f-8520-4d2e-9492-c7f0ae19fbd8.
    Successfully updated payload for point ID 29dd8cd9-5450-4ab3-8c39-e8eb56e7fee8.
    Successfully updated payload for point ID 325dd5af-0fc4-42f6-ad55-bb498c496b2a.
    Successfully updated payload for point ID 32dd484e-6413-4074-81d0-7233337469ef.
    Successfully updated payload for point ID 48f42368-c969-4fcb-91d3-770cd966294f.
    Successfully updated payload for point ID 48fd93c4-3e61-4e77-af86-d633535db061.
    Successfully updated payload for point ID 591594ef-76ab-4aca-803b-0dfe09ffd0e4.
    Successfully updated payload for point ID 5a0504c6-56f4-4667-8c98-2f31f136640a.
    Successfully updated payload for point ID 7a7ed2a6-b3b4-4d8a-9dd7-6687602b5b68.
    Successfully updated payload for point ID 7ed0d4c8-42be-4afb-9dc2-ab33d7d5f62e.
    Successfully updated payload for point ID 8efd04f0-3abc-4e10-92b5-d577451a135d.
    Successfully updated payload for point ID a3b96045-6c99-4541-b8f6-4cc291e35581.
    Successfully updated payload for point ID a64f34f6-b44b-4694-b357-cdec17ecd644.
    Successfully updated payload for point ID aa0519a6-80de-4e20-9757-811dc4fbaca7.
    Successfully updated payload for point ID c5a26ed3-4d5d-4325-b193-7fed809d2665.
    Successfully updated payload for point ID cb314628-bb64-4472-8cc8-ebacfab47262.
    Successfully updated payload for point ID d6bb59e4-9a20-4e6e-8591-235600b5165b.
    Successfully updated payload for point ID e46ed917-dc43-431e-aa1e-f1d28e25ff25.
    Successfully updated payload for point ID eda5363f-bb0a-4259-9c60-9bc50e46fc2a.
    Successfully updated payload for point ID fa6e2b4f-f698-4773-81fa-557b4073464d.
    Successfully updated payload for point ID fd58693b-fc06-40a0-aab8-638c6dfe9f2f.
    Successfully updated payload for point ID ffeb408c-ef72-4963-b5d3-d7035c788566.
    Update operation completed.
</pre>

### Similarity Search Options

When using `QdrantVectorStore`, you have three options for performing similarity searches. You can select the desired search mode using the retrieval_mode parameter when you set up the class. The available modes are:

- Dense Vector Search (Default)
- Sparse Vector Search
- Hybrid Search

### Dense Vector Search

To perform a search using only dense vectors:

The `retrieval_mode` parameter must be set to `RetrievalMode.DENSE`. This is also the default setting.
You need to provide a [dense embeddings](https://python.langchain.com/docs/integrations/text_embedding/) value through the embedding parameter.

```python
from langchain_qdrant import RetrievalMode

query = "What is the significance of the rose in The Little Prince?"

# Initialize QdrantVectorStore
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    url=url,
    api_key=api_key,
    collection_name="dense_collection",
    retrieval_mode=RetrievalMode.DENSE,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=1,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* "Go and look again at the roses. You will understand now that yours is unique in all the world. Then come back to say goodbye to me, and I will make you a present of a secret." 
    The little prince went
     [{'source': './data/the_little_prince.txt', '_id': 'b024fac2-620e-4102-bf55-a53becd3d174', '_collection_name': 'dense_collection'}]
    
    
</pre>

### Sparse Vector Search

To search with only sparse vectors,

The `retrieval_mode` parameter should be set to `RetrievalMode.SPARSE` .
An implementation of the [SparseEmbeddings](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any sparse embeddings provider has to be provided as value to the `sparse_embedding` parameter.
The `langchain-qdrant` package provides a FastEmbed based implementation out of the box.

To use it, install the [FastEmbed](https://github.com/qdrant/fastembed) package.

pip install fastembed

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

query = "What is the significance of the rose in The Little Prince?"

# Initialize QdrantVectorStore
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url=url,
    api_key=api_key,
    collection_name="sparse_collection",
    retrieval_mode=RetrievalMode.SPARSE,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=1,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* [ Chapter 20 ]
    - the little prince discovers a garden of roses
    But it happened that after walking for a long time through sand, and rocks, and snow, the little prince at last came upon a road. And all
     [{'source': './data/the_little_prince.txt', '_id': '9b772687-0981-4e0b-acc6-a13b76746665', '_collection_name': 'sparse_collection'}]
    
    
</pre>

### Hybrid Vector Search
To perform a hybrid search using dense and sparse vectors with score fusion,

- The `retrieval_mode` parameter should be set to `RetrievalMode.HYBRID` .
- A [ `dense embeddings` ](https://python.langchain.com/docs/integrations/text_embedding/) value should be provided to the `embedding` parameter.
- An implementation of the [ `SparseEmbeddings` ](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) interface using any sparse embeddings provider has to be provided as value to the `sparse_embedding` parameter.

Note that if you've added documents with the `HYBRID` mode, you can switch to any retrieval mode when searching. Since both the dense and sparse vectors are available in the collection.

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings

query = "What is the significance of the rose in The Little Prince?"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Initialize QdrantVectorStore
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding,
    sparse_embedding=sparse_embeddings,
    url=url,
    api_key=api_key,
    collection_name="hybrid_collection",
    retrieval_mode=RetrievalMode.HYBRID,
    batch_size=10,
)

# Perform similarity search in the vector store
results = vector_store.similarity_search(
    query=query,
    k=1,
)

for res in results:
    print(f"* {res.page_content[:200]}\n [{res.metadata}]\n\n")
```

<pre class="custom">* [ Chapter 20 ]
    - the little prince discovers a garden of roses
    But it happened that after walking for a long time through sand, and rocks, and snow, the little prince at last came upon a road. And all
     [{'source': './data/the_little_prince.txt', '_id': '6540d214-84f2-4505-b2f1-7aa937f7e2d0', '_collection_name': 'hybrid_collection'}]
    
    
</pre>

```python

```

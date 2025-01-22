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

# Evaluation using RAGAS

- Author: [Sungchul Kim](https://github.com/rlatjcj)
- Peer Review: [Yoonji](https://github.com/samdaseuss), [Sunyoung Park](https://github.com/architectyou)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/16-Evaluations/02-Evaluation-using-RAGAS.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/16-Evaluations/02-Evaluation-using-RAGAS.ipynb)

## Overview
This tutorial will show you how to evaluate the quality of your LLM output using RAGAS.

Before starting this tutorial, let's review metrics to be used in this tutorial, **Context Recall**, **Context Precision**, **Answer Relevancy**, and **Faithfulness** first.

### Context Recall

It estimates "how well the retrieved context matches the LLM-generated answer".  
It is calculated using question, ground truth, and retrieved context. The value is between 0 and 1, and higher values indicate better performance. To estimate context recall from the ground truth answer, each claim in the ground truth answer is analyzed to see if it can be attributed to the retrieved context. In the ideal scenario, all claims in the ground truth answer should be able to be attributed to the retrieved context.

$$\text{Context Recall} = \frac{|\text{GT claims that can be attributed to context}|}{|\text{Number of claims in GT}|}$$


### Context Precision

It estimates "whether ground-truth related items in contexts are ranked at the top".

Ideally, all relevant chunks should appear in the top ranks. This metric is calculated using question, ground_truth, and contexts, with values ranging from 0 to 1. Higher scores indicate better precision.

The formula for Context Precision@K is as follows:

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total number of relevant items in the top K results}}$$

Here, Precision@k is calculated as follows:

$$\text{Precision@k} = \frac{\text{true positives@k}}{(\text{true positives@k + false positives@k})}$$

K is the total number of chunks in contexts, and $v_k \in \{0, 1\}$ is the relevance indicator at rank k.

This metric is used to evaluate the quality of the retrieved context in information retrieval systems. It measures how well relevant information is placed in the top ranks, allowing for performance assessment.


### Answer Relevancy (Response Relevancy)

It is a metric that evaluates "how well the generated answer matches the given prompt".

The main features and calculation methods of this metric are as follows:

1. Purpose: Evaluate the relevance of the generated answer.
2. Score interpretation: Lower scores indicate incomplete or duplicate information in the answer, while higher scores indicate better relevance.
3. Elements used in calculation: question, context, answer

The calculation method for Answer Relevancy is defined as the average cosine similarity between the original question and the generated synthetic questions.

$$\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^N \cos(E_{g_i}, E_o)$$

or

$$\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^N \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}$$

Here:
- $E_{g_i}$ is the embedding of the generated question $i$
- $E_o$ is the embedding of the original question
- $N$ is the number of generated questions (default value is 3)

Note:
- The actual score is mostly between 0 and 1, but mathematically it can be between -1 and 1 due to the characteristics of cosine similarity.

This metric is useful for evaluating the performance of question-answering systems, particularly for measuring how well the generated answer reflects the original question's intent.


### Faithfulness

It is a metric that evaluates "the factual consistency of the generated answer compared to the given context".

The main features and calculation methods of this metric are as follows:

1. Purpose: Evaluate the factual consistency of the generated answer compared to the given context.
2. Calculation elements: Use the generated answer and the retrieved context.
3. Score range: Adjusted between 0 and 1, with higher values indicating better performance.

The calculation method for Faithfulness score is as follows:

$$\text{Faithfulness score} = \frac{|\text{Number of claims in the generated answer that can be inferred from given context}|}{|\text{Total number of claims in the generated answer}|}$$

Calculation process:
1. Identify claims in the generated answer.
2. Verify each claim against the given context to check if it can be inferred from the context.
3. Use the above formula to calculate the score.

Example:
- Question: "When and where was Einstein born?"
- Context: "Albert Einstein (born March 14, 1879) is a German-born theoretical physicist, widely considered one of the most influential scientists of all time."
- High faithfulness answer: "Einstein was born in Germany on March 14, 1879."
- Low faithfulness answer: "Einstein was born in Germany on March 20, 1879."

This metric is useful for evaluating the performance of question-answering systems, particularly for measuring how well the generated answer reflects the given context.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Load saved RAGAS dataset](#load-saved-ragas-dataset)
- [Evaluate the answers](#evaluate-the-answers)

### References

- [RAGAS Documentation](https://docs.ragas.io/en/stable/)
- [RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/)
- [RAGAS Metrics - Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [RAGAS Metrics - Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [RAGAS Metrics - Answer Relevancy (Response Relevancy)](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance)
- [RAGAS Metrics - Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

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
        "langsmith",
        "langchain_core",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
        "ragas",
        "pymupdf",
        "faiss-cpu",
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
        "LANGCHAIN_PROJECT": "Evaluation-using-RAGAS",
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



## Load saved RAGAS dataset

`# TODO (sungchul): update the filename & link`  
Load the RAGAS dataset that you saved in the previous step ([16-Evaluations/01-Test-Dataset-Generator-RAGAS.ipynb](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/16-Evaluations/01-Test-Dataset-Generator-RAGAS.ipynb)).

```python
import pandas as pd

df = pd.read_csv("data/ragas_synthetic_dataset.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the role of generative AI in the conte...</td>
      <td>["Agents\nThis combination of reasoning,\nlogi...</td>
      <td>Generative AI models can be trained to use too...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the essential components of an agent'...</td>
      <td>['Agents\nWhat is an agent?\nIn its most funda...</td>
      <td>The essential components in an agent's cogniti...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What are the key considerations for selecting ...</td>
      <td>['Agents\nFigure 1. General agent architecture...</td>
      <td>When selecting a model for an agent, it is cru...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How does retrieval augmented generation enhanc...</td>
      <td>['Agents\nThe tools\nFoundational models, desp...</td>
      <td>Retrieval augmented generation (RAG) significa...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>In the context of AI agents, how does the CoT ...</td>
      <td>['Agents\nAgents vs. models\nTo gain a clearer...</td>
      <td>The CoT framework enhances reasoning capabilit...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
  </tbody>
</table>
</div>



```python
from datasets import Dataset

test_dataset = Dataset.from_pandas(df)
test_dataset
```




<pre class="custom">Dataset({
        features: ['user_input', 'reference_contexts', 'reference', 'synthesizer_name'],
        num_rows: 10
    })</pre>



```python
import ast

# Convert contexts column from string to list
def convert_to_list(example):
    contexts = ast.literal_eval(example["reference_contexts"])
    return {"reference_contexts": contexts}

test_dataset = test_dataset.map(convert_to_list)
print(test_dataset)
```

<pre class="custom">Map: 100%|██████████| 10/10 [00:00<00:00, 721.48 examples/s]</pre>

    Dataset({
        features: ['user_input', 'reference_contexts', 'reference', 'synthesizer_name'],
        num_rows: 10
    })
    

    
    

```python
test_dataset[1]["reference_contexts"]
```




<pre class="custom">['Agents\nWhat is an agent?\nIn its most fundamental form, a Generative AI agent can be defined as an application that\nattempts to achieve a goal by observing the world and acting upon it using the tools that it\nhas at its disposal. Agents are autonomous and can act independently of human intervention,\nespecially when provided with proper goals or objectives they are meant to achieve. Agents\ncan also be proactive in their approach to reaching their goals. Even in the absence of\nexplicit instruction sets from a human, an agent can reason about what it should do next to\nachieve its ultimate goal. While the notion of agents in AI is quite general and powerful, this\nwhitepaper focuses on the specific types of agents that Generative AI models are capable of\nbuilding at the time of publication.\nIn order to understand the inner workings of an agent, let’s first introduce the foundational\ncomponents that drive the agent’s behavior, actions, and decision making. The combination\nof these components can be described as a cognitive architecture, and there are many\nsuch architectures that can be achieved by the mixing and matching of these components.\nFocusing on the core functionalities, there are three essential components in an agent’s\ncognitive architecture as shown in Figure 1.\nSeptember 2024 5\n']</pre>



```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Step 1: Load Documents
loader = PyMuPDFLoader("data/Newwhitepaper_Agents2.pdf")
docs = loader.load()

# Step 2: Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# Step 3: Create Embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Create DB and Save
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Step 5: Create Retriever
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

# Step 7: Create LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Create batch dataset. Batch dataset is useful when you want to process a large number of questions at once.

- Reference for `batch`: [Link](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb)

```python
batch_dataset = [question for question in test_dataset["user_input"]]
batch_dataset[:3]
```




<pre class="custom">['What is the role of generative AI in the context of agents?',
     "What are the essential components of an agent's cognitive architecture as of September 2024?",
     'What are the key considerations for selecting a model for an agent in the context of advancements expected by September 2024?']</pre>



Call `batch()` to get answers for the batch dataset.

```python
answer = chain.batch(batch_dataset)
answer[:3]
```




<pre class="custom">['The role of generative AI in the context of agents is to extend the capabilities of language models by leveraging tools to access real-time information, suggest real-world actions, and autonomously plan and execute complex tasks. Generative AI models can be trained to use external tools to access specific information or perform actions, such as making API calls to send emails or complete transactions. These agents are autonomous, capable of acting independently of human intervention, and can proactively reason about what actions to take to achieve their goals. The orchestration layer, a cognitive architecture, structures the reasoning, planning, and decision-making processes of these agents.',
     "The essential components of an agent's cognitive architecture as of September 2024 include the core functionalities that drive the agent’s behavior, actions, and decision-making. These components can be described as a cognitive architecture, which involves the orchestration layer that structures reasoning, planning, decision-making, and guides the agent's actions.",
     "The key considerations for selecting a model for an agent, in the context of advancements expected by September 2024, include the model's ability to reason and act on various tasks, its capability to select the right tools, and how well those tools have been defined. Additionally, the model should be able to interact with external data and services through tools, which can significantly extend its capabilities beyond what the foundational model alone can achieve. This involves using tools like web API methods (GET, POST, PATCH, DELETE) to access and process real-world information, thereby supporting more specialized systems like retrieval augmented generation (RAG). The iterative approach to building complex agent architectures, focusing on experimentation and refinement, is also crucial to finding solutions for specific business cases and organizational needs."]</pre>



Store the answers generated by the LLM in the 'answer' column.

```python
# Overwrite or add 'answer' column
if "answer" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["answer"]).add_column("answer", answer)
else:
    test_dataset = test_dataset.add_column("answer", answer)
```

## Evaluate the answers

Using `ragas.evaluate()`, we can evaluate the answers.

```python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# Format dataset structure
formatted_dataset = []
for item in test_dataset:
    formatted_item = {
        "question": item["user_input"],
        "answer": item["answer"],
        "reference": item["answer"],
        "contexts": item["reference_contexts"],
        "retrieved_contexts": item["reference_contexts"],
    }
    formatted_dataset.append(formatted_item)

# Convert to RAGAS dataset
ragas_dataset = Dataset.from_list(formatted_dataset)

result = evaluate(
    dataset=ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

result
```

<pre class="custom">Evaluating: 100%|██████████| 40/40 [00:46<00:00,  1.16s/it]
</pre>




    {'context_precision': 1.0000, 'faithfulness': 0.5894, 'answer_relevancy': 0.9694, 'context_recall': 0.7167}



```python
result_df = result.to_pandas()
result_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>context_precision</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the role of generative AI in the conte...</td>
      <td>[Agents\nThis combination of reasoning,\nlogic...</td>
      <td>The role of generative AI in the context of ag...</td>
      <td>The role of generative AI in the context of ag...</td>
      <td>1.0</td>
      <td>0.470588</td>
      <td>1.000000</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the essential components of an agent'...</td>
      <td>[Agents\nWhat is an agent?\nIn its most fundam...</td>
      <td>The essential components of an agent's cogniti...</td>
      <td>The essential components of an agent's cogniti...</td>
      <td>1.0</td>
      <td>0.400000</td>
      <td>1.000000</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What are the key considerations for selecting ...</td>
      <td>[Agents\nFigure 1. General agent architecture ...</td>
      <td>The key considerations for selecting a model f...</td>
      <td>The key considerations for selecting a model f...</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How does retrieval augmented generation enhanc...</td>
      <td>[Agents\nThe tools\nFoundational models, despi...</td>
      <td>Retrieval Augmented Generation (RAG) enhances ...</td>
      <td>Retrieval Augmented Generation (RAG) enhances ...</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.919411</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>In the context of AI agents, how does the CoT ...</td>
      <td>[Agents\nAgents vs. models\nTo gain a clearer ...</td>
      <td>The Chain-of-Thought (CoT) framework enhances ...</td>
      <td>The Chain-of-Thought (CoT) framework enhances ...</td>
      <td>1.0</td>
      <td>0.076923</td>
      <td>0.944423</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
result_df.to_csv("data/ragas_evaluation_result.csv", index=False)
```

```python
result_df.loc[:, "context_precision":"context_recall"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>context_precision</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.470588</td>
      <td>1.000000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.400000</td>
      <td>1.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.919411</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.076923</td>
      <td>0.944423</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.923077</td>
      <td>0.938009</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.984205</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.920000</td>
      <td>0.971321</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>0.352941</td>
      <td>0.963824</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.916667</td>
      <td>0.972590</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



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

# LangSmith Custom LLM Evaluation

- Author: [HeeWung Song(Dan)](https://github.com/kofsitho87)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb)

## Overview

**LangSmith Custom LLM Evaluation** is a customizable evaluation framework in LangChain that enables users to assess LLM application outputs based on their specific requirements.

1. **Custom Evaluation Logic**: 
   - Define your own evaluation criteria
   - Create specific scoring mechanisms

2. **Easy Integration**:
   - Works with LangChain's RAG systems
   - Compatible with LangSmith for tracking

3. **Evaluation Methods**:
   - Simple metric-based evaluation
   - Advanced LLM-based assessment

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [RAG System Setup](#rag-system-setup)
- [Basic Custom Evaluator](#basic-custom-evaluator)
- [Custom LLM-as-Judge](#custom-llm-as-judge)


### References

- [LangChain Get started with LangSmith](https://docs.smith.langchain.com/)
- [LangChain How to define a custom evaluator](https://docs.smith.langchain.com/evaluation/how_to_guides/custom_evaluator)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial pandas
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
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
        "LANGCHAIN_PROJECT": "LangSmith-Custom-LLM-Evaluation",
    }
)
```

Alternatively, you can set and load `OPENAI_API_KEY` from a `.env` file.

**[Note]** This is only necessary if you haven't already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## RAG System Setup

We'll build a basic **RAG** (Retrieval-Augmented Generation) system to test **Custom Evaluators**. This implementation creates a question-answering system based on PDF documents, which will serve as our foundation for evaluation purposes.

This **RAG** system will be used to evaluate answer quality and accuracy through **Custom Evaluators** in later sections.

### RAG System Preparation

1. **Document Processing**
   - `load_documents()`: Loads PDF documents using `PyMuPDFLoader`
   - `split_documents()`: Splits documents into appropriate sizes using `RecursiveCharacterTextSplitter`

2. **Vector Store Creation**
   - `create_vectorstore()`: Creates vector DB using `OpenAIEmbeddings` and `FAISS`
   - `create_retriever()`: Generates a retriever based on the vector store

3. **QA Chain Configuration**
   - `create_chain()`: Creates a chain that answers questions based on retrieved context
   - Includes prompt template for question-answering tasks

```python
from myrag import PDFRAG
from langchain_openai import ChatOpenAI

# Create PDFRAG object
rag = PDFRAG(
    "data/Newwhitepaper_Agents2.pdf",
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Create Retriever
retriever = rag.create_retriever()

# Create Chain
chain = rag.create_chain(retriever)

# Generate answer for question
chain.invoke("List up the name of the authors")
```




<pre class="custom">'The authors are Julia Wiesinger, Patrick Marlow, and Vladimir Vuskovic.'</pre>



We'll create a function called `ask_question` that takes a dictionary `inputs` as a parameter and returns a dictionary with an `answer` key. This function will serve as our question-answering interface.

```python
# Create function to answer question
def ask_question(inputs: dict):
    return {"answer": chain.invoke(inputs["question"])}
```

## Basic Custom Evaluator

Let's explore the fundamental concepts of creating **Custom Evaluators**. **Custom Evaluators** are evaluation tools in LangChain's LangSmith evaluation system that users can define according to their specific requirements. LangSmith provides a comprehensive platform for monitoring, evaluating, and improving LLM applications.

### Understanding Evaluator Arguments

Custom Evaluator functions can use the following arguments:

- `run (Run)`: The complete Run object generated by the application
- `example (Example)`: Dataset example containing inputs, outputs, and metadata
- `inputs (dict)`: Input dictionary for a single example from the dataset
- `outputs (dict)`: Output dictionary generated by the application for given inputs
- `reference_outputs (dict)`: Reference output dictionary associated with the example

In most cases, `inputs`, `outputs`, and `reference_outputs` are sufficient. The `run` and `example` objects are only needed when additional metadata is required.

### Understanding Output Types

**Custom Evaluators** can return results in the following formats:

1. **Dictionary Format** (Recommended)
   ```python
   {"key": "metric_name", "score": value}
   ```

2. **Basic Types** (Python)
   - `int`, `float`, `bool`: Continuous numerical metrics
   - `str`: Categorical metrics
   
3. **Multiple Metrics**
   ```python
   [{"key": "metric1", "score": value1}, {"key": "metric2", "score": value2}]
   ```

### Random Score Evaluator Example

Now, let's create a simple **Custom Evaluator** example. This evaluator will return a random score between 1 and 10, regardless of the answer content.

**Random Score Evaluator Implementation**
- Takes `Run` and `Example` objects as input parameters
- Returns a dictionary in the format: **{"key": "score_name", "score": score}**

Here's the basic implementation of a random score evaluator:

```python
from langsmith.schemas import Run, Example
import random


def random_score_evaluator(run: Run, example: Example) -> dict:
    # Return random score
    score = random.randint(1, 10)
    return {"key": "random_score", "score": score}
```

```python
from langsmith.evaluation import evaluate

# Set dataset name
dataset_name = "RAG_EVAL_DATASET"

# Run
experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[random_score_evaluator],
    experiment_prefix="CUSTOM-EVAL",
    # Set experiment metadata
    metadata={
        "variant": "Random Score Evaluator",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'CUSTOM-EVAL-565330e1' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=d0296986-a186-4dc6-a327-659c1e00169c
    
    
</pre>


    0it [00:00, ?it/s]


```python
experiment_results.to_pandas()
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
      <th>inputs.question</th>
      <th>outputs.answer</th>
      <th>error</th>
      <th>reference.answer</th>
      <th>feedback.random_score</th>
      <th>execution_time</th>
      <th>example_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>The three targeted learnings to enhance model ...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>4</td>
      <td>3.112384</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>ae36f6a7-86a2-4f0a-89d2-8be9671ca3cb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>6</td>
      <td>4.077394</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>6c65f286-a103-4a60-b906-555fd405ea7e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>List up the name of the authors</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>7</td>
      <td>1.172011</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>429dad1e-f68c-4f67-ae36-cc2171c4c6a0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is Tree-of-thoughts?</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>5</td>
      <td>1.374912</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>be337bef-90b0-4b6a-b9ab-941562ab4b44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>7</td>
      <td>1.821961</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>9cff92b1-04e7-49f5-ab2a-85763468e6cb</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How do agents differ from standalone language ...</td>
      <td>Agents differ from standalone language models ...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1</td>
      <td>2.135424</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>3fbe6fa6-88bf-46de-bdfa-0f39eac18c78</td>
    </tr>
  </tbody>
</table>
</div>



![CUSTOM-EVAL-FOR-RANDOM-SCORE](./img/07-LangSmith-Custom-LLM-Evaluation-01.png)

## Custom LLM-as-Judge

Now, we'll create a LLM Chain to use as an evaluator. 

First, let's define a function that returns `context`, `answer`, and `question`:

```python
# Function to return RAG results with `context`, `answer`, and `question`
def context_answer_rag_answer(inputs: dict):
    # Get context from Vector Store Retriever
    context = retriever.invoke(inputs["question"])
    # Get answer from RAG Chain in PDFRAG
    answer = chain.invoke(inputs["question"])
    return {
        "context": "\n".join([doc.page_content for doc in context]),
        "answer": answer,
        "question": inputs["question"],
    }
```

Let's run our evaluation using LangSmith's evaluate function. We'll use our custom evaluator to assess the RAG system's performance across our test dataset.

We'll use the "teddynote/context-answer-evaluator" prompt template from LangChain Hub, which provides a structured evaluation framework for RAG systems.

The evaluator uses the following criteria:
- **Accuracy (0-10)**: How well the answer aligns with the context
- **Comprehensiveness (0-10)**: How complete and detailed the answer is
- **Context Precision (0-10)**: How effectively the context information is used

The final score is normalized to a 0-1 scale using the formula:
`Final Score = (Accuracy + Comprehensiveness + Context Precision) / 30`

This evaluation framework helps us quantitatively assess the quality of our RAG system's responses.

```python
from langchain import hub

# Get evaluator Prompt
llm_evaluator_prompt = hub.pull("teddynote/context-answer-evaluator")
llm_evaluator_prompt.pretty_print()
```

<pre class="custom">
    As an LLM evaluator (judge), please assess the LLM's response to the given question. Evaluate the response's accuracy, comprehensiveness, and context precision based on the provided context. After your evaluation, return only the numerical scores in the following format:
    Accuracy: [score]
    Comprehensiveness: [score]
    Context Precision: [score]
    Final: [normalized score]
    Grading rubric:
    
    Accuracy (0-10 points):
    Evaluate how well the answer aligns with the information provided in the given context.
    
    0 points: The answer is completely inaccurate or contradicts the provided context
    4 points: The answer partially aligns with the context but contains significant inaccuracies
    7 points: The answer mostly aligns with the context but has minor inaccuracies or omissions
    10 points: The answer fully aligns with the provided context and is completely accurate
    
    
    Comprehensiveness (0-10 points):
    
    0 points: The answer is completely inadequate or irrelevant
    3 points: The answer is accurate but too brief to fully address the question
    7 points: The answer covers main aspects but lacks detail or misses minor points
    10 points: The answer comprehensively covers all aspects of the question
    
    
    Context Precision (0-10 points):
    Evaluate how precisely the answer uses the information from the provided context.
    
    0 points: The answer doesn't use any information from the context or uses it entirely incorrectly
    4 points: The answer uses some information from the context but with significant misinterpretations
    7 points: The answer uses most of the relevant context information correctly but with minor misinterpretations
    10 points: The answer precisely and correctly uses all relevant information from the context
    
    
    Final Normalized Score:
    Calculate by summing the scores for accuracy, comprehensiveness, and context precision, then dividing by 30 to get a score between 0 and 1.
    Formula: (Accuracy + Comprehensiveness + Context Precision) / 30
    
    #Given question:
    [33;1m[1;3m{question}[0m
    
    #LLM's response:
    [33;1m[1;3m{answer}[0m
    
    #Provided context:
    [33;1m[1;3m{context}[0m
    
    Please evaluate the LLM's response according to the criteria above. 
    
    In your output, include only the numerical scores for FINAL NORMALIZED SCORE without any additional explanation or reasoning.
    ex) 0.81
    
    #Final Normalized Score(Just the number):
    
    
</pre>

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Create evaluator
custom_llm_evaluator = (
    llm_evaluator_prompt
    | ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    | StrOutputParser()
)
```

Let's evaluate our system using the previously created `context_answer_rag_answer` function. We'll pass the generated answer and context to our `custom_llm_evaluator` for assessment.

```python
# Generate answer
output = context_answer_rag_answer(
    {"question": "What are the three targeted learnings to enhance model performance?"}
)

# Run evaluator
custom_llm_evaluator.invoke(output)
```




<pre class="custom">'0.87'</pre>



Let's define our `custom_evaluator` function.

- `run.outputs`: Gets the `answer`, `context`, and `question` generated by the RAG chain
- `example.outputs`: Gets the reference answer from our dataset

```python
from langsmith.schemas import Run, Example

def custom_evaluator(run: Run, example: Example) -> dict:
    # Get LLM generated answer and reference answer
    llm_answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.outputs.get("question", "")

    # Return custom score
    score = custom_llm_evaluator.invoke(
        {"question": question, "answer": llm_answer, "context": context}
    )
    return {"key": "custom_score", "score": float(score)}
```

Let's run our evaluation using LangSmith's evaluate function.

```python
from langsmith.evaluation import evaluate

# Set dataset name
dataset_name = "RAG_EVAL_DATASET"

# Run
experiment_results = evaluate(
    context_answer_rag_answer,
    data=dataset_name,
    evaluators=[custom_evaluator],
    experiment_prefix="CUSTOM-LLM-EVAL",
    # Set experiment metadata
    metadata={
        "variant": "Evaluation using Custom LLM Evaluator",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'CUSTOM-LLM-EVAL-e33ee0a7' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=156ad2c4-b8ec-4ada-b76c-44b09a527b50
    
    
</pre>


    0it [00:00, ?it/s]


```python
experiment_results.to_pandas()
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
      <th>inputs.question</th>
      <th>outputs.context</th>
      <th>outputs.answer</th>
      <th>outputs.question</th>
      <th>error</th>
      <th>reference.answer</th>
      <th>feedback.custom_score</th>
      <th>execution_time</th>
      <th>example_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>The three targeted learnings to enhance model ...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>0.87</td>
      <td>3.603254</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>85ddbfcb-8c49-4551-890a-f137d7b413b8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>0.93</td>
      <td>4.028933</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>0b423bb6-c722-41af-ae6e-c193ebc3ff8a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>List up the name of the authors</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>0.87</td>
      <td>1.885114</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>54e0987b-502f-48a7-877f-4b3d56bd82cf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>What is Tree-of-thoughts?</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>0.87</td>
      <td>1.732563</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>f0b02411-b377-4eaa-821a-2108b8b4836f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>0.83</td>
      <td>2.651672</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>38d34eb6-1ec5-44ea-a7d0-c7c98d46b0bc</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>Agents differ from standalone language models ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>0.93</td>
      <td>2.519094</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>49b26b38-e499-4c71-bdcb-eccfa44a1beb</td>
    </tr>
  </tbody>
</table>
</div>



![CUSTOM-EVAL-FOR-CUSTOM-SCORE](./img/07-LangSmith-Custom-LLM-Evaluation-02.png)

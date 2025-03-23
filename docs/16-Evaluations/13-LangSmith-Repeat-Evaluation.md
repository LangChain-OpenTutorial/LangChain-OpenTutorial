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

# LangSmith Repeat Evaluation

- Author: [Hwayoung Cha](https://github.com/forwardyoung)
- Design: 
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

> Repeat evaluation is a method for measuring the performance of a model more accurately by performing multiple evaluations on the same dataset.

You can add repetition to the experiment. This notebook demonstrates how to use `LangSmith` for repeatable evaluations of language models. It covers setting up evaluation workflows, running evaluations on different datasets, and analyzing results to ensure consistency. The focus is on leveraging `LangSmith`'s tools for reproducible and scalable model evaluation.

This allows the evaluation to be repeated multiple times, which is useful in the following cases:

- For larger evaluation sets
- For chains that can generate variable responses
- For evaluations that can produce variable scores (e.g., `llm-as-judge`)

You can learn how to run an evaluation from [this site](https://docs.smith.langchain.com/evaluation/how_to_guides/evaluate_llm_application#evaluate-on-a-dataset-with-repetitions).

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Performing Repetitive Evaluations with num_repetitions](#performing-repetitive-evaluations-with-num_repetitions)
- [Define a function for RAG performance testing](#define-a-function-for-rag-performance-testing)
- [Repetitive evaluation of RAG using GPT models](#repetitive-evaluation-of-rag-using-gpt-models)
- [Repetitive evaluation of RAG using Ollama models](#repetitive-evaluation-of-rag-using-ollama-models)

## References
- [How to run an evaluation](https://docs.smith.langchain.com/evaluation/how_to_guides/evaluate_llm_application#evaluate-on-a-dataset-with-repetitions)
- [How to evaluate with repetitions](https://docs.smith.langchain.com/evaluation/how_to_guides/repetition)
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

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langchain_community",
        "langchain_ollama",
        "faiss-cpu",
        "pymupdf",
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
        "LANGSMITH_TRACING_V2": "true",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_PROJECT": "Repeat-Evaluations"
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set OPENAI_API_KEY in .env file and load it.

[Note] This is not necessary if you've already set OPENAI_API_KEY in previous steps.

```python
# Configuration file to manage API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Performing Repetitive Evaluations with `num_repetitions`

`LangSmith` offers a simple way to perform repetitive evaluations using the `num_repetitions` parameter in the evaluate function. This parameter specifies how many times each example in your dataset should be evaluated.

When you set `num_repetitions=N`, `LangSmith` will:

Run each example in your dataset N times.

Aggregate the results to provide a more accurate measure of your model's performance.

For example:

If your dataset has 10 examples and you set `num_repetitions=5`, each example will be evaluated 5 times, resulting in a total of 50 runs.

## Define a function for RAG performance testing

Create a RAG system to use for performance testing.

```python
from myrag import PDFRAG


# Create a function to generate responses to questions.
def ask_question_with_llm(llm):
    # Create a PDFRAG object
    rag = PDFRAG(
        "data/Newwhitepaper_Agents2.pdf",
        llm,
    )

    # Create a retriever
    retriever = rag.create_retriever()

    # Create a chain
    rag_chain = rag.create_chain(retriever)

    def _ask_question(inputs: dict):
        # Context retrieval for the question
        context = retriever.invoke(inputs["question"])
        # Combine the retrieved documents into a single string.
        context = "\n".join([doc.page_content for doc in context])
        # Return a dictionary containing the question, context, and answer.
        return {
            "question": inputs["question"],
            "context": context,
            "answer": rag_chain.invoke(inputs["question"]),
        }

    return _ask_question
```

In this tutorial, we use the `llama3.2` model for repetitive evaluations. Make sure to install [`Ollama`](https://ollama.com/) on your local machine and run `ollama pull llama3.2` to download the model before proceeding with this tutorial.

```python
!ollama pull llama3.2
```

<pre class="custom">pulling manifest ⠙ pulling manifest ⠙ pulling manifest ⠹ pulling manifest ⠸ pulling manifest ⠼ pulling manifest ⠴ pulling manifest ⠦ pulling manifest 
    pulling dde5aa3fc5ff... 100% ▕████████████████▏ 2.0 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 56bb8bd477a5... 100% ▕████████████████▏   96 B                         
    pulling 34bb5ab01051... 100% ▕████████████████▏  561 B                         
    verifying sha256 digest 
    writing manifest 
    success 
</pre>


Below is an example of loading and invoking the model:

```python
from langchain_ollama import ChatOllama

# Load the Ollama model
ollama = ChatOllama(model="llama3.2")

# Call the Ollama model
ollama.invoke("hello") 
```




<pre class="custom">AIMessage(content='Hello! How can I assist you today?', additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-02-17T06:53:39.1001407Z', 'done': True, 'done_reason': 'stop', 'total_duration': 640983000, 'load_duration': 31027500, 'prompt_eval_count': 26, 'prompt_eval_duration': 288000000, 'eval_count': 10, 'eval_duration': 319000000, 'message': Message(role='assistant', content='', images=None, tool_calls=None)}, id='run-e563e830-e561-4333-a402-ef1227d68222-0', usage_metadata={'input_tokens': 26, 'output_tokens': 10, 'total_tokens': 36})</pre>



```python
from langchain_openai import ChatOpenAI

gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=1.0))

# Load the Ollama model.
ollama_chain = ask_question_with_llm(ChatOllama(model="llama3.2"))
```

## Repetitive evaluation of RAG using GPT models

This section demonstrates the process of conducting repetitive evaluations of a RAG system using GPT models. It focuses on setting up and executing repeated tests to assess the consistency and performance of the RAG system across various scenarios, helping to identify potential areas for improvement and ensure reliable outputs.

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Create a QA evaluator
cot_qa_evalulator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": ChatOpenAI(model="gpt-4o-mini", temperature=0)},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],
        "input": example.inputs["question"],
    },
)

dataset_name = "RAG_EVAL_DATASET"

# Run the evaluation
evaluate(
    gpt_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="REPEAT_EVAL",
    # Specify the experiment metadata.
    metadata={
        "variant": "Perform repeat evaluation. GPT-4o-mini model (cot_qa)",
    },
    num_repetitions=3,
)
```

<pre class="custom">View the evaluation results for experiment: 'REPEAT_EVAL-9906ae0d' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=4ca1ec21-cda0-4b78-abda-f3ad3b42edc5
    
    
</pre>


    0it [00:00, ?it/s]





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
      <th>outputs.question</th>
      <th>outputs.context</th>
      <th>outputs.answer</th>
      <th>error</th>
      <th>reference.answer</th>
      <th>feedback.COT Contextual Accuracy</th>
      <th>execution_time</th>
      <th>example_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>The three targeted learnings to enhance model ...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>1</td>
      <td>13.151277</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>510240bb-4c28-4440-a769-929be7edb98f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>1</td>
      <td>4.226702</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>60c42896-89fe-4a57-b8e3-e5cdacabae30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The authors of the document are Julia Wiesinge...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1</td>
      <td>2.524669</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>d9a3335b-06d6-46a0-bcb1-3a84d3d56c66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1</td>
      <td>2.944406</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>0d8cc590-0518-4098-b006-b0613d5e7cb8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>The framework used for reasoning and planning ...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>1</td>
      <td>2.452457</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>155ef405-4754-441f-a178-177922122d63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>Agents differ from standalone language models ...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1</td>
      <td>2.868793</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>e0d61836-a440-463d-82c0-c32053b6337b</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>The three targeted learnings to enhance model ...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>1</td>
      <td>3.615821</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>65fb7cdf-4545-4330-b4b4-055fdfe710cb</td>
    </tr>
    <tr>
      <th>7</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>1</td>
      <td>2.201849</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>9d587a12-e035-45d6-9a8b-64c58ae4dd67</td>
    </tr>
    <tr>
      <th>8</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The authors listed are Julia Wiesinger, Patric...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1</td>
      <td>1.720297</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>eaff2aba-0e70-4a7c-b47f-912ac6318016</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1</td>
      <td>2.107871</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>7029baaf-2e66-4d71-98c5-443577b5c430</td>
    </tr>
    <tr>
      <th>10</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>1</td>
      <td>2.265368</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>04b223a3-5ae5-4180-a0c0-db818a9e28af</td>
    </tr>
    <tr>
      <th>11</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>Agents differ from standalone language models ...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1</td>
      <td>2.088294</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>676c6265-8cc1-41ac-828c-e294ac3f4a10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>The three targeted learning approaches mention...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>1</td>
      <td>3.550540</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>1b92081d-ca19-4679-906e-187dea30a5dc</td>
    </tr>
    <tr>
      <th>13</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>1</td>
      <td>4.070889</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>07b70cac-203f-4d39-998d-befef6bc0bd8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1</td>
      <td>1.588084</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>0f6ccf7a-f79f-4fdb-ab00-4831930e6e98</td>
    </tr>
    <tr>
      <th>15</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1</td>
      <td>2.138192</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>bd0f5f68-215e-4756-b87b-0aef5e4f01ab</td>
    </tr>
    <tr>
      <th>16</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>1</td>
      <td>2.071085</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>826d6013-987c-4095-80dd-612591271c2f</td>
    </tr>
    <tr>
      <th>17</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>Agents differ from standalone language models ...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1</td>
      <td>2.863684</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>5b172bbf-abe0-4a71-8a32-d2f05e4039bb</td>
    </tr>
  </tbody>
</table>
</div>



![13-langsmith-repeat-evaluation-01](./img/13-langsmith-repeat-evaluation-01.png)

## Repetitive evaluation of RAG using Ollama

This part focuses on performing repetitive evaluations of the RAG system using Ollama. It illustrates the process of setting up and running multiple tests with Ollama, allowing for a comprehensive evaluation of the RAG system's performance with these specific models.

```python
# Create a QA evaluator
cot_qa_evalulator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": ChatOllama(model="llama3.2", temperature=0)},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],
        "input": example.inputs["question"],
    },
)

dataset_name = "RAG_EVAL_DATASET"

# Run the evaluation
evaluate(
    ollama_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="REPEAT_EVAL",
    # Specify the experiment metadata.
    metadata={
        "variant": "Perform repeat evaluation. Ollama(llama3.2) (cot_qa)",
    },
    num_repetitions=3,
)
```

<pre class="custom">View the evaluation results for experiment: 'REPEAT_EVAL-8279cd53' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=cee9221e-93d8-40fd-9585-519466fa7f99
    
    
</pre>


    0it [00:00, ?it/s]





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
      <th>outputs.question</th>
      <th>outputs.context</th>
      <th>outputs.answer</th>
      <th>error</th>
      <th>reference.answer</th>
      <th>feedback.COT Contextual Accuracy</th>
      <th>execution_time</th>
      <th>example_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>In-context learning, Fine-tuning based learning.</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>0.0</td>
      <td>2.527441</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>96233779-b37d-484f-85a8-22a7320ff72b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>Based on the retrieved context, it appears tha...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>0.0</td>
      <td>7.891397</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>5f761c37-3bf0-4b64-91bf-0b1167165184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The names of the authors are:\n\n1. Julia Wies...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1.0</td>
      <td>3.461620</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>5e56e10f-9220-4107-b0da-cfd206e4cd27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts is a prompt engineering frame...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1.0</td>
      <td>3.017406</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>4f0f23af-2cf3-4de2-923f-d8cbbd184a47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>Based on the provided context, it appears that...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>0.0</td>
      <td>8.636841</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>f729da06-0b0e-42ff-88f1-64676e19d1b0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>According to the context, agents differ from s...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1.0</td>
      <td>6.293883</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>045cdaba-4dc0-46ad-a955-4d00944bfabd</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>The two methods mentioned for enhancing model ...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>0.0</td>
      <td>3.524431</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>e1f26ba7-cd91-4e4f-8684-4af4262b8c17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>Based on the retrieved context, the key functi...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>NaN</td>
      <td>5.473330</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>10df33b1-8936-454f-9c13-9baedb8d557a</td>
    </tr>
    <tr>
      <th>8</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The names of the authors are:\n\n1. Julia Wies...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1.0</td>
      <td>2.525374</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>77e497f6-3f3e-400d-a385-72063096f879</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1.0</td>
      <td>2.907534</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>a6b767b3-b831-4cbb-a62f-2e351a948a01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>Based on the retrieved context, it appears tha...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>0.0</td>
      <td>6.760531</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>c00fd2ce-4108-45e8-8b0d-0e2419e883f3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>Based on the provided context, it appears that...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>1.0</td>
      <td>6.969271</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>239706b7-f82c-49dd-a4ba-15d845d40f3e</td>
    </tr>
    <tr>
      <th>12</th>
      <td>What are the three targeted learnings to enhan...</td>
      <td>What are the three targeted learnings to enhan...</td>
      <td>Agents\n33\nSeptember 2024\nEnhancing model pe...</td>
      <td>In-context learning and Fine-tuning based lear...</td>
      <td>None</td>
      <td>The three targeted learning approaches to enha...</td>
      <td>0.0</td>
      <td>2.515873</td>
      <td>0e661de4-636b-425d-8f6e-0a52b8070576</td>
      <td>bad8da17-774d-43e4-b0f1-9436f4a6f516</td>
    </tr>
    <tr>
      <th>13</th>
      <td>What are the key functions of an agent's orche...</td>
      <td>What are the key functions of an agent's orche...</td>
      <td>implementation of the agent orchestration laye...</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>None</td>
      <td>The key functions of an agent's orchestration ...</td>
      <td>0.0</td>
      <td>6.819861</td>
      <td>3561c6fe-6ed4-4182-989a-270dcd635f32</td>
      <td>a08170c2-8953-450f-9e49-1b431f87f506</td>
    </tr>
    <tr>
      <th>14</th>
      <td>List up the name of the authors</td>
      <td>List up the name of the authors</td>
      <td>Agents\nAuthors: Julia Wiesinger, Patrick Marl...</td>
      <td>The names of the authors are:\n\n1. Julia Wies...</td>
      <td>None</td>
      <td>The authors are Julia Wiesinger, Patrick Marlo...</td>
      <td>1.0</td>
      <td>2.512632</td>
      <td>b03e98d1-44ad-4142-8dfa-7b0a31a57096</td>
      <td>e7b1221e-23fe-4715-8315-daa7375dd73f</td>
    </tr>
    <tr>
      <th>15</th>
      <td>What is Tree-of-thoughts?</td>
      <td>What is Tree-of-thoughts?</td>
      <td>weaknesses depending on the specific applicati...</td>
      <td>Tree-of-Thoughts (ToT) is a prompt engineering...</td>
      <td>None</td>
      <td>Tree-of-thoughts (ToT) is a prompt engineering...</td>
      <td>1.0</td>
      <td>3.005581</td>
      <td>be18ec98-ab18-4f30-9205-e75f1cb70844</td>
      <td>9c043533-e24e-498d-a27c-02b5499fd27e</td>
    </tr>
    <tr>
      <th>16</th>
      <td>What is the framework used for reasoning and p...</td>
      <td>What is the framework used for reasoning and p...</td>
      <td>reasoning frameworks (CoT, ReAct, etc.) to \nf...</td>
      <td>Based on the provided context, it seems that t...</td>
      <td>None</td>
      <td>The frameworks used for reasoning and planning...</td>
      <td>0.0</td>
      <td>4.558945</td>
      <td>eb4b29a7-511c-4f78-a08f-2d5afeb84320</td>
      <td>8875837e-fca5-4bf8-bf94-2fc733ae7387</td>
    </tr>
    <tr>
      <th>17</th>
      <td>How do agents differ from standalone language ...</td>
      <td>How do agents differ from standalone language ...</td>
      <td>1.\t Agents extend the capabilities of languag...</td>
      <td>According to the retrieved context, agents dif...</td>
      <td>None</td>
      <td>Agents can use tools to access real-time data ...</td>
      <td>0.0</td>
      <td>5.888388</td>
      <td>f4a5a0cf-2d2e-4e15-838a-bc8296eb708b</td>
      <td>1889177c-ea36-488d-9327-26147f4e83ee</td>
    </tr>
  </tbody>
</table>
</div>



![13-langsmith-repeat-evaluation-02](./img/13-langsmith-repeat-evaluation-02.png)

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

# Compare experiment evaluations 

- Author: [Yejin Park](https://github.com/ppakyeah)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial demonstrates the process of backtesting and comparing model evaluations using LangSmith, focusing on assessing RAG system performance between GPT-4 and Ollama models.

Through practical examples, you'll learn how to create evaluation datasets from production data, implement evaluation metrics, and analyze results using LangSmith's comparison features. 


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Backtesting with LangSmith](#backtesting-with-langsmith)

### References

- [LangSmith: Backtesting](https://docs.smith.langchain.com/evaluation/tutorials/backtesting)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_ollama",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_core",
        "pymupdf",
        "faiss-cpu"
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

**[Note]** If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "09-CompareEvaluation",
        }
    )
```

## Backtesting with LangSmith
Backtesting involves assessing new versions of your application using historical data and comparing the new outputs to the original ones.

Compared to evaluations using pre-production datasets, backtesting offers a clearer indication of whether the new version of your application is an improvement over the current deployment.

Here are the basic steps for backtesting:

1. Select sample runs from your production tracing project to test against.
2. Transform the run inputs into a dataset and record the run outputs as an initial experiment against that dataset.
3. Execute your new system on the new dataset and compare the results of the experiments.

You can easily compare the results of your experiments by utilizing the Compare feature provided by LangSmith.

### Define Functions for RAG Performance Testing

Let's create a RAG system to utilize for testing.

```python
from myrag import PDFRAG
from langchain_openai import ChatOpenAI


# Function to answer questions
def ask_question_with_llm(llm):
    # Create PDFRAG object
    rag = PDFRAG(
        "data/Newwhitepaper_Agents2.pdf",
        llm,
    )

    # Create retriever
    retriever = rag.create_retriever()

    # Create chain
    rag_chain = rag.create_chain(retriever)

    def _ask_question(inputs: dict):
        context = retriever.invoke(inputs["question"])
        context = "\n".join([doc.page_content for doc in context])
        return {
            "question": inputs["question"],
            "context": context,
            "answer": rag_chain.invoke(inputs["question"]),
        }

    return _ask_question
```

This tutorial uses the `llama3.2` 1b model. Please make sure you have Ollama installed.

For detailed information about Ollama, refer to the [GitHub tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/10-Ollama.ipynb).

```python
from langchain_ollama import ChatOllama

ollama = ChatOllama(model="llama3.2:1b")

# Call Ollama model
ollama.invoke("Hello?")
```




<pre class="custom">AIMessage(content='Hello. Is there something I can help you with or would you like to chat?', additional_kwargs={}, response_metadata={'model': 'llama3.2:1b', 'created_at': '2025-01-15T06:23:09.277041Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1404527875, 'load_duration': 566634125, 'prompt_eval_count': 27, 'prompt_eval_duration': 707000000, 'eval_count': 18, 'eval_duration': 127000000, 'message': Message(role='assistant', content='', images=None, tool_calls=None)}, id='run-01a8d197-dc3a-4471-855a-36daac538e8b-0', usage_metadata={'input_tokens': 27, 'output_tokens': 18, 'total_tokens': 45})</pre>



Create a function that utilizes the GPT-4o-mini model and the Ollama model to generate answers to your questions.

```python
gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))
ollama_chain = ask_question_with_llm(ChatOllama(model="llama3.2:1b"))
```

Then, evaluate the answers using the GPT-4o-mini model and the Ollama model.

Do this for each of the two chains.

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Create QA evaluator
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

# Run gpt evaluation
experiment_results1 = evaluate(
    gpt_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="MODEL_COMPARE_EVAL",
    metadata={
        "variant": "GPT-4o-mini Evaluation (cot_qa)",
    },
)

# Run ollama evaluation
experiment_results2 = evaluate(
    ollama_chain,
    data=dataset_name,
    evaluators=[cot_qa_evalulator],
    experiment_prefix="MODEL_COMPARE_EVAL",
    metadata={
        "variant": "Ollama(llama3.2:1b) Evaluation (cot_qa)",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'MODEL_COMPARE_EVAL-05b6496b' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=33fa8084-b82f-45ee-a3dd-c374caad16e0
    
    
</pre>


    0it [00:00, ?it/s]


    View the evaluation results for experiment: 'MODEL_COMPARE_EVAL-c264adb7' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=f784a8c4-88ab-4a35-89a7-3aba5367f182
    
    
    


    0it [00:00, ?it/s]


### Comparing the result

Use the Compare view to inspect the results.

1. In the Experiment tab of the dataset, select the experiment you want to compare.
2. Click the “Compare” button at the bottom.
3. The comparison view is displayed.

![compare-view-01.png](./img/09-compare-evaluation-compare-view-01.png)
![compare-view-02png](./img/09-compare-evaluation-compare-view-02.png)

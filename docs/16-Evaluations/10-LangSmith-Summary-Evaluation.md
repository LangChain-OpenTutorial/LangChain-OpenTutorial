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

# Summary Evaluators

- Author: [Youngjun Cho](https://github.com/choincnp)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This document provides a comprehensive guide to building and evaluating RAG systems using `LangChain` tools. It demonstrates how to define RAG performance testing functions, and utilize summary evaluators for relevance assessment. By leveraging models like `GPT-4o-mini` and `Ollama` , you can evaluate the relevance of generated answers and questions effectively.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Defining a function for rag performance testing](#defining-a-function-for-rag-performance-testing)
- [Summary evaluator for relevance assessment](#summary-evaluator-for-relevance-assessment)

### References

- [How to define a summary evaluator](https://docs.smith.langchain.com/evaluation/how_to_guides/summary)
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
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_ollama",
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

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
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com/",
            "LANGCHAIN_PROJECT": "10-LangSmith-Summary-Evaluation",  # set the project name same as the title
        }
    )
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Defining a Function for RAG Performance Testing

We’ll create a RAG system for testing purposes.

```python
from myrag import PDFRAG

# Function to generate answers for questions
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
        # Retrieve context for the question
        context = retriever.invoke(inputs["question"])
        # Combine retrieved documents into a single string
        context = "\n".join([doc.page_content for doc in context])
        # Return a dictionary with the question, context, and answer
        return {
            "question": inputs["question"],
            "context": context,
            "answer": rag_chain.invoke(inputs["question"]),
        }

    return _ask_question
```

We’ll create functions using `GPT-4o-mini` and `Ollama model` to answer questions.

```python
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

gpt_chain = ask_question_with_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0))
ollama_chain = ask_question_with_llm(ChatOllama(model="llama3.2:1b"))
```

The `OpenAIRelevanceGrader` evaluates whether the **question** , **context** , and **answer** are relevant.
- `target="retrieval-question"` : Evaluates the relevance of the **question** to the **context** .
- `target="retrieval-answer"` : Evaluates the relevance of the **answer** to the **context** .

We first need to define OpenAIRelevanceGrader.

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# Data Models
class GradeRetrievalQuestion(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the question."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the question, 'yes' or 'no'"
    )


# Data Models
class GradeRetrievalAnswer(BaseModel):
    """A binary score to determine the relevance of the retrieved documents to the answer."""

    score: str = Field(
        description="Whether the retrieved context is relevant to the answer, 'yes' or 'no'"
    )


class OpenAIRelevanceGrader:
    """
    OpenAI-based relevance grader class.

    This class evaluates how relevant a retrieved document is to a given question or answer.
    It operates in two modes: 'retrieval-question' or 'retrieval-answer'.

    Attributes:
        llm: The language model instance to use
        structured_llm_grader: LLM instance generating structured outputs
        grader_prompt: Prompt template for evaluation

    Args:
        llm: The language model instance to use
        target (str): Target of the evaluation ('retrieval-question' or 'retrieval-answer')
    """

    def __init__(self, llm, target="retrieval-question"):
        """
        Initialization method for the OpenAIRelevanceGrader class.

        Args:
            llm: The language model instance to use
            target (str): Target of the evaluation ('retrieval-question' or 'retrieval-answer')

        Raises:
            ValueError: Raised if an invalid target value is provided
        """
        self.llm = llm

        if target == "retrieval-question":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalQuestion
            )
        elif target == "retrieval-answer":
            self.structured_llm_grader = llm.with_structured_output(
                GradeRetrievalAnswer
            )
        else:
            raise ValueError(f"Invalid target: {target}")

        # Prompt
        target_variable = (
            "user question" if target == "retrieval-question" else "answer"
        )
        system = f"""You are a grader assessing relevance of a retrieved document to a {target_variable}. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the {target_variable}, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to {target_variable}."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Retrieved document: \n\n {{context}} \n\n {target_variable}: {{input}}",
                ),
            ]
        )
        self.grader_prompt = grade_prompt

    def create(self):
        """
        Creates and returns the relevance grader.

        Returns:
            Chain object for performing relevance evaluation
        """

        retrieval_grader_oai = self.grader_prompt | self.structured_llm_grader
        return retrieval_grader_oai


class GroundnessQuestionScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the answer is relevant to the question else answer 'no'"
    )


class GroundnessAnswerRetrievalScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the answer is relevant to the retrieved document else answer 'no'"
    )


class GroundnessQuestionRetrievalScore(BaseModel):
    """Binary scores for relevance checks"""

    score: str = Field(
        description="relevant or not relevant. Answer 'yes' if the question is relevant to the retrieved document else answer 'no'"
    )


class GroundednessChecker:
    """
    GroundednessChecker class evaluates the accuracy of a document.

    This class evaluates whether a given document is accurate.
    It returns one of two values: 'yes' or 'no'.

    Attributes:
        llm (BaseLLM): The language model instance to use
        target (str): Evaluation target ('retrieval-answer', 'question-answer', or 'question-retrieval')
    """

    def __init__(self, llm, target="retrieval-answer"):
        """
        Constructor for the GroundednessChecker class.

        Args:
            llm (BaseLLM): The language model instance to use
            target (str): Evaluation target ('retrieval-answer', 'question-answer', or 'question-retrieval')
        """
        self.llm = llm
        self.target = target

    def create(self):
        """
        Creates a chain for groundedness evaluation.

        Returns:
            Chain: Object for performing groundedness evaluation
        """
        # Parser
        if self.target == "retrieval-answer":
            llm = self.llm.with_structured_output(GroundnessAnswerRetrievalScore)
        elif self.target == "question-answer":
            llm = self.llm.with_structured_output(GroundnessQuestionScore)
        elif self.target == "question-retrieval":
            llm = self.llm.with_structured_output(GroundnessQuestionRetrievalScore)
        else:
            raise ValueError(f"Invalid target: {self.target}")

        # Prompt selection
        if self.target == "retrieval-answer":
            template = """You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {context} \n\n
                Here is the answer: {answer} \n
                If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer."""
            input_vars = ["context", "answer"]

        elif self.target == "question-answer":
            template = """You are a grader assessing whether an answer appropriately addresses the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the answer: {answer} \n
                If the answer directly addresses the question and provides relevant information, grade it as relevant. \n
                Consider both semantic meaning and factual accuracy in your assessment. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question."""
            input_vars = ["question", "answer"]

        elif self.target == "question-retrieval":
            template = """You are a grader assessing whether a retrieved document is relevant to the given question. \n
                Here is the question: \n\n {question} \n\n
                Here is the retrieved document: \n\n {context} \n
                If the document contains information that could help answer the question, grade it as relevant. \n
                Consider both semantic meaning and potential usefulness for answering the question. \n
                
                Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the question."""
            input_vars = ["question", "context"]

        else:
            raise ValueError(f"Invalid target: {self.target}")

        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=input_vars,
        )

        # Chain
        chain = prompt | llm
        return chain
```

Then, set `retrieval-question grader` and `retriever-answer grader`.

```python
from langchain_openai import ChatOpenAI


rq_grader = OpenAIRelevanceGrader(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="retrieval-question"
).create()

ra_grader = OpenAIRelevanceGrader(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="retrieval-answer"
).create()
```

Invoke the graders.

```python
rq_grader.invoke(
    {
        "input": "What are the three targeted learnings to enhance model performance?",
        "context": """
        The three targeted learning approaches to enhance model performance mentioned in the context are:
            1. Pre-training-based learning
            2. Fine-tuning based learning
            3. Using External Memory
        """,
    }
)

ra_grader.invoke(
    {
        "input": """
        The three targeted learning approaches to enhance model performance mentioned in the context are:
            1. In-context learning
            2. Fine-tuning based learning
            3. Using External Memory
        """,
        "context": """
        The three targeted learning approaches to enhance model performance mentioned in the context are:
            1. Pre-training-based learning
            2. Fine-tuning based learning
            3. Using External Memory
        """,
    }
)
```




<pre class="custom">GradeRetrievalAnswer(score='yes')</pre>



## Summary Evaluator for Relevance Assessment

Certain metrics can only be defined at the experiment level rather than for individual runs of an experiment.

For example, you may want to **calculate the evaluation score of a classifier across all runs** initiated from a dataset.

This is referred to as `summary_evaluators`.

These evaluators take lists of Runs and Examples instead of single instances.

```python
from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

def relevance_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
    rq_scores = 0  # Question relevance score
    ra_scores = 0  # Answer relevance score

    for run, example in zip(runs, examples):
        question = example.inputs["question"]
        context = run.outputs["context"]
        prediction = run.outputs["answer"]

        # Evaluate question relevance
        rq_score = rq_grader.invoke(
            {
                "input": question,
                "context": context,
            }
        )
        # Evaluate answer relevance
        ra_score = ra_grader.invoke(
            {
                "input": prediction,
                "context": context,
            }
        )

        # Accumulate relevance scores
        if rq_score.score == "yes":
            rq_scores += 1
        if ra_score.score == "yes":
            ra_scores += 1

    # Calculate the final relevance score (average of question and answer relevance)
    final_score = ((rq_scores / len(runs)) + (ra_scores / len(runs))) / 2

    return {"key": "relevance_score", "score": final_score}
```

Now, Let's evaluate.

```python
dataset_name = "RAG_EVAL_DATASET"

experiment_result1 = evaluate(
    gpt_chain,
    data=dataset_name,
    summary_evaluators=[relevance_score_summary_evaluator],
    experiment_prefix="SUMMARY_EVAL",
    metadata={
        "variant": "Using GPT-4o-mini: relevance evaluation with summary_evaluator",
    },
)

experiment_result2 = evaluate(
    ollama_chain,
    data=dataset_name,
    summary_evaluators=[relevance_score_summary_evaluator],
    experiment_prefix="SUMMARY_EVAL",
    metadata={
        "variant": "Using Ollama(llama3.2): relevance evaluation with summary_evaluator",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'SUMMARY_EVAL-6a120022' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=09484df0-8405-4452-9fda-0dc268a0de44
    
    
</pre>


    0it [00:00, ?it/s]


    View the evaluation results for experiment: 'SUMMARY_EVAL-a04c0d31' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=0ed4c0f5-1fff-4ea4-ac81-7cafe03a66d7
    
    
    


    0it [00:00, ?it/s]


Check the result.

[ **Note** ]  
Results are not available for individual datasets but can be reviewed at the experiment level.

![](./assets/10-LangSmith-Summary-Evaluation-01.png)

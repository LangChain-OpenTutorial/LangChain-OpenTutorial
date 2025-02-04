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

# Routing

- Author: [Jinu Cho](https://github.com/jinucho), [Lee Jungbin](https://github.com/leebeanbin)
- Peer Review: [Teddy Lee](https://github.com/teddylee777), [김무상](https://github.com/musangk), [전창원](https://github.com/changwonjeon)
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb)

## Overview

This tutorial introduces three key tools in LangChain: `RunnableSequence`, `RunnableBranch`, and `RunnableLambda`, essential for building efficient and powerful AI applications.

`RunnableSequence` is a fundamental component that enables sequential processing pipelines, allowing structured and efficient handling of AI-related tasks. It provides automatic data flow management, error handling, and seamless integration with other LangChain components.

`RunnableBranch` enables structured decision-making by routing input through predefined conditions, simplifying complex branching scenarios.

`RunnableLambda` offers a flexible, function-based approach, ideal for lightweight transformations and inline processing.

**Key Features of these components:**

- **`RunnableSequence`:**
  - Sequential processing pipeline creation
  - Automatic data flow management
  - Error handling and monitoring
  - Support for async operations  

- **`RunnableBranch`:**
  - Dynamic routing based on conditions
  - Structured decision trees
  - Complex branching logic

- **`RunnableLambda`:**
  - Lightweight transformations
  - Function-based processing
  - Inline data manipulation

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is the RunnableSequence](#what-is-the-runnablesequence)
- [What is the RunnableBranch](#what-is-the-runnablebranch)
- [RunnableLambda](#runnablelambda)
- [RunnableBranch](#runnablebranch)
- [Comparison of RunnableBranch and RunnableLambda](#comparison-of-runnablesequence-runnablebranch-and-runnablelambda)

### References
- [RunnableSequence API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSequence.html)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/interface)
- [RunnableBranch API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.branch.RunnableBranch.html)  
- [RunnableLambda API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)  
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

[Note]
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "pydantic",
    ],
    verbose=False,
    upgrade=True,
)
```

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "04-Routing",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Load environment variables
# Reload any variables that need to be overwritten from the previous cell

from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is the RunnableSequence

`RunnableSequence` is a fundamental component in LangChain that enables the creation of sequential processing pipelines. It allows developers to chain multiple operations together where the output of one step becomes the input of the next step.

### Key Concepts

1. **Sequential Processing**
   - Ordered execution of operations
   - Automatic data flow between steps
   - Clear pipeline structure

2. **Data Transformation**
   - Input preprocessing
   - State management
   - Output formatting

3. **Error Handling**
   - Pipeline-level error management
   - Step-specific error recovery
   - Fallback mechanisms

Let's explore these concepts with practical examples.

### Simple Example

First, we will create a Chain that classifies incoming questions into one of three categories: math, science, or other.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Basic Example: Text Processing Pipeline
basic_chain = (
    # Step 1: Input handling and prompt creation
    PromptTemplate.from_template("Summarize this text in three sentences: {text}")
    # Step 2: LLM processing
    | ChatOpenAI(temperature=0)
    # Step 3: Output parsing
    | StrOutputParser()
)

# Example usage
result = basic_chain.invoke({"text": "This is a sample text to process."})
print(result)
```

<pre class="custom">This text is a sample for processing purposes. It is likely being used as an example for a specific task or function. The content of the text is not specified beyond being a sample.
</pre>

### Basic Pipeline Creation

In this section, we'll explore how to create fundamental pipelines using RunnableSequence. We'll start with a simple text generation pipeline and gradually build more complex functionality.

**Understanding Basic Pipeline Structure**  
- Sequential Processing: How data flows through the pipeline
- Component Integration: Combining different LangChain components
- Data Transformation: Managing input/output between steps

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
Basic Text Generation Pipeline
This demonstrates the fundamental way to chain components in RunnableSequence.

Flow:
1. PromptTemplate -> Creates the prompt with specific instructions
2. ChatOpenAI -> Processes the prompt and generates content
3. StrOutputParser -> Cleans and formats the output
"""

# Step 1: Define the basic text generation chain
basic_generation_chain = (
    # Create prompt template for AI content generation
        PromptTemplate.from_template(
            """Generate a detailed technical explanation about {topic} in AI/ML field.
            Include:
            - Core technical concepts
            - Implementation details
            - Real-world applications
            - Technical challenges
            """
        )
        # Process with LLM
        | ChatOpenAI(temperature=0.7)
        # Convert output to clean string
        | StrOutputParser()
)

# Example usage
basic_result = basic_generation_chain.invoke({"topic": "Transformer architecture in LLMs"})
print("Generated Content:", result)
```

<pre class="custom">Generated Content: This text is a sample for processing purposes. It is likely being used as an example for a specific task or function. The content of the text is not specified beyond being a sample.
</pre>

### Advanced Analysis Pipeline


Building upon our basic pipeline, we'll now create a more sophisticated analysis system that processes and evaluates the generated content.

**Key Features**
- State Management: Maintaining context throughout the pipeline
- Structured Analysis: Organizing output in a clear format
- Error Handling: Basic error management implementation

```python
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time

# Step 1: Define the analysis prompt template
analysis_prompt = PromptTemplate.from_template(
    """Analyze this technical content and extract the most crucial insights:
    
    {generated_basic_content}
    
    Provide a concise analysis focusing only on the most important aspects:
    (Importance : You should use Notion Syntax and try highliting with underlines, bold, emoji for title or something you describe context)
    
    Output format markdown outlet:
    # Key Technical Analysis
    
    ## Core Concept Summary
    [Extract and explain the 2-3 most fundamental concepts]
    
    ## Critical Implementation Insights
    [Focus on crucial implementation details that make this technology work]
    
    ## Key Challenges & Solutions
    [Identify the most significant challenges and their potential solutions]
    """
)

# Step 2: Define the critical analysis chain
analysis_chain = RunnableSequence(
    first=analysis_prompt,
    middle=[ChatOpenAI(temperature=0)],
    last=StrOutputParser()
)

# Step 3: Define the basic generation chain
generation_prompt = RunnableLambda(lambda x: f"""Generate technical content about: {x['topic']}""")

basic_generation_chain = RunnableSequence(
    first=RunnablePassthrough(),
    middle=[generation_prompt],
    last=ChatOpenAI(temperature=0.7)
)

# Step 4: Define the state initialization function
def init_state(x):
    return {
        "topic": x["topic"],
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S')
    }

init_step = RunnableLambda(init_state)

# Step 5: Define the content generation function
def generated_basic_content(x):
    content = basic_generation_chain.invoke({"topic": x["topic"]})
    return {
        **x,
        # "generated_basic_content": content.content
        # To create a comprehensive wrap-up, you can combine the previous basic result with new annotated analysis.
        "generated_basic_content": basic_result
    }

generate_step = RunnableLambda(generated_basic_content)

# Step 6: Define the analysis function
def perform_analysis(x):
    analysis = analysis_chain.invoke({"generated_basic_content": x["generated_basic_content"]})
    return {
        **x,
        "key_insights": analysis
    }

analysis_step = RunnableLambda(perform_analysis)

# Step 7: Define the output formatting function
def format_output(x):
    return {
        "timestamp": x["start_time"],
        "topic": x["topic"],
        "content": x["generated_basic_content"],
        "analysis": x["key_insights"],
        "formatted_output": f"""
# Technical Analysis Summary
Generated: {x['start_time']}

## Original Technical Content
{x['generated_basic_content']}

---

{x['key_insights']}
"""
    }

format_step = RunnableLambda(format_output)

# Step 8: Create the complete analysis pipeline
analysis_pipeline = RunnableSequence(
    first=init_step,
    middle=[
        generate_step,
        analysis_step
    ],
    last=format_step
)
```

<p align="left">
 <img src = "./assets/04-routing-runnable-pipeline.png">
</p>

```python
# Example usage
def run_analysis(topic: str):
    result = analysis_pipeline.invoke({"topic": topic})

    print("Analysis Timestamp:", result["timestamp"])
    print("\nTopic:", result["topic"])
    print("\nFormatted Output:", result["formatted_output"])

if __name__ == "__main__":
    run_analysis("Transformer attention mechanisms")
```

<pre class="custom">Analysis Timestamp: 2025-01-16 00:01:15
    
    Topic: Transformer attention mechanisms
    
    Formatted Output: 
    # Technical Analysis Summary
    Generated: 2025-01-16 00:01:15
    
    ## Original Technical Content
    Transformer architecture in Language Model (LLM) is a type of neural network architecture that has gained popularity in the field of artificial intelligence and machine learning for its ability to handle sequential data efficiently. The core technical concept behind the Transformer architecture is the use of self-attention mechanisms to capture long-range dependencies in the input data.
    
    In a Transformer network, the input sequence is divided into tokens, which are then passed through multiple layers of self-attention and feedforward neural networks. The self-attention mechanism allows each token to attend to all other tokens in the input sequence, capturing the contextual information necessary for understanding the relationship between different parts of the input data. This enables the model to learn complex patterns in the data and generate more accurate predictions.
    
    The implementation of the Transformer architecture involves designing the network with multiple layers of self-attention and feedforward neural networks. Each layer consists of a multi-head self-attention mechanism, which allows the model to attend to different parts of the input data simultaneously. The output of the self-attention mechanism is then passed through a feedforward neural network with activation functions such as ReLU or GELU to introduce non-linearity into the model.
    
    Real-world applications of Transformer architecture in LLMs include natural language processing tasks such as language translation, text generation, and sentiment analysis. Transformers have shown state-of-the-art performance in these tasks, outperforming traditional recurrent neural networks and convolutional neural networks in terms of accuracy and efficiency. Companies like Google, OpenAI, and Facebook have used Transformer-based models in their products and services to improve language understanding and generation capabilities.
    
    However, there are also technical challenges associated with the Transformer architecture, such as the high computational cost of training and inference. Transformers require a large amount of memory and computational resources to process input sequences efficiently, making them computationally expensive to train and deploy. Researchers are actively working on developing more efficient versions of the Transformer architecture, such as the Transformer-XL and the Reformer, to address these challenges and make LLMs more accessible to a wider range of applications.
    
    ---
    
    # Key Technical Analysis
    
    ## Core Concept Summary
    - **Transformer Architecture**: Utilizes self-attention mechanisms to capture long-range dependencies in input data efficiently.
    - **Self-Attention Mechanism**: Allows each token to attend to all other tokens in the input sequence, enabling the model to understand relationships and learn complex patterns.
    
    ## Critical Implementation Insights
    - **Multi-Layer Design**: Transformer network consists of multiple layers of self-attention and feedforward neural networks.
    - **Multi-Head Self-Attention**: Enables the model to attend to different parts of the input data simultaneously, enhancing contextual understanding.
    - **Activation Functions**: Utilized in feedforward neural networks to introduce non-linearity into the model for better predictions.
    
    ## Key Challenges & Solutions
    - **High Computational Cost**: Training and inference in Transformers require significant memory and computational resources.
    - **Solutions**: Ongoing research focuses on developing more efficient versions like Transformer-XL and Reformer to address computational challenges and broaden application possibilities.
    
</pre>

### Structured Evaluation Pipeline

In this section, we'll add structured evaluation capabilities to our pipeline, including proper error handling and validation.

**Features**
- Structured Output: Using schema-based parsing
- Validation: Input and output validation
- Error Management: Comprehensive error handling

```python
"""
Structured Evaluation Pipeline

This demonstrates:
1. Custom output parsing with schema validation
2. Error handling at each pipeline stage
3. Comprehensive validation system
"""
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI
import json
import time

# Step 1: Define structured output schema
response_schemas = [
    ResponseSchema(
        name="technical_evaluation",
        description="Technical evaluation of the content",
        type="object",
        properties={
            "core_concepts": {
                "type": "array",
                "description": "Key technical concepts identified"
            },
            "implementation_details": {
                "type": "object",
                "properties": {
                    "complexity": {"type": "string"},
                    "requirements": {"type": "array"},
                    "challenges": {"type": "array"}
                }
            },
            "quality_metrics": {
                "type": "object",
                "properties": {
                    "technical_accuracy": {"type": "number"},
                    "completeness": {"type": "number"},
                    "clarity": {"type": "number"}
                }
            }
        }
    )
]

evaluation_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Step 2: Create basic generation chain
generation_prompt = RunnableLambda(lambda x: f"""Generate technical content about: {x['topic']}""")
basic_generation_chain = RunnableSequence(
    first=RunnablePassthrough(),
    middle=[generation_prompt],
    last=ChatOpenAI(temperature=0.7)
)

# Step 3: Create analysis chain
analysis_prompt = RunnableLambda(lambda x: f"""Analyze the following content: {x['generated_content']}""")
analysis_chain = RunnableSequence(
    first=RunnablePassthrough(),
    middle=[analysis_prompt],
    last=ChatOpenAI(temperature=0)
)

# Step 4: Create evaluation chain
evaluation_prompt = RunnableLambda(
    lambda x: f"""
    Evaluate the following AI technical content:
    {x['generated_content']}
    
    Provide a structured evaluation following these criteria:
    1. Identify and list core technical concepts
    2. Assess implementation details
    3. Rate quality metrics (1-10)
    
    {evaluation_parser.get_format_instructions()}
    """
)

evaluation_chain = RunnableSequence(
    first=RunnablePassthrough(),
    middle=[evaluation_prompt, ChatOpenAI(temperature=0)],
    last=evaluation_parser
)

# Helper function for error handling
def try_or_error(func, error_list):
    try:
        return func()
    except Exception as e:
        error_list.append(str(e))
        return None

# Step 5: Create pipeline components
def init_state(x):
    return {
        "topic": x["topic"],
        "errors": [],
        "start_time": time.time()
    }

def generate_content(x):
    return {
        **x,
        "generated_content": try_or_error(
            lambda: basic_generation_chain.invoke({"topic": x["topic"]}).content,
            x["errors"]
        )
    }

def perform_analysis(x):
    return {
        **x,
        "analysis": try_or_error(
            lambda: analysis_chain.invoke({"generated_content": x["generated_content"]}).content,
            x["errors"]
        )
    }

def perform_evaluation(x):
    return {
        **x,
        "evaluation": try_or_error(
            lambda: evaluation_chain.invoke(x),
            x["errors"]
        ) if not x["errors"] else None
    }

def finalize_output(x):
    return {
        **x,
        "completion_time": time.time() - x["start_time"],
        "status": "success" if not x["errors"] else "error"
    }

# Step 6: Create integrated pipeline
def create_evaluation_pipeline():
    return RunnableSequence(
        first=RunnableLambda(init_state),
        middle=[
            RunnableLambda(generate_content),
            RunnableLambda(perform_analysis),
            RunnableLambda(perform_evaluation)
        ],
        last=RunnableLambda(finalize_output)
    )

# Example usage
def demonstrate_evaluation():
    pipeline = create_evaluation_pipeline()
    result = pipeline.invoke({"topic": "Transformer attention mechanisms"})

    print("Pipeline Status:", result["status"])
    if result["status"] == "success":
        print("\nEvaluation Results:", json.dumps(result["evaluation"], indent=2))
    else:
        print("\nErrors Encountered:", result["errors"])

    print(f"\nProcessing Time: {result['completion_time']:.2f} seconds")

if __name__ == "__main__":
    demonstrate_evaluation()
```

<pre class="custom">Pipeline Status: success
    
    Evaluation Results: {
      "technical_evaluation": {
        "core_technical_concepts": [
          "Transformer model",
          "Attention mechanisms",
          "Input sequence processing",
          "Long-range dependencies",
          "Context-specific attention patterns"
        ],
        "implementation_details": "The content provides a clear explanation of how the attention mechanism works in the Transformer model, including how attention scores are computed and used to generate the final output. It also highlights the advantages of the Transformer model over traditional RNNs and CNNs in capturing long-range dependencies and learning context-specific patterns.",
        "quality_metrics": {
          "accuracy": 9,
          "clarity": 8,
          "relevance": 10,
          "depth": 8
        }
      }
    }
    
    Processing Time: 9.55 seconds
</pre>

## What is the RunnableBranch

`RunnableBranch` is a powerful tool that allows dynamic routing of logic based on input. It enables developers to flexibly define different processing paths depending on the characteristics of the input data.  

`RunnableBranch` helps implement complex decision trees in a simple and intuitive way. This greatly improves code readability and maintainability while promoting logic modularization and reusability.  

Additionally, `RunnableBranch` can dynamically evaluate branching conditions at runtime and select the appropriate processing routine, enhancing the system's adaptability and scalability.  

Due to these features, `RunnableBranch` can be applied across various domains and is particularly useful for developing applications with high input data variability and volatility.

By effectively utilizing `RunnableBranch`, developers can reduce code complexity and improve system flexibility and performance.

### Dynamic Logic Routing Based on Input

This section covers how to perform routing in LangChain Expression Language.

Routing allows you to create non-deterministic chains where the output of a previous step defines the next step. This helps bring structure and consistency to interactions with LLMs.

There are two primary methods for performing routing:

1. Returning a Conditionally Executable Object from `RunnableLambda` (*Recommended*)  
2. Using `RunnableBranch`

Both methods can be explained using a two-step sequence, where the first step classifies the input question as related to math, science, or other, and the second step routes it to the corresponding prompt chain.

### Simple Example

First, we will create a Chain that classifies incoming questions into one of three categories: math, science, or other.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Classify the given user question into one of `math`, `science`, or `other`. Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
)

# Create the chain.
chain = (
    prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()  # Use a string output parser.
)
```

Use the created chain to classify the question.

```python
# Invoke the chain with a question.
chain.invoke({"question": "What is 2+2?"})
```




<pre class="custom">'math'</pre>



```python
# Invoke the chain with a question.
chain.invoke({"question": "What is the law of action and reaction?"})
```




<pre class="custom">'science'</pre>



```python
# Invoke the chain with a question.
chain.invoke({"question": "What is LangChain?"})
```




<pre class="custom">'other'</pre>



## RunnableLambda  

`RunnableLambda` is a type of `Runnable` designed to simplify the execution of a single transformation or operation using a lambda (anonymous) function. 

It is primarily used for lightweight, stateless operations where defining an entire custom `Runnable` class would be overkill.  

Unlike `RunnableBranch`, which focuses on conditional branching logic, `RunnableLambda` excels in straightforward data transformations or function applications.

Syntax  
- `RunnableLambda` is initialized with a single lambda function or callable object.  
- When invoked, the input value is passed directly to the lambda function.  
- The lambda function processes the input and returns the result.  

Now, let's create three sub-chains.

```python
math_chain = (
    PromptTemplate.from_template(
        """You are an expert in math. \
Always answer questions starting with "Pythagoras once said...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

science_chain = (
    PromptTemplate.from_template(
        """You are an expert in science. \
Always answer questions starting with "Isaac Newton once said...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question concisely:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)
```

### Using Custom Functions  

This is the recommended approach in the official LangChain documentation. You can wrap custom functions with `RunnableLambda` to handle routing between different outputs.

```python
# Return each chain based on the contents included in the topic.
def route(info):
    if "math" in info["topic"].lower():
        return math_chain
    elif "science" in info["topic"].lower():
        return science_chain
    else:
        return general_chain
```

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

full_chain = (
    {"topic": chain, "question": itemgetter("question")}
    | RunnableLambda(
        # Pass the routing function as an argument.
        route
    )
    | StrOutputParser()
)
```

```python
# Invoke the chain with a math-related question.
full_chain.invoke({"question": "Please explain the concept of calculus."})
```




<pre class="custom">"Pythagoras once said that understanding the relationships between different quantities is essential for grasping the universe's complexities. Calculus is the branch of mathematics that investigates how things change and helps us understand the concept of motion and rates of change. It consists of two main branches: differential calculus, which focuses on the concept of the derivative, measuring how a function changes as its input changes, and integral calculus, which deals with accumulation, essentially summing up small parts to find whole quantities, like areas under curves. Together, these tools allow us to analyze complex systems, model real-world phenomena, and solve problems involving continuous change."</pre>



```python
# Invoke the chain with a science-related question.
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," reflecting his profound understanding of gravity. To calculate gravitational acceleration, we typically use the formula derived from Newton\'s law of universal gravitation. The gravitational acceleration \\( g \\) at the surface of a celestial body, such as Earth, can be calculated using the equation:\n\n\\[\ng = \\frac{G \\cdot M}{r^2}\n\\]\n\nwhere \\( G \\) is the universal gravitational constant (\\(6.674 \\times 10^{-11} \\, \\text{m}^3 \\text{kg}^{-1} \\text{s}^{-2}\\)), \\( M \\) is the mass of the celestial body, and \\( r \\) is the radius from the center of the mass to the point where gravitational acceleration is being calculated. For Earth, this results in an approximate value of \\( 9.81 \\, \\text{m/s}^2 \\). Thus, gravitational acceleration can be understood as the force of gravity acting on a unit mass near the surface of a large body.'</pre>



```python
# Invoke the chain with a general question.
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'RAG (Retrieval Augmented Generation) is a model framework that combines information retrieval and natural language generation. It retrieves relevant documents or information from a large database and uses that information to generate more accurate and contextually relevant text responses. This approach enhances the generation process by grounding it in concrete data, improving both the quality and relevance of the output.'</pre>



## RunnableBranch

`RunnableBranch` is a special type of `Runnable` that allows you to define conditions and corresponding Runnable objects based on input values.

However, it does not provide functionality that cannot be achieved with custom functions, so using custom functions is generally recommended.

Syntax

- `RunnableBranch` is initialized with a list of (condition, Runnable) pairs and a default Runnable.
- When invoked, the input value is passed to each condition sequentially.
- The first condition that evaluates to True is selected, and the corresponding Runnable is executed with the input value.
- If no condition matches, the `default Runnable` is executed.

```python
from operator import itemgetter
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # Check if the topic contains "math" and execute math_chain if true.
    (lambda x: "math" in x["topic"].lower(), math_chain),
    # Check if the topic contains "science" and execute science_chain if true.
    (lambda x: "science" in x["topic"].lower(), science_chain),
    # If none of the above conditions match, execute general_chain.
    general_chain,
)

# Define the full chain that takes a topic and question, routes it, and parses the output.
full_chain = (
    {"topic": chain, "question": itemgetter("question")} | branch | StrOutputParser()
)
```

Execute the full chain with each question.

```python
full_chain.invoke({"question": "Please explain the concept of calculus."})
```




<pre class="custom">'Pythagoras once said that understanding the world around us often requires us to look deeper into the relationships between various elements. Calculus, much like the geometric principles he championed, is a branch of mathematics that studies how things change. It is fundamentally divided into two main areas: differentiation and integration.\n\nDifferentiation focuses on the concept of the derivative, which represents the rate of change of a quantity. For instance, if you think of a car’s velocity as the rate of change of its position over time, calculus allows us to analyze and predict this kind of change in different contexts.\n\nIntegration, on the other hand, deals with the accumulation of quantities, which can be thought of as the total size or area under a curve. It answers questions like how much distance is traveled over time, given a particular speed.\n\nTogether, these two concepts allow us to model and understand a vast array of phenomena—from physics to economics—enabling us to explain how systems evolve and interact over time. Just as Pythagoras sought to uncover the hidden relationships within numbers and shapes, calculus seeks to reveal the intricate patterns of change in our world.'</pre>



```python
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," which reflects the fundamental principle of gravitational attraction. Gravitational acceleration, often denoted as \\( g \\), can be calculated using the formula:\n\n\\[\ng = \\frac{G \\cdot M}{r^2}\n\\]\n\nwhere \\( G \\) is the universal gravitational constant (approximately \\( 6.674 \\times 10^{-11} \\, \\text{N m}^2/\\text{kg}^2 \\)), \\( M \\) is the mass of the object creating the gravitational field (like the Earth), and \\( r \\) is the distance from the center of the mass to the point where the acceleration is being measured (which is the radius of the Earth when calculating gravitational acceleration at its surface). For Earth, this results in a standard gravitational acceleration of approximately \\( 9.81 \\, \\text{m/s}^2 \\).'</pre>



```python
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'RAG (Retrieval-Augmented Generation) is a framework that combines retrieval and generative models to improve the quality and relevance of generated text. It first retrieves relevant documents or information from a knowledge base and then uses this data to enhance the generation of responses, making the output more informative and contextually accurate.'</pre>



## Building an AI Learning Assistant

Let's apply what we've learned about Runnable components to build a practical AI Learning Assistant. This system will help students by providing tailored responses based on their questions.

First, let's set up our core components:

```python
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import json
import asyncio

# Question Classification Component
question_classifier = RunnableSequence(
    first=PromptTemplate.from_template(
        """Classify this question into one of: beginner, intermediate, advanced
        Consider:
        - Complexity of concepts
        - Prior knowledge required
        - Technical depth needed
        
        Question: {question}
        
        Return only the classification word in lowercase."""
    ),
    middle=[ChatOpenAI(temperature=0)],
    last=StrOutputParser()
)

# Example Generator Component
example_generator = RunnableSequence(
    first=PromptTemplate.from_template(
        """Generate a practical example for this concept.
        Level: {level}
        Question: {question}
        
        If code is needed, provide it in appropriate markdown format."""
    ),
    middle=[ChatOpenAI(temperature=0.7)],
    last=StrOutputParser()
)
```

Next, let's create our response generation strategy:

```python
# Response Generation Strategy
response_strategy = RunnableBranch(
    (
        lambda x: x["level"] == "beginner",
        RunnableSequence(
            first=PromptTemplate.from_template(
                """Explain in simple terms for a beginner:
                Question: {question}
                
                Use simple analogies and avoid technical jargon."""
            ),
            middle=[ChatOpenAI(temperature=0.3)],
            last=StrOutputParser()
        )
    ),
    (
        lambda x: x["level"] == "intermediate",
        RunnableSequence(
            first=PromptTemplate.from_template(
                """Provide a detailed explanation with practical examples:
                Question: {question}
                
                Include relevant technical concepts and use cases."""
            ),
            middle=[ChatOpenAI(temperature=0.3)],
            last=StrOutputParser()
        )
    ),
    # Default case (advanced)
    RunnableSequence(
        first=PromptTemplate.from_template(
            """Give an in-depth technical explanation:
            Question: {question}
            
            Include advanced concepts and detailed technical information."""
        ),
        middle=[ChatOpenAI(temperature=0.3)],
        last=StrOutputParser()
    )
)
```

Now, let's create our main pipeline:

```python
def format_response(x):
    return {
        "question": x["question"],
        "level": x["level"],
        "explanation": x["response"],
        "example": x["example"],
        "metadata": {
            "difficulty": x["level"],
            "timestamp": datetime.now().isoformat()
        }
    }

# Main Learning Assistant Pipeline
learning_assistant = RunnableSequence(
    first=RunnableLambda(lambda x: {"question": x["question"]}),
    middle=[
        RunnableLambda(lambda x: {
            **x,
            "level": question_classifier.invoke({"question": x["question"]})
        }),
        RunnableLambda(lambda x: {
            **x,
            "response": response_strategy.invoke(x),
            "example": example_generator.invoke(x)
        })
    ],
    last=RunnableLambda(format_response)
)
```

Let's try out our assistant:

```python
async def run_assistant():
    # Example questions for different levels
    questions = [
        "What is a variable in Python?",
        "How does dependency injection work?",
        "Explain quantum computing qubits"
    ]
    
    for question in questions:
        result = await learning_assistant.ainvoke({"question": question})
        print(f"\nQuestion: {result['question']}")
        print(f"Difficulty Level: {result['level']}")
        print(f"\nExplanation: {result['explanation']}")
        print(f"\nExample: {result['example']}")
        print("\n" + "="*50)

# For Jupyter environments
import nest_asyncio
nest_asyncio.apply()

# Run the assistant
if __name__ == "__main__":
    asyncio.run(run_assistant())
```

<pre class="custom">
    Question: What is a variable in Python?
    Difficulty Level: beginner
    
    Explanation: In Python, a variable is like a container that holds information. Just like a box can hold toys, a variable can hold different types of data like numbers, text, or lists. You can give a variable a name, like "age" or "name", and then store information in it to use later in your program.Variables are used to store and manipulate data in a program.
    
    Example: A variable in Python is a placeholder for storing data values. It can be assigned a value which can be changed or accessed throughout the program.
    
    Example:
    ```python
    # Assigning a value to a variable
    x = 5
    
    # Accessing the value of the variable
    print(x)  # Output: 5
    
    # Changing the value of the variable
    x = 10
    
    # Accessing the updated value of the variable
    print(x)  # Output: 10
    ```
    
    ==================================================
    
    Question: How does dependency injection work?
    Difficulty Level: intermediate
    
    Explanation: Dependency injection is a design pattern commonly used in object-oriented programming to achieve loose coupling between classes. It is a technique where one object supplies the dependencies of another object. This helps in making the code more modular, maintainable, and testable.
    
    There are three main types of dependency injection: constructor injection, setter injection, and interface injection.
    
    1. Constructor Injection: In constructor injection, the dependencies are provided through the class constructor. This is the most common type of dependency injection. Here is an example in Java:
    
    ```java
    public class UserService {
        private UserRepository userRepository;
    
        public UserService(UserRepository userRepository) {
            this.userRepository = userRepository;
        }
    
        // Other methods of UserService that use userRepository
    }
    ```
    
    2. Setter Injection: In setter injection, the dependencies are provided through setter methods. Here is an example in Java:
    
    ```java
    public class UserService {
        private UserRepository userRepository;
    
        public void setUserRepository(UserRepository userRepository) {
            this.userRepository = userRepository;
        }
    
        // Other methods of UserService that use userRepository
    }
    ```
    
    3. Interface Injection: In interface injection, the dependent object implements an interface that defines the method(s) to inject the dependency. Here is an example in Java:
    
    ```java
    public interface UserRepositoryInjector {
        void injectUserRepository(UserRepository userRepository);
    }
    
    public class UserService implements UserRepositoryInjector {
        private UserRepository userRepository;
    
        @Override
        public void injectUserRepository(UserRepository userRepository) {
            this.userRepository = userRepository;
        }
    
        // Other methods of UserService that use userRepository
    }
    ```
    
    Dependency injection is commonly used in frameworks like Spring, where dependencies are managed by the framework and injected into the classes at runtime. This allows for easier configuration and management of dependencies.
    
    Overall, dependency injection helps in promoting code reusability, testability, and maintainability by decoupling the classes and their dependencies. It also makes it easier to switch out dependencies or mock them for testing purposes.
    
    Example: Dependency injection is a design pattern in which the dependencies of a class are provided externally. This helps in making the code more modular, testable and maintainable.
    
    Here is a practical example of how dependency injection works in Java:
    
    ```java
    // Interface for the dependency
    interface Logger {
        void log(String message);
    }
    
    // Class that depends on the Logger interface
    class UserService {
        private Logger logger;
    
        // Constructor injection
        public UserService(Logger logger) {
            this.logger = logger;
        }
    
        public void doSomething() {
            logger.log("Doing something...");
        }
    }
    
    // Implementation of the Logger interface
    class ConsoleLogger implements Logger {
        @Override
        public void log(String message) {
            System.out.println(message);
        }
    }
    
    public class Main {
        public static void main(String[] args) {
            // Creating an instance of the Logger implementation
            Logger logger = new ConsoleLogger();
    
            // Passing the Logger implementation to the UserService class through constructor injection
            UserService userService = new UserService(logger);
    
            // Calling a method on the UserService class
            userService.doSomething();
        }
    }
    ```
    
    In this example, the `UserService` class depends on the `Logger` interface. Instead of creating an instance of the `Logger` implementation (`ConsoleLogger`) inside the `UserService` class, we provide the `Logger` implementation externally through constructor injection. This allows us to easily swap out different implementations of the `Logger` interface without modifying the `UserService` class.
    
    ==================================================
    
    Question: Explain quantum computing qubits
    Difficulty Level: intermediate
    
    Explanation: Quantum computing qubits are the fundamental building blocks of quantum computers. Unlike classical computers, which use bits to represent information as either a 0 or a 1, quantum computers use qubits to represent information as a combination of 0 and 1 simultaneously. This property, known as superposition, allows quantum computers to perform complex calculations much faster than classical computers.
    
    One of the key concepts in quantum computing is entanglement, which allows qubits to be correlated with each other in such a way that the state of one qubit can instantly affect the state of another qubit, regardless of the distance between them. This property enables quantum computers to perform parallel computations and solve certain problems exponentially faster than classical computers.
    
    There are several types of qubits that can be used in quantum computing, including superconducting qubits, trapped ions, and topological qubits. Each type of qubit has its own advantages and disadvantages, and researchers are actively working to develop new qubit technologies that can overcome existing limitations and improve the performance of quantum computers.
    
    One practical example of quantum computing qubits is in the field of cryptography. Quantum computers have the potential to break many of the encryption algorithms that are currently used to secure sensitive information, such as credit card numbers and government communications. By leveraging the power of qubits and quantum algorithms, researchers are developing new encryption techniques that are resistant to attacks from quantum computers.
    
    Another use case for quantum computing qubits is in the field of drug discovery. Quantum computers have the ability to simulate the behavior of molecules at the quantum level, which can help researchers design new drugs more efficiently and accurately. By using qubits to model the interactions between atoms and molecules, scientists can identify potential drug candidates and optimize their properties before conducting costly and time-consuming experiments in the lab.
    
    In conclusion, quantum computing qubits are a revolutionary technology that has the potential to transform many industries and solve complex problems that are currently beyond the reach of classical computers. By harnessing the power of superposition and entanglement, quantum computers can perform calculations at speeds that were previously thought impossible, opening up new possibilities for innovation and discovery.
    
    Example: Practical example:
    
    Imagine you have a classical computer with a bit that can be in one of two states: 0 or 1. This bit can represent a single piece of information. Now, imagine you have a quantum computer with a qubit. A qubit can be in a superposition of both 0 and 1 states at the same time. This means that a qubit can represent multiple pieces of information simultaneously.
    
    For example, if you have 3 qubits, they can be in a superposition of 8 different states (2^3 = 8). This allows quantum computers to perform complex calculations much faster than classical computers.
    
    ```markdown
    // Example code in Qiskit for creating a quantum circuit with qubits
    
    from qiskit import QuantumCircuit
    
    # Create a quantum circuit with 3 qubits
    qc = QuantumCircuit(3)
    
    # Apply operations to the qubits
    qc.h(0)  # Apply a Hadamard gate to qubit 0
    qc.cx(0, 1)  # Apply a CNOT gate between qubit 0 and qubit 1
    qc.measure_all()  # Measure all qubits in the circuit
    
    print(qc)
    ```
    
    ==================================================
</pre>

## Comparison of RunnableSequence, RunnableBranch, and RunnableLambda

| Criteria | RunnableSequence | RunnableBranch | RunnableLambda |
|----------|------------------|----------------|----------------|
| Primary Purpose | Sequential pipeline processing | Conditional routing and branching | Simple transformations and functions |
| Condition Definition | No conditions, sequential flow | Each condition defined as `(condition, runnable)` pair | All conditions within single function (`route`) |
| Structure | Linear chain of operations | Tree-like branching structure | Function-based transformation |
| Readability | Very clear for sequential processes | Becomes clearer as conditions increase | Very clear for simple logic |
| Maintainability | Easy to maintain step-by-step flow | Clear separation between conditions and runnables | Can become complex if function grows large |
| Flexibility | Flexible for linear processes | Must follow `(condition, runnable)` pattern | Allows flexible condition writing |
| Scalability | Add or modify pipeline steps | Requires adding new conditions and runnables | Expandable by modifying function |
| Error Handling | Pipeline-level error management | Branch-specific error handling | Basic error handling |
| State Management | Maintains state throughout pipeline | State managed per branch | Typically stateless |
| Recommended Use Case | When you need ordered processing steps | When there are many conditions or maintainability is priority | When conditions are simple or function-based |
| Complexity Level | Medium to High | Medium | Low |
| Async Support | Full async support | Limited async support | Basic async support |

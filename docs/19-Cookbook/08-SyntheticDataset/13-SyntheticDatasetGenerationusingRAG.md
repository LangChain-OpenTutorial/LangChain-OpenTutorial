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

# Synthetic Dataset Generation using RAG

- Author: [Ash-hun](https://github.com/ash-hun)
- Design: 
- Peer Review: [syshin0116](https://github.com/syshin0116), [Kane](https://github.com/HarryKane11)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers an example of generating a synthetic dataset using RAG. Typically, it is used to create evaluation datasets for Domain Specific RAG pipelines or to generate synthetic data for model training. This tutorial will focus on the following two features. While the structure is the same, their intended use and purpose differ.

**Features**

- Domain Specific RAG Evaluation Dataset : Generates a domain specific synthetic dataset (Context, Question, Answer) for evaluating the RAG pipeline.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Domain Specific RAG Evaluation Dataset](#domain-specific-rag-evaluation-dataset)


### References

- [autoRAG github](https://github.com/Marker-Inc-Korea/AutoRAG?tab=readme-ov-file#3-qa-creation)
- [ragas github : singlehop question](https://github.com/explodinggradients/ragas/blob/main/src/ragas/testset/synthesizers/single_hop/prompts.py)
- [huggingface : RAG Evaluation Dataset Prompt](https://huggingface.co/datasets/Ash-Hun/Create_RAG_Evalauation_Data)
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
        "langchain",
        "langchain_core",
        "langchain_openai",
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
        "LANGCHAIN_PROJECT": "Synthetic Dataset Generation using RAG",
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



## Domain Specific RAG Evaluation Dataset

Generates a synthetic dataset (```Context```, ```Question```, ```Answer```) for evaluating the Domain Specific RAG pipeline.

- ```Context```: A context randomly selected from documents in a specific domain is used as the ground truth.
- ```Question```: A question that can be answered using the ```Context```.
- ```Answer```: An answer generated based on the ```Context``` and the ```Question```.

```python
# Import Library
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

### Question Generating Prompt

A prompt for generating questions from a given ```context``` using the RAG (Retriever Augmented Generation) technique is structured as follows.  
It consists of four main sections—```Instruction```, ```Requirements```, ```Style```, and ```Example```—along with an Indicator section where actual variable values are mapped. Each section is explained below: 
- ```Instruction```: Provides overall guidance for the prompt, including the purpose of the task and an explanation of the structured prompt sections.
- ```Requirements```: Lists essential conditions that must be met when performing the task.
- ```Style```: Specifies the stylistic guidelines for the generated output.
- ```Example```: Includes actual execution examples.

```python
Q_GEN = """
[ Instruction ] : 
- Your mission is to generate detailed ONE QUESTION that can yield correct answers from the GIVEN CONTEXT.
- When creating a QUESTION, you should carefully consider the following items:
  - Requirements : Essential requirements that must be included
  - Style : The form and style of the generated question
  - Think : Elements and procedures you need to self-examine for the created question

<Requirements>
- The questions you generate must always maintain high quality.
- Please do not print and generate any other unnecessary words.
- The Questions are created from the given context, but it must be created with an appropriate balance between general content and domain-specific content.
- If the given context related figure, you must generate the question related figure data.
- Finally, verify that the generated question contains only ONE QUESTION itself without any unnecessary description or content.
</Requirements> 

<Style>
- The expressions you use should either be inferred from the given context or be directly used expressions.
- Any expressions involving formulas must always be enclosed within $ symbols.
- Text that is conceptually represented with subscripts or superscripts should be expressed as mathematical formulas.
- You should compose questions that are as natural as possible within the context.
</Style>


Now, It's your turn. You must generate long and detailed high-quality questions from the given context while following the mentioned <Requirements> and <Style>.
The examples below consist of positive samples and negative samples. Please refer to the given examples to generate your answer.
When you generated QUESTION, you should take a deep breath and think step-by-step and generate the most natural Korean question.

<Example>
  - Given Context : Advancements in smart manufacturing technology have significantly contributed to improving production efficiency and product quality. In particular, automation systems utilizing artificial intelligence (AI) and the Internet of Things (IoT) enable real-time data analysis to detect abnormalities in machinery and facilitate preventive maintenance. These technologies play a crucial role in minimizing production downtime and reducing costs. Additionally, by implementing digital twin technology, manufacturers can simulate product performance in a virtual environment, allowing them to anticipate and address potential issues before they arise in the production process.
  - Question : What role do artificial intelligence (AI) and the Internet of Things (IoT) play in smart manufacturing, and what benefits do they offer to businesses?
</Example>

- Given Context : {context}
- Question : 
"""
```

```python
# Step 01. Generation Question from Domain Specific Context

def question_generate(context:list, q_gen_prompt:str=Q_GEN) -> str:
    # Create an OpenAI object
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant"),
            ("user", q_gen_prompt),
        ]
    )
    
    # Combine the prompt, model, and JsonOutputParser into a chain
    chain = prompt | model
    
    # Prepare inputs for batch processing
    inputs = [{"context": ctx} for ctx in context]

    results = chain.batch(inputs)
    return [result.content for result in results]
```

```python
sample_context = [
    "Smart manufacturing systems utilizing robotics and artificial intelligence (AI) are being rapidly adopted in the automotive industry. Automated assembly lines enhance production speed while maintaining high precision, and AI-powered vision systems for quality inspection help detect defects in real time, reducing defect rates. Additionally, with the advancement of electric vehicles (EVs) and autonomous driving technology, the importance of battery manufacturing and sensor technology is growing, leading to the continuous development of optimized production processes.", 
    "In the pharmaceutical industry, quality and safety are considered the most critical factors, and smart manufacturing technologies are actively utilized to ensure them. Automated production lines enable precise formulation and packaging of medications, while artificial intelligence (AI) is used for real-time quality inspections to detect impurities. Additionally, data analytics help optimize manufacturing processes and ensure compliance with regulatory requirements. In particular, for biopharmaceuticals, precise temperature and humidity control is essential, making real-time monitoring with IoT sensors a crucial component."
]
sample_question = question_generate(context=sample_context)
sample_question
```




<pre class="custom">["How are robotics and artificial intelligence (AI) utilized in the automotive industry's smart manufacturing systems to improve production speed and quality, and what impact do advancements in electric vehicles (EVs) and autonomous driving technology have on battery manufacturing and sensor technology?",
     'How do smart manufacturing technologies, including automated production lines and artificial intelligence (AI), contribute to ensuring quality and safety in the pharmaceutical industry, and why is real-time monitoring with IoT sensors particularly important for biopharmaceuticals?']</pre>



### Question Evolving Prompt


A prompt that utilizes the RAG (Retriever Augmented Generation) technique to correct inaccurate information or generate more evolved questions based on a given ```context``` and a draft ```question``` is structured as follows.  
It consists of four main sections—```Instruction```, ```Evolving```, and ```Example``` along with an Indicator section where actual variable values are mapped. Each section is explained below:  

- ```Instruction```: Provides the overall guidance for the prompt.
- ```Evolving```: Contains step-by-step instructions to achieve the purpose of the prompt.
- ```Example```: Includes actual execution examples.

```python
Q_EVOLVE="""
[ Instruction ] : 
- Your mission is to review & check the context and the question to determine whether the answer to the question can be obtained from the given context, and then evolve the following question.
- If the question can be fully answered using information from the given context, please return the same question.
- If the question cannot be answered using information from the given context, please modify the question.
- This process is called <Evolving> and should be carried out according to the procedure below, ultimately returning either the original or the EVOLVING QUESTION.

<Evolving>
1. Understand the given context and question.
2. Determine whether all the information required to answer the question is present in the context.
  2-1. If you conclude that all necessary information is available in the context:
    2-1-1. Treat the given question as the EVOLVING QUESTION.
  2-2. If you conclude that not all necessary information is present in the context:
    2-2-1. Modify the form or content of the question so that it can be answered using only the information provided in the context.
    2-2-2. Treat the modified question as the EVOLVING QUESTION.
3. Return the evolving question.
</Evolving> 

You can refer to the following examples when performing the <Evolving> process, however you must never explain the <Evolving> process. Only provide the EVOLVING QUESTION.
Take sufficient time to think logically through each step as you proceed. 
If you create a high-quality EVOLVING QUESTION, you may receive a small tip. 
Now it's your turn. Take a deep breath and start!

<Example>
  - Given Context : Advancements in smart manufacturing technology have significantly contributed to improving production efficiency and product quality. In particular, automation systems utilizing artificial intelligence (AI) and the Internet of Things (IoT) enable real-time data analysis to detect abnormalities in machinery and facilitate preventive maintenance. These technologies play a crucial role in minimizing production downtime and reducing costs. Additionally, by implementing digital twin technology, manufacturers can simulate product performance in a virtual environment, allowing them to anticipate and address potential issues before they arise in the production process.
  - Question : What role do artificial intelligence (AI) and the Internet of Things (IoT) play in smart manufacturing, and what benefits do they offer to businesses?
  - EVOLVING QUESTION : What role do artificial intelligence (AI) and the Internet of Things (IoT) play in smart manufacturing, and how do they contribute to improving production efficiency and reducing costs?
</Example>

- Given Context : {context}
- Question :{question}
- EVOLVING QUESTION :
"""
```

```python
# Step 02. Evolving Question from Domain Specific Context & Question
def question_evolving(context:str, question:str, evolving_prompt:str=Q_EVOLVE) -> str:
    # Create an OpenAI object
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant"),
            ("user", evolving_prompt),
        ]
    )
    
    # Combine the prompt, model, and JsonOutputParser into a chain
    chain = prompt | model

    # Prepare inputs for batch processing
    inputs = [{"context": ctx, "question": q} for ctx, q in zip(context, question)]

    results = chain.batch(inputs)
    return [result.content for result in results]
```

```python
evolved_question = question_evolving(context=sample_context, question=sample_question)
evolved_question
```




<pre class="custom">["How are robotics and artificial intelligence (AI) utilized in the automotive industry's smart manufacturing systems to improve production speed and quality?",
     'How do smart manufacturing technologies, including automated production lines and artificial intelligence (AI), contribute to ensuring quality and safety in the pharmaceutical industry, and why is real-time monitoring with IoT sensors particularly important for biopharmaceuticals?']</pre>



### Answer Generating Prompt

A prompt that uses the RAG (Retriever Augmented Generation) technique to generate the final answer based on a given ```context``` and ```question``` is structured as follows. It consists of two main sections—```Instruction``` and ```Example```—along with an Indicator section where actual variable values are mapped. Each section is explained below:  

- ```Instruction```: Provides the overall guidance for the prompt.
- ```Example```: Includes actual execution examples.

```python
A_GEN = """
[ Instruction ] : 
- Your mission is to produce an accurate answer by reviewing the provided CONTEXT and QUESTION.
- When creating your ANSWER, you must use the information from the CONTEXT and strive to make your response as detailed and high-quality as possible.
- Please refer to the <Example> below to create your ANSWER.

<Example>
    - Given Context : Advancements in smart manufacturing technology have significantly contributed to improving production efficiency and product quality. In particular, automation systems utilizing artificial intelligence (AI) and the Internet of Things (IoT) enable real-time data analysis to detect abnormalities in machinery and facilitate preventive maintenance. These technologies play a crucial role in minimizing production downtime and reducing costs. Additionally, by implementing digital twin technology, manufacturers can simulate product performance in a virtual environment, allowing them to anticipate and address potential issues before they arise in the production process.
    - QUESTION : What role do artificial intelligence (AI) and the Internet of Things (IoT) play in smart manufacturing, and how do they contribute to improving production efficiency and reducing costs?
    - Answer : Artificial intelligence (AI) and the Internet of Things (IoT) play a key role in smart manufacturing by enabling real-time data analysis to detect abnormalities in machinery. These technologies help facilitate preventive maintenance, which minimizes production downtime and ensures that equipment operates efficiently. By continuously monitoring production processes, AI and IoT systems identify potential issues before they lead to failures, thus reducing the likelihood of costly repairs and unplanned stoppages. Additionally, this proactive approach helps reduce operational costs by optimizing the use of resources and improving overall production efficiency, ultimately leading to better product quality and cost savings for businesses.
</Example>

Take sufficient time to think logically through each step as you proceed. 
If you create a detail high-quality ANSWER, you may receive a small tip. 
Now it's your turn. Take a deep breath and start!

- Given Context : {context}
- Question :{question}
- Answer :

"""
```

```python
# Step 03. Generate Answer from Domain Specific Context & Evolved Question
def answer_generate(context:str, evolved_question:str, a_gen_prompt:str=A_GEN) -> str:
    # Create an OpenAI object
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant"),
            ("user", a_gen_prompt),
        ]
    )
    # Combine the prompt, model, and JsonOutputParser into a chain
    chain = prompt | model

    # Prepare inputs for batch processing
    inputs = [{"context": ctx, "question": q} for ctx, q in zip(context, evolved_question)]

    results = chain.batch(inputs)
    return [result.content for result in results]
```

```python
answer = answer_generate(context=sample_context, evolved_question=evolved_question)
answer
```




<pre class="custom">["Robotics and artificial intelligence (AI) are integral components of smart manufacturing systems in the automotive industry, significantly enhancing both production speed and quality. Automated assembly lines, powered by robotics, streamline the manufacturing process by performing repetitive tasks with high precision and speed, which accelerates production rates while maintaining consistency. This automation reduces the likelihood of human error, ensuring that each component is assembled accurately.\n\nAI further augments these systems through AI-powered vision systems used for quality inspection. These systems are capable of detecting defects in real time, allowing for immediate corrective actions. By identifying and addressing defects as they occur, AI helps reduce defect rates, ensuring that only high-quality products reach the market.\n\nMoreover, as the automotive industry shifts towards electric vehicles (EVs) and autonomous driving technology, the role of AI and robotics becomes even more critical. The production of EV batteries and advanced sensors requires highly optimized and precise manufacturing processes, which are facilitated by these technologies. Robotics and AI contribute to the continuous development and refinement of these processes, supporting the industry's evolution towards more advanced and efficient vehicle technologies.",
     'Smart manufacturing technologies play a pivotal role in ensuring quality and safety in the pharmaceutical industry by leveraging automated production lines and artificial intelligence (AI). Automated production lines are crucial for the precise formulation and packaging of medications, which is essential to maintain the integrity and efficacy of pharmaceutical products. These automated systems reduce human error and ensure consistency in the production process, which is vital for meeting stringent quality standards.\n\nArtificial intelligence (AI) further enhances quality assurance by conducting real-time quality inspections. AI systems are capable of detecting impurities and anomalies in the production process, allowing for immediate corrective actions. This real-time inspection capability ensures that only products meeting the highest quality standards reach the market, thereby safeguarding consumer safety and maintaining regulatory compliance.\n\nIn the realm of biopharmaceuticals, real-time monitoring with IoT sensors is particularly important due to the sensitive nature of these products. Biopharmaceuticals often require precise environmental conditions, such as specific temperature and humidity levels, to maintain their stability and effectiveness. IoT sensors provide continuous monitoring of these critical parameters, ensuring that any deviations are quickly identified and addressed. This real-time data collection and analysis help prevent potential quality issues that could arise from environmental fluctuations, thereby ensuring the safety and efficacy of biopharmaceutical products. Overall, the integration of smart manufacturing technologies, including AI and IoT, is essential for maintaining the high standards of quality and safety required in the pharmaceutical industry.']</pre>



### SyntheticGenerator

Based on the above tutorial, I have written it as a single class called `SyntheticGenerator()`.  
The overall flow is the same as the tutorial, and it can be used by executing the `run()` method. By specifying the desired path in save_path, the generated data will be saved as a CSV file, with each (Context, Query, Answer) triple-pair stored in a single row.

```python
class SyntheticGenerator():
    def __init__(self, data:list|None=None, Gen_prompt_question:str=Q_GEN, Gen_prompt_evolving:str=Q_EVOLVE, Gen_prompt_answer:str=A_GEN)->None:
        self.data = data
        self.q_prompt = Gen_prompt_question
        self.e_prompt = Gen_prompt_evolving
        self.a_prompt = Gen_prompt_answer

    def _generate_question(self, data:list[str]) -> list:
        # Create an OpenAI object
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.q_prompt),
            ]
        )
        
        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model
        
        # Prepare inputs for batch processing
        inputs = [{"context": ctx} for ctx in data]

        results = chain.batch(inputs)
        return [[ctx, result.content] for ctx, result in zip(data, results)]

    def _evolving_question(self, data:list[list[str]]) -> list[list[str]]:
        # Create an OpenAI object
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.e_prompt),
            ]
        )
        
        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model

        # Prepare inputs for batch processing
        inputs = [{"context": each[0], "question": each[1]} for each in data]

        results = chain.batch(inputs)
        return [[each[0], result.content] for each, result in zip(data, results)]

    def _generate_answer(self, data:list[list[str]]) -> list[list[str]]:
        # Create an OpenAI object
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.a_prompt),
            ]
        )
        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model

        # Prepare inputs for batch processing
        inputs = [{"context": ctx, "question": q} for ctx, q in data]

        results = chain.batch(inputs)
        return [[each[0], each[1], result.content] for each, result in zip(data, results)]

    def run(self, data:list|None=None, save_path:str|None=None) -> list:
        raw_data = data if self.data is None else self.data
        print(raw_data)
        if raw_data is None:
            raise ValueError("Empty Data")
        else:
            ctx_data_q = self._generate_question(data=raw_data)
            ctx_data_eq = self._evolving_question(data=ctx_data_q)
            ctx_data_eq_a = self._generate_answer(data=ctx_data_eq)
        
        if save_path is None:
            return ctx_data_eq_a
        else:
            import pandas as pd
            pd.DataFrame(ctx_data_eq_a, columns=['context', 'question', 'answer']).to_csv(save_path, index=False)
            return ctx_data_eq_a

```

```python
sample_context = [
    "Smart manufacturing systems utilizing robotics and artificial intelligence (AI) are being rapidly adopted in the automotive industry. Automated assembly lines enhance production speed while maintaining high precision, and AI-powered vision systems for quality inspection help detect defects in real time, reducing defect rates. Additionally, with the advancement of electric vehicles (EVs) and autonomous driving technology, the importance of battery manufacturing and sensor technology is growing, leading to the continuous development of optimized production processes.", 
    "In the pharmaceutical industry, quality and safety are considered the most critical factors, and smart manufacturing technologies are actively utilized to ensure them. Automated production lines enable precise formulation and packaging of medications, while artificial intelligence (AI) is used for real-time quality inspections to detect impurities. Additionally, data analytics help optimize manufacturing processes and ensure compliance with regulatory requirements. In particular, for biopharmaceuticals, precise temperature and humidity control is essential, making real-time monitoring with IoT sensors a crucial component."
]

generator = SyntheticGenerator(data=sample_context)
synthetic_data = generator.run(save_path='./sample_data.csv')
synthetic_data
```

<pre class="custom">['Smart manufacturing systems utilizing robotics and artificial intelligence (AI) are being rapidly adopted in the automotive industry. Automated assembly lines enhance production speed while maintaining high precision, and AI-powered vision systems for quality inspection help detect defects in real time, reducing defect rates. Additionally, with the advancement of electric vehicles (EVs) and autonomous driving technology, the importance of battery manufacturing and sensor technology is growing, leading to the continuous development of optimized production processes.', 'In the pharmaceutical industry, quality and safety are considered the most critical factors, and smart manufacturing technologies are actively utilized to ensure them. Automated production lines enable precise formulation and packaging of medications, while artificial intelligence (AI) is used for real-time quality inspections to detect impurities. Additionally, data analytics help optimize manufacturing processes and ensure compliance with regulatory requirements. In particular, for biopharmaceuticals, precise temperature and humidity control is essential, making real-time monitoring with IoT sensors a crucial component.']
</pre>




    [['Smart manufacturing systems utilizing robotics and artificial intelligence (AI) are being rapidly adopted in the automotive industry. Automated assembly lines enhance production speed while maintaining high precision, and AI-powered vision systems for quality inspection help detect defects in real time, reducing defect rates. Additionally, with the advancement of electric vehicles (EVs) and autonomous driving technology, the importance of battery manufacturing and sensor technology is growing, leading to the continuous development of optimized production processes.',
      'How are robotics and artificial intelligence (AI) integrated into smart manufacturing systems in the automotive industry, and what impact do they have on production speed, precision, and defect rates?',
      'Robotics and artificial intelligence (AI) are integral components of smart manufacturing systems in the automotive industry, significantly impacting production speed, precision, and defect rates. Automated assembly lines, powered by robotics, enhance production speed by streamlining the manufacturing process, allowing for faster assembly of automotive components. These robotic systems are designed to perform repetitive tasks with high precision, ensuring that each component is assembled accurately and consistently, which is crucial in maintaining the quality standards of automotive manufacturing.\n\nAI-powered vision systems play a critical role in quality inspection by detecting defects in real time. These systems use advanced algorithms to analyze visual data from cameras and sensors, identifying any anomalies or defects in the components being produced. This real-time defect detection capability helps reduce defect rates by allowing manufacturers to address issues immediately, preventing defective products from progressing further down the production line.\n\nFurthermore, the growing focus on electric vehicles (EVs) and autonomous driving technology has increased the importance of battery manufacturing and sensor technology. This has led to the continuous development of optimized production processes, where robotics and AI are used to enhance the efficiency and quality of these critical components. Overall, the integration of robotics and AI in smart manufacturing systems not only boosts production speed and precision but also significantly reduces defect rates, contributing to more efficient and reliable automotive manufacturing.'],
     ['In the pharmaceutical industry, quality and safety are considered the most critical factors, and smart manufacturing technologies are actively utilized to ensure them. Automated production lines enable precise formulation and packaging of medications, while artificial intelligence (AI) is used for real-time quality inspections to detect impurities. Additionally, data analytics help optimize manufacturing processes and ensure compliance with regulatory requirements. In particular, for biopharmaceuticals, precise temperature and humidity control is essential, making real-time monitoring with IoT sensors a crucial component.',
      'How are smart manufacturing technologies utilized to ensure quality and safety in the pharmaceutical industry, and why is real-time monitoring particularly important in the production of biopharmaceuticals?',
      'Smart manufacturing technologies are utilized in the pharmaceutical industry to ensure quality and safety by integrating automation, artificial intelligence (AI), and data analytics into the production processes. Automated production lines are employed to achieve precise formulation and packaging of medications, which is crucial for maintaining the consistency and efficacy of pharmaceutical products. AI plays a significant role in conducting real-time quality inspections, allowing for the detection of impurities and ensuring that only products meeting stringent quality standards are released to the market. This real-time inspection capability helps prevent defective products from reaching consumers, thereby safeguarding public health.\n\nData analytics further enhance the manufacturing process by optimizing operations and ensuring compliance with regulatory requirements. By analyzing production data, manufacturers can identify areas for improvement, streamline processes, and maintain adherence to industry standards and regulations, which are critical for ensuring the safety and efficacy of pharmaceutical products.\n\nIn the production of biopharmaceuticals, real-time monitoring is particularly important due to the sensitive nature of these products. Biopharmaceuticals often require precise control of environmental conditions, such as temperature and humidity, to maintain their stability and effectiveness. IoT sensors are employed to provide continuous, real-time monitoring of these conditions, ensuring that any deviations are quickly detected and corrected. This level of control is essential to prevent degradation of the biopharmaceuticals, which could compromise their safety and therapeutic value. Thus, real-time monitoring with IoT technology is a crucial component in maintaining the quality and safety of biopharmaceutical products.']]



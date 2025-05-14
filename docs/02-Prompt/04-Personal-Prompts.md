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

# Personal Prompts for LangChain

- Author: [Eun](https://github.com/yuneun92)
- Peer Review: [jeong-wooseok](https://github.com/jeong-wooseok), [r14minji](https://github.com/r14minji)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/04-Personal-Prompts.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/04-Personal-Prompts.ipynb)

## Overview

This cookbook contains a comprehensive collection of specialized prompts designed for various professional domains using LangChain. The prompts are crafted to leverage the power of large language models while maintaining domain expertise and professional standards.

> The primary goals of this project are to:
- Provide standardized, high-quality prompts for different professional domains
- Enable consistent and reliable outputs from language models
- Facilitate domain-specific knowledge extraction and analysis
- Support automated report generation and content creation
- Maintain professional standards across various fields

### Table of Contents

- [Overview](##overview)
- [Prompt Generating Tips](##prompt-generating-tips)
- [Basic Prompts](##basic-prompts)
- [Advanced Prompts](##advanced-prompts)
- [Specialized Prompts](##specialized-prompts)
- [Professional Domain Prompts](##professional-domain-prompts)

### References

- [Prompt Engineering Guide: Gemini](https://www.promptingguide.ai/models/gemini)
- [Google: Prompting Guide 101](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)
- [Anthropic: Prompt Engineering - Use XML tags to structure your prompts
](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)
- [Anthropic: Prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Anthropic: Anthropic's Prompt Engineering Interactive Tutorial](https://docs.google.com/spreadsheets/d/19jzLgRruG9kjUQNKtCg1ZjdD6l6weA6qRXG5zLIAhC8/edit?gid=1733615301#gid=1733615301)
- [Github: prompt-eng-interactive-tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)
- [The Decoder: Chat GPT Guide](https://the-decoder.com/chatgpt-guide-prompt-strategies/)
- [Dorik: How to Write Prompts for ChatGPT (with Examples)](https://dorik.com/blog/how-to-write-prompts-for-chatgpt)
- [Coursera: How To Write ChatGPT Prompts: Your 2025 Guide](https://www.coursera.org/articles/how-to-write-chatgpt-prompts)
- [LangSmith: Prompt Hub](https://docs.smith.langchain.com/old/hub/dev-setup)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install dotenv langchain-opentutorial langchain langchainhub
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchainhub"
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
        # Get an API key for your Personal organization if you have not yet. The hub will not work with your non-personal organization's api key!
        # If you already have LANGCHAIN_API_KEY set to a personal organization’s api key from LangSmith, you can skip this.
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Personal Prompts for LangChain",
    }
)
```

```python
# import os

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "Personal Prompts for LangChain"
```

## Prompt Generating Tips


### **Model Comparison at a Glance:**

| Feature                | **ChatGPT**                                      | **Claude**                                      | **Gemini**                                      |
|------------------------|--------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **Strengths**           | Conversational, logical reasoning               | Handles structured formats, logical responses  | Works well with detailed tasks and examples    |
| **Best Practice**       | Clear, focused prompts                          | XML-style structured prompts                   | Detailed instructions and examples             |
| **Example Use Case**    | Writing emails, casual conversations            | Analytical tasks, structured outputs           | Summaries, detailed reports, multimodal tasks  |

By following these tailored tips, you can maximize the strengths of each model and achieve optimal performance in your LangChain projects.


### **1. ChatGPT (OpenAI's GPT-4)**  
ChatGPT is a powerful language model known for its conversational ability and logical reasoning.

> **Prompt Tips:**
- **Keep it Clear and Focused:**  Clearly define what you want the model to do. Don’t overload it with too much background information.
- **Ask for a Specific Format:**  If you need the response in bullet points, tables, or paragraphs, mention it.
- **Assign a Role:**  Tell ChatGPT who it is (e.g., "You are a project manager") to get more tailored answers.


```python
# Example Prompt for GPT4
"You are a professional email writer. Write a polite email to a client informing them of a project delay of one month due to supply chain issues. The tone should be apologetic but confident."
```




<pre class="custom">'You are a professional email writer. Write a polite email to a client informing them of a project delay of one month due to supply chain issues. The tone should be apologetic but confident.'</pre>





### **2. Claude (Anthropic's Model)**  
Claude excels in structured thinking and understanding detailed tasks. It often works well with **XML-style formatting** for prompts.

> **Prompt Tips:**
- **Use Structured Formats:**  Use XML tags to organize the instructions, which helps Claude interpret them better.
- **Provide Context and Examples:**  Add a clear task and examples to guide the model's response.


```python
# Example Prompt for Claude
"""
<context>
  <project>
    <name>Website Redesign</name>
    <deadline>March 15, 2025</deadline>
  </project>
</context>
<instructions>
  Write an email to the client explaining the project will be delayed by one month due to supply chain issues. Apologize and propose a new deadline.
</instructions>
<example>
  Dear [Client Name],

  Due to supply chain challenges, we regret to inform you that the project will be delayed. The new expected completion date is April 15, 2025. We apologize for the inconvenience and appreciate your understanding.

  Best regards,
  [Your Name]
</example>
"""
```




<pre class="custom">'\n<context>\n  <project>\n    <name>Website Redesign</name>\n    <deadline>March 15, 2025</deadline>\n  </project>\n</context>\n<instructions>\n  Write an email to the client explaining the project will be delayed by one month due to supply chain issues. Apologize and propose a new deadline.\n</instructions>\n<example>\n  Dear [Client Name],\n\n  Due to supply chain challenges, we regret to inform you that the project will be delayed. The new expected completion date is April 15, 2025. We apologize for the inconvenience and appreciate your understanding.\n\n  Best regards,  \n  [Your Name]\n</example>\n'</pre>





### **3. Gemini (Google’s AI Model)**  
Gemini is a cutting-edge multimodal AI designed to work across text, images, and other data types. It handles detailed and structured tasks effectively.

> **Prompt Tips:**
- **Be Detailed and Specific:**  Clearly explain the task and provide any necessary background details.
- **Break Complex Tasks into Steps:**  If the task is complicated, split it into smaller, sequential steps.
- **Add Examples:**  Providing examples helps Gemini align its output with your expectations.



```python
# Exmple Prompt for Gemini
"You are a marketing strategist. Write a 200-word summary of the key milestones achieved in a project, emphasizing the team’s performance and results. Use a professional tone."
```




<pre class="custom">'You are a marketing strategist. Write a 200-word summary of the key milestones achieved in a project, emphasizing the team’s performance and results. Use a professional tone.'</pre>



---
## Basic Prompts

The Basic Prompts chapter covers summarization tasks that are most commonly used across all domains. These prompts can be used individually or combined in a pipeline:


1. **Sequential Processing**
   ```python
   documents → Summary Prompt → Map Prompt → Reduce Prompt → Final Output
   ```

2. **Parallel Processing**
   ```python
   documents → Multiple Summary Prompts (parallel)
            → Map Prompts (parallel)
            → Single Reduce Prompt
            → Final Output
   ```

3. **Hybrid Processing**
   ```python
   documents → Summary Prompt
            → Map Prompt (for themes)
            → Reduce Prompt (for final synthesis)
            → Additional Summary Prompt (for final polish)
   ```



### 1. Summary Prompt

The Summary Prompt is designed to create concise, informative summaries of documents while maintaining key information and context.



```python
PROMPT_OWNER = "eun"
```

```python
from langchain import hub
from langchain.prompts import PromptTemplate

# Let's upload the prompt to the LangChain Hub.
# Don't forget to enter the LangSmith API as an environment variable.
prompt_title = "summarize_document"

summarize_prompt = """
Please summarize the sentence according to the following REQUEST.
REQUEST:
1. Summarize the main points in bullet points.
2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
3. Use various emojis to make the summary more interesting.
4. DO NOT include any unnecessary information.

CONTEXT:
{context}

SUMMARY:"
"""
prompt = PromptTemplate.from_template(summarize_prompt)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template='\nPlease summarize the sentence according to the following REQUEST.\nREQUEST:\n1. Summarize the main points in bullet points.\n2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.\n3. Use various emojis to make the summary more interesting.\n4. DO NOT include any unnecessary information.\n\nCONTEXT:\n{context}\n\nSUMMARY:"\n')</pre>



```python
# To upload a prompt to Hub:
#
# Private Repository:
# - Simply pass the prompt title as the first argument
# hub.push(prompt_title, prompt, new_repo_is_public=False)
#
# Public Repository:
# - First create a Hub Handle at LangSmith (smith.langchain.com)
# - Include your handle in the prompt title path
# hub.push(f"{PROMPT_OWNER}/{prompt_title}", prompt, new_repo_is_public=True)

hub.push(f"{PROMPT_OWNER}/{prompt_title}", prompt, new_repo_is_public=True)
```




<pre class="custom">'https://smith.langchain.com/prompts/summarize_document/129da0ee?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



You can find the uploaded prompt in your LangSmith. Please go to the site address as output.


```python
# You can import and use prompts as follows.
prompt = hub.pull("eun/summarize_document:129da0ee")
prompt
```




<pre class="custom">PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, metadata={'lc_hub_owner': 'eun', 'lc_hub_repo': 'summarize_document', 'lc_hub_commit_hash': '129da0ee7cc02d076cd26692334f58a4aa898f5c40916847e8d808adb31f0263'}, template='\nPlease summarize the sentence according to the following REQUEST.\nREQUEST:\n1. Summarize the main points in bullet points.\n2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.\n3. Use various emojis to make the summary more interesting.\n4. DO NOT include any unnecessary information.\n\nCONTEXT:\n{context}\n\nSUMMARY:"\n')</pre>



### 2. Map Prompt

The Map Prompt is used to extract and organize main themes from documents, creating a structured representation of the content.



```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "map-prompt"

map_prompt = """
You are a helpful expert journalist in extracting the main themes from a GIVEN DOCUMENTS below.
Please provide a comprehensive summary of the GIVEN DOCUMENTS in numbered list format.
The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format.
Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.

GIVEN DOCUMENTS:
{docs}

FORMAT:
1. main theme 1
2. main theme 2
3. main theme 3
...

CAUTION:
- DO NOT list more than 5 main themes.

Helpful Answer:
"""
prompt = PromptTemplate.from_template(map_prompt)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['docs'], input_types={}, partial_variables={}, template='\nYou are a helpful expert journalist in extracting the main themes from a GIVEN DOCUMENTS below.\nPlease provide a comprehensive summary of the GIVEN DOCUMENTS in numbered list format.\nThe summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format.\nPlease ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.\nThe length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.\n\nGIVEN DOCUMENTS:\n{docs}\n\nFORMAT:\n1. main theme 1\n2. main theme 2\n3. main theme 3\n...\n\nCAUTION:\n- DO NOT list more than 5 main themes.\n\nHelpful Answer:\n')</pre>



```python
hub.push(prompt_title, prompt, new_repo_is_public=False)
```




<pre class="custom">'https://smith.langchain.com/prompts/map-prompt/1535fbd6?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



### 3. Reduce Prompt

The Reduce Prompt combines and synthesizes multiple summaries into a single, coherent output, particularly useful for processing large document sets.



```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "reduce-prompt"

reduce_prompt = """
You are a helpful expert in summary writing.
You are given numbered lists of summaries.
Extract top 10 most important insights and create a unified summary.

LIST OF SUMMARIES:
{doc_summaries}

REQUIREMENTS:
1. Identify key insights across summaries
2. Maintain coherence and flow
3. Eliminate redundancy
4. Preserve important details
5. Create a unified narrative

OUTPUT FORMAT:
1. Main insights (bullet points)
2. Synthesized summary
3. Key takeaways
"""
prompt = PromptTemplate.from_template(reduce_prompt)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['doc_summaries'], input_types={}, partial_variables={}, template='\nYou are a helpful expert in summary writing.\nYou are given numbered lists of summaries.\nExtract top 10 most important insights and create a unified summary.\n\nLIST OF SUMMARIES:\n{doc_summaries}\n\nREQUIREMENTS:\n1. Identify key insights across summaries\n2. Maintain coherence and flow\n3. Eliminate redundancy\n4. Preserve important details\n5. Create a unified narrative\n\nOUTPUT FORMAT:\n1. Main insights (bullet points)\n2. Synthesized summary\n3. Key takeaways\n')</pre>



```python
hub.push(prompt_title, prompt, new_repo_is_public=False)
```




<pre class="custom">'https://smith.langchain.com/prompts/reduce-prompt/17ed176f?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



---
## Advanced Prompts

The Advanced Prompts chapter explores sophisticated techniques that enhance the quality and specificity of language model outputs. These prompts are designed to handle complex tasks requiring deeper analysis and more nuanced responses.

### 1. Chain of Density Summarization

Chain of Density Summarization iteratively refines summaries to achieve higher information density while maintaining readability and key insights.


```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "chain-of-density"

chain_density_prompt = """
Given the input text, generate increasingly dense summaries through the following steps:

INPUT PARAMETERS:
- Text: {text}
- Iteration Count: {iterations}
- Target Length: {length}

PROCESS:
1. Initial Summary
2. Entity Identification
3. Density Enhancement
4. Quality Check

OUTPUT REQUIREMENTS:
1. Maintain consistent length
2. Increase information density
3. Preserve key entities
4. Ensure readability

Please provide the summary following this structure:

FORMAT:
{
    "initial_summary": str,
    "entity_map": list,
    "refined_summaries": list,
    "final_summary": str
}
"""

prompt = PromptTemplate.from_template(chain_density_prompt)
```

```python
hub.push(prompt_title, prompt, new_repo_is_public=False)
```




<pre class="custom">'https://smith.langchain.com/prompts/chain-of-density/a9eac13f?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



### 1.1. Chain of Density (Multilingual)
Generate increasingly dense summaries in any specified language through iterative refinement while maintaining semantic accuracy.

```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate

prompt_title = "chain-of-density-multilingual"

chain_density_multilingual = """
Article: {ARTICLE}
Language: {LANGUAGE}

You will generate increasingly concise, entity-dense summaries of the above article in the specified language.

Repeat the following 2 steps 5 times.

Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

A missing entity is:
- relevant to the main story,
- specific yet concise (100 words or fewer),
- novel (not in the previous summary),
- faithful (present in the article),
- anywhere (can be located anywhere in the article).

Guidelines:
- The first summary should be long (8-10 sentences, ~200 words) yet highly non-specific
- Make every word count: rewrite the previous summary to improve flow
- Make space with fusion, compression, and removal of uninformative phrases
- The summaries should become highly dense and concise yet self-contained
- Missing entities can appear anywhere in the new summary
- Never drop entities from the previous summary

OUTPUT FORMAT:
[
    {
        "Missing_Entities": str,
        "Denser_Summary": str
    }
]

Provide the output in the specified language: {LANGUAGE}
"""

prompt = ChatPromptTemplate.from_template(chain_density_multilingual)

# Usage Example:
response = chain_density_multilingual.format(
    ARTICLE="Your article text here", LANGUAGE="Spanish"
)
```

### 1.2. Chain of Density Map (Multilingual)

 Create mapped summaries with increasing density in any specified language, focusing on key entity extraction and relationship mapping.


```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate

prompt_title = "chain-of-density-map-multilingual"

chain_density_map_multilingual = """
Article: {ARTICLE}
Language: {LANGUAGE}

You will generate increasingly concise, entity-dense summaries of the above article in the specified language.

Repeat the following 2 steps 3 times.

Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previous summary.
Step 2. Write a new, denser summary of identical length covering all previous entities plus new ones.

A missing entity is:
- relevant to the main story,
- specific yet concise (100 words or fewer),
- novel (not in the previous summary),
- faithful (present in the article),
- anywhere (can be located anywhere in the article).

Guidelines:
- First summary: 8-10 sentences (~200 words), non-specific with fillers
- Optimize word usage and improve flow
- Remove uninformative phrases
- Maintain density and self-containment
- Preserve all previous entities

OUTPUT FORMAT:
Text format for "Missing Entities" and "Denser_Summary"

Provide the output in the specified language: {LANGUAGE}
"""

prompt = ChatPromptTemplate.from_template(chain_density_map_multilingual)

# Usage Example:
response_map = chain_density_map_multilingual.format(
    ARTICLE="Your article text here", LANGUAGE="Japanese"  # or any other language
)
```

### 2. Key Information Extraction

Extract and structure critical information from various document types with high precision and consistency.




```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "key-information-extraction"

extraction_prompt = """
Extract key information from the provided document according to these specifications:

INPUT:
- Document: {document}
- Target Fields: {fields}
- Context Requirements: {context}

EXTRACTION REQUIREMENTS:
1. Identify specified data points
2. Maintain contextual relationships
3. Validate extracted information
4. Format according to schema

OUTPUT FORMAT:
{
    "extracted_data": dict,
    "confidence_scores": dict,
    "validation_results": dict,
    "metadata": dict
}
"""

prompt = PromptTemplate.from_template(extraction_prompt)
hub.push(prompt_title, prompt, new_repo_is_public=False)
```


### 3. Metadata Tagging
Automatically generate relevant tags and metadata to enhance content organization and searchability.






```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "metadata-tagger"

metadata_prompt = """
Generate comprehensive metadata tags for the given content:

CONTENT PARAMETERS:
- Type: {content_type}
- Domain: {domain}
- Context: {context}

TAGGING REQUIREMENTS:
1. Generate relevant tags
2. Create hierarchical categories
3. Identify key topics
4. Map relationships
5. Optimize for search

OUTPUT FORMAT:
{
    "primary_tags": list,
    "categories": dict,
    "relationships": dict,
    "search_terms": list
}
"""

prompt = PromptTemplate.from_template(metadata_prompt)
hub.push(prompt_title, prompt, new_repo_is_public=False)
```




<pre class="custom">'https://smith.langchain.com/prompts/metadata-tagger/9bf50dec?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



---
## Specialized Prompts

### 1. RAG Prompts

### 1.1. RAG Document Analysis

Process and answer questions based on retrieved document contexts with high accuracy and relevance.



```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate

prompt_title = "rag-document-analysis"

system = """You are a precise and helpful AI assistant specializing in question-answering tasks based on provided context.
Your primary task is to:
1. Analyze the provided context thoroughly
2. Answer questions using ONLY the information from the context
3. Preserve technical terms and proper nouns exactly as they appear
4. If the answer cannot be found in the context, respond with: 'The provided context does not contain information to answer this question.'
5. Format responses in clear, readable paragraphs with relevant examples when available
6. Focus on accuracy and clarity in your responses
"""

human = """#Question:
{question}

#Context:
{context}

#Answer:
Please provide a focused, accurate response that directly addresses the question using only the information from the provided context."""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

prompt
```




<pre class="custom">ChatPromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template="You are a precise and helpful AI assistant specializing in question-answering tasks based on provided context.\nYour primary task is to:\n1. Analyze the provided context thoroughly\n2. Answer questions using ONLY the information from the context\n3. Preserve technical terms and proper nouns exactly as they appear\n4. If the answer cannot be found in the context, respond with: 'The provided context does not contain information to answer this question.'\n5. Format responses in clear, readable paragraphs with relevant examples when available\n6. Focus on accuracy and clarity in your responses\n"), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template='#Question:\n{question}\n\n#Context:\n{context}\n\n#Answer:\nPlease provide a focused, accurate response that directly addresses the question using only the information from the provided context.'), additional_kwargs={})])</pre>



```python
hub.push(prompt_title, prompt, new_repo_is_public=False)
```




<pre class="custom">'https://smith.langchain.com/prompts/rag-document-analysis/f7a42fa8?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



### 1.2. RAG with Source Attribution

Enhanced RAG implementation with detailed source tracking and citation for improved accountability and verification.


```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate

prompt_title = "rag-with-sources"

system = """You are a precise and thorough AI assistant that provides well-documented answers with source attribution.
Your responsibilities include:
1. Analyzing provided context thoroughly
2. Generating accurate answers based solely on the given context
3. Including specific source references for each key point
4. Preserving technical terminology exactly as presented
5. Maintaining clear citation format [source: page/document]
6. If information is not found in the context, state: 'The provided context does not contain information to answer this question.'

Format your response as:
1. Main Answer
2. Sources Used (with specific locations)
3. Confidence Level (High/Medium/Low)"""

human = """#Question:
{question}

#Context:
{context}

#Answer:
Please provide a detailed response with source citations using only information from the provided context."""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
PROMPT_OWNER = "eun"
hub.push(f"{PROMPT_OWNER}/{prompt_title}", prompt, new_repo_is_public=True)
```




<pre class="custom">'https://smith.langchain.com/prompts/rag-with-sources/67246bf3?organizationId=f2bffb3c-dd45-53ac-b23b-5d696451d11c'</pre>



### 2. LLM Response Evaluation

Comprehensive evaluation of LLM responses based on multiple quality metrics with detailed scoring methodology.


```python
from langchain import hub
from langchain.prompts import PromptTemplate

prompt_title = "llm-response-evaluation"

evaluation_prompt = """Evaluate the LLM's response based on the following criteria:

INPUT:
Question: {question}
Context: {context}
LLM Response: {answer}

EVALUATION CRITERIA:
1. Accuracy (0-10)
- Perfect (10): Completely accurate, perfectly aligned with context
- Good (7-9): Minor inaccuracies
- Fair (4-6): Some significant inaccuracies
- Poor (0-3): Major inaccuracies or misalignment

2. Completeness (0-10)
- Perfect (10): Comprehensive coverage of all relevant points
- Good (7-9): Covers most important points
- Fair (4-6): Missing several key points
- Poor (0-3): Critically incomplete

3. Context Relevance (0-10)
- Perfect (10): Optimal use of context
- Good (7-9): Good use with minor omissions
- Fair (4-6): Partial use of relevant context
- Poor (0-3): Poor context utilization

4. Clarity (0-10)
- Perfect (10): Exceptionally clear and well-structured
- Good (7-9): Clear with minor issues
- Fair (4-6): Somewhat unclear
- Poor (0-3): Confusing or poorly structured

SCORING METHOD:
1. Calculate individual scores
2. Compute weighted average:
   - Accuracy: 40%
   - Completeness: 25%
   - Context Relevance: 25%
   - Clarity: 10%
3. Normalize to 0-1 scale

OUTPUT FORMAT:
{
    "individual_scores": {
        "accuracy": float,
        "completeness": float,
        "context_relevance": float,
        "clarity": float
    },
    "weighted_score": float,
    "normalized_score": float,
    "evaluation_notes": string
}

Return ONLY the normalized_score as a decimal between 0 and 1."""

prompt = PromptTemplate.from_template(evaluation_prompt)
```

---
## Professional Domain Prompts

Each professional domain prompt is carefully crafted to address specific industry needs and requirements.

This part requires optimization of prompts, especially according to domain data and format. Therefore, it is recommended that you test multiple prompts with Playground on websites such as OpenAI or Anthropic and use the most appropriate prompts. Below is an example of prompts in each field.

### 1. Academic Research Analysis Prompt
```python
PROMPT_TEMPLATE = """
As an expert academic researcher, analyze the academic content with:

INPUT:
- Content Type: {content_type}
- Field of Study: {field}
- Analysis Depth: {depth}

ANALYZE:
1. Research methodology and design
2. Key findings and significance
3. Theoretical framework
4. Statistical validity
5. Study limitations
6. Future directions

OUTPUT FORMAT:
{
    "executive_summary": str,
    "methodology_analysis": dict,
    "findings_analysis": dict,
    "quality_assessment": dict
}
"""
```

### 2. Clinical Case Analysis Prompt
```python
PROMPT_TEMPLATE = """
As a medical professional, analyze clinical cases with:

INPUT:
- Patient Information: {patient_data}
- Clinical Notes: {clinical_notes}

PROVIDE:
1. Clinical Assessment
2. Diagnostic Process
3. Treatment Plan
4. Risk Assessment

OUTPUT FORMAT:
{
    "clinical_summary": str,
    "differential_diagnosis": list,
    "treatment_plan": dict,
    "risk_assessment": dict
}
"""
```

### 3. Market Research Analysis Prompt
```python
PROMPT_TEMPLATE = """
As a market research analyst, analyze:

PARAMETERS:
- Industry: {industry}
- Market Segment: {segment}
- Region: {region}
- Time Period: {time_period}

COMPONENTS:
1. Market Overview
2. Competitive Analysis
3. Customer Analysis
4. SWOT Analysis
5. Financial Analysis
6. Recommendations

OUTPUT FORMAT:
{
    "market_overview": dict,
    "competitive_landscape": dict,
    "customer_insights": dict,
    "strategic_recommendations": list
}
"""
```

### 4. Educational Content Development Prompt
```python
PROMPT_TEMPLATE = """
As an educational content developer, create:

PARAMETERS:
- Subject: {subject}
- Grade Level: {grade_level}
- Learning Objectives: {objectives}
- Duration: {duration}

DELIVER:
1. Course Structure
2. Learning Materials
3. Assessment Components
4. Differentiation Strategies
5. Support Resources

OUTPUT FORMAT:
{
    "course_outline": dict,
    "lesson_plans": list,
    "assessments": dict,
    "support_resources": dict
}
"""
```

### 5. Legal Document Analysis Prompt
```python
PROMPT_TEMPLATE = """
As a legal professional, analyze:

PARAMETERS:
- Document Type: {doc_type}
- Jurisdiction: {jurisdiction}
- Legal Domain: {domain}

ANALYZE:
1. Document Overview
2. Key Provisions
3. Risk Assessment
4. Compliance Check
5. Recommendations

OUTPUT FORMAT:
{
    "document_summary": str,
    "key_provisions": dict,
    "risk_analysis": dict,
    "recommendations": list
}
"""
```

### 6. UX Research Analysis Prompt
```python
PROMPT_TEMPLATE = """
As a UX researcher, analyze:

PARAMETERS:
- Research Type: {research_type}
- Product/Service: {product}
- User Segment: {segment}
- Research Goals: {goals}

PROVIDE:
1. User Behavior Analysis
2. Usability Assessment
3. User Experience Mapping
4. Accessibility Evaluation
5. Recommendations

OUTPUT FORMAT:
{
    "behavioral_insights": dict,
    "usability_metrics": dict,
    "experience_mapping": dict,
    "design_recommendations": list
}
"""
```

### 7. Environmental Impact Assessment Prompt
```python
PROMPT_TEMPLATE = """
As an environmental specialist, assess:

PARAMETERS:
- Project Type: {project_type}
- Location: {location}
- Scale: {scale}
- Duration: {duration}

ANALYZE:
1. Environmental Baseline
2. Impact Analysis
3. Resource Assessment
4. Mitigation Strategies
5. Monitoring Plan

OUTPUT FORMAT:
{
    "assessment_summary": str,
    "impact_analysis": dict,
    "mitigation_plan": dict,
    "monitoring_framework": dict
}
"""
```


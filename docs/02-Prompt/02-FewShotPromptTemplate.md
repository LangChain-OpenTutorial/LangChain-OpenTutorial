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

# Few-Shot Templates

- Author: [hong-seongmin](https://github.com/hong-seongmin)
- Peer Review: [Hye-yoon Jeong](https://github.com/Hye-yoonJeong), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain OpenTutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/02-FewShotPromptTemplate.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/02-FewShotPromptTemplate.ipynb)

## Overview

LangChain's few-shot prompting provides a robust framework for guiding language models to generate high-quality outputs by supplying carefully selected examples. This technique minimizes the need for extensive model fine-tuning while ensuring precise, context-aware results across diverse use cases.

- **Few-shot prompt templates** : Define the structure and format of prompts by embedding illustrative examples, guiding the model to produce consistent outputs.
- **Example selection strategies** : Dynamically select the most relevant examples for a given query, enhancing the model's contextual understanding and response accuracy.
- **Chroma vector store** : A powerful utility for storing and retrieving examples based on semantic similarity, enabling scalable and efficient prompt construction.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [FewShotPromptTemplate](#FewShotPromptTemplate)
- [Dynamic Example Selection with Chroma](#dynamic-example-selection-with-chroma)
- [FewShotChatMessagePromptTemplate](#FewShotChatMessagePromptTemplate)

### References

- [LangChain Documentation: Few-shot prompting](https://python.langchain.com/docs/concepts/few_shot_prompting)
- [How to better prompt when doing SQL question-answering](https://python.langchain.com/docs/how_to/sql_prompting/#few-shot-examples)

---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

<pre class="custom">WARNING: Ignoring invalid distribution -angchain-community (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -treamlit (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Error parsing dependencies of torchsde: .* suffix can only be used with `==` or `!=` operators
        numpy (>=1.19.*) ; python_version >= "3.7"
               ~~~~~~~^
    WARNING: Ignoring invalid distribution -angchain-community (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -treamlit (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -angchain-community (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -orch (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
    WARNING: Ignoring invalid distribution -treamlit (c:\users\user\appdata\local\programs\python\python310\lib\site-packages)
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain-chroma",
    ],
    verbose=False,
    upgrade=False,
)
```

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "09-FewShot-Prompt-Templates",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## ```FewShotPromptTemplate```

Few-shot prompting is a powerful technique that guides language models to produce accurate and contextually relevant outputs by providing a small set of carefully designed examples. LangChain's ```FewShotPromptTemplate``` streamlines this process, allowing users to construct flexible and reusable prompts for various tasks like question answering, summarization, and correction.

1. **Designing Few-Shot Prompts**

   - Define examples that illustrate the desired structure and style of the output.
   - Ensure examples cover edge cases to improve model understanding and performance.

2. **Dynamic Example Selection**

   - Use semantic similarity techniques or vector-based search to select the most relevant examples for a query.

3. **Integrating Few-Shot Prompts**

   - Combine prompt templates with language models to create robust chains for generating responses.

### ```FewShotPromptTemplate``` Example

The ```FewShotPromptTemplate``` allows you to provide a language model with a small set of examples that demonstrate the desired structure and format of its output. By leveraging these examples, the model can better understand the context and generate more accurate responses for new queries. This technique is especially useful for tasks like question answering, summarization, or generating structured outputs.

Below, we define a few examples to help the model answer questions more effectively by breaking them down into intermediate steps. We then use the ```FewShotPromptTemplate``` to format the prompt dynamically based on the query.

---

```python
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(
    temperature=0,  # Creativity
    model_name="gpt-4o-mini",  # Use a valid model name
)

# User query
question = "What is the capital of United States of America?"

# Query the model
response = llm.predict(question)

# Print the response
print(response)
```

<pre class="custom">The capital of the United States of America is Washington, D.C.
</pre>

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define examples for the few-shot prompt
examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: At what age did Steve Jobs die?
Intermediate Answer: Steve Jobs died at the age of 56.
Additional Question: At what age did Einstein die?
Intermediate Answer: Einstein died at the age of 76.
The final answer is: Einstein
""",
    },
    {
        "question": "When was the founder of Naver born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the founder of Naver?
Intermediate Answer: Naver was founded by Lee Hae-jin.
Additional Question: When was Lee Hae-jin born?
Intermediate Answer: Lee Hae-jin was born on June 22, 1967.
The final answer is: June 22, 1967
""",
    },
    {
        "question": "Who was the reigning king when Yulgok Yi's mother was born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is Yulgok Yi's mother?
Intermediate Answer: Yulgok Yi's mother is Shin Saimdang.
Additional Question: When was Shin Saimdang born?
Intermediate Answer: Shin Saimdang was born in 1504.
Additional Question: Who was the king of Joseon in 1504?
Intermediate Answer: The king of Joseon in 1504 was Yeonsangun.
The final answer is: Yeonsangun
""",
    },
    {
        "question": "Are the directors of Oldboy and Parasite from the same country?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the director of Oldboy?
Intermediate Answer: The director of Oldboy is Park Chan-wook.
Additional Question: Which country is Park Chan-wook from?
Intermediate Answer: Park Chan-wook is from South Korea.
Additional Question: Who is the director of Parasite?
Intermediate Answer: The director of Parasite is Bong Joon-ho.
Additional Question: Which country is Bong Joon-ho from?
Intermediate Answer: Bong Joon-ho is from South Korea.
The final answer is: Yes
""",
    },
]
```

```python
# Create an example prompt template
example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

# Print the first formatted example
print(example_prompt.format(**examples[0]))
```

<pre class="custom">Question:
    Who lived longer, Steve Jobs or Einstein?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: At what age did Steve Jobs die?
    Intermediate Answer: Steve Jobs died at the age of 56.
    Additional Question: At what age did Einstein die?
    Intermediate Answer: Einstein died at the age of 76.
    The final answer is: Einstein
    
</pre>

```python
# Initialize the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

# Example question
question = "How old was Bill Gates when Google was founded?"

# Generate the final prompt
final_prompt = few_shot_prompt.format(question=question)
print(final_prompt)
```

<pre class="custom">Question:
    Who lived longer, Steve Jobs or Einstein?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: At what age did Steve Jobs die?
    Intermediate Answer: Steve Jobs died at the age of 56.
    Additional Question: At what age did Einstein die?
    Intermediate Answer: Einstein died at the age of 76.
    The final answer is: Einstein
    
    
    Question:
    When was the founder of Naver born?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is the founder of Naver?
    Intermediate Answer: Naver was founded by Lee Hae-jin.
    Additional Question: When was Lee Hae-jin born?
    Intermediate Answer: Lee Hae-jin was born on June 22, 1967.
    The final answer is: June 22, 1967
    
    
    Question:
    Who was the reigning king when Yulgok Yi's mother was born?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is Yulgok Yi's mother?
    Intermediate Answer: Yulgok Yi's mother is Shin Saimdang.
    Additional Question: When was Shin Saimdang born?
    Intermediate Answer: Shin Saimdang was born in 1504.
    Additional Question: Who was the king of Joseon in 1504?
    Intermediate Answer: The king of Joseon in 1504 was Yeonsangun.
    The final answer is: Yeonsangun
    
    
    Question:
    Are the directors of Oldboy and Parasite from the same country?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is the director of Oldboy?
    Intermediate Answer: The director of Oldboy is Park Chan-wook.
    Additional Question: Which country is Park Chan-wook from?
    Intermediate Answer: Park Chan-wook is from South Korea.
    Additional Question: Who is the director of Parasite?
    Intermediate Answer: The director of Parasite is Bong Joon-ho.
    Additional Question: Which country is Bong Joon-ho from?
    Intermediate Answer: Bong Joon-ho is from South Korea.
    The final answer is: Yes
    
    
    Question:
    How old was Bill Gates when Google was founded?
    Answer:
</pre>

```python
# Query the model with the final prompt
response = llm.predict(final_prompt)
print(response)
```

<pre class="custom">Does this question require additional questions: Yes.  
    Additional Question: When was Google founded?  
    Intermediate Answer: Google was founded on September 4, 1998.  
    Additional Question: What year was Bill Gates born?  
    Intermediate Answer: Bill Gates was born on October 28, 1955.  
    Additional Question: How old was Bill Gates on September 4, 1998?  
    Intermediate Answer: Bill Gates was 42 years old when Google was founded.  
    The final answer is: 42 years old.
</pre>

## Dynamic Example Selection with ```Chroma```

Sometimes we need to go through multiple steps of thinking to evaluate a single question. Breaking down the question into steps and guiding towards the desired answer can lead to better quality responses.
```Chroma``` provides an efficient way to store and retrieve examples based on semantic similarity, enabling dynamic example selection in workflows.

1. **Embedding and Vector Store Initialization**

   - Use ```OpenAIEmbeddings``` to embed examples.
   - Store the embeddings in a ```Chroma``` vector store for efficient retrieval.

2. **Example Storage**

   - Examples are stored with both their content and metadata.
   - Metadata can include details such as the question and answer, which are used for further processing after retrieval.

3. **Similarity Search**

   - Query the vector store to retrieve the most similar examples based on the input.
   - Enables context-aware dynamic prompt creation.

Import the necessary packages and create a vector store.

```python
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
chroma = Chroma(persist_directory="example_selector", embedding_function=embeddings)
```

Create example question-answer processes needed to derive the answer.

```python
examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: At what age did Steve Jobs die?
Intermediate Answer: Steve Jobs died at the age of 56.
Additional Question: At what age did Einstein die?
Intermediate Answer: Einstein died at the age of 76.
The final answer is: Einstein
""",
    },
    {
        "question": "When was the founder of Google born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the founder of Google?
Intermediate Answer: Google was founded by Larry Page and Sergey Brin.
Additional Question: When was Larry Page born?
Intermediate Answer: Larry Page was born on March 26, 1973.
Additional Question: When was Sergey Brin born?
Intermediate Answer: Sergey Brin was born on August 21, 1973.
The final answer is: Larry Page was born on March 26, 1973, and Sergey Brin was born on August 21, 1973.
""",
    },
    {
        "question": "Who was the President when Donald Trump's mother was born?",  # 쉼표 추가
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is Donald Trump's mother?
Intermediate Answer: Donald Trump's mother is Mary Anne MacLeod Trump.
Additional Question: When was Mary Anne MacLeod Trump born?
Intermediate Answer: Mary Anne MacLeod Trump was born on May 10, 1912.
Additional Question: Who was the U.S. President in 1912?
Intermediate Answer: William Howard Taft was President in 1912.
The final answer is: William Howard Taft
""",
    },
    {
        "question": "Are the directors of Oldboy and Parasite from the same country?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the director of Oldboy?
Intermediate Answer: The director of Oldboy is Park Chan-wook.
Additional Question: Which country is Park Chan-wook from?
Intermediate Answer: Park Chan-wook is from South Korea.
Additional Question: Who is the director of Parasite?
Intermediate Answer: The director of Parasite is Bong Joon-ho.
Additional Question: Which country is Bong Joon-ho from?
Intermediate Answer: Bong Joon-ho is from South Korea.
The final answer is: Yes
""",
    },
]
```

Create a vector store and define the ```DynamicFewShotLearning``` template.

```python
# Add examples to the vector store
texts = [example["question"] for example in examples]
metadatas = [example for example in examples]
chroma.add_texts(texts=texts, metadatas=metadatas)

# Set up Example Selector
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=chroma,  # Only vectorstore is needed
    k=1  # Number of examples to select
)

# Define Few-Shot Prompt Template
example_prompt_template = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}\n"
)
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt_template,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

```

Let's run it to verify if it produces answers through our desired process.

```python
# Query input and process
query = {"question": "How old was Elon Musk when he made Paypal?"}
formatted_prompt = prompt.format(**query)

# Print the constructed prompt
print("Constructed Prompt:\n")
print(formatted_prompt)

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Query the language model with the constructed prompt
response = llm.predict(formatted_prompt)
print("\nResponse:\n")
print(response)
```

<pre class="custom">Constructed Prompt:
    
    Question:
    When was the founder of Google born?
    Answer:
    Does this question require additional questions: Yes.
    Additional Question: Who is the founder of Google?
    Intermediate Answer: Google was founded by Larry Page and Sergey Brin.
    Additional Question: When was Larry Page born?
    Intermediate Answer: Larry Page was born on March 26, 1973.
    Additional Question: When was Sergey Brin born?
    Intermediate Answer: Sergey Brin was born on August 21, 1973.
    The final answer is: Larry Page was born on March 26, 1973, and Sergey Brin was born on August 21, 1973.
    
    
    
    Question:
    How old was Elon Musk when he made Paypal?
    Answer:
    
    Response:
    
    Does this question require additional questions: Yes.  
    Additional Question: When was PayPal founded?  
    Intermediate Answer: PayPal was founded in December 1998.  
    Additional Question: When was Elon Musk born?  
    Intermediate Answer: Elon Musk was born on June 28, 1971.  
    Additional Question: How old was Elon Musk in December 1998?  
    Intermediate Answer: In December 1998, Elon Musk was 27 years old.  
    The final answer is: Elon Musk was 27 years old when he made PayPal.
</pre>

## ```FewShotChatMessagePromptTemplate```

Creating prompts for each situation is a complex and tedious task.
The ```FewShotChatMessagePromptTemplate``` leverages a few-shot learning approach to dynamically generate chat-based prompts by combining relevant examples with a structured format. This method is especially effective for tasks like summarization, meeting minute creation, or document processing.


```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize the Vector DB
chroma = Chroma(persist_directory="fewshot_chat", embedding_function=embeddings)

# Define examples for few-shot prompting : 
examples = [
    {
        "instruction": "You are an expert in writing meeting minutes. Please write meeting minutes based on the given information",
        "input": "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.",
        "answer": """
Meeting Minutes: XYZ Company Marketing Strategy Meeting
Date: December 25, 2023
Location: XYZ Company Conference Room
Attendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)

1. Opening
   - Meeting began with opening remarks by Team Leader John Smith.
   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.

2. Market Trend Overview (John Smith)
   - Team Leader John Smith presented analysis of recent market trends.
   - Shared insights on consumer behavior changes and competitor strategies.

3. Digital Marketing Strategy (Sarah Johnson)
   - Sarah Johnson presented digital marketing strategy.
   - Focused on online advertising and SEO optimization methods.

4. Social Media Campaign (Mike Wilson)
   - Mike Wilson proposed ideas for new social media campaigns.
   - Explained plans for influencer marketing and content strategy.

5. Comprehensive Discussion
   - Team members shared and discussed ideas.
   - Discussed budget and resource allocation for each strategy.

6. Closing
   - Confirmed next meeting date and time.
   - Sarah Johnson assigned to compile and distribute meeting minutes.
""",
    },
    {
        "instruction": "You are a summarization expert. Please summarize the content based on the given information",
        "input": "This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.",
        "answer": """
Document Summary: Strategy Report for Sustainable Urban Development

- Importance: Emphasizes the necessity of sustainable urban development and its social, economic, and environmental benefits.
- Current Issues: Analyzes major problems in current urbanization processes, such as environmental pollution, resource depletion, and increasing inequality.
- Strategies: Presents various strategies to achieve sustainable urban development. These include eco-friendly construction, public transportation improvements, energy efficiency enhancement, and strengthening community engagement.
- Case Studies: Introduces successful sustainable development cases from cities worldwide. For example, explains achievable strategies through cases like Copenhagen, Denmark and Yokohama, Japan.
- Lessons: Summarizes key lessons learned from these cases. Emphasized lessons include the importance of multi-faceted approaches, cooperation with local communities, and the need for long-term planning.

This report provides an in-depth analysis of how sustainable urban development can be realized in practical and effective forms.
""",
    },
    {
        "instruction": "You are a sentence correction expert. Please correct the following sentences",
        "input": "Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.",
        "answer": "This company expects to improve customer communication more effectively by introducing a new marketing strategy.",
    },
]
```

```python
# Add examples to the vector store
texts = [example["instruction"] + " " + example["input"] for example in examples]
metadatas = examples
chroma.add_texts(texts=texts, metadatas=metadatas)

# Example query
query = {
    "instruction": "Please write the meeting minutes",
    "input": "On December 26, 2023, the product development team of ABC Technology Company held a weekly progress meeting for a new mobile application project. The meeting was attended by Project Manager Choi Hyun-soo, Lead Developer Hwang Ji-yeon, and UI/UX Designer Kim Tae-young. The main purpose of the meeting was to review the current progress of the project and establish plans for upcoming milestones. Each team member provided updates on their respective areas of work, and the team set goals for the next week.",
}

# Perform similarity search
query_text = query["instruction"] + " " + query["input"]
results = chroma.similarity_search(query_text, k=1)
print(results)
```

<pre class="custom">[Document(metadata={'answer': '\nMeeting Minutes: XYZ Company Marketing Strategy Meeting\nDate: December 25, 2023\nLocation: XYZ Company Conference Room\nAttendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)\n\n1. Opening\n   - Meeting began with opening remarks by Team Leader John Smith.\n   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.\n\n2. Market Trend Overview (John Smith)\n   - Team Leader John Smith presented analysis of recent market trends.\n   - Shared insights on consumer behavior changes and competitor strategies.\n\n3. Digital Marketing Strategy (Sarah Johnson)\n   - Sarah Johnson presented digital marketing strategy.\n   - Focused on online advertising and SEO optimization methods.\n\n4. Social Media Campaign (Mike Wilson)\n   - Mike Wilson proposed ideas for new social media campaigns.\n   - Explained plans for influencer marketing and content strategy.\n\n5. Comprehensive Discussion\n   - Team members shared and discussed ideas.\n   - Discussed budget and resource allocation for each strategy.\n\n6. Closing\n   - Confirmed next meeting date and time.\n   - Sarah Johnson assigned to compile and distribute meeting minutes.\n', 'input': "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.", 'instruction': 'You are an expert in writing meeting minutes. Please write meeting minutes based on the given information'}, page_content="You are an expert in writing meeting minutes. Please write meeting minutes based on the given information On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.")]
</pre>

```python
# Print the most similar example
print(f"Most similar example to the input:\n{query_text}\n")
for result in results:
    print(f'Instruction:\n{result.metadata["instruction"]}')
    print(f'Input:\n{result.metadata["input"]}')
    print(f'Answer:\n{result.metadata["answer"]}')
```

<pre class="custom">Most similar example to the input:
    Please write the meeting minutes On December 26, 2023, the product development team of ABC Technology Company held a weekly progress meeting for a new mobile application project. The meeting was attended by Project Manager Choi Hyun-soo, Lead Developer Hwang Ji-yeon, and UI/UX Designer Kim Tae-young. The main purpose of the meeting was to review the current progress of the project and establish plans for upcoming milestones. Each team member provided updates on their respective areas of work, and the team set goals for the next week.
    
    Instruction:
    You are an expert in writing meeting minutes. Please write meeting minutes based on the given information
    Input:
    On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    Answer:
    
    Meeting Minutes: XYZ Company Marketing Strategy Meeting
    Date: December 25, 2023
    Location: XYZ Company Conference Room
    Attendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)
    
    1. Opening
       - Meeting began with opening remarks by Team Leader John Smith.
       - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.
    
    2. Market Trend Overview (John Smith)
       - Team Leader John Smith presented analysis of recent market trends.
       - Shared insights on consumer behavior changes and competitor strategies.
    
    3. Digital Marketing Strategy (Sarah Johnson)
       - Sarah Johnson presented digital marketing strategy.
       - Focused on online advertising and SEO optimization methods.
    
    4. Social Media Campaign (Mike Wilson)
       - Mike Wilson proposed ideas for new social media campaigns.
       - Explained plans for influencer marketing and content strategy.
    
    5. Comprehensive Discussion
       - Team members shared and discussed ideas.
       - Discussed budget and resource allocation for each strategy.
    
    6. Closing
       - Confirmed next meeting date and time.
       - Sarah Johnson assigned to compile and distribute meeting minutes.
    
</pre>

```python
# Create the final prompt template
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        (
            "human",
            "{instruction}:\n{input}",
        ),
    ]
)

# Combine the prompt and the best example
best_example = results[0].metadata
filled_prompt = final_prompt.format_messages(**best_example)

# Print the filled prompt
print("\nFilled Prompt:\n")
for message in filled_prompt:
    # Determine message type and extract content
    message_type = type(message).__name__  # e.g., SystemMessage, HumanMessage, AIMessage
    content = message.content
    print(f"{message_type}:\n{content}\n")
```

<pre class="custom">
    Filled Prompt:
    
    SystemMessage:
    You are a helpful assistant.
    
    HumanMessage:
    You are an expert in writing meeting minutes. Please write meeting minutes based on the given information:
    On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    
</pre>

```python
# Query the model with the filled prompt
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
formatted_prompt = "\n".join([f"{type(message).__name__}:\n{message.content}" for message in filled_prompt])
response = llm.predict(formatted_prompt)

# Print the model's response
print("\nResponse:\n")
print(response)
```

<pre class="custom">
    Response:
    
    **Meeting Minutes**
    
    **Date:** December 25, 2023  
    **Time:** 3:00 PM  
    **Location:** XYZ Company Conference Room  
    
    **Attendees:**  
    - John Smith, Marketing Team Leader  
    - Sarah Johnson, Digital Marketing Manager  
    - Mike Wilson, Social Media Manager  
    
    **Agenda:**  
    1. Overview of recent market trends  
    2. Discussion of marketing strategies for the first half of 2024  
    3. Ideas for new social media campaigns  
    
    **Minutes:**  
    
    1. **Call to Order:**  
       The meeting was called to order by John Smith at 3:00 PM.
    
    2. **Overview of Recent Market Trends:**  
       - John Smith provided a brief overview of the current market trends affecting the industry. He highlighted key insights that will inform the marketing strategies for the upcoming year.
    
    3. **Presentations on Strategic Ideas:**  
       - **Sarah Johnson (Digital Marketing Manager):**  
         - Presented her strategic ideas focusing on enhancing digital presence through targeted online advertising and SEO optimization.  
         - Suggested exploring partnerships with influencers to broaden reach.
    
       - **Mike Wilson (Social Media Manager):**  
         - Discussed potential new social media campaigns aimed at increasing engagement and brand awareness.  
         - Proposed a series of interactive posts and contests to drive user participation.
    
    4. **Discussion:**  
       - The team engaged in a collaborative discussion regarding the proposed strategies and campaigns. Feedback was exchanged, and additional ideas were generated to refine the marketing approach.
    
    5. **Next Steps:**  
       - Each team member will further develop their proposals and present a detailed plan in the next meeting scheduled for January 15, 2024.  
       - John Smith will compile the insights from this meeting and prepare a summary report for upper management.
    
    6. **Adjournment:**  
       - The meeting was adjourned at 4:30 PM.
    
    **Action Items:**  
    - Sarah Johnson to refine digital marketing strategies.  
    - Mike Wilson to develop a detailed plan for social media campaigns.  
    - John Smith to prepare a summary report for management.
    
    **Next Meeting:**  
    - Date: January 15, 2024  
    - Time: TBD  
    
    **Minutes Prepared by:** [Your Name]  
    **Date of Preparation:** December 25, 2023
</pre>

### Resolving Similarity Search Issues in Example Selector

When calculating similarity, both ```instruction``` and ```input``` are used. However, searching based only on ```instruction``` does not yield accurate similarity results.

To resolve this, a custom class for similarity calculation is defined.

Below is an example of incorrectly retrieved results.


```python
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize the Vector DB
chroma = Chroma(persist_directory="fewshot_chat", embedding_function=embeddings)

```

```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize the Vector DB
chroma = Chroma(persist_directory="fewshot_chat", embedding_function=embeddings)

# Define examples for few-shot prompting
examples = [
    {
        "instruction": "You are an expert in writing meeting minutes. Please write meeting minutes based on the given information",
        "input": "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader Kim Su-jin, Digital Marketing Manager Park Ji-min, and Social Media Manager Lee Jun-ho. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader Kim Su-jin provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.",
        "answer": """
Meeting Minutes: XYZ Company Marketing Strategy Meeting
Date: December 25, 2023
Location: XYZ Company Conference Room
Attendees: Kim Su-jin (Marketing Team Leader), Park Ji-min (Digital Marketing Manager), Lee Jun-ho (Social Media Manager)

1. Opening
   - Meeting began with opening remarks by Team Leader Kim Su-jin.
   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.

2. Market Trend Overview (Kim Su-jin)
   - Team Leader Kim Su-jin presented analysis of recent market trends.
   - Shared insights on consumer behavior changes and competitor strategies.

3. Digital Marketing Strategy (Park Ji-min)
   - Park Ji-min presented digital marketing strategy.
   - Focused on online advertising and SEO optimization methods.

4. Social Media Campaign (Lee Jun-ho)
   - Lee Jun-ho proposed ideas for new social media campaigns.
   - Explained plans for influencer marketing and content strategy.

5. Comprehensive Discussion
   - Team members shared and discussed ideas.
   - Discussed budget and resource allocation for each strategy.

6. Closing
   - Confirmed next meeting date and time.
   - Park Ji-min assigned to compile and distribute meeting minutes.
""",
    },
    {
        "instruction": "You are a summarization expert. Please summarize the content based on the given information",
        "input": "This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.",
        "answer": """
Document Summary: Strategy Report for Sustainable Urban Development

- Importance: Emphasizes the necessity of sustainable urban development and its social, economic, and environmental benefits.
- Current Issues: Analyzes major problems in current urbanization processes, such as environmental pollution, resource depletion, and increasing inequality.
- Strategies: Presents various strategies to achieve sustainable urban development. These include eco-friendly construction, public transportation improvements, energy efficiency enhancement, and strengthening community engagement.
- Case Studies: Introduces successful sustainable development cases from cities worldwide. For example, explains achievable strategies through cases like Copenhagen, Denmark and Yokohama, Japan.
- Lessons: Summarizes key lessons learned from these cases. Emphasized lessons include the importance of multi-faceted approaches, cooperation with local communities, and the need for long-term planning.

This report provides an in-depth analysis of how sustainable urban development can be realized in practical and effective forms.
""",
    },
    {
        "instruction": "You are a sentence correction expert. Please correct the following sentences",
        "input": "Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.",
        "answer": "This company expects to improve customer communication more effectively by introducing a new marketing strategy.",
    },
]
```

```python
# Add examples to the vector store
texts = [example["instruction"] + " " + example["input"] for example in examples]
metadatas = examples
chroma.add_texts(texts=texts, metadatas=metadatas)

```




<pre class="custom">['d6022d36-36bd-4f7d-b60f-fdf3954141d8',
     '4596fe6f-b3c8-4184-9e47-cca4e6e749b6',
     '326ce3b0-8a76-4ad4-b83c-78e9af0728f1']</pre>



```python
# Use Chroma for example selection
def select_examples(query):
    query_text = query["instruction"] + " " + query.get("input", "")
    results = chroma.similarity_search(query_text, k=1)
    print("\n[select_examples Output]")
    print(f"Query Text: {query_text}")
    for i, result in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"Page Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
    return results

# Example selector using Chroma
def custom_selector(query):
    results = select_examples(query)
    selected_examples = [
        {
            "instruction": result.metadata["instruction"],
            "input": result.metadata["input"],
            "answer": result.metadata["answer"],
        }
        for result in results
    ]
    print("\n[custom_selector Output]")
    for i, example in enumerate(selected_examples, start=1):
        print(f"Selected Example {i}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input']}")
        print(f"Answer: {example['answer']}")
    return selected_examples

# Example query to test
query = {
    "instruction": "Please write the meeting minutes",
    "input": "On December 26, 2023, the product development team of ABC Technology Company held a weekly progress meeting for a new mobile application project. The meeting was attended by Project Manager Choi Hyun-soo, Lead Developer Hwang Ji-yeon, and UI/UX Designer Kim Tae-young. The main purpose of the meeting was to review the current progress of the project and establish plans for upcoming milestones. Each team member provided updates on their respective areas of work, and the team set goals for the next week.",
}

# Test the functions
custom_selector(query)

```

<pre class="custom">
    [select_examples Output]
    Query Text: Please write the meeting minutes On December 26, 2023, the product development team of ABC Technology Company held a weekly progress meeting for a new mobile application project. The meeting was attended by Project Manager Choi Hyun-soo, Lead Developer Hwang Ji-yeon, and UI/UX Designer Kim Tae-young. The main purpose of the meeting was to review the current progress of the project and establish plans for upcoming milestones. Each team member provided updates on their respective areas of work, and the team set goals for the next week.
    Result 1:
    Page Content: You are an expert in writing meeting minutes. Please write meeting minutes based on the given information On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader Kim Su-jin, Digital Marketing Manager Park Ji-min, and Social Media Manager Lee Jun-ho. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader Kim Su-jin provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    Metadata: {'answer': '\nMeeting Minutes: XYZ Company Marketing Strategy Meeting\nDate: December 25, 2023\nLocation: XYZ Company Conference Room\nAttendees: Kim Su-jin (Marketing Team Leader), Park Ji-min (Digital Marketing Manager), Lee Jun-ho (Social Media Manager)\n\n1. Opening\n   - Meeting began with opening remarks by Team Leader Kim Su-jin.\n   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.\n\n2. Market Trend Overview (Kim Su-jin)\n   - Team Leader Kim Su-jin presented analysis of recent market trends.\n   - Shared insights on consumer behavior changes and competitor strategies.\n\n3. Digital Marketing Strategy (Park Ji-min)\n   - Park Ji-min presented digital marketing strategy.\n   - Focused on online advertising and SEO optimization methods.\n\n4. Social Media Campaign (Lee Jun-ho)\n   - Lee Jun-ho proposed ideas for new social media campaigns.\n   - Explained plans for influencer marketing and content strategy.\n\n5. Comprehensive Discussion\n   - Team members shared and discussed ideas.\n   - Discussed budget and resource allocation for each strategy.\n\n6. Closing\n   - Confirmed next meeting date and time.\n   - Park Ji-min assigned to compile and distribute meeting minutes.\n', 'input': "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader Kim Su-jin, Digital Marketing Manager Park Ji-min, and Social Media Manager Lee Jun-ho. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader Kim Su-jin provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.", 'instruction': 'You are an expert in writing meeting minutes. Please write meeting minutes based on the given information'}
    
    [custom_selector Output]
    Selected Example 1:
    Instruction: You are an expert in writing meeting minutes. Please write meeting minutes based on the given information
    Input: On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader Kim Su-jin, Digital Marketing Manager Park Ji-min, and Social Media Manager Lee Jun-ho. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader Kim Su-jin provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    Answer: 
    Meeting Minutes: XYZ Company Marketing Strategy Meeting
    Date: December 25, 2023
    Location: XYZ Company Conference Room
    Attendees: Kim Su-jin (Marketing Team Leader), Park Ji-min (Digital Marketing Manager), Lee Jun-ho (Social Media Manager)
    
    1. Opening
       - Meeting began with opening remarks by Team Leader Kim Su-jin.
       - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.
    
    2. Market Trend Overview (Kim Su-jin)
       - Team Leader Kim Su-jin presented analysis of recent market trends.
       - Shared insights on consumer behavior changes and competitor strategies.
    
    3. Digital Marketing Strategy (Park Ji-min)
       - Park Ji-min presented digital marketing strategy.
       - Focused on online advertising and SEO optimization methods.
    
    4. Social Media Campaign (Lee Jun-ho)
       - Lee Jun-ho proposed ideas for new social media campaigns.
       - Explained plans for influencer marketing and content strategy.
    
    5. Comprehensive Discussion
       - Team members shared and discussed ideas.
       - Discussed budget and resource allocation for each strategy.
    
    6. Closing
       - Confirmed next meeting date and time.
       - Park Ji-min assigned to compile and distribute meeting minutes.
    
</pre>




    [{'instruction': 'You are an expert in writing meeting minutes. Please write meeting minutes based on the given information',
      'input': "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader Kim Su-jin, Digital Marketing Manager Park Ji-min, and Social Media Manager Lee Jun-ho. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader Kim Su-jin provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.",
      'answer': '\nMeeting Minutes: XYZ Company Marketing Strategy Meeting\nDate: December 25, 2023\nLocation: XYZ Company Conference Room\nAttendees: Kim Su-jin (Marketing Team Leader), Park Ji-min (Digital Marketing Manager), Lee Jun-ho (Social Media Manager)\n\n1. Opening\n   - Meeting began with opening remarks by Team Leader Kim Su-jin.\n   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.\n\n2. Market Trend Overview (Kim Su-jin)\n   - Team Leader Kim Su-jin presented analysis of recent market trends.\n   - Shared insights on consumer behavior changes and competitor strategies.\n\n3. Digital Marketing Strategy (Park Ji-min)\n   - Park Ji-min presented digital marketing strategy.\n   - Focused on online advertising and SEO optimization methods.\n\n4. Social Media Campaign (Lee Jun-ho)\n   - Lee Jun-ho proposed ideas for new social media campaigns.\n   - Explained plans for influencer marketing and content strategy.\n\n5. Comprehensive Discussion\n   - Team members shared and discussed ideas.\n   - Discussed budget and resource allocation for each strategy.\n\n6. Closing\n   - Confirmed next meeting date and time.\n   - Park Ji-min assigned to compile and distribute meeting minutes.\n'}]



```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define a new example prompt template with the custom selector
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}:\n{input}"),
        ("ai", "{answer}"),
    ]
)

# Create a prompt template with selected examples
def create_fewshot_prompt(query):
    selected_examples = custom_selector(query)
    fewshot_prompt = ""
    for example in selected_examples:
        fewshot_prompt += example_prompt.format(
            instruction=example["instruction"],
            input=example["input"],
            answer=example["answer"],
        )
    return fewshot_prompt

```

```python
# Create the final prompt for the chain
def create_final_prompt(query):
    fewshot_prompt = create_fewshot_prompt(query)
    final_prompt = (
        "You are a helpful professional assistant.\n\n" + fewshot_prompt + "\n\n"
        f"Human:\n{query['instruction']}:\n{query['input']}\nAssistant:"
    )
    return final_prompt

# Example queries
queries = [
    {
        "instruction": "Draft meeting minutes for the following discussion",
        "input": "On December 26, 2023, ABC Tech's product development team held their weekly progress meeting regarding the new mobile app project. Present were Project Manager John Davis, Lead Developer Emily Chen, and UI/UX Designer Michael Brown. The team reviewed current project milestones and established next steps. Each team member provided updates on their respective workstreams, and the team set deliverables for the upcoming week.",
    },
    {
        "instruction": "Please provide a summary of the following document",
        "input": "This comprehensive 30-page report titled 'Global Economic Outlook 2023' covers sustainable urban development trends, current urbanization challenges, and strategic approaches to sustainable city planning. The document includes case studies of successful urban development initiatives from various countries and key takeaways from these implementations.",
    },
    {
        "instruction": "Please review and correct these sentences",
        "input": "The company anticipates revenue growth this fiscal year. The new strategic initiatives are showing positive results.",
    },
]

# Process each query and print the responses
for query in queries:
    final_prompt = create_final_prompt(query)
    print("\nFinal Prompt:\n")
    print(final_prompt)
    response = llm.predict(final_prompt)
    print("\nModel Response:\n", response)
    print("\n---\n")
```

<pre class="custom">
    [select_examples Output]
    Query Text: Draft meeting minutes for the following discussion On December 26, 2023, ABC Tech's product development team held their weekly progress meeting regarding the new mobile app project. Present were Project Manager John Davis, Lead Developer Emily Chen, and UI/UX Designer Michael Brown. The team reviewed current project milestones and established next steps. Each team member provided updates on their respective workstreams, and the team set deliverables for the upcoming week.
    Result 1:
    Page Content: You are an expert in writing meeting minutes. Please write meeting minutes based on the given information On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    Metadata: {'answer': '\nMeeting Minutes: XYZ Company Marketing Strategy Meeting\nDate: December 25, 2023\nLocation: XYZ Company Conference Room\nAttendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)\n\n1. Opening\n   - Meeting began with opening remarks by Team Leader John Smith.\n   - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.\n\n2. Market Trend Overview (John Smith)\n   - Team Leader John Smith presented analysis of recent market trends.\n   - Shared insights on consumer behavior changes and competitor strategies.\n\n3. Digital Marketing Strategy (Sarah Johnson)\n   - Sarah Johnson presented digital marketing strategy.\n   - Focused on online advertising and SEO optimization methods.\n\n4. Social Media Campaign (Mike Wilson)\n   - Mike Wilson proposed ideas for new social media campaigns.\n   - Explained plans for influencer marketing and content strategy.\n\n5. Comprehensive Discussion\n   - Team members shared and discussed ideas.\n   - Discussed budget and resource allocation for each strategy.\n\n6. Closing\n   - Confirmed next meeting date and time.\n   - Sarah Johnson assigned to compile and distribute meeting minutes.\n', 'input': "On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.", 'instruction': 'You are an expert in writing meeting minutes. Please write meeting minutes based on the given information'}
    
    [custom_selector Output]
    Selected Example 1:
    Instruction: You are an expert in writing meeting minutes. Please write meeting minutes based on the given information
    Input: On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    Answer: 
    Meeting Minutes: XYZ Company Marketing Strategy Meeting
    Date: December 25, 2023
    Location: XYZ Company Conference Room
    Attendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)
    
    1. Opening
       - Meeting began with opening remarks by Team Leader John Smith.
       - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.
    
    2. Market Trend Overview (John Smith)
       - Team Leader John Smith presented analysis of recent market trends.
       - Shared insights on consumer behavior changes and competitor strategies.
    
    3. Digital Marketing Strategy (Sarah Johnson)
       - Sarah Johnson presented digital marketing strategy.
       - Focused on online advertising and SEO optimization methods.
    
    4. Social Media Campaign (Mike Wilson)
       - Mike Wilson proposed ideas for new social media campaigns.
       - Explained plans for influencer marketing and content strategy.
    
    5. Comprehensive Discussion
       - Team members shared and discussed ideas.
       - Discussed budget and resource allocation for each strategy.
    
    6. Closing
       - Confirmed next meeting date and time.
       - Sarah Johnson assigned to compile and distribute meeting minutes.
    
    
    Final Prompt:
    
    You are a helpful professional assistant.
    
    Human: You are an expert in writing meeting minutes. Please write meeting minutes based on the given information:
    On December 25, 2023, XYZ Company's marketing strategy meeting began at 3 PM. The meeting was attended by Marketing Team Leader John Smith, Digital Marketing Manager Sarah Johnson, and Social Media Manager Mike Wilson. The main purpose of the meeting was to establish marketing strategies for the first half of 2024 and discuss ideas for new social media campaigns. Team Leader John Smith provided a brief overview of recent market trends, followed by presentations from each team member about strategic ideas in their respective fields.
    AI: 
    Meeting Minutes: XYZ Company Marketing Strategy Meeting
    Date: December 25, 2023
    Location: XYZ Company Conference Room
    Attendees: John Smith (Marketing Team Leader), Sarah Johnson (Digital Marketing Manager), Mike Wilson (Social Media Manager)
    
    1. Opening
       - Meeting began with opening remarks by Team Leader John Smith.
       - Meeting purpose: Establish marketing strategies for H1 2024 and discuss new social media campaign ideas.
    
    2. Market Trend Overview (John Smith)
       - Team Leader John Smith presented analysis of recent market trends.
       - Shared insights on consumer behavior changes and competitor strategies.
    
    3. Digital Marketing Strategy (Sarah Johnson)
       - Sarah Johnson presented digital marketing strategy.
       - Focused on online advertising and SEO optimization methods.
    
    4. Social Media Campaign (Mike Wilson)
       - Mike Wilson proposed ideas for new social media campaigns.
       - Explained plans for influencer marketing and content strategy.
    
    5. Comprehensive Discussion
       - Team members shared and discussed ideas.
       - Discussed budget and resource allocation for each strategy.
    
    6. Closing
       - Confirmed next meeting date and time.
       - Sarah Johnson assigned to compile and distribute meeting minutes.
    
    
    Human:
    Draft meeting minutes for the following discussion:
    On December 26, 2023, ABC Tech's product development team held their weekly progress meeting regarding the new mobile app project. Present were Project Manager John Davis, Lead Developer Emily Chen, and UI/UX Designer Michael Brown. The team reviewed current project milestones and established next steps. Each team member provided updates on their respective workstreams, and the team set deliverables for the upcoming week.
    Assistant:
    
    Model Response:
     Meeting Minutes: ABC Tech Product Development Team Weekly Progress Meeting  
    Date: December 26, 2023  
    Location: ABC Tech Conference Room  
    Attendees: John Davis (Project Manager), Emily Chen (Lead Developer), Michael Brown (UI/UX Designer)  
    
    1. Opening  
       - The meeting commenced at 10 AM with opening remarks by Project Manager John Davis.  
       - Purpose: Review progress on the new mobile app project and establish next steps.
    
    2. Project Milestones Review  
       - The team reviewed current project milestones and assessed progress against the project timeline.  
       - Discussed any challenges encountered and potential impacts on the schedule.
    
    3. Workstream Updates  
       - **Emily Chen (Lead Developer)**: Provided an update on the development progress, highlighting completed features and ongoing tasks.  
       - **Michael Brown (UI/UX Designer)**: Presented updates on the design aspects, including user feedback and adjustments made to the interface.
    
    4. Next Steps and Deliverables  
       - The team established deliverables for the upcoming week, ensuring alignment on priorities and deadlines.  
       - Each member committed to specific tasks to be completed by the next meeting.
    
    5. Closing  
       - John Davis summarized the key points discussed and confirmed the next meeting date and time.  
       - Action items were assigned, and the meeting adjourned at 11 AM.  
    
    6. Action Items  
       - Emily Chen to continue development on assigned features.  
       - Michael Brown to finalize UI adjustments based on user feedback.  
       - John Davis to monitor overall project progress and address any emerging issues.  
    
    **Next Meeting:** January 2, 2024, at 10 AM.  
    **Minutes Prepared by:** [Your Name] (if applicable)
    
    ---
    
    
    [select_examples Output]
    Query Text: Please provide a summary of the following document This comprehensive 30-page report titled 'Global Economic Outlook 2023' covers sustainable urban development trends, current urbanization challenges, and strategic approaches to sustainable city planning. The document includes case studies of successful urban development initiatives from various countries and key takeaways from these implementations.
    Result 1:
    Page Content: You are a summarization expert. Please summarize the content based on the given information This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.
    Metadata: {'answer': '\nDocument Summary: Strategy Report for Sustainable Urban Development\n\n- Importance: Emphasizes the necessity of sustainable urban development and its social, economic, and environmental benefits.\n- Current Issues: Analyzes major problems in current urbanization processes, such as environmental pollution, resource depletion, and increasing inequality.\n- Strategies: Presents various strategies to achieve sustainable urban development. These include eco-friendly construction, public transportation improvements, energy efficiency enhancement, and strengthening community engagement.\n- Case Studies: Introduces successful sustainable development cases from cities worldwide. For example, explains achievable strategies through cases like Copenhagen, Denmark and Yokohama, Japan.\n- Lessons: Summarizes key lessons learned from these cases. Emphasized lessons include the importance of multi-faceted approaches, cooperation with local communities, and the need for long-term planning.\n\nThis report provides an in-depth analysis of how sustainable urban development can be realized in practical and effective forms.\n', 'input': "This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.", 'instruction': 'You are a summarization expert. Please summarize the content based on the given information'}
    
    [custom_selector Output]
    Selected Example 1:
    Instruction: You are a summarization expert. Please summarize the content based on the given information
    Input: This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.
    Answer: 
    Document Summary: Strategy Report for Sustainable Urban Development
    
    - Importance: Emphasizes the necessity of sustainable urban development and its social, economic, and environmental benefits.
    - Current Issues: Analyzes major problems in current urbanization processes, such as environmental pollution, resource depletion, and increasing inequality.
    - Strategies: Presents various strategies to achieve sustainable urban development. These include eco-friendly construction, public transportation improvements, energy efficiency enhancement, and strengthening community engagement.
    - Case Studies: Introduces successful sustainable development cases from cities worldwide. For example, explains achievable strategies through cases like Copenhagen, Denmark and Yokohama, Japan.
    - Lessons: Summarizes key lessons learned from these cases. Emphasized lessons include the importance of multi-faceted approaches, cooperation with local communities, and the need for long-term planning.
    
    This report provides an in-depth analysis of how sustainable urban development can be realized in practical and effective forms.
    
    
    Final Prompt:
    
    You are a helpful professional assistant.
    
    Human: You are a summarization expert. Please summarize the content based on the given information:
    This document is a 20-page report on 'Strategies for Sustainable Urban Development'. The report comprehensively covers the importance of sustainable urban development, current urbanization issues, and various strategies to make urban development sustainable. The report also introduces successful sustainable urban development cases from multiple countries and summarizes the lessons learned from these cases.
    AI: 
    Document Summary: Strategy Report for Sustainable Urban Development
    
    - Importance: Emphasizes the necessity of sustainable urban development and its social, economic, and environmental benefits.
    - Current Issues: Analyzes major problems in current urbanization processes, such as environmental pollution, resource depletion, and increasing inequality.
    - Strategies: Presents various strategies to achieve sustainable urban development. These include eco-friendly construction, public transportation improvements, energy efficiency enhancement, and strengthening community engagement.
    - Case Studies: Introduces successful sustainable development cases from cities worldwide. For example, explains achievable strategies through cases like Copenhagen, Denmark and Yokohama, Japan.
    - Lessons: Summarizes key lessons learned from these cases. Emphasized lessons include the importance of multi-faceted approaches, cooperation with local communities, and the need for long-term planning.
    
    This report provides an in-depth analysis of how sustainable urban development can be realized in practical and effective forms.
    
    
    Human:
    Please provide a summary of the following document:
    This comprehensive 30-page report titled 'Global Economic Outlook 2023' covers sustainable urban development trends, current urbanization challenges, and strategic approaches to sustainable city planning. The document includes case studies of successful urban development initiatives from various countries and key takeaways from these implementations.
    Assistant:
    
    Model Response:
     Document Summary: Global Economic Outlook 2023
    
    - Overview: The report provides an in-depth analysis of sustainable urban development trends and challenges in the context of the global economy for 2023.
    - Urbanization Challenges: It identifies and discusses current challenges faced by urban areas, including rapid population growth, infrastructure strain, and environmental degradation.
    - Strategic Approaches: The document outlines strategic approaches for sustainable city planning, focusing on integrated policies, innovative technologies, and community involvement to foster resilience and sustainability.
    - Case Studies: It features case studies of successful urban development initiatives from various countries, highlighting effective practices and innovative solutions.
    - Key Takeaways: The report concludes with key takeaways from the case studies, emphasizing the importance of collaboration among stakeholders, adaptive planning, and the integration of sustainability into economic frameworks.
    
    This report serves as a valuable resource for understanding the dynamics of urban development and the strategies necessary for fostering sustainable cities in the face of ongoing global challenges.
    
    ---
    
    
    [select_examples Output]
    Query Text: Please review and correct these sentences The company anticipates revenue growth this fiscal year. The new strategic initiatives are showing positive results.
    Result 1:
    Page Content: You are a sentence correction expert. Please correct the following sentences Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.
    Metadata: {'answer': 'This company expects to improve customer communication more effectively by introducing a new marketing strategy.', 'input': 'Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.', 'instruction': 'You are a sentence correction expert. Please correct the following sentences'}
    
    [custom_selector Output]
    Selected Example 1:
    Instruction: You are a sentence correction expert. Please correct the following sentences
    Input: Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.
    Answer: This company expects to improve customer communication more effectively by introducing a new marketing strategy.
    
    Final Prompt:
    
    You are a helpful professional assistant.
    
    Human: You are a sentence correction expert. Please correct the following sentences:
    Our company is planning to introduce a new marketing strategy. Through this, communication with customers will become more effective.
    AI: This company expects to improve customer communication more effectively by introducing a new marketing strategy.
    
    Human:
    Please review and correct these sentences:
    The company anticipates revenue growth this fiscal year. The new strategic initiatives are showing positive results.
    Assistant:
    
    Model Response:
     The company anticipates revenue growth for this fiscal year, and the new strategic initiatives are yielding positive results.
    
    ---
    
</pre>

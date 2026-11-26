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

# Basic Example: Prompt+Model+OutputParser

- Author: [ChangJun Lee](https://www.linkedin.com/in/cjleeno1/)
- Peer Review: [Erika Park](https://www.linkedin.com/in/yeonseo-park-094193198/), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/06-LangChain-Expression-Language(LCEL).ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/06-LangChain-Expression-Language(LCEL).ipynb)


## Overview

The most fundamental and commonly used case involves linking a prompt template with a model. To illustrate how this works, let us create a chain that asks for the capital cities of various countries.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Utilizing Prompt Templates](#utilizing-prompt-templates)
- [Chain Creation](#chain-creation)

### References

- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [LangChain Core Output Parsers](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.list.CommaSeparatedListOutputParser.html#)
- [Python List Tutorial](https://docs.python.org/3.13/tutorial/datastructures.html)
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

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_openai",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

---
You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
# Configuration File for Managing API Key as an Environment Variable
from dotenv import load_dotenv

# Load API KEY Information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Set up LangSmith tracking: https://smith.langchain.com
from langsmith import utils

utils.tracing_is_enabled()
```




<pre class="custom">True</pre>



## Utilizing Prompt Templates

```PromptTemplate```

- A prompt template is used to create a complete prompt string by incorporating the user's input variables.
- Usage
  - ```template```: A template string is a predefined format where curly braces '{}' are used to represent variables.

  - ```input_variables```: The names of the variables to be inserted within the curly braces are defined as a list.

```input_variables```

- ```input_variables``` is a list that defines the names of the variables used in the ```PromptTemplate```.

```python
from langchain_core.prompts import PromptTemplate
```

The ```from_template()``` method is used to create a ```PromptTemplate``` object.

```python
# Define template
template = "What is the capital of {country}?"

# Create a `PromptTemplate` object using the `from_template` method.
prompt_template = PromptTemplate.from_template(template)
prompt_template
```




<pre class="custom">PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')</pre>



```python
# Generate the prompt.
prompt = prompt_template.format(country="Korea")
prompt
```




<pre class="custom">'What is the capital of Korea?'</pre>



```python
# Generate the prompt.
prompt = prompt_template.format(country="USA")
prompt
```




<pre class="custom">'What is the capital of USA?'</pre>



```python
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
```

## Chain Creation

### LCEL (LangChain Expression Language)

Here, we use LCEL to combine various components into a single chain.

![lcel.png](./img/02-langchain-expression-language.png)

```
chain = prompt | model | output_parser
```

The ```|``` symbol works similarly to the [Unix pipe operator](<https://en.wikipedia.org/wiki/Pipeline_(Unix)>), linking different components and passing the output of one component as the input to the next.

In this chain, user input is passed to the prompt template, and the output from the prompt template is then forwarded to the model. By examining each component individually, you can understand what happens at each step.

```python
# Create the prompt as a `PromptTemplate` object.
prompt = PromptTemplate.from_template("Please explain {topic} in simple terms.")


# Combine the prompt and model into a chain
chain = prompt | model
```

### Calling ```invoke()```

- Input values are provided in the form of a Python dictionary (key-value pairs).  
- When calling the ```invoke()``` function, these input values are passed as arguments.

```python
# Set the topic in the `input` dictionary to 'The Principles of Learning in Artificial Intelligence Models'.
input = {"topic": "The Principles of Learning in Artificial Intelligence Models"}
```

```python
# Connect the `prompt` object and the `model` object using the pipe (`|`) operator.
# Use the `invoke` method to pass the `input`.
# This will return the message generated by the AI model.
chain.invoke(input)
```




<pre class="custom">AIMessage(content='Sure! The principles of learning in artificial intelligence (AI) models can be understood as the basic ideas that guide how these models learn from data. Here are some key principles explained in simple terms:\n\n1. **Data is Key**: AI models learn from data. The more relevant and high-quality data they have, the better they can learn and make predictions. Think of it like a student learning from textbooks; the better the books, the more they learn.\n\n2. **Patterns and Features**: AI looks for patterns in the data. It identifies important features (characteristics) that help it understand the information. For example, if it’s learning to recognize cats in pictures, it might focus on features like fur texture, ear shape, and eye color.\n\n3. **Training and Testing**: AI models go through a training phase where they learn from a set of data. After training, they are tested on new data to see how well they learned. This is like practicing for a test and then taking the actual exam.\n\n4. **Feedback Loop**: AI models improve through feedback. When they make mistakes, they learn from those errors to adjust their understanding. This is similar to how a student learns from corrections on their homework.\n\n5. **Generalization**: The goal of an AI model is to generalize, meaning it should perform well not just on the data it was trained on, but also on new, unseen data. This is like a student who studies a variety of problems and can solve new ones on the exam.\n\n6. **Overfitting and Underfitting**: Overfitting happens when a model learns too much from the training data, including noise and outliers, making it less effective on new data. Underfitting occurs when it doesn’t learn enough. It’s like a student who either memorizes answers without understanding or doesn’t study enough to grasp the concepts.\n\n7. **Algorithms**: Different algorithms (methods) are used for learning. Some are better for certain types of data or tasks. Choosing the right algorithm is like picking the best study method for a subject.\n\n8. **Continuous Learning**: AI can continue to learn over time as it receives more data. This is similar to lifelong learning, where a person keeps gaining knowledge and skills throughout their life.\n\nIn summary, AI models learn from data by identifying patterns, receiving feedback, and adjusting their understanding to perform well on new information. The principles of learning help guide this process to make AI more effective and accurate.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 507, 'prompt_tokens': 21, 'total_tokens': 528, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0aa8d3e20b', 'finish_reason': 'stop', 'logprobs': None}, id='run-c6e9d1b4-7fae-4af5-8217-1456c5b23d24-0', usage_metadata={'input_tokens': 21, 'output_tokens': 507, 'total_tokens': 528, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



Below is an example of outputting a streaming response:

```python
# Request for Streaming Output
answer = chain.stream(input)

# Streaming Output
for token in answer:
    print(token.content, end="", flush=True)
```

<pre class="custom">Sure! The Principles of Learning in Artificial Intelligence (AI) Models can be understood as the basic ideas that guide how AI systems learn from data and improve their performance over time. Here are some key principles explained in simple terms:
    
    1. **Data is Key**: AI models learn from data. The more quality data they have, the better they can learn. Think of it like a student studying for a test; the more information they have, the better they can do.
    
    2. **Learning from Examples**: AI models learn by looking at examples. For instance, if you show an AI many pictures of cats and dogs, it can learn to tell the difference between them. This is similar to how humans learn by observing and practicing.
    
    3. **Feedback Loop**: AI models improve through feedback. When they make mistakes, they can adjust their understanding based on the corrections. This is like a teacher giving feedback to a student to help them learn from their errors.
    
    4. **Generalization**: AI models aim to generalize from the examples they see. This means they should be able to apply what they've learned to new, unseen data. For example, if an AI learns to recognize cats from specific pictures, it should still recognize a cat it has never seen before.
    
    5. **Optimization**: AI models often use optimization techniques to improve their performance. This involves tweaking their internal settings to minimize errors. It’s like fine-tuning a musical instrument to get the best sound.
    
    6. **Adaptability**: Good AI models can adapt to new information. If the environment changes or new data comes in, they can adjust their learning accordingly. This is similar to how people learn to adapt to new situations.
    
    7. **Scalability**: AI models should be able to handle increasing amounts of data without losing performance. This means they can learn from more examples as they become available, much like how a student can learn more as they read more books.
    
    8. **Transfer Learning**: Sometimes, knowledge gained from one task can help with another task. For example, if an AI learns to recognize animals, it might use that knowledge to help recognize different types of objects. This is like how learning to ride a bike can help you learn to ride a motorcycle.
    
    9. **Exploration vs. Exploitation**: AI models need to balance exploring new possibilities (trying new things) and exploiting what they already know (using what works well). This is similar to how a person might try new foods while still enjoying their favorite dishes.
    
    These principles help guide the development and training of AI models, making them more effective and efficient in learning from data and performing tasks.</pre>

### Output Parser

An **Output Parser** is a tool designed to transform or process the responses from an AI model into a specific format. Since the model's output is typically provided as free-form text, an **Output Parser** is essential to convert it into a structured format or extract the required data.


```python
from langchain_core.output_parsers import StrOutputParser

output_parser = (
    StrOutputParser()
)  # Directly returns the model's response as a string without modification.
```

An output parser is added to the chain.

```python
# A processing chain is constructed by connecting the prompt, model, and output parser.
chain = prompt | model | output_parser
```

```python
# Use the invoke method of the chain object to pass the input
chain.invoke(input)
```




<pre class="custom">"Sure! The Principles of Learning in Artificial Intelligence (AI) Models can be understood as the basic ideas that guide how AI systems learn from data and improve their performance over time. Here are some key principles explained in simple terms:\n\n1. **Data is Key**: AI models learn from data. The more relevant and high-quality data they have, the better they can learn. Think of it like a student studying for a test; the more information they have, the better they can prepare.\n\n2. **Learning from Examples**: AI models often learn by looking at examples. For instance, if you want to teach an AI to recognize cats in pictures, you show it many pictures of cats and non-cats. The model learns to identify patterns that distinguish cats from other objects.\n\n3. **Feedback Loop**: AI models improve through feedback. After making predictions or decisions, they receive feedback on whether they were right or wrong. This feedback helps them adjust and learn from their mistakes, similar to how a coach helps an athlete improve.\n\n4. **Generalization**: A good AI model can generalize from the examples it has seen to make predictions about new, unseen data. For example, if it has learned to recognize cats from various pictures, it should be able to identify a new cat picture it hasn't seen before.\n\n5. **Overfitting and Underfitting**: These are common problems in AI learning. Overfitting happens when a model learns too much from the training data, including noise and outliers, making it perform poorly on new data. Underfitting occurs when a model is too simple and doesn’t learn enough from the data. The goal is to find a balance where the model learns well without being too specific to the training data.\n\n6. **Continuous Learning**: AI models can continue to learn and improve over time. This means they can adapt to new data and changing environments, much like how people keep learning throughout their lives.\n\n7. **Algorithms Matter**: The methods or algorithms used to train AI models are crucial. Different algorithms can lead to different learning outcomes, just like different teaching methods can affect how well students learn.\n\n8. **Evaluation and Testing**: To know how well an AI model is learning, it needs to be tested on separate data that it hasn’t seen before. This helps to evaluate its performance and ensure it’s not just memorizing the training data.\n\nIn summary, the principles of learning in AI models revolve around using data effectively, learning from examples, receiving feedback, generalizing knowledge, avoiding common pitfalls, and continuously improving. These principles help AI systems become more accurate and useful over time."</pre>



```python
# Request for Streaming Output
answer = chain.stream(input)

# Streaming Output
for token in answer:
    print(token, end="", flush=True)
```

<pre class="custom">Sure! The Principles of Learning in Artificial Intelligence (AI) Models can be understood as the basic ideas that guide how AI systems learn from data and improve their performance over time. Here are some key principles explained in simple terms:
    
    1. **Data is Key**: AI models learn from data. The more quality data they have, the better they can learn. Think of it like a student studying for a test; the more information they have, the better they can do.
    
    2. **Learning from Examples**: AI models often learn by looking at examples. For instance, if you want an AI to recognize cats in pictures, you show it many pictures of cats and non-cats. Over time, it learns to tell the difference.
    
    3. **Feedback Loop**: AI models improve through feedback. When they make mistakes, they can adjust their understanding based on the corrections. This is similar to how a teacher helps a student learn from their errors.
    
    4. **Generalization**: A good AI model can apply what it has learned to new, unseen data. For example, if it learns to recognize cats from specific pictures, it should still recognize a cat in a different picture. This ability to generalize is crucial for effective learning.
    
    5. **Optimization**: AI models often use algorithms to find the best way to make predictions or decisions. They adjust their internal settings (like tuning a musical instrument) to minimize errors and improve accuracy.
    
    6. **Transfer Learning**: Sometimes, AI can use knowledge gained from one task to help with another task. For example, if an AI learns to recognize animals, it might use that knowledge to help recognize different types of animals more easily.
    
    7. **Continuous Learning**: AI can keep learning over time. As it gets more data or experiences, it can update its knowledge and improve its performance. This is like how people continue to learn and grow throughout their lives.
    
    8. **Exploration vs. Exploitation**: AI models often face a choice between exploring new possibilities (trying new things) and exploiting what they already know (using what works best). Balancing these two is important for effective learning.
    
    In summary, the principles of learning in AI models revolve around using data, learning from examples, receiving feedback, generalizing knowledge, optimizing performance, transferring skills, continuously learning, and balancing exploration with exploitation. These principles help AI systems become smarter and more effective over time.</pre>

### Applying and Modifying Templates

- The prompt content below can be **modified** as needed for testing purposes.  
- The ```model_name``` can also be adjusted for testing.

```python
template = """
You are a seasoned English teacher with 10 years of experience. Please write an English conversation suitable for the given situation.  
Refer to the [FORMAT] for the structure.

#SITUATION:
{question}

#FORMAT:
- Dialogue in English:
- Explanation of the Dialogue: 
"""

# Generate the prompt using the PromptTemplate
prompt = PromptTemplate.from_template(template)

# Initialize the ChatOpenAI model.
model = ChatOpenAI(model_name="gpt-4o-mini")

# Initialize the string output parser.
output_parser = StrOutputParser()
```

```python
# Construct the chain.
chain = prompt | model | output_parser
```

```python
# Execute the completed Chain to obtain a response.
print(chain.invoke({"question": "I want to go to a restaurant and order food."}))
```

<pre class="custom">- Dialogue in English:
    **Waiter:** Good evening! Welcome to La Bella Italia. How many are in your party?  
    **Customer:** Just one, please.  
    **Waiter:** Right this way. Here’s your menu. Can I get you something to drink while you look?  
    **Customer:** Yes, I’ll have a glass of water, please.  
    **Waiter:** Sure! Are you ready to order, or do you need more time?  
    **Customer:** I think I’m ready. I’d like the spaghetti carbonara, please.  
    **Waiter:** Excellent choice! Would you like any appetizers or desserts with that?  
    **Customer:** I’ll have a side salad to start, and maybe a slice of tiramisu for dessert.  
    **Waiter:** Great! I’ll put that order in for you. Anything else I can get you?  
    **Customer:** No, that’s all for now, thank you.  
    **Waiter:** You’re welcome! I’ll be back shortly with your order.  
    
    - Explanation of the Dialogue: 
    In this conversation, the customer arrives at a restaurant and interacts with the waiter. The waiter greets the customer and asks how many people are dining, establishing a friendly atmosphere. The customer indicates they are alone and is guided to a table. The waiter then offers the menu and suggests getting a drink, showing attentiveness to the customer's needs. The customer requests water, indicating a simple preference, before deciding on their meal. The waiter encourages further orders by suggesting appetizers and desserts, showcasing good customer service. The customer ultimately chooses a main course and additional items, and the waiter confirms the order while maintaining a courteous demeanor. The dialogue highlights common phrases and interactions typical in a restaurant setting, providing a practical example for learners of English.
</pre>

```python
# Execute the completed Chain to obtain a response
# Request for Streaming Output
answer = chain.stream({"question": "I want to go to a restaurant and order food."})

# Streaming Output
for token in answer:
    print(token, end="", flush=True)
```

<pre class="custom">- Dialogue in English:
    
    **Customer:** Hi there! Can I get a table for two, please?  
    
    **Host:** Sure! Right this way. Here’s your menu.  
    
    **Customer:** Thank you! What do you recommend for a starter?  
    
    **Host:** Our bruschetta is very popular, and the garlic shrimp is a favorite as well.  
    
    **Customer:** Sounds delicious! We'll have the bruschetta to start.  
    
    **Host:** Great choice! And for the main course?  
    
    **Customer:** I’d like the grilled salmon, please. How about you?  
    
    **Friend:** I’ll have the steak, medium rare.  
    
    **Host:** Excellent choices! Would you like to add any sides?  
    
    **Customer:** Yes, we’ll take a side of roasted vegetables and a Caesar salad.  
    
    **Host:** Perfect! I’ll get that order started for you.  
    
    **Customer:** Thank you!  
    
    **Host:** You’re welcome! Enjoy your meal.  
    
    ---
    
    - Explanation of the Dialogue: 
    
    In this dialogue, the customer arrives at a restaurant and requests a table for two. The host guides them to their table and hands them the menu. The customer asks for recommendations on starters, indicating they are interested in the menu. After deciding on the bruschetta, the customer moves on to the main course, where they order grilled salmon while their friend opts for steak. The host then inquires about side dishes, and the customer adds roasted vegetables and a Caesar salad, showing an understanding of meal complementing. Finally, the host confirms the order and wishes them an enjoyable meal, completing the dining experience. This dialogue showcases typical restaurant interactions, emphasizing polite communication and decision-making.</pre>

```python
# This time, set the question to 'Ordering Pizza in the US' and execute it.
# Request for Streaming Output
answer = chain.stream({"question": "Ordering Pizza in the US"})

# Streaming Output
for token in answer:
    print(token, end="", flush=True)
```

<pre class="custom">- Dialogue in English:
    
    **Customer:** Hi there! I’d like to order a pizza, please.
    
    **Pizza Server:** Of course! What size would you like? We have small, medium, large, and extra-large.
    
    **Customer:** I’ll take a large, please. 
    
    **Pizza Server:** Great choice! What type of pizza do you want? We have pepperoni, cheese, veggie, and a few specialty pizzas.
    
    **Customer:** I’ll go with pepperoni. Can I add extra cheese?
    
    **Pizza Server:** Absolutely! Extra cheese on a large pepperoni pizza. Would you like anything else? 
    
    **Customer:** Yes, can I also get a side of garlic bread and a two-liter soda?
    
    **Pizza Server:** Sure! We have a few soda options. We have cola, diet cola, lemon-lime, and root beer. Which one would you like?
    
    **Customer:** I’ll take a cola, please.
    
    **Pizza Server:** Great! So that’s one large pepperoni pizza with extra cheese, a side of garlic bread, and a cola. Would you like to add any dipping sauces?
    
    **Customer:** Yes, please! Can I get a marinara sauce and a ranch dressing?
    
    **Pizza Server:** Absolutely! Your total comes to $25.99. How would you like to pay?
    
    **Customer:** I’ll pay with my credit card.
    
    **Pizza Server:** Perfect! I’ll take that and have your order ready in about 30 minutes.
    
    **Customer:** Thank you! I appreciate it.
    
    **Pizza Server:** You’re welcome! Enjoy your meal!
    
    ---
    
    - Explanation of the Dialogue: 
    
    This dialogue captures a typical conversation when ordering pizza in the US. It begins with the customer greeting the server and expressing their intention to place an order. The server asks about the size of the pizza, which is a common initial query. The customer specifies they want a large pepperoni pizza and adds a request for extra cheese, demonstrating how customers can customize their orders.
    
    The server continues by offering additional items, like garlic bread and soda, which reflects the common practice of upselling in food service. The server also provides options for the soda, showcasing the variety available. The customer selects their preferences and further adds dipping sauces, which is a popular choice when ordering pizza.
    
    Finally, the server confirms the order and provides the total cost, with the customer choosing to pay by credit card, a common payment method. The exchange ends on a polite note, with both parties expressing gratitude, which is typical in customer service interactions. This dialogue is a practical example of everyday communication in a restaurant setting, highlighting key phrases and vocabulary related to food ordering.</pre>

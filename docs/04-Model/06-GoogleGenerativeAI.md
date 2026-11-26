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

# Google Generative AI

- Author: [HyeonJong Moon](https://github.com/hj0302)
- Peer Review : [effort-type](https://github.com/effort-type)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/06-GoogleGenerativeAI.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/06-GoogleGenerativeAI.ipynb)
## Overview

You can use the ```ChatGoogleGenerativeAI``` class from the [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) integration package to access not only Google AI’s ```gemini``` and ```gemini-vision``` models, but also other generative models.

### Table of Contents

- [Overview](#overview)
- [Create API key](#Create-API-Key)
- [ChatGoogleGenerativeAI](#ChatGoogleGenerativeAI)
- [Safety Settings](#Safety-Settings)
- [Streaming and Batching](#Streaming-and-Batching)
- [Multimodeal Model](#Multimodeal-Model)

### References

- [LangChain ChatGoogleGenerativeAI Reference](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)

----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_core",
        "langchain_google_genai",
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
        "GOOGLE_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "05-GoogleGenerativeAI",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Create API Key

- Please create an API KEY from [link](https://aistudio.google.com/app/apikey?hl=en).
- Set the user's Google API key as the environment variable ```GOOGLE_API_KEY```.

You can alternatively set ```GOOGLE_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```GOOGLE_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## ChatGoogleGenerativeAI

Import the ```ChatGoogleGenerativeAI``` class from the ```langchain_google_genai``` package.

The ```ChatGoogleGenerativeAI``` class is used to implement conversational AI systems using Google’s Generative AI models. Through this class, users can interact with Google’s conversational AI model. Conversations with the model take place in a chat format, and the model generates appropriate responses based on user input.

Because the ```ChatGoogleGenerativeAI``` class is integrated with the LangChain framework, it can be used alongside other LangChain components. 

For information about supported models, see: https://ai.google.dev/gemini-api/docs/models/gemini?hl=en

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Pass the prompt to generate an output
answer = llm.stream("Please explain about Langchain(in three lines)")

# Print the result
for token in answer:
    print(token.content, end="", flush=True)
```

<pre class="custom">LangChain is a framework for developing applications powered by large language models (LLMs).  It simplifies building applications by connecting LLMs to other sources of data and computation.  This enables creation of sophisticated chains of prompts and actions, going beyond single LLM calls.
</pre>

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Create an instance of ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create a prompt
prompt = PromptTemplate.from_template("Answer with Yes or No. Is {question} a fruit?")

# Create the chain
chain = prompt | model

# Print the result
response = chain.invoke({"question": "Apple"})
print(response.content)
```

<pre class="custom">Yes
    
</pre>

## Safety Settings

Gemini models have default safety settings that can be overridden. If you are receiving lots of "Safety Warnings" from your models, you can try tweaking the ```safety_settings``` attribute of the model. For example, to turn off safety blocking for dangerous content, you can construct your LLM as follows:

```python
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    safety_settings={
        # Set threshold levels for blocking harmful content.
        # In this case, the settings indicate not to block harmful content. (However, there could still be basic blocking.)
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

response = llm.invoke("Please explain about Gemini model")
print(response.content)
```

<pre class="custom">Gemini is Google's large multimodal AI model, designed to handle various types of information, including text, code, audio, and images.  It's positioned as a competitor to OpenAI's GPT models and aims to be a more versatile and powerful tool.  Here's a breakdown of key aspects:
    
    **Key Features and Capabilities:**
    
    * **Multimodality:** This is a crucial differentiator.  Gemini can understand and generate responses across different modalities, unlike models primarily focused on text.  This means it can process and create text, translate languages, write different kinds of creative content, and answer your questions in an informative way, even if they are open ended, challenging, or strange.  It can also analyze images and potentially other forms of data in the future.
    
    * **Reasoning and Problem-Solving:** Google emphasizes Gemini's improved reasoning abilities compared to previous models.  This allows it to tackle more complex tasks and provide more accurate and nuanced answers.
    
    * **Scalability and Efficiency:**  Gemini is built to be highly scalable, meaning it can be adapted and deployed for various applications and tasks.  Google also focuses on its efficiency, aiming for optimal performance with lower computational costs.
    
    * **Integration with Google Ecosystem:**  Gemini is designed to integrate seamlessly with other Google services and products, potentially enhancing functionalities across various platforms.
    
    
    **Different Versions/Sizes:**
    
    Google has announced different sizes or versions of Gemini, each with varying capabilities and computational resources needed to run them.  This is similar to how different GPT models (like GPT-3.5, GPT-4) exist with different strengths and weaknesses.  Larger models generally possess more capabilities but require more computational power.
    
    **Applications:**
    
    The potential applications of Gemini are vast and span various fields:
    
    * **Chatbots and Conversational AI:**  Creating more natural and engaging conversational experiences.
    * **Code Generation and Assistance:**  Helping developers write and debug code more efficiently.
    * **Content Creation:**  Assisting in writing, translating, and summarizing text.
    * **Image Understanding and Analysis:**  Analyzing images and extracting information from them.
    * **Scientific Research:**  Potentially accelerating research in various fields by analyzing large datasets.
    
    
    **Limitations:**
    
    Despite its advancements, Gemini, like all large language models, has limitations:
    
    * **Bias and Safety Concerns:**  Large language models can inherit biases present in the data they are trained on.  Addressing these biases and ensuring safe and responsible use is crucial.
    * **Hallucinations:**  The model might sometimes generate incorrect or nonsensical information, a phenomenon known as "hallucination."
    * **Computational Cost:**  Running large models like Gemini requires significant computational resources, making it expensive to deploy and maintain.
    
    
    **In summary:**
    
    Gemini represents a significant advancement in AI technology, offering a multimodal approach with improved reasoning capabilities.  Its potential applications are vast, but it's important to be aware of its limitations and the ongoing challenges in developing and deploying responsible AI.  Google's ongoing development and refinements of Gemini will likely further expand its capabilities and address its limitations over time.
    
</pre>

## Streaming and Batching

```ChatGoogleGenerativeAI``` natively supports streaming and batching. Below is an example.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Use the llm.batch() method to request multiple queries at once
for chunk in llm.stream("Can you recommend 5 travel destinations in California?"):
    print(chunk.content)
    print("---")
```

<pre class="custom">California
    ---
     offers a diverse range of experiences. Here are 5 travel destinations, each offering
    ---
     something different:
    
    1. **Yosemite National Park (Nature & Hiking
    ---
    ):** Iconic granite cliffs, giant sequoia trees, waterfalls, and challenging hikes make this a classic California experience.  Best for those who enjoy outdoor activities and
    ---
     stunning natural beauty.
    
    2. **San Francisco (City & Culture):** A vibrant city with iconic landmarks like the Golden Gate Bridge, Alcatraz Island,
    ---
     and cable cars. Offers a blend of history, culture, diverse food scenes, and a bustling atmosphere.
    
    3. **Santa Barbara (Beach & Relaxation):**  A charming coastal city with beautiful beaches, Spanish architecture, and a relaxed
    ---
     atmosphere. Perfect for those seeking a more laid-back vacation with opportunities for swimming, sunbathing, and exploring a quaint town.
    
    4. **Joshua Tree National Park (Desert & Stargazing):** Unique desert landscape with bizarre rock
    ---
     formations, Joshua trees, and incredible stargazing opportunities. Ideal for those who appreciate unusual scenery and enjoy hiking and photography.
    
    5. **Napa Valley (Wine Country & Gastronomy):** World-renowned wine region with rolling vineyards, charming towns, and luxurious resorts. Perfect for wine enthusiasts, foodies,
    ---
     and those seeking a relaxing getaway with opportunities for wine tasting and gourmet dining.
    
    
    These are just suggestions, and the best destination for you will depend on your interests and travel style.
    
    ---
</pre>

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Use the llm.batch() method to request multiple queries at once
results = llm.batch(
    [
        "What is the capital of the United States?",
        "What is the capital of South Korea?",
    ]
)

for res in results:
    print(res.content)
```

<pre class="custom">The capital of the United States is **Washington, D.C.**
    
    The capital of South Korea is **Seoul**.
    
</pre>

## Multimodeal Model

To provide an image, pass a human message with contents of type ```List[dict]```, where each dict contains either an image value (type of ```image_url```) or a text (type of text) value. The value of ```image_url``` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A local file path
- A base64 encoded image (e.g., ```data:image/png;base64,abcd124```)
- A PIL image

```python
# CASE - A pulbic image URL
import requests
from IPython.display import Image

image_url = "https://picsum.photos/seed/picsum/300/300"
content = requests.get(image_url).content
Image(content)
```




    
![jpeg](./img/output_17_0.jpg)
    



```python
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# The content is provided as a list, which can include both text and an image URL object
message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": image_url},
    ]
)

response = llm.invoke([message])
print(response.content)
```

<pre class="custom">That's a picture of the Matterhorn mountain in Switzerland.  The image shows the iconic pyramidal peak covered in snow, set against a dramatic, softly colored sunset or sunrise sky.  The foreground features a gently sloping snow-covered landscape.
    
</pre>

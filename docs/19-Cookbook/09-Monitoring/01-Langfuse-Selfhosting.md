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

# Langfuse Selfhosting

- Author: [JeongGi Park](https://github.com/jeongkpa)
- Design: []()
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb)

## Overview
In this tutorial, you’ll learn how to run Langfuse locally using Docker Compose and integrate it with your LLM-based applications (e.g., those built with LangChain). Langfuse provides comprehensive tracking and observability for:

- Token usage
- Execution time and performance metrics
- Error rates and unexpected behaviors
- Agent and chain interactions

By the end of this tutorial, you’ll have a local Langfuse instance running on http://localhost:3000 and be ready to send tracking events from your application.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Prerequisites](#prerequisites)
- [Clone the Langfuse Repository](#Clone-the-Langfuse-Repository)
- [Start Langfuse with Docker Compose](#Start-Langfuse-with-Docker-Compose)
- [Initial Setup & Usage](#Initial-Setup-&-Usage)
- [Sending Traces from Your Application](#Sending-Traces-from-Your-Application)
- [Managing & Monitoring Your Local Instance](#Managing-&-Monitoring-Your-Local-Instance)
- [Upgrading & Maintenance](#Upgrading-&-Maintenance)
- [Set KEY and run LANGFUSE locally](#set-key-and-run-langfuse-locally)



### References

- [Langfuse DOC](https://langfuse.com/self-hosting/local)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

set environment variables is in .env.

Copy the contents of .env_sample and load it into your .env with the key you set.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```


<pre class="custom">True</pre>


```python
%%capture --no-stderr
!pip install langchain-opentutorial
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
        "langchain-openai",
        "langchain",
        "python-dotenv",
        "langchain-core",
        "langfuse",
        "openai"
    ],
    verbose=False,
    upgrade=False,
)
```

## Prerequisites

1. Docker & Docker Compose

- Docker Desktop (Mac/Windows) or
- Docker Engine & Docker Compose plugin (Linux)

2. Git

- Make sure you can run git clone from your terminal or command prompt.

3. Sufficient System Resources
- Running Langfuse locally requires a few GB of memory available for Docker containers.

## Clone the Langfuse Repository

open your terminal and run the following command:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```
This will download the Langfuse repository to your local machine. and move into the directory.



## Start Langfuse with Docker Compose

Inside the `langfuse` directory, simply run:

```bash
docker compose up
```

### What's happening?
- Docker Compose will pull and start multiple containers for the Langfuse stack:
    - Database (Postgres)
    - Langfuse backend services
    - Langfuse web/UI
- This is the simplest way to run Langfuse locally and explore its features.

### Wait for Ready State
After 2-3 minutes, you should see logs indicating that the 'langfuse-web-1' container is Ready. At that point, open your browser to:

```
http://localhost:3000
```

You will be greeted by the Langfuse login/setup sceen or homepage.


## Initial Setup & Usage

Follow the on-screen prompts to configure your local Langfuse instance(e.g., create an initial admin user). Once the setup completes, you can log in to the Langfuse UI.

From here, you will be able to:
- Create projects to track events from your LLM apps
- Invite team members (if applicable)
- Adjust various settings like API keys, encryption or networking


## Sending Traces from Your Application

To start collecting trace data in your local Langfuse instance, you’ll need to send events from your application code:

1. Obtain your Project Key / API Key from the Langfuse UI.
2. Install the appropriate Langfuse SDK in your application environment (e.g., npm install @langfuse/node for Node.js, or pip install langfuse for Python—if available).
3. Initialize the Langfuse client and pass in the host (pointing to your local instance) and the API key.
4. Record events (e.g., chain steps, LLM requests, agent actions) in your code to send structured data to Langfuse.

(Refer to Langfuse’s official docs for language-specific integration examples.)

## Managing & Monitoring Your Local Instance

Once you’re up and running:

- UI: Go to http://localhost:3000 for the Langfuse dashboard.
- Logs: Monitor the Docker logs in your terminal for any errors or system messages.
- Configuration: Check the configuration guide for advanced setups:
    - Authentication & SSO
    - Encryption
    - Headless Initialization
    - Networking
    - Organization Creators
    - UI Customization

**Stopping/Restarting**

To stop the containers, press Ctrl + C in the same terminal window running Docker Compose. If you used a detached mode (-d), then:

```bash
docker compose down
```

Add the -v flag to remove volumes as well (which also deletes stored data).


## Upgrading & Maintenance

To upgrade to the latest Langfuse version, simply:

```bash
docker compose down
docker compose up --pull always
```

This will pull the newest images and restart Langfuse. Refer to the upgrade guide for more detailed steps.


## Set KEY and run LANGFUSE locally

You can try out langfuse by running the code below.


```python
import os
 
# get keys for your project from https://cloud.langfuse.com
# os.environ["LANGFUSE_PUBLIC_KEY"] = "YOUR_LANGFUSE_PUBLIC_KEY"
# os.environ["LANGFUSE_SECRET_KEY"] = "YOUR_LANGFUSE_SECRET_KEY"

 
# Your host, defaults to https://cloud.langfuse.com
# For US data region, set to "https://us.cloud.langfuse.com"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
```

```python
from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
 
@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        messages=[
          {"role": "system", "content": "You are a great storyteller."},
          {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
    ).choices[0].message.content
 
@observe()
def main():
    return story()
 
main()
```




<pre class="custom">"there existed a peaceful planet called Zoraxia, home to a diverse array of alien species living in harmony. The inhabitants of Zoraxia were known for their love of music and had created a beautiful symphony that could be heard throughout the entire galaxy.\n\nHowever, one fateful day, a dark shadow fell upon Zoraxia as an evil sorcerer named Malakar arrived with his army of shadowy creatures. Malakar sought to harness the power of the planet's music"</pre>



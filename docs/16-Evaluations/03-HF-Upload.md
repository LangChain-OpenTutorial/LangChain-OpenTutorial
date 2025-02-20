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

# HF-Upload

- Author: [Sun Hyoung Lee](https://github.com/LEE1026icarus)
- Design: 
- Peer Review : 
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embeeding/04-UpstageEmbeddings.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embeeding/04-UpstageEmbeddings.ipynb)

## Overview

The process involves loading a local CSV file, converting it to a HuggingFace Dataset format, and uploading it to the Hugging Face Hub as a private dataset. This process allows for easy sharing and access of the dataset through the HuggingFace infrastructure.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Upload Generated Dataset](#upload-generated-dataset)
- [Upload to HuggingFace Dataset](#upload-to-huggingface-dataset)


### References
- [Huggingface / Share a dataset to the Hub](https://huggingface.co/docs/datasets/upload_dataset)
---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

 **[Note]** 
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

### API Key Configuration
To use `HuggingFace Dataset` , you need to [obtain a HuggingFace write token](https://huggingface.co/settings/tokens).

Once you have your API key, set it as the value for the variable `HUGGINGFACEHUB_API_TOKEN` .

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    ["datasets"],
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
if not load_dotenv(override=True):
    set_env(
        {
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "", # set the project name same as the title
            "HUGGINGFACEHUB_API_TOKEN": "",
        }
    )

```

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Upload Generated Dataset
Import the pandas library for data upload

```python
import pandas as pd

df = pd.read_csv("data/ragas_synthetic_dataset.csv")
df.head()
```




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
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wht is an API?</td>
      <td>["Agents\nThis combination of reasoning,\nlogi...</td>
      <td>An API can be used by a model to make various ...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What are the three essential components in an ...</td>
      <td>['Agents\nWhat is an agent?\nIn its most funda...</td>
      <td>The three essential components in an agent's c...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What Chain-of-Thought do in agent model, how i...</td>
      <td>['Agents\nFigure 1. General agent architecture...</td>
      <td>Chain-of-Thought is a reasoning and logic fram...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Waht is the DELETE method used for?</td>
      <td>['Agents\nThe tools\nFoundational models, desp...</td>
      <td>The DELETE method is a common web API method t...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do foundational components contribute to t...</td>
      <td>['&lt;1-hop&gt;\n\nAgents\ncombining specialized age...</td>
      <td>Foundational components contribute to the cogn...</td>
      <td>NewMultiHopQuery</td>
    </tr>
  </tbody>
</table>
</div>



## Upload to HuggingFace Dataset
Convert a Pandas DataFrame to a Hugging Face Dataset and proceed with the upload.

```python
from datasets import Dataset

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Check the dataset
print(dataset)
```

<pre class="custom">Dataset({
        features: ['user_input', 'reference_contexts', 'reference', 'synthesizer_name'],
        num_rows: 10
    })
</pre>

```python
from datasets import Dataset
import os

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Set dataset name (change to your desired name)
hf_username = "icarus1026"  # Your Hugging Face Username(ID)
dataset_name = f"{hf_username}/rag-synthetic-dataset"

# Upload dataset
dataset.push_to_hub(
    dataset_name,
    private=True,  # Set private=False for a public dataset
    split="test_v1",  # Enter dataset split name
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
```


<pre class="custom">Pushing dataset shards to the dataset hub:   0%|          | 0/1 [00:00<?, ?it/s]</pre>



    Creating parquet from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]


[Note] The Dataset Viewer may take some time to display.


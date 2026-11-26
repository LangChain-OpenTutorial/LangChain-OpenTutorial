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

# Video Q&A LLM (Gemini)

- Author: [Youngin Kim](https://github.com/Normalist-K)
- Design: [Teddy](https://github.com/teddylee777)
- Peer Review :
- Proofread : [frimer](https://github.com/brian604)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/12-Gemini-Video.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/12-Gemini-Video.ipynb)
## Overview

This tutorial demonstrates how to use the ```Gemini API``` to process and analyze video content. 

Specifically, it shows how to **upload a video file** using ```File API```, and then use a generative model to extract descriptive information about the video. 

The workflow utilizes the ```gemini-1.5-flash``` model to generate a text-based description of a given video clip.

Additionally, it provides an example of Integrating the Gemini Model into a LangChain Workflow for Video Data, showcasing how to build a chain that processes and analyzes video content seamlessly within the LangChain framework.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Upload and Preprocess video using Gemini API](#upload-and-preprocess-video-using-gemini-api)
- [Generate content (Gemini API)](#generate-content-gemini-api)
- [Integrating the Gemini Model into a LangChain Workflow for Video Data](#integrating-the-gemini-model-into-a-langchain-workflow-for-video-data)
- [File Deletion](#file-deletion)

### References

- [Gemini API (Cookbook) - Video](https://ai.google.dev/api/generate-content#video)
- [LangChain Components - google_generative_ai](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
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
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_google_genai",
        "google-generativeai",
    ],
    verbose=False,
    upgrade=False,
)
```

**API KEY Issuance**

- Obtain an API KEY from the [link](https://makersuite.google.com/app/apikey?hl=en).

**Important:**

- The ```File API``` used in this tutorial requires ```API keys``` for authentication and access.

- Uploaded files are linked to the cloud project associated with the ```API key```.

Unlike other ```Gemini APIs```, the ```API key``` also grants access to data uploaded via the ```File API```, so it's crucial to store the ```API key``` securely.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "GOOGLE_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Video-Q&A-LLM-Gemini",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Data preparation

license free video from pexels
- author: [SwissHumanity Stories](https://www.pexels.com/ko-kr/@swisshumanity-stories-1686058/)
- link: [SwissHumanity Stories's pexels](https://www.pexels.com/video/drone-footage-of-a-train-traveling-on-a-valley-8290926/)

Please download the video and copy it to the ```./data``` folder for the tutorial

```python
# Set video file name
video_path = "data/sample-video.mp4"
```

## Upload and Preprocess video using Gemini API

Next, use the File API to upload the video file.

```python
import google.generativeai as genai

print("Uploading files...")

# Upload the file and return the file object
video_file = genai.upload_file(path=video_path)

print(f"Upload complete: {video_file.uri}")
```

<pre class="custom">Uploading files...
    Upload complete: https://generativelanguage.googleapis.com/v1beta/files/ycq94nkeb9gd
</pre>

After uploading the file, you can call ```get_file``` to verify if the API has successfully processed the file.

```get_file``` allows you to check the uploaded file associated with the File API in the cloud project linked to the API key.

```python
import time

# Videos need to be processed before you can use them.
while video_file.state.name == "PROCESSING":
    print("Please wait while the video upload and preprocessing are completed...")
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

# Raise an exception if the processing fails
if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)

# Print completion message
print(
    f"\nVideo processing is complete!\nYou can now start the conversation: {video_file.uri}"
)
```

<pre class="custom">Please wait while the video upload and preprocessing are completed...
    
    Video processing is complete!
    You can now start the conversation: https://generativelanguage.googleapis.com/v1beta/files/ycq94nkeb9gd
</pre>

## Generate content (Gemini API)

After the video is preprocessed, you can use the ```generate_content``` function from Gemini API to request questions about the video.

```python
# Prompt message
prompt = "Describe this video clip"

# Set model to Gemini 1.5 Flash
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# request response to LLM
response = model.generate_content(
    [prompt, video_file], request_options={"timeout": 600}
)
# print response
print(response.text)
```

<pre class="custom">Here's a description of the video clip:
    
    The video shows an aerial, high-angle view of a red passenger train traveling along a railway line that runs parallel to a road through a picturesque valley. 
    
    
    Here's a breakdown of the scene:
    
    * **The Train:** A long, red passenger train is the central focus, moving from the bottom to the middle of the frame.  It's a fairly modern-looking train.
    
    * **The Valley:** The valley is lush green, with fields dotted with yellow wildflowers (likely dandelions).  The grass is vibrant and appears to be well-maintained pastureland. Several farmhouses and buildings are scattered throughout the valley. A small stream or river meanders alongside the road and tracks.
    
    * **The Mountains:** Towering mountains, partially snow-capped, form a dramatic backdrop. The mountains are steep and rocky, showcasing a mix of textures and shades of green and grey.
    
    * **The Atmosphere:** The overall atmosphere is peaceful and idyllic, with clear blue skies and abundant sunlight suggesting a pleasant spring or summer day.
    
    The video appears to be drone footage, smoothly following the train's progress through the valley. The camera angle provides a sweeping perspective that showcases the beauty of the landscape and the integration of the train within the environment. The entire scene evokes a sense of serene beauty and the charm of rural Switzerland.
    
</pre>

Below is an example of stream output (with the ```stream=True``` option added).

```python
# Prompt message
prompt = "What type of train is shown in this video, and what color is it?"

# Set model to Gemini 1.5 Flash
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# request stream response to LLM
response = model.generate_content(
    [prompt, video_file], request_options={"timeout": 600}, stream=True
)

# print stream response
for chunk in response:
    print(chunk.text, end="", flush=True)
```

<pre class="custom">That's a narrow-gauge railway train.  More specifically, it appears to be a type of railcar used on the  Appenzell Bahn (AB) in Switzerland.  The train is primarily red in color, with some black and white accents.
</pre>

## Integrating the Gemini Model into a LangChain Workflow for Video Data

Here is an example of using LangChain with the Gemini model.

The model is loaded via ```ChatGoogleGenerativeAI``` from ```langchain_google_genai```, allowing multimodal data to be included in the content of ```HumanMessage``` using the ```media``` type.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize the Gemini model with the specified version
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create a message to send to the model and attach the video file as media input
message = HumanMessage(
    content=[
        {"type": "text", "text": "Please analyze the content of this video."},
        {
            "type": "media",
            "mime_type": video_file.mime_type,
            "file_uri": video_file.uri,
        },
    ]
)

# Stream the response and process each chunk
for chunk in llm.stream([message]):
    print(chunk.content)
```

<pre class="custom">This
     video shows an aerial view of a red train traveling along a railway line that runs
     parallel to a road through a picturesque valley in what appears to be the Swiss Alps
    . 
    
    
    Here's a breakdown of the content:
    
    * **Scenery:** The valley is lush green, with fields dotted with yellow wildflowers, likely
     dandelions.  The valley is surrounded by steep, verdant hillsides and majestic snow-capped mountains in the background. A small stream or river runs
     alongside the road and railway.  Several farmhouses or chalets are scattered throughout the valley.  The overall impression is one of idyllic rural Switzerland.
    
    * **Transportation:** A long red train is the central focus, moving steadily along the
     railway tracks. The train appears to be a passenger train, given its length and the typical design.  A road runs parallel to the tracks, offering a contrasting mode of transportation.
    
    * **Camera Work:** The video is shot from a
     drone, providing a high-angle, sweeping perspective. The camera follows the train as it moves through the valley, giving viewers a sense of the scale and beauty of the landscape. The drone maintains a relatively constant distance and speed to follow the train.
    
    * **Overall Impression:** The video is visually stunning and evokes a
     sense of tranquility and the beauty of nature in a mountainous region.  It's a perfect example of promotional footage for tourism, showcasing Switzerland's landscapes and transportation infrastructure.
    
</pre>

## File Deletion

Files are automatically deleted after 2 days, or you can manually delete them using files.delete().

```python
# File deletion
genai.delete_file(video_file.name)

print(f"The video has been deleted: {video_file.uri}")
```

<pre class="custom">The video has been deleted: https://generativelanguage.googleapis.com/v1beta/files/ycq94nkeb9gd
</pre>

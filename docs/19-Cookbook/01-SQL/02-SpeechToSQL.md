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

# SpeechToSQL 
- Author: [Dooil Kwak](https://github.com/back2zion)
- Design: -
- Peer Review : [Ilgyun Jeong](https://github.com/johnny9210), [Jaehun Choi](https://github.com/ash-hun)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/01-SQL/02-SpeechToSQL.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/01-SQL/02-SpeechToSQL.ipynb)


## Overview

The Speech to SQL system is a powerful tool that converts spoken language into SQL queries. It combines advanced speech recognition with natural language processing to enable hands-free database interactions.

**Key Features**:

- **Real-time Speech Processing**: 
  Captures and processes voice input in real-time, supporting various microphone configurations.

- **Accurate Speech Recognition**: 
  Uses Whisper model for reliable speech-to-text conversion with support for clear English queries.

- **SQL Query Generation**: 
  Transforms natural language questions into properly formatted SQL queries.

**System Requirements**:
- Python 3.8 or higher
- Working microphone

### Table of Contents 

- [Overview](#overview)
- [Installation and Setup](#installation-and-setup)
- [Audio Device Configuration](#audio-device-configuration)
- [Speech Recognition Setup](#speech-recognition-setup)
- [Basic Usage](#basic-usage)
- [Advanced Usage and Troubleshooting](#advanced-usage-and-troubleshooting)

### References
- [Faster Whisper Documentation > Python API Reference](https://github.com/guillaumekln/faster-whisper)
- [SoundDevice Documentation > Python API Reference](https://python-sounddevice.readthedocs.io/en/0.4.6/)
- [Wavio Documentation > Audio File Handling](https://github.com/WarrenWeckesser/wavio)
- [NumPy Documentation > Audio Processing](https://numpy.org/doc/stable/reference/routines.html#audio-processing)

## Installation and Setup

Before we begin, let's install all necessary packages. This tutorial requires several Python packages for speech processing, SQL operations, and machine learning:

1. LangChain Components:
   - `langchain-community`: Core LangChain functionality and community components
   - `langchain-openai`: OpenAI integration
   - `langchain-core`: Essential LangChain components

2. Database and API:
   - `openai`: For OpenAI API access
   - `sqlalchemy`: For database operations
   - `python-dotenv`: For environment variable management
   - `torch`: For faster-whisper

3. Audio Processing:
   - `sounddevice`: For audio capture
   - `numpy`: For data processing
   - `wavio`: For audio file handling
   - `faster-whisper`: For speech recognition

4. Additional dependencies:
   - `blosc2`: For data compression
   - `cython`: For Python-C integration
   - `black`: For code formatting
   
After running the installation cell, you may need to restart the kernel for the changes to take effect. We'll verify the installation in the next step.   

### Windows Users: Important Note
If you encounter a permission error during installation such as "Access is denied", you have two options:

1. Use the `--user` option with pip (recommended):
   - This installs packages in your user directory, avoiding permission issues
   - We've already included this option in the installation command

2. Alternative: Run Jupyter as Administrator:
   - Only if the first option doesn't work
   - Right-click on Jupyter Notebook
   - Select "Run as administrator"
   - Then try the installation again

After installation, you'll need to restart the kernel regardless of which method you use.

### Verification
After installation and kernel restart, run the verification cell below to ensure everything is set up correctly:

Run the following cells to install all required packages:

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages with compatible versions
# -*- coding: utf-8 -*-
import subprocess
import sys

def install_packages():
   packages = [
       'langchain-core>=0.3.29,<0.4.0',
       'langchain-community==0.0.24',
       'langchain-openai==0.0.5', 
       'openai==1.12.0',
       'sqlalchemy==2.0.27',
       'python-dotenv==1.0.1',
       'sounddevice==0.4.6',
       'numpy==1.24.3',
       'wavio==0.0.8',
       'faster-whisper==0.10.0',
       'blosc2~=2.0.0',
       'cython>=0.29.21',
       'black>=22.3.0'
   ]
   
   for package in packages:
       try:
           subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
       except subprocess.CalledProcessError:
           print(f"Failed to install {package}")
           continue
   
   print("✓ Installation complete!")

install_packages()
```

<pre class="custom">✓ Installation complete!
</pre>

### Important Note About Package Installation
After running the installation cell, you might see messages like:
This is normal! Here's what you need to do:

'Note: you may need to restart the kernel to use updated packages.'

1. First, look for the "✓ All packages installed successfully!" message to confirm the installation worked
2. Then, restart the Jupyter kernel to ensure all packages are properly loaded:
   - Click on the "Kernel" menu at the top
   - Select "Restart Kernel..."
   - Click "Restart" when prompted

After restarting the kernel, run the following verification cell to make sure everything is set up correctly:

Now let's verify that everything is ready to use:

```python
try:
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    print("✓ All set! Let's move on to the next step.")
except ImportError as e:
    print(f"✗ Something's missing. Please try running the installation command again.")
```

<pre class="custom">✓ All set! Let's move on to the next step.
</pre>

### Verifying Package Installation

After installing the packages and restarting the kernel, let's verify that everything is set up correctly.

If you see any ✗ marks, it means that package wasn't installed correctly. Try these steps:
1. Run the installation cell again
2. Restart the kernel
3. Run the verification cell again

If you still see errors, make sure you have sufficient permissions and a stable internet connection.

```python
# Import necessary libraries
import sounddevice as sd
import numpy as np
import wavio
import os
import time
from faster_whisper import WhisperModel
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Audio Device Configuration

A crucial first step is selecting the correct audio input device. Let's identify and configure your system's microphone.

**Note**: You'll see a filtered list of input devices only, making it easier to choose the correct microphone.

```python
def list_audio_input_devices():
    """Display only audio input devices with clear formatting."""
    print("\nAvailable Audio Input Devices:")
    print("=" * 50)
    
    input_devices = []
    for idx, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:  # Only show input devices
            # Skip duplicate devices (different APIs)
            device_name = device['name'].split(',')[0]  # Remove API information
            if not any(d['name'].startswith(device_name) for d in input_devices):
                input_devices.append({
                    'index': idx,
                    'name': device_name,
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
                
                print(f"Device {idx}: {device_name}")
                print(f"  Channels: {device['max_input_channels']}")
                print(f"  Sample Rate: {device['default_samplerate']}Hz")
                print("-" * 50)
    
    return input_devices

# List available input devices
input_devices = list_audio_input_devices()
```

<pre class="custom">
    Available Audio Input Devices:
    ==================================================
    Device 0: Microsoft 사운드 매퍼 - Input
      Channels: 2
      Sample Rate: 44100.0Hz
    --------------------------------------------------
    Device 1: 마이크 배열(디지털 마이크용 인텔® 스마트 사운드 기술)
      Channels: 2
      Sample Rate: 44100.0Hz
    --------------------------------------------------
    Device 4: 주 사운드 캡처 드라이버
      Channels: 2
      Sample Rate: 44100.0Hz
    --------------------------------------------------
    Device 8: Realtek ASIO
      Channels: 2
      Sample Rate: 44100.0Hz
    --------------------------------------------------
    Device 13: PC Speaker (Realtek HD Audio output with SST)
      Channels: 2
      Sample Rate: 48000.0Hz
    --------------------------------------------------
    Device 14: Input 1 (Realtek HD Audio Mic input with SST)
      Channels: 2
      Sample Rate: 48000.0Hz
    --------------------------------------------------
    Device 15: Input 2 (Realtek HD Audio Mic input with SST)
      Channels: 4
      Sample Rate: 16000.0Hz
    --------------------------------------------------
    Device 16: Stereo Mix (Realtek HD Audio Stereo input)
      Channels: 2
      Sample Rate: 48000.0Hz
    --------------------------------------------------
    Device 18: Headset (@System32\drivers\bthhfenum.sys
      Channels: 1
      Sample Rate: 8000.0Hz
    --------------------------------------------------
    Device 19: Microphone Array 1 ()
      Channels: 2
      Sample Rate: 48000.0Hz
    --------------------------------------------------
    Device 20: Microphone Array 2 ()
      Channels: 4
      Sample Rate: 16000.0Hz
    --------------------------------------------------
    Device 22: Headset Microphone (@System32\drivers\bthhfenum.sys
      Channels: 1
      Sample Rate: 8000.0Hz
    --------------------------------------------------
</pre>

```python
def test_audio_device(device_index, duration=1):
    """
    Test if an audio device works properly.
    Args:
        device_index (int): The index of the device to test
        duration (float): Test duration in seconds
    Returns:
        bool: True if device works, False otherwise
    """
    try:
        print(f"Testing audio device {device_index}...")
        with sd.InputStream(device=device_index, channels=1, samplerate=16000):
            print("✓ Device initialized successfully")
            return True
    except Exception as e:
        print(f"✗ Device test failed: {str(e)}")
        return False
```

### Audio Device Selection and Testing

After viewing the available devices above, you'll need to select and test your microphone. Choose a device with input channels (marked as "Channels: X" where X > 0).

**Important Tips**:
- Choose a device with clear device name (avoid generic names like "Default Input")
- Prefer devices with 1 or 2 input channels
- If using a USB microphone, make sure it's properly connected
- Test the device before proceeding to actual recording

```python
# Let's test the first available input device as default
if input_devices:
    default_device = input_devices[0]
    print(f"\nTesting default device: {default_device['name']}")
    if test_audio_device(default_device['index']):
        # Set as default device
        os.environ['DEFAULT_DEVICE'] = str(default_device['index'])
        os.environ['SAMPLE_RATE'] = str(int(default_device['sample_rate']))
        print(f"\nDefault device set to: {default_device['name']}")
        print(f"Sample rate: {default_device['sample_rate']}Hz")
    else:
        print("\nPlease select a different device and try again.")
else:
    print("\nNo input devices found. Please check your microphone connection.")
```

<pre class="custom">
    Testing default device: Microsoft 사운드 매퍼 - Input
    Testing audio device 0...
    ✓ Device initialized successfully
    
    Default device set to: Microsoft 사운드 매퍼 - Input
    Sample rate: 44100.0Hz
</pre>

## Speech Recognition Setup

Now let's set up the speech recognition component using the Whisper model. 

**Note**: The first time you run this, it will download the Whisper model. This might take a few minutes depending on your internet connection.

```python
def initialize_whisper():
    """Initialize the Whisper model."""
    try:
        # Initialize Whisper model with base configuration
        model = WhisperModel(
            model_size_or_path="base",  # Using 'base' model for faster CPU processing
            device="cpu",
            compute_type="int8"  # Optimized for CPU
        )
        print("✓ Whisper model initialized successfully")
        return model
    except Exception as e:
        print(f"✗ Error initializing Whisper model: {str(e)}")
        print("Please make sure all packages are installed correctly.")
        return None

model = initialize_whisper()
```

<pre class="custom">✓ Whisper model initialized successfully
</pre>

## Basic Usage

Let's implement the core components for speech-to-SQL conversion. We'll create a robust system that can:
1. Record audio from your microphone
2. Convert speech to text
3. Transform the text into SQL queries

### **Step 1: Record Audio from Your Microphone**

The `AudioRecorder` class records audio input from the user's microphone and saves it as a temporary audio file.

```python
# 1. Record audio from your microphone
import sounddevice as sd
import numpy as np
import wavio
import tempfile

class AudioRecorder:
    def __init__(self):
        self._samplerate = 16000
        self.audio_data = []
        self.recording = False
        self.stream = None

    def start_recording(self, device_id=0):
        """Start recording audio"""
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self._samplerate,
                callback=self._audio_callback,
                device=device_id
            )
            self.audio_data = []
            self.recording = True
            self.stream.start()
            print("Recording started. Speak now!")
            return True
        except Exception as e:
            print(f"Recording failed: {str(e)}")
            return False

    def _audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def stop_and_process(self):
        """Stop recording and save audio data"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            if len(self.audio_data) > 0:
                audio = np.concatenate(self.audio_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    wavio.write(tmpfile.name, audio, self._samplerate, sampwidth=2)
                return tmpfile.name
        return None
```

### **Step 2: Convert Speech to Text**

We use the Whisper model for accurate transcription of recorded audio into text.

```python
# 2. Convert speech to text

from faster_whisper import WhisperModel

def initialize_whisper():
    """Initialize the Whisper model with English language setting"""
    return WhisperModel("base", device="cpu", compute_type="int8")

class AudioProcessor:
    def __init__(self, model):
        self.model = model

    def transcribe_audio(self, audio_file):
        """Transcribe audio to text using Whisper with English language enforcement"""
        try:
            segments, _ = self.model.transcribe(audio_file, language="en")  
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            return None

```

### **Step 3: Transform Text into SQL Queries**

We use the LangChain library to transform natural language text into SQL queries.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage  
from langchain_openai import ChatOpenAI

class SQLQueryGenerator:
   def __init__(self):
       self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
       self.template = ChatPromptTemplate.from_messages([
           ("system", "You are an SQL query generator."),
           ("human", "{query_text}")
       ])

   def generate_sql(self, query_text):
       try:
           prompt = self.template.format_messages(query_text=query_text)
           result = self.llm.invoke(prompt)
           return result.content
       except Exception as e:
           return f"Error: {str(e)}"
```

### **Step 4: Putting It All Together**

Finally, we combine all the components into a single process that listens for audio input, transcribes it, and generates an SQL query.

```python
import time

def process_speech_to_sql(duration=5):
    """Main function for speech-to-SQL conversion"""
    print("\n=== Starting Speech to SQL Process ===")
    print("Recording will start in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Step 1: Record Audio
    recorder = AudioRecorder()
    if recorder.start_recording():
        print("\nSpeak your query now... (10 seconds)")
        time.sleep(duration)
        audio_file = recorder.stop_and_process()
        print(f"Saved audio file: {audio_file}")

        # Step 2: Speech-to-Text
        if audio_file:
            model = initialize_whisper()
            processor = AudioProcessor(model)
            print("Processing audio...")
            text = processor.transcribe_audio(audio_file)
            print(f"Transcribed Text: {text}")

            # Step 3: SQL Query Generation
            sql_generator = SQLQueryGenerator()
            sql_query = sql_generator.generate_sql(text)
            print(f"Generated SQL Query: {sql_query}")
            return sql_query
```

Let's try it out! Run this command to start recording:

```python
query_text = process_speech_to_sql(duration=10)  # 10 seconds recording
```

<pre class="custom">
    === Starting Speech to SQL Process ===
    Recording will start in:
    3...
    2...
    1...
    Recording started. Speak now!
    
    Speak your query now... (10 seconds)
    Saved audio file: C:\Users\rhkre\AppData\Local\Temp\tmp47m8xvlx.wav
    Processing audio...
    Transcribed Text:  Find Top 10 Customers by Revenue
    Generated SQL Query: To find the top 10 customers by revenue, you would typically need a table that contains customer information and a table that records transactions or orders, including the revenue generated by each transaction. Assuming you have a `customers` table and an `orders` table, where the `orders` table includes a `customer_id` and a `revenue` column, you can use the following SQL query:
    
    ```sql
    SELECT c.customer_id, c.customer_name, SUM(o.revenue) AS total_revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name
    ORDER BY total_revenue DESC
    LIMIT 10;
    ```
    
    This query does the following:
    - Joins the `customers` table with the `orders` table on the `customer_id`.
    - Groups the results by each customer to calculate the total revenue for each customer.
    - Orders the results in descending order based on the total revenue.
    - Limits the results to the top 10 customers. 
    
    Make sure to replace `customer_name` and `revenue` with the actual column names used in your database schema if they differ.
</pre>

## Example Queries

Here are some example queries you can try with the system:

1. "Show sales figures for the last quarter"
2. "Find top 10 customers by revenue"
3. "List all products with inventory below 100 units"
4. "Calculate total sales by region"
5. "Get employee performance metrics for 2023"

These queries demonstrate the range of SQL operations our system can handle.

## Advanced Usage and Troubleshooting

### Common Issues and Solutions

1. **No audio device found**
   - Check if your microphone is properly connected
   - Try unplugging and reconnecting your microphone
   - Verify microphone permissions in your OS settings

2. **Poor recognition accuracy**
   - Speak clearly and at a moderate pace
   - Minimize background noise
   - Keep the microphone at an appropriate distance

3. **Device initialization errors**
   - Try selecting a different audio device
   - Restart your Python kernel
   - Check if another application is using the microphone

----

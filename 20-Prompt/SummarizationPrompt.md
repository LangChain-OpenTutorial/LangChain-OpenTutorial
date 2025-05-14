# **Summarization Prompt**

## **Description**

This prompt is designed to help you create **concise yet comprehensive** summaries from various types of documents (e.g., transcripts, research papers, articles). By analyzing the provided text, it extracts **key ideas**, **context**, and **vital details**, delivering a coherent overview that emphasizes the **most critical points**.

*   Suitable for:
    *   Research overviews
    *   Meeting or interview transcripts
    *   News articles or blog posts
    *   Academic papers

---

## **Input**

### SYSTEM

You are an expert summarizer. Please read the document below and summarize the most critical information. Focus on core ideas, statistics, or facts that should not be overlooked. Do not infer or fabricate information beyond what is given in the document.

Your summary must adhere to the following key requirements:
- **Include relevant numbers, statistics, or contextual dates**: e.g., "On March 12, revenue increased by 15%..."
- **Ensure coherence**: The summary must read smoothly and maintain a logical flow between points.
- **Avoid excessive repetition**: Do not restate the same point multiple times.
- **Do not omit vital details**: Key facts, critical outcomes, or unique data points should not be lost.
- **Do not invent or assume facts**: Summaries must reflect the source text accurately, without adding new information.

Generate your summary using both of the following formats:
(1) Bullet-point style: Highlight key points clearly and concisely.
(2) Concise executive summary: Summarize the document in 1–5 sentences.

### **HUMAN**

[Paste your text here, in paragraphs or bullet points if relevant]

> **Note**: For very long texts, consider splitting the content into manageable chunks (e.g., ~500 words each) with an overlap of ~50 words to maintain context (see “Tools” section below). After summarizing each chunk, you can merge partial summaries into one final summary, ensuring no duplication and a coherent flow.

---

## **Output**

Generate a summary in **one** of the following two formats:

1.  **Bullet-point format**
    *   Use either numbered or bulleted lists.
    *   Highlight critical insights, figures, or statistics.

2.  **Concise executive summary (1-5 sentences)**
    *   Capture the essence of the original text in a short paragraph.
    *   Avoid missing any pivotal data or context.

---

## **Example Output**

### **Option 1: Key Points (Bullet-Point Summary)**

1.  **First critical point** — Introduce the main argument or data.
2.  **Second critical point** — Show specific numbers or research findings.
3.  **Third critical point** — Provide additional context or outcome.

> **Example**:
>
> 1.  The Transformer architecture processes tokens in parallel, reducing training times significantly.
> 2.  Neural network parameters reached 100M in the latest iteration, a 20% increase from the previous version.
> 3.  The method demonstrated a 5% improvement in accuracy on the benchmark dataset.

### **Option 2: Executive Summary (1-5 sentences)**

> "The primary goal of this study was to evaluate the Transformer’s efficiency in processing sequence data. By removing recurrent operations and relying on self-attention, the model significantly reduced training time while improving accuracy metrics. This approach could pave the way for more scalable and parallelizable architectures in natural language processing."

---

## **Final Checks**

### **1. Merge Partial Summaries** (if chunking was used):
*   Combine the individual chunk summaries into one final overview.
*   Remove any duplicate information and ensure a smooth flow from one point to the next.

### **2. Verify Essential Details:**
*   Make sure all important data, statistics, or dates from the original source are preserved.
*   Check that the summary contains no contradictions or fabricated details.

### **3. Language & Style Consistency:**
*   If multiple sources or languages are involved, maintain consistent tone and language usage as required.

---

## **Tool (Optional)**

### **1. Document Loader**

*   **Description**: Loads a document or transcript from a specified source.
*   **Input Types**:
    *   **File path** (e.g., `"./data/document.txt"`)
    *   **URL** (e.g., `"https://example.com/document"`)
    *   **Direct text input** (as a string or text buffer)

```python
# Usage Example:
loader = DocumentLoader("path/to/file.txt")
text_content = loader.load()

loader = DocumentLoader("https://example.com/document")
web_content = loader.load()

loader = DocumentLoader("Raw text content here...")
raw_text_content = loader.load()
```

### **2. Text Chunking**
*   **Input Types**:
    *  JSON configuration for `chunkSize` (number of words per chunk)
    *  JSON configuration for `overlapSize` (number of overlapping words)

```json
{
  "chunk_size": {
    "unit": "words",
    "value": 500
  },
  "overlap_size": {
    "unit": "words",
    "value": 50
  },
  "min_chunk_size": {
    "unit": "words",
    "value": 100
  },
  "encoding": "utf-8"
}
```

```python
# Initialize chunking assistant with configuration
chunking_assistant = ChunkingAssistant(
    chunk_size=500,
    overlap_size=50,
    min_chunk_size=100
)

# Process text and get chunks
try:
    chunked_texts = chunking_assistant.process(text)
    
    # Process each chunk
    summaries = []
    for chunk in chunked_texts:
        summary = summarize_chunk(chunk)  # Implement your summarization logic
        summaries.append(summary)
    
    # Combine summaries
    final_summary = combine_summaries(summaries)
    
except ValueError as e:
    print(f"Error during chunking: {e}")
```
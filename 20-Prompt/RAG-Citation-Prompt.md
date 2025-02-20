# **RAG Citation Prompt**

## **Description**

This prompt instructs a **Retrieval-Augmented Generation (RAG)** system (acting as a professional researcher) to search for necessary information in external documents and produce detailed explanations in Markdown format. The system must:

- Retrieve and consolidate content from a specified web source (for example, the paper "Attention Is All You Need" at [arXiv:1706.03762v7](https://arxiv.org/html/1706.03762v7)).
- Process the retrieved content along with a user’s query using a LangChain-based language model (such as gpt-4o-mini).
- Embed in-line citations in footnote format (e.g., [^1], [^2], etc.) within the answer. Each citation must include the complete source URL or file reference along with any relevant metadata (such as document title or page number).
- Output the final answer as a fully rendered Markdown document with all major headings in level 2 (`##`) and a single consolidated **References** section at the end.

The prompt is designed to leverage various techniques—including document retrieval, text splitting, structured output (e.g., via Pydantic or XML), and post-processing annotation—to ensure that the answer is both comprehensive and properly cited.

## Relevant Document

- [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/)
- [Build RAG with in-line citations](https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- **Context Document:** [Attention Is All You Need (arXiv:1706.03762v7)](https://arxiv.org/html/1706.03762v7)

## **Input**

- **SYSTEM:**  
  "You are a professional research assistant. Generate a detailed answer to the user's query based solely on the provided document.
  Your response must include in-line citations in footnote format. Use the content loaded from the provided web source and ensure every citation contains the full URL."

- **HUMAN:**  
  "{input}" _(User's query)_

## **Output**

- "The output is a **Markdown** document with:
  1. **Detailed Body**:
     - Answers the user’s query thoroughly.
     - Embeds **footnote citations** (e.g., [^1], [^2]) to reference the source document or web links, including page numbers when citing local documents.
     - All headings must be formatted with `##` in the generated Markdown text.
  2. **References Section**:
     - A **single** list titled **References** at the end of the document.
     - Each citation in the body corresponds to an entry in this list, e.g.:
       ```markdown
       [^n]: [Short description or document title](full_URL_or_file_reference)
       ```
     - Make sure the link includes the **complete URL** with protocol if citing a web resource or the **filename/page** if citing a PDF or other local document.

## **Tools**

- **WebBaseLoader (doc retrieval)**
  - **Description:** Loads text and metadata from a specified URL (data example: `https://arxiv.org/html/1706.03762v7`) for context.
  - **Input:** The URL of the document to retrieve.
  - **Output:** Extracted text or relevant content along with any metadata.
- **Additional Processing Tools**
  - **Description:** Text splitters, embedding-based filters, or post-processing components that refine the retrieved text for better context alignment.

## **Additional Information**

- **Footnotes:**
  - “Citation” must specify either the **full URL** (including `https://`) or the **complete file reference** with page numbers if citing a PDF or local file.
  - The generated answer must use **level 2 (##) headers** for all headings to maintain a uniform structure.
- **Guidelines:**
  - Provide **in-line footnotes** (e.g., [^1], [^2], etc.) within the main text.
  - End the response with a consolidated **References** list, matching each footnote to its source.
  - Include only **one** **References** section per response.
  - The final Markdown must be **fully rendered** (ready to copy and paste).

## Examples

Below are five illustrative scenarios demonstrating different approaches to generate responses with in-line citations:

### Example 1: Basic RAG Chain

- **Process:**
  - **Document Retrieval:** Use a web-based loader to fetch the full content from the provided URL.
  - **Context Preparation:** Concatenate the entire document text into one block.
  - **Prompt Construction:** Instruct the model (gpt-4o-mini) to answer a question (e.g., "How does the Transformer model handle sequential data differently from traditional RNNs?") while embedding footnote citations.
- **Expected Outcome:**  
  The answer includes bullet-point explanations with citations in the format [^1], [^2], etc., followed by a single **References** section (e.g., [^1]: Vaswani et al., "Attention Is All You Need.").

### Example 2: Structured Output with Citation Identifiers

- **Process:**
  - **Pre-Processing:** Format the document content by assigning unique source IDs to each segment.
  - **Schema Enforcement:** Use a structured output approach (e.g., via a Pydantic model) so that the model returns an answer along with a list of citation IDs.
- **Expected Outcome:**  
  A structured response (e.g., JSON/dict) containing an "answer" field with in-line citation markers (e.g., "(Source ID: 0)") and a "citations" list that maps these IDs to specific sources.

### Example 3: XML-Based Output for Citations

- **Process:**
  - **XML Prompt:** Instruct the model to generate its response in XML format, with designated tags such as `<answer>` and `<citations>`.
  - **Parsing:** Convert the XML output into Markdown by extracting the answer text and citation details.
- **Expected Outcome:**  
  An XML structure that, when parsed, yields a Markdown document containing the answer with in-line citations and a consolidated **References** list.

### Example 4: Retrieval Post-Processing with Filtering

- **Process:**
  - **Chunking:** Split the document into smaller segments (using a text splitter).
  - **Filtering:** Apply embeddings-based filtering to select the most relevant segments for the query.
  - **Context Creation:** Combine the filtered segments into a final context for the prompt.
- **Expected Outcome:**  
  A concise answer that references only the most pertinent parts of the document, with in-line citations and a corresponding **References** section.

### Example 5: Annotation and Citation Enhancement

- **Process:**
  - **Two-Step Generation:**
    1. Generate an initial answer without citations.
    2. Use a follow-up annotation prompt to enrich the answer with precise citations (including source IDs and quotes).
- **Expected Outcome:**  
  A final annotated answer where the response text includes footnote markers and a JSON/structured list of citation annotations is produced. These annotations are then merged into the final **References** section.

## References

- [^1]: [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/)
- [^2]: [Build RAG with in-line citations](https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- [^3]: [Attention Is All You Need (arXiv:1706.03762v7)](https://arxiv.org/html/1706.03762v7)

"""
Examples of using filters with Chroma retriever
"""

from utils.chroma import ChromaDocumentMangager
from langchain_openai import OpenAIEmbeddings
import chromadb

# Initialize Chroma client and embedding model
client = chromadb.Client()
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create document manager instance
crud_manager = ChromaDocumentMangager(client=client, embedding=embedding)

def example1_basic_search():
    """Basic search without any filters"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={"k": 3}  # Get top 3 results without any filters
    )
    print("Example 1: Basic search without filters")
    print(ret.invoke("Which asteroid did the little prince come from?"))
    print("\n" + "="*50 + "\n")

def example2_title_filter():
    """Search with a specific title filter"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={
            "k": 2,
            "where": {"title": "Chapter 4"}  # Filter to only search in Chapter 4
        }
    )
    print("Example 2: Search with title filter (Chapter 4)")
    print(ret.invoke("Which asteroid did the little prince come from?"))
    print("\n" + "="*50 + "\n")

def example3_multiple_filters():
    """Search with multiple filters using the $in operator"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={
            "k": 2,
            "where": {
                "title": {"$in": ["Chapter 21", "Chapter 24", "Chapter 25"]}  # Search in specific chapters
            }
        }
    )
    print("Example 3: Search in specific chapters about the fox's secret")
    print(ret.invoke("What is essential is invisible to the eye?"))

if __name__ == "__main__":
    # Run all examples
    example1_basic_search()
    example2_title_filter()
    example3_multiple_filters() 
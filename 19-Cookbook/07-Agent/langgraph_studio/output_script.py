import sys
import subprocess

def pip_install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    packages = ['langchain-opentutorial']
    for pkg in packages:
        pip_install(pkg)


# Install required packages
from langchain_opentutorial import package

package.install(
    ["langsmith", "langchain_anthropic", "langgraph", "tavily-python", "kora"],
    verbose=False,
    upgrade=False,
)

# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "ANTHROPIC_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "20-LangGraphStudio-MultiAgent",
    }
)

# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_anthropic import ChatAnthropic

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10, 
)

llm = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Get tavily api key
from tavily import AsyncTavilyClient
# Search
tavily_async_client = AsyncTavilyClient()


from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator
from pydantic import BaseModel

class Person(BaseModel):
    """A class representing a person to research."""

    name: Optional[str] = None
    """The name of the person."""
    company: Optional[str] = None
    """The current company of the person."""
    linkedin: Optional[str] = None
    """The Linkedin URL of the person."""
    email: str
    """The email of the person."""
    role: Optional[str] = None
    """The current title of the person."""

@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research provided by the user."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    # Add default values for required fields
    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
    Questions: list[str] = field(default=None)

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"
    
    combined_notes: str = field(default_factory=str)
    "Consolidated research notes combining all gathered information into a single coherent document."

    project_queries: list[str] = field(default_factory=list)
    "List of search queries specifically focused on finding project-related information."

    company_search_queries: list[str] = field(default=None)
    "List of search queries generated for gathering company-specific information."

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )

class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=True
):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()



import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig

@dataclass(kw_only=True)
class Configuration:
    max_search_queries: int = 3
    max_search_results: int = 3
    max_reflection_steps: int = 0

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            "max_search_queries": configurable.get("max_search_queries", 3),
            "max_search_results": configurable.get("max_search_results", 3),
            "max_reflection_steps": configurable.get("max_reflection_steps", 0),
        }
        return cls(**values)

QUERY_WRITER_PROMPT = """ You are a search query generator tasked with creating targeted search queries to gather specific information about a person.

  Here is the person you are researching: {person}

  Generate at most {max_search_queries} search queries that will help gather the following information:

  <user_notes>
  {user_notes}
  </user_notes>

  Your query should:
  1. Make sure to look up the right name
  2. Use context clues as to the company the person works at (if it isn't concretely provided)
  3. Do not hallucinate search terms that will make you miss the persons profile entirely
  4. Take advantage of the Linkedin URL if it exists, you can include the raw URL in your search query as that will lead you to the correct page guaranteed.

  Create a focused query that will maximize the chances of finding schema-relevant information about the person.
  Remember we are interested in determining their work experience mainly."""


# generate queries
from langchain_core.runnables import RunnableConfig
from typing import cast

def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" LinkedIn URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"

    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


SEARCH_COMPANY_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific information about a company.

  Person and Company Information: {person}

  Additional Context:
  {user_notes}

  Your query should:
  1. Focus on finding detailed company information and recent developments
  2. Look for company products, services, and main business areas
  3. Search for company culture, work environment, and growth trajectory
  4. Include specific company name variations and common abbreviations if known

  Constraints:
  1. Generate at most {max_search_queries} unique and meaningful queries
  2. Do not use generic terms that might dilute search results
  3. Prioritize recent company information (within last 2 years)
  4. Include company name in each query for better targeting

  Create focused queries that will maximize the chances of finding comprehensive company information.
  Remember we are interested in understanding the company's business context, culture, and market position.
"""

import asyncio

async def generate_queries_for_company(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate search queries specifically for the company, 
    possibly merging with user-provided queries (company_queries in state).
    """

    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" LinkedIn URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"


    structured_llm = llm.with_structured_output(Queries)  # Queries: pydantic model for array of strings

    prompt = SEARCH_COMPANY_PROMPT.format(
        person=person_str,
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    llm_result = cast(
        Queries,
        await structured_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Generate company-related search queries now.",
                },
            ]
        )
    )

    generated_queries = [q for q in llm_result.queries]

    return {"company_search_queries": generated_queries}

COMPANY_INFO_PROMPT = """You are doing web research on company.

  The following schema shows the type of information we're interested in:

  You have just scraped website content. Your task is to take clear, organized notes about a company, focusing on topics relevant to our interests.

  <Website contents>
  {content}
  </Website contents>

  Here are any additional notes from the user:
  <user_notes>
  {user_notes}
  </user_notes>

  Please provide detailed research notes that:
  1. Are well-organized and easy to read
  2. Include specific facts, dates, and figures when available
  3. Maintain accuracy of the original content
  4. Note when important information appears to be missing or unclear

  Remember: Just take clear notes that capture all relevant information."""

async def research_company(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """
    Use 'company_search_queries' to search relevant info about the company,
    then produce 'company_notes'.
    """

    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Web search
    search_tasks = []
    for query in state.company_search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    p = COMPANY_INFO_PROMPT.format(
        content=source_str,
        user_notes=state.user_notes,
    )
    result = await llm.ainvoke(p)
    return {"company_notes": str(result.content)}

INFO_PROMPT = """You are doing web research on people, {people}.

  The following schema shows the type of information we're interested in:

  You have just scraped website content. Your task is to take clear, organized notes about a person, focusing on topics relevant to our interests.

  <Website contents>
  {content}
  </Website contents>

  Here are any additional notes from the user:
  <user_notes>
  {user_notes}
  </user_notes>

  Please provide detailed research notes that:
  1. Are well-organized and easy to read
  2. Focus on topics mentioned in the schema
  3. Include specific facts, dates, and figures when available
  4. Maintain accuracy of the original content
  5. Note when important information appears to be missing or unclear

  Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""


async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Web search
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        content=source_str,
        people=state.person,
        user_notes=state.user_notes,
    )
    result = await llm.ainvoke(p)
    return {"completed_notes": [str(result.content)]}


SEARCH_PROJECTS_PROMPT = """You have the following notes about a person:

{completed_notes}

From these notes, extract up to 3 specific search queries that would help us find more detailed information
about projects, case studies, or portfolio items mentioned or implied in the notes.

Generate at most {max_search_queries} search queries that will help gather the following information:

Output them as JSON in the format: {{"queries": ["query1", "query2", ...]}}.
"""

async def extract_project_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """
    Examine `state.completed_notes` to identify potential project-based keywords or queries,
    then return them as a list so we can further search Tavily.
    """
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # 2) We'll ask LLM to parse out project-related queries
    structured_llm = llm.with_structured_output(Queries)  # Re-using your Queries pydantic model

    prompt = SEARCH_PROJECTS_PROMPT.format(
        completed_notes = state.completed_notes,
        max_search_queries = max_search_queries,
    )

    llm_result = await structured_llm.ainvoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please provide the project-related queries now."},
        ]
    )

    # 3) Return the queries
    return {"project_queries": llm_result.queries if llm_result else []}


PROJECT_INFO_PROMPT = """You are doing additional web research on projects by {people}.

Focus on details about important projects they have worked on. Gather:
- project name
- relevant context or goals
- timeline
- major accomplishments
- technologies used
- challenges faced

Website contents:
{content}

Return your notes in a well-structured format that could be used for interview preparation.
"""

async def research_projects(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """
    Execute a multi-step web search focusing on the person's projects,
    guided by the project queries extracted from `completed_notes`.
    """

    # 1) Retrieve the queries from state
    project_queries = getattr(state, "project_queries", [])
    if not project_queries:
        return {"project_notes": "No project queries found. Please run extract_project_queries first."}

    # 2) Set up concurrency
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    search_tasks = []
    for query in project_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # 3) Execute all searches concurrently
    search_results = await asyncio.gather(*search_tasks)

    # 4) Format & deduplicate
    source_str = deduplicate_and_format_sources(
        search_results, max_tokens_per_source=1000, include_raw_content=True
    )

    # 5) Summarize the project details with LLM
    prompt = PROJECT_INFO_PROMPT.format(
        people=state.person,
        content=source_str,
    )
    llm_result = await llm.ainvoke(prompt)

    return {"project_notes": str(llm_result.content)}


COMBINE_ALL_PROMPT = """We have three sets of notes:

1) Person Notes:
{person_notes}

2) Company Notes:
{company_notes}

3) Project Notes:
{project_notes}

Please merge these notes into a single coherent summary. 
Remove any duplicate or conflicting information. 
The result should flow logically, covering person's details, the company background, and key project info.
"""

async def combine_notes(state: OverallState) -> dict[str, Any]:
    """
    Merge the notes from research_person (person_notes) and research_projects (project_notes)
    into a single cohesive note.
    """
    person_notes = getattr(state, "completed_notes", "")
    company_notes = getattr(state, "company_notes", "")
    project_notes = getattr(state, "project_notes", "")

    prompt = COMBINE_ALL_PROMPT.format(
        person_notes=person_notes,
        company_notes=company_notes,
        project_notes=project_notes,
    )
    llm_result = await llm.ainvoke(prompt) 

    return {"combined_notes": str(llm_result.content)}
    

GENERATE_QUESTIONS_PROMPT = """Based on the following combined note:
{combined_notes}

Create a list of interview questions that focus on this person's experiences, 
skills, and the project/product details mentioned. Format them as:
Q1: ...
Q2: ...
Q3: ...
"""

async def generate_questions(state: OverallState) -> dict[str, Any]:
    """
    Use the combined notes to generate a set of interview questions.
    """
    combined_notes = getattr(state, "combined_notes", "")
    if not combined_notes:
        return {"Questions": "No combined notes available. Please run combine_notes first."}

    prompt = GENERATE_QUESTIONS_PROMPT.format(
        combined_notes=combined_notes
    )
    llm_result = await llm.ainvoke(prompt)

    return {"Questions": str(llm_result.content)}

REFLECTION_PROMPT = """
  You are verifying if the following interview question is reasonably supported by the combined notes.
  If the question is even partially related to or inspired by the notes, set `is_satisfactory` to True.
  Only set `is_satisfactory` to False if it clearly contradicts or has no logical connection.

  <Question>
  {Question}
  </Question>

  <combined_notes>
  {combine_notes}
  </combined_notes>

  Important:
  - If any part of the combined notes is relevant to this question, consider it supported.
  - Only answer with 'true' or 'false' as is_satisfactory. Provide a brief explanation.
"""

def reflection(state: OverallState) -> dict[str, Any]:
    """
    Analyze the quality and completeness of the gathered information.
    """
    structured_llm = llm.with_structured_output(ReflectionOutput)
    system_prompt = REFLECTION_PROMPT.format(
        Question=state.Questions,
        combine_notes=state.combined_notes
    )
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Produce a structured reflection output."},
        ]
    )
    result = cast(ReflectionOutput, result)
    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
            "reasoning": result.reasoning
        }

from typing import Literal
from langgraph.graph import END

def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "generate_questions"]:  # type: ignore
    """Route the graph based on the reflection output."""
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "generate_questions"

    return END

from langgraph.graph import StateGraph, START

# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)

builder.add_node("generate_queries", generate_queries)
builder.add_node("research_person", research_person)
builder.add_node("combine_notes", combine_notes)
builder.add_node("research_projects", research_projects)
builder.add_node("generate_questions", generate_questions)
builder.add_node("reflection", reflection)
builder.add_node("generate_queries_for_company", generate_queries_for_company)
builder.add_node("research_company", research_company)
builder.add_node("extract_project_queries", extract_project_queries)

# -- Node Connections (Edges) ---

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_person")
builder.add_edge("research_person", "extract_project_queries")
builder.add_edge("extract_project_queries", "research_projects")

builder.add_edge(START, "generate_queries_for_company")
builder.add_edge("generate_queries_for_company", "research_company")

builder.add_edge(["research_company", "research_projects"], "combine_notes")

builder.add_edge("combine_notes", "generate_questions")
builder.add_edge("generate_questions", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile the graph
graph = builder.compile()


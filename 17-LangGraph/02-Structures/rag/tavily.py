from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing import Literal, Sequence, Optional, List
import json
import os


class TavilySearchInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="검색 쿼리")


def format_search_result(result: dict, include_raw_content: bool = False) -> str:
    """
    Utility functions for formatting search results.

    Args:
        result (dict): 원본 검색 결과

    Returns:
        str: XML 형식으로 포맷팅된 검색 결과
    """
    # 한글 인코딩 처리를 위해 json.dumps() 사용
    title = json.dumps(result["title"], ensure_ascii=False)[1:-1]
    content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
    raw_content = ""
    if (
        include_raw_content
        and "raw_content" in result
        and result["raw_content"] is not None
        and len(result["raw_content"].strip()) > 0
    ):
        raw_content = f"<raw>{result['raw_content']}</raw>"

    return f"<document><title>{title}</title><url>{result['url']}</url><content>{content}</content>{raw_content}</document>"


class TavilySearch(BaseTool):
    """
    Tool that queries the Tavily Search API and gets back json
    """

    name: str = "tavily_web_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. [IMPORTANT] Input(query) should be over 5 characters."
    )
    args_schema: type[BaseModel] = TavilySearchInput
    client: TavilyClient = None
    include_domains: list = []
    exclude_domains: list = []
    max_results: int = 3
    topic: Literal["general", "news"] = "general"
    days: int = 3
    search_depth: Literal["basic", "advanced"] = "basic"
    include_answer: bool = False
    include_raw_content: bool = True
    include_images: bool = False
    format_output: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        include_domains: list = [],
        exclude_domains: list = [],
        max_results: int = 3,
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
        format_output: bool = False,
    ):
        """
        TavilySearch 클래스의 인스턴스를 초기화합니다.

        Args:
            api_key (str): Tavily API 키
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_results (int): 기본 검색 결과 수
        """
        super().__init__()
        if api_key is None:
            api_key = os.environ.get("TAVILY_API_KEY", None)

        if api_key is None:
            raise ValueError("Tavily API key is not set.")

        self.client = TavilyClient(api_key=api_key)
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.max_results = max_results
        self.topic = topic
        self.days = days
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.format_output = format_output

    def _run(self, query: str) -> str:
        """BaseTool의 _run 메서드 구현"""
        results = self.search(query)
        return results
        # return json.dumps(results, ensure_ascii=False)

    def search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = None,
        topic: Literal["general", "news"] = None,
        days: int = None,
        max_results: int = None,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = None,
        include_raw_content: bool = None,
        include_images: bool = None,
        format_output: bool = None,
        **kwargs,
    ) -> list:
        """
        검색을 수행하고 결과를 반환합니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            include_answer (bool): 답변 포함 여부
            include_raw_content (bool): 원본 콘텐츠 포함 여부
            include_images (bool): 이미지 포함 여부
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            list: 검색 결과 목록
        """
        # 기본값 설정
        params = {
            "query": query,
            "search_depth": search_depth or self.search_depth,
            "topic": topic or self.topic,
            "max_results": max_results or self.max_results,
            "include_domains": include_domains or self.include_domains,
            "exclude_domains": exclude_domains or self.exclude_domains,
            "include_answer": (
                include_answer if include_answer is not None else self.include_answer
            ),
            "include_raw_content": (
                include_raw_content
                if include_raw_content is not None
                else self.include_raw_content
            ),
            "include_images": (
                include_images if include_images is not None else self.include_images
            ),
            **kwargs,
        }

        # days 파라미터 처리
        if days is not None:
            if params["topic"] == "general":
                print(
                    "Warning: days parameter is ignored for 'general' topic search. Set topic parameter to 'news' to use days."
                )
            else:
                params["days"] = days

        # API 호출
        response = self.client.search(**params)

        # 결과 포맷팅
        format_output = (
            format_output if format_output is not None else self.format_output
        )
        if format_output:
            return [
                format_search_result(r, params["include_raw_content"])
                for r in response["results"]
            ]
        else:
            return response["results"]

    def get_search_context(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = 5,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        max_tokens: int = 4000,
        format_output: bool = True,
        **kwargs,
    ) -> str:
        """
        검색 쿼리에 대한 컨텍스트를 가져옵니다. 웹사이트에서 관련 콘텐츠만 가져오는 데 유용하며,
        컨텍스트 추출과 제한을 직접 처리할 필요가 없습니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_tokens (int): 반환할 최대 토큰 수 (openai 토큰 계산 기준). 기본값은 4000입니다.
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            str: 컨텍스트 제한까지의 검색 컨텍스트를 포함하는 JSON 문자열
        """
        response = self.client.search(
            query,
            search_depth=search_depth,
            topic=topic,
            days=days,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            **kwargs,
        )

        sources = response.get("results", [])
        if format_output:
            context = [
                format_search_result(source, include_raw_content=False)
                for source in sources
            ]
        else:
            context = [
                {
                    "url": source["url"],
                    "content": json.dumps(
                        {"title": source["title"], "content": source["content"]},
                        ensure_ascii=False,
                    ),
                }
                for source in sources
            ]

        # max_tokens 처리 로직은 여기에 구현해야 합니다.
        # 현재는 간단히 모든 컨텍스트를 반환합니다.
        return json.dumps(context, ensure_ascii=False)

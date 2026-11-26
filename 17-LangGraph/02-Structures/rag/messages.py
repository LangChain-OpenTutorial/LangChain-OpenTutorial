from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid


def random_uuid():
    return str(uuid.uuid4())


def stream_response(response, return_output=False):
    """
    AI 모델로부터의 응답을 스트리밍하여 각 청크를 처리하면서 출력합니다.

    이 함수는 `response` 이터러블의 각 항목을 반복 처리합니다. 항목이 `AIMessageChunk`의 인스턴스인 경우,
    청크의 내용을 추출하여 출력합니다. 항목이 문자열인 경우, 문자열을 직접 출력합니다. 선택적으로, 함수는
    모든 응답 청크의 연결된 문자열을 반환할 수 있습니다.

    매개변수:
    - response (iterable): `AIMessageChunk` 객체 또는 문자열일 수 있는 응답 청크의 이터러블입니다.
    - return_output (bool, optional): True인 경우, 함수는 연결된 응답 문자열을 문자열로 반환합니다. 기본값은 False입니다.

    반환값:
    - str: `return_output`이 True인 경우, 연결된 응답 문자열입니다. 그렇지 않으면, 아무것도 반환되지 않습니다.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


# 도구 호출 시 실행되는 콜백 함수입니다.
def tool_callback(tool) -> None:
    print("[도구 호출]")
    print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
    if tool_input := tool.get("tool_input"):  # 도구에 입력된 값이 있다면
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # 입력값의 키와 값을 출력합니다.
    print(f"Log: {tool.get('log')}")  # 도구 실행 로그를 출력합니다.


# 관찰 결과를 출력하는 콜백 함수입니다.
def observation_callback(observation) -> None:
    print("[관찰 내용]")
    print(f"Observation: {observation.get('observation')}")  # 관찰 내용을 출력합니다.


# 최종 결과를 출력하는 콜백 함수입니다.
def result_callback(result: str) -> None:
    print("[최종 답변]")
    print(result)  # 최종 답변을 출력합니다.


@dataclass
class AgentCallbacks:
    """
    에이전트 콜백 함수들을 포함하는 데이터 클래스입니다.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): 도구 사용 시 호출되는 콜백 함수
        observation_callback (Callable[[Dict[str, Any]], None]): 관찰 결과 처리 시 호출되는 콜백 함수
        result_callback (Callable[[str], None]): 최종 결과 처리 시 호출되는 콜백 함수
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    에이전트의 스트림 출력을 파싱하고 처리하는 클래스입니다.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """
        AgentStreamParser 객체를 초기화합니다.

        Args:
            callbacks (AgentCallbacks, optional): 파싱 과정에서 사용할 콜백 함수들. 기본값은 AgentCallbacks()입니다.
        """
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        에이전트의 단계를 처리합니다.

        Args:
            step (Dict[str, Any]): 처리할 에이전트 단계 정보
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """
        에이전트의 액션들을 처리합니다.

        Args:
            actions (List[Any]): 처리할 액션 리스트
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """
        도구 호출을 처리합니다.

        Args:
            action (Any): 처리할 도구 호출 액션
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """
        관찰 결과들을 처리합니다.

        Args:
            observations (List[Any]): 처리할 관찰 결과 리스트
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """
        최종 결과를 처리합니다.

        Args:
            result (str): 처리할 최종 결과
        """
        self.callbacks.result_callback(result)
        self.output = result


def pretty_print_messages(messages: list[BaseMessage]):
    for message in messages:
        message.pretty_print()


# 각 깊이 수준에 대해 미리 정의된 색상 (ANSI 이스케이프 코드 사용)
depth_colors = {
    1: "\033[96m",  # 밝은 청록색 (눈에 잘 띄는 첫 계층)
    2: "\033[93m",  # 노란색 (두 번째 계층)
    3: "\033[94m",  # 밝은 초록색 (세 번째 계층)
    4: "\033[95m",  # 보라색 (네 번째 계층)
    5: "\033[92m",  # 밝은 파란색 (다섯 번째 계층)
    "default": "\033[96m",  # 기본값은 밝은 청록색으로
    "reset": "\033[0m",  # 기본 색상으로 재설정
}


def is_terminal_dict(data):
    """말단 딕셔너리인지 확인합니다."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True


def format_terminal_dict(data):
    """말단 딕셔너리를 포맷팅합니다."""
    items = []
    for key, value in data.items():
        if isinstance(value, str):
            items.append(f'"{key}": "{value}"')
        else:
            items.append(f'"{key}": {value}')
    return "{" + ", ".join(items) + "}"


def _display_message_tree(data, indent=0, node=None, is_root=False):
    """
    JSON 객체의 트리 구조를 타입 정보 없이 출력합니다.
    """
    spacing = " " * indent * 4
    color = depth_colors.get(indent + 1, depth_colors["default"])

    if isinstance(data, dict):
        if not is_root and node is not None:
            if is_terminal_dict(data):
                print(
                    f'{spacing}{color}{node}{depth_colors["reset"]}: {format_terminal_dict(data)}'
                )
            else:
                print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        if not is_root and node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')

        for index, item in enumerate(data):
            print(f'{spacing}    {color}index [{index}]{depth_colors["reset"]}')
            _display_message_tree(item, indent + 1)

    elif hasattr(data, "__dict__") and not is_root:
        if node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
        _display_message_tree(data.__dict__, indent)

    else:
        if node is not None:
            if isinstance(data, str):
                value_str = f'"{data}"'
            else:
                value_str = str(data)

            print(f'{spacing}{color}{node}{depth_colors["reset"]}: {value_str}')


def display_message_tree(message):
    """
    메시지 트리를 표시하는 주 함수입니다.
    """
    if isinstance(message, BaseMessage):
        _display_message_tree(message.__dict__, is_root=True)
    else:
        _display_message_tree(message, is_root=True)


class ToolChunkHandler:
    """Tool Message 청크를 처리하고 관리하는 클래스"""

    def __init__(self):
        self._reset_state()

    def _reset_state(self) -> None:
        """상태 초기화"""
        self.gathered = None
        self.first = True
        self.current_node = None
        self.current_namespace = None

    def _should_reset(self, node: str | None, namespace: str | None) -> bool:
        """상태 리셋 여부 확인"""
        # 파라미터가 모두 None인 경우 초기화하지 않음
        if node is None and namespace is None:
            return False

        # node만 설정된 경우
        if node is not None and namespace is None:
            return self.current_node != node

        # namespace만 설정된 경우
        if namespace is not None and node is None:
            return self.current_namespace != namespace

        # 둘 다 설정된 경우
        return self.current_node != node or self.current_namespace != namespace

    def process_message(
        self,
        chunk: AIMessageChunk,
        node: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """
        메시지 청크 처리

        Args:
            chunk: 처리할 AI 메시지 청크
            node: 현재 노드명 (선택사항)
            namespace: 현재 네임스페이스 (선택사항)
        """
        if self._should_reset(node, namespace):
            self._reset_state()

        self.current_node = node if node is not None else self.current_node
        self.current_namespace = (
            namespace if namespace is not None else self.current_namespace
        )

        self._accumulate_chunk(chunk)
        return self._display_tool_calls()

    def _accumulate_chunk(self, chunk: AIMessageChunk) -> None:
        """청크 누적"""
        self.gathered = chunk if self.first else self.gathered + chunk
        self.first = False

    def _display_tool_calls(self) -> None:
        """도구 호출 정보 출력"""
        if (
            self.gathered
            and not self.gathered.content
            and self.gathered.tool_call_chunks
            and self.gathered.tool_calls
        ):
            return self.gathered.tool_calls[0]["args"]


def get_role_from_messages(msg):
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages):
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )


def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
        if not node_names or curr_node in node_names:
            # 콜백 함수가 있는 경우 실행
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            # 콜백이 없는 경우 기본 출력
            else:
                # 노드가 변경된 경우에만 구분선 출력
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node


def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph 앱의 실행 결과를 예쁘게 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs=True 를 통해 서브그래프의 출력도 포함
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # node_names가 비어있지 않은 경우에만 필터링
            if len(node_names) > 0 and node_name not in node_names:
                continue

            # 콜백 함수가 있는 경우 실행
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            # 콜백이 없는 경우 기본 출력
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)

                # 노드의 청크 데이터 출력
                for k, v in node_chunk.items():
                    if isinstance(v, BaseMessage):
                        v.pretty_print()
                    elif isinstance(v, list):
                        for list_item in v:
                            if isinstance(list_item, BaseMessage):
                                list_item.pretty_print()
                            else:
                                print(list_item)
                    elif isinstance(v, dict):
                        for node_chunk_key, node_chunk_value in node_chunk.items():
                            print(f"{node_chunk_key}:\n{node_chunk_value}")
                print("=" * 50)

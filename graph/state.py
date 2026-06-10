from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    query: str
    agents_to_use: List[str]
    sub_tasks: dict
    pdf_answer: Optional[str]
    live_answer: Optional[str]
    final_answer: Optional[str]
    quality_score: Optional[float]
    retry_count: int


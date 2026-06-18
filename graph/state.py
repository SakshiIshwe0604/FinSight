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
    # ── new compare fields ──
    is_compare: Optional[bool]
    company1: Optional[str]
    company2: Optional[str]
    pdf_answer_2: Optional[str]
    live_answer_2: Optional[str]
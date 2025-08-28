from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Card:
    card_id: int
    question: str
    answer: str
    alpha_param: float = 1.0
    beta_param: float = 1.0
    knowledge_state: float = 0.1
    learning_rate: float = 0.3
    slip_probability: float = 0.1
    guess_probability: float = 0.2
    forgetting_rate: float = 0.1
    last_reviewed: Optional[datetime] = None
    last_knowledge_update: Optional[datetime] = None
    review_count: int = 0
    # optional additional fields
    avg_response_time: float = 5.0
    difficulty_level: float = 0.5
    next_due_at: Optional[datetime] = None




@dataclass
class UserLearningParams:
    user_id: int
    base_learning_rate: float = 0.3
    base_forgetting_rate: float = 0.1
    exploration_weight: float = 0.6
    knowledge_weight: float = 0.4
    optimal_session_length: int = 20
    # extra fields for more advanced scheduler
    response_time_weight: float = 0.15
    auto_adapt_params: bool = True
    last_adaptation: Optional[datetime] = None
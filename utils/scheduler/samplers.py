import numpy as np
from typing import Tuple

class ThompsonSampler:
    """Handles Thompson Sampling for flashcard selection"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def sample_recall_probability(self, alpha: float, beta: float) -> float:
        alpha = max(0.01, alpha)
        beta = max(0.01, beta)
        return float(self.rng.beta(alpha, beta))

    def update_parameters(self, alpha: float, beta: float, success: bool, confidence: int = 2) -> Tuple[float, float]:
        # Confidence-based weight mapping
        confidence_weights = {0: 0.5, 1: 0.7, 2: 1.0, 3: 1.3}
        weight = confidence_weights.get(confidence, 1.0)
        if success:
            return alpha + weight, beta
        else:
            return alpha, beta + weight

    def get_uncertainty(self, alpha: float, beta: float) -> float:
        alpha = max(0.01, alpha)
        beta = max(0.01, beta)
        total = alpha + beta
        return (alpha * beta) / (total**2 * (total + 1))


import numpy as _np
from typing import Optional

class KnowledgeTracer:
    """Lightweight Bayesian Knowledge Tracing utilities"""

    def update_knowledge_state(self, current_knowledge: float, success: bool,
                               learning_rate: float, slip_prob: float, guess_prob: float) -> float:
        # Clip inputs
        slip_prob = _np.clip(slip_prob, 0.01, 0.5)
        guess_prob = _np.clip(guess_prob, 0.01, 0.5)
        learning_rate = _np.clip(learning_rate, 0.01, 0.8)

        if success:
            denom = current_knowledge * (1 - slip_prob) + (1 - current_knowledge) * guess_prob
            if denom > 1e-6:
                p_know_given_correct = (current_knowledge * (1 - slip_prob)) / denom
            else:
                p_know_given_correct = current_knowledge
            new_knowledge = p_know_given_correct + (1 - p_know_given_correct) * learning_rate
        else:
            denom = current_knowledge * slip_prob + (1 - current_knowledge) * (1 - guess_prob)
            if denom > 1e-6:
                p_know_given_incorrect = (current_knowledge * slip_prob) / denom
            else:
                p_know_given_incorrect = current_knowledge * 0.5
            new_knowledge = p_know_given_incorrect * (1 - learning_rate * 0.1)

        return float(_np.clip(new_knowledge, 0.0, 1.0))

    def apply_time_decay(self, knowledge: float, time_delta_hours: float, forgetting_rate: float) -> float:
        forgetting_rate = max(0.001, forgetting_rate)
        decay_factor = _np.exp(-forgetting_rate * time_delta_hours / 24.0)
        return float(knowledge * decay_factor)

    def get_recall_probability(self, knowledge: float, slip_prob: float, guess_prob: float) -> float:
        return float(knowledge * (1 - slip_prob) + (1 - knowledge) * guess_prob)



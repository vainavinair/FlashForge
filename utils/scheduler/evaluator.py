
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

class SchedulerEvaluator:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def evaluate_prediction_accuracy(self, user_id: int, days: int = 30) -> float:
        conn = sqlite3.connect(self.scheduler.db_path)
        cur = conn.cursor()
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        cur.execute("""
        SELECT sampled_theta, user_response
        FROM review_history
        WHERE user_id = ? AND timestamp > ? AND sampled_theta IS NOT NULL
        """, (user_id, start_date))
        rows = cur.fetchall()
        conn.close()
        if len(rows) < 5:
            return 0.0
        preds = np.array([r[0] for r in rows])
        actuals = np.array([float(r[1]) for r in rows])
        n_bins = 5
        edges = np.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        for i in range(n_bins):
            mask = (preds >= edges[i]) & (preds < edges[i+1])
            if mask.sum() > 0:
                avg_pred = preds[mask].mean()
                avg_act = actuals[mask].mean()
                calibration_error += (mask.sum() / len(preds)) * abs(avg_pred - avg_act)
        return max(0.0, 1.0 - calibration_error)

    def suggest_parameter_adjustments(self, user_id: int) -> Dict[str, float]:
        analytics = self.scheduler.get_learning_analytics(user_id)
        current_params = self.scheduler.get_user_params(user_id)
        suggestions: Dict[str, float] = {}
        if analytics['accuracy_rate'] > 0.85:
            suggestions['exploration_weight'] = min(current_params.exploration_weight + 0.1, 0.9)
            suggestions['knowledge_weight'] = max(current_params.knowledge_weight - 0.1, 0.1)
        elif analytics['accuracy_rate'] < 0.65:
            suggestions['exploration_weight'] = max(current_params.exploration_weight - 0.1, 0.1)
            suggestions['knowledge_weight'] = min(current_params.knowledge_weight + 0.1, 0.9)
        if analytics['avg_response_time'] > 10.0:
            suggestions['optimal_session_length'] = max(current_params.optimal_session_length - 5, 10)
        elif analytics['accuracy_rate'] > 0.8 and analytics['total_reviews'] > 20:
            suggestions['optimal_session_length'] = min(current_params.optimal_session_length + 5, 50)
        return suggestions
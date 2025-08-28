

from typing import List
from datetime import datetime, timedelta
import numpy as np
import logging

from models import Card, UserLearningParams
from samplers import ThompsonSampler, KnowledgeTracer
import db_utils as dbu

logger = logging.getLogger(__name__)

class HybridScheduler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.thompson_sampler = ThompsonSampler()
        self.knowledge_tracer = KnowledgeTracer()
        self.knowledge_threshold = 0.7

    def get_user_params(self, user_id: int) -> UserLearningParams:
        row = dbu.get_user_params_row(user_id)
        if row:
            # map row -> UserLearningParams (assumes same column order)
            return UserLearningParams(
                user_id=row[0],
                base_learning_rate=row[1],
                base_forgetting_rate=row[2],
                exploration_weight=row[3],
                knowledge_weight=row[4],
                optimal_session_length=row[5]
            )
        else:
            params = UserLearningParams(user_id=user_id)
            dbu.insert_user_params_row(user_id, params)
            return params

    def get_deck_cards_enhanced(self, deck_id: int) -> List[Card]:
        rows = dbu.fetch_deck_cards_basic(deck_id)
        cards: List[Card] = []
        for row in rows:
            card = Card(
                card_id=row[0], question=row[1], answer=row[2],
                alpha_param=row[3] or 1.0, beta_param=row[4] or 1.0,
                knowledge_state=row[5] or 0.1, learning_rate=row[6] or 0.3,
                slip_probability=row[7] or 0.1, guess_probability=row[8] or 0.2,
                forgetting_rate=row[9] or 0.1,
                last_reviewed=datetime.fromisoformat(row[10]) if row[10] else None,
                last_knowledge_update=datetime.fromisoformat(row[11]) if row[11] else None,
                review_count=row[12] or 0
            )
            cards.append(card)
        return cards

    def update_knowledge_with_time_decay(self, cards: List[Card]) -> List[Card]:
        now = datetime.now()
        for card in cards:
            if card.last_knowledge_update:
                hours = (now - card.last_knowledge_update).total_seconds() / 3600
                card.knowledge_state = self.knowledge_tracer.apply_time_decay(card.knowledge_state, hours, card.forgetting_rate)
        return cards

    def calculate_hybrid_priority(self, cards: List[Card], user_params: UserLearningParams):
        priorities = []
        now = datetime.now()
        for card in cards:
            sampled = self.thompson_sampler.sample_recall_probability(card.alpha_param, card.beta_param)
            thompson_score = 1.0 - sampled
            knowledge_urgency = max(0.0, self.knowledge_threshold - card.knowledge_state)
            hybrid_score = user_params.exploration_weight * thompson_score + user_params.knowledge_weight * knowledge_urgency
            if card.last_reviewed:
                hours_since = (now - card.last_reviewed).total_seconds() / 3600
                recency_boost = min(hours_since / 24.0, 2.0)
                hybrid_score *= (1 + recency_boost * 0.5)
            else:
                hybrid_score *= 2.0
            priorities.append((card, hybrid_score))
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def select_cards_for_review(self, deck_id: int, user_id: int, num_cards: int = 10) -> List[Card]:
        user_params = self.get_user_params(user_id)
        cards = self.get_deck_cards_enhanced(deck_id)
        if not cards:
            return []
        cards = self.update_knowledge_with_time_decay(cards)
        priorities = self.calculate_hybrid_priority(cards, user_params)
        num_candidates = min(num_cards * 2, len(priorities))
        top_candidates = priorities[:num_candidates]
        scores = [s for _, s in top_candidates]
        total = sum(scores)
        if total > 0:
            probs = [s / total for s in scores]
            idxs = np.random.choice(len(top_candidates), size=min(num_cards, len(top_candidates)), replace=False, p=probs)
            selected = [top_candidates[i][0] for i in idxs]
        else:
            selected = [card for card, _ in priorities[:num_cards]]
        return selected

    def update_after_review(self, card_id: int, user_id: int, success: bool, response_time: float = None, confidence: int = None):
        # Read current card fields
        rows = dbu.fetch_deck_cards_basic(None)  # not ideal; directly query single card instead
        # Simpler direct fetch
        conn = __import__('utils.datamodels.db').db.get_connection() if False else None
        # We'll implement a minimal inline query for the single card to keep parity with prior code
        import sqlite3
        from utils.datamodels.db import get_connection
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT alpha_param, beta_param, knowledge_state, learning_rate,
                   slip_probability, guess_probability, forgetting_rate, review_count
            FROM cards WHERE card_id = ?
        """, (card_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return
        alpha, beta, knowledge, lr, slip, guess, forget, review_count = row

        pre_k = knowledge
        sampled_theta = self.thompson_sampler.sample_recall_probability(alpha, beta)
        new_alpha, new_beta = self.thompson_sampler.update_parameters(alpha, beta, success)
        new_knowledge = self.knowledge_tracer.update_knowledge_state(knowledge, success, lr, slip, guess)

        now_iso = datetime.now().isoformat()
        # update cards
        dbu.update_card_row(card_id, {
            'alpha_param': new_alpha,
            'beta_param': new_beta,
            'knowledge_state': new_knowledge,
            'last_reviewed': now_iso,
            'last_knowledge_update': now_iso,
            'review_count': review_count + 1
        })

        # update scheduler_state (store interval as 0 for now)
        dbu.upsert_scheduler_state(card_id, new_alpha, new_beta, 0.0, review_count + 1, ('good' if success else 'again'), now_iso, None)

        # insert review history
        dbu.insert_review_history(card_id, user_id, int(success), response_time, confidence, pre_k, new_knowledge, sampled_theta)

    def get_learning_analytics(self, user_id: int, days: int = 7) -> dict:
        # Minimal implementation using SQL inside this method for readability
        import sqlite3
        from utils.datamodels.db import get_connection
        conn = get_connection()
        cur = conn.cursor()
        start = (datetime.now() - timedelta(days=days)).isoformat()
        cur.execute("""
        SELECT COUNT(*) as total_reviews,
               AVG(CAST(user_response as FLOAT)) as accuracy_rate,
               AVG(response_time) as avg_response_time,
               AVG(post_review_knowledge_state) as avg_knowledge
        FROM review_history
        WHERE user_id = ? AND timestamp > ?
        """, (user_id, start))
        stats = cur.fetchone()

        cur.execute("""
        SELECT AVG(knowledge_state) as avg_knowledge,
               COUNT(CASE WHEN knowledge_state > 0.7 THEN 1 END) as mastered_cards,
               COUNT(*) as total_cards
        FROM cards c
        JOIN decks d ON c.deck_id = d.deck_id
        WHERE d.user_id = ?
        """, (user_id,))
        knowledge_stats = cur.fetchone()
        conn.close()
        return {
            'total_reviews': stats[0] or 0,
            'accuracy_rate': stats[1] or 0.0,
            'avg_response_time': stats[2] or 0.0,
            'avg_knowledge': stats[3] or 0.0,
            'avg_deck_knowledge': knowledge_stats[0] or 0.0,
            'mastered_cards': knowledge_stats[1] or 0,
            'total_cards': knowledge_stats[2] or 0,
            'mastery_rate': (knowledge_stats[1] or 0) / max(knowledge_stats[2] or 1, 1)
        }
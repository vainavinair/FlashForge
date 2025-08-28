# utils/hybrid_scheduler.py
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class UserLearningParams:
    user_id: int
    base_learning_rate: float = 0.3
    base_forgetting_rate: float = 0.1
    exploration_weight: float = 0.6
    knowledge_weight: float = 0.4
    optimal_session_length: int = 20

class ThompsonSampler:
    """Handles Thompson Sampling for flashcard selection"""
    
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def sample_recall_probability(self, alpha: float, beta: float) -> float:
        """Sample recall probability from Beta distribution"""
        return self.rng.beta(alpha, beta)
    
    def update_parameters(self, alpha: float, beta: float, success: bool) -> Tuple[float, float]:
        """Update Beta parameters based on recall outcome"""
        if success:
            return alpha + 1, beta
        else:
            return alpha, beta + 1
    
    def get_uncertainty(self, alpha: float, beta: float) -> float:
        """Calculate uncertainty (variance) of Beta distribution"""
        total = alpha + beta
        return (alpha * beta) / (total**2 * (total + 1))

class KnowledgeTracer:
    """Handles Bayesian Knowledge Tracing"""
    
    def __init__(self):
        pass
    
    def update_knowledge_state(self, current_knowledge: float, success: bool, 
                             learning_rate: float, slip_prob: float, guess_prob: float) -> float:
        """Update knowledge state based on review outcome"""
        if success:
            # Probability of knowing given correct response
            p_know_given_correct = (current_knowledge * (1 - slip_prob)) / (
                current_knowledge * (1 - slip_prob) + (1 - current_knowledge) * guess_prob
            )
            # Update with learning
            new_knowledge = p_know_given_correct + (1 - p_know_given_correct) * learning_rate
        else:
            # Probability of knowing given incorrect response
            p_know_given_incorrect = (current_knowledge * slip_prob) / (
                current_knowledge * slip_prob + (1 - current_knowledge) * (1 - guess_prob)
            )
            # Knowledge decreases slightly on failure
            new_knowledge = p_know_given_incorrect * (1 - learning_rate * 0.1)
        
        return np.clip(new_knowledge, 0.0, 1.0)
    
    def apply_time_decay(self, knowledge: float, time_delta_hours: float, 
                        forgetting_rate: float) -> float:
        """Apply forgetting curve decay"""
        decay_factor = np.exp(-forgetting_rate * time_delta_hours / 24.0)  # per day
        return knowledge * decay_factor
    
    def get_recall_probability(self, knowledge: float, slip_prob: float, guess_prob: float) -> float:
        """Calculate predicted recall probability"""
        return knowledge * (1 - slip_prob) + (1 - knowledge) * guess_prob

class HybridScheduler:
    """Main scheduler combining Thompson Sampling and Knowledge Tracing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.thompson_sampler = ThompsonSampler()
        self.knowledge_tracer = KnowledgeTracer()
        self.knowledge_threshold = 0.7
    
    def get_user_params(self, user_id: int) -> UserLearningParams:
        """Get or create user-specific learning parameters"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT * FROM user_learning_params WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        
        if row:
            params = UserLearningParams(
                user_id=row[0],
                base_learning_rate=row[1],
                base_forgetting_rate=row[2],
                exploration_weight=row[3],
                knowledge_weight=row[4],
                optimal_session_length=row[5]
            )
        else:
            # Create default parameters for new user
            params = UserLearningParams(user_id=user_id)
            cur.execute("""
            INSERT INTO user_learning_params 
            (user_id, base_learning_rate, base_forgetting_rate, exploration_weight, knowledge_weight, optimal_session_length)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, params.base_learning_rate, params.base_forgetting_rate, 
                  params.exploration_weight, params.knowledge_weight, params.optimal_session_length))
            conn.commit()
        
        conn.close()
        return params
    
    def get_deck_cards_enhanced(self, deck_id: int) -> List[Card]:
        """Get all cards from deck with enhanced parameters"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
        SELECT card_id, question, answer, 
               alpha_param, beta_param, knowledge_state, learning_rate,
               slip_probability, guess_probability, forgetting_rate, 
               last_reviewed, last_knowledge_update, review_count
        FROM cards WHERE deck_id = ?
        """, (deck_id,))
        
        cards = []
        for row in cur.fetchall():
            card = Card(
                card_id=row[0],
                question=row[1],
                answer=row[2],
                alpha_param=row[3],
                beta_param=row[4],
                knowledge_state=row[5],
                learning_rate=row[6],
                slip_probability=row[7],
                guess_probability=row[8],
                forgetting_rate=row[9],
                last_reviewed=datetime.fromisoformat(row[10]) if row[10] else None,
                last_knowledge_update=datetime.fromisoformat(row[11]) if row[11] else None,
                review_count=row[12]
            )
            cards.append(card)
        
        conn.close()
        return cards
    
    def update_knowledge_with_time_decay(self, cards: List[Card]) -> List[Card]:
        """Apply time decay to all cards' knowledge states"""
        now = datetime.now()
        
        for card in cards:
            if card.last_knowledge_update:
                time_delta = now - card.last_knowledge_update
                hours_passed = time_delta.total_seconds() / 3600
                
                # Apply time decay
                card.knowledge_state = self.knowledge_tracer.apply_time_decay(
                    card.knowledge_state, hours_passed, card.forgetting_rate
                )
        
        return cards
    
    def calculate_hybrid_priority(self, cards: List[Card], user_params: UserLearningParams) -> List[Tuple[Card, float]]:
        """Calculate priority scores using hybrid approach"""
        priorities = []
        
        for card in cards:
            # Thompson Sampling component
            sampled_theta = self.thompson_sampler.sample_recall_probability(
                card.alpha_param, card.beta_param
            )
            thompson_score = 1 - sampled_theta  # Lower recall prob = higher priority
            
            # Knowledge Tracing component
            knowledge_urgency = max(0, self.knowledge_threshold - card.knowledge_state)
            
            # Combine scores
            hybrid_score = (user_params.exploration_weight * thompson_score + 
                          user_params.knowledge_weight * knowledge_urgency)
            
            # Boost score for cards not reviewed recently
            if card.last_reviewed:
                hours_since_review = (datetime.now() - card.last_reviewed).total_seconds() / 3600
                recency_boost = min(hours_since_review / 24.0, 2.0)  # Max 2x boost after 2 days
                hybrid_score *= (1 + recency_boost * 0.5)
            else:
                hybrid_score *= 2.0  # New cards get high priority
            
            priorities.append((card, hybrid_score))
        
        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities
    
    def select_cards_for_review(self, deck_id: int, user_id: int, num_cards: int = 10) -> List[Card]:
        """Select optimal cards for review session"""
        # Get user parameters
        user_params = self.get_user_params(user_id)
        
        # Get all cards from deck
        cards = self.get_deck_cards_enhanced(deck_id)
        
        if not cards:
            return []
        
        # Apply time decay to knowledge states
        cards = self.update_knowledge_with_time_decay(cards)
        
        # Calculate priorities
        priorities = self.calculate_hybrid_priority(cards, user_params)
        
        # Select top cards, but ensure some randomization to prevent deterministic patterns
        num_candidates = min(num_cards * 2, len(priorities))
        top_candidates = priorities[:num_candidates]
        
        # Sample from top candidates with probability proportional to score
        scores = [score for _, score in top_candidates]
        total_score = sum(scores)
        
        if total_score > 0:
            probabilities = [score / total_score for score in scores]
            selected_indices = np.random.choice(
                len(top_candidates), 
                size=min(num_cards, len(top_candidates)),
                replace=False,
                p=probabilities
            )
            selected_cards = [top_candidates[i][0] for i in selected_indices]
        else:
            # Fallback: select first N cards
            selected_cards = [card for card, _ in priorities[:num_cards]]
        
        logger.info(f"Selected {len(selected_cards)} cards for review")
        return selected_cards
    
    def update_after_review(self, card_id: int, user_id: int, success: bool, 
                          response_time: float = None, confidence: int = None):
        """Update card parameters after review"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get current card data
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
        
        # Store pre-review state
        pre_knowledge = knowledge
        sampled_theta = self.thompson_sampler.sample_recall_probability(alpha, beta)
        
        # Update Thompson Sampling parameters
        new_alpha, new_beta = self.thompson_sampler.update_parameters(alpha, beta, success)
        
        # Update Knowledge Tracing state
        new_knowledge = self.knowledge_tracer.update_knowledge_state(
            knowledge, success, lr, slip, guess
        )
        
        # Update database
        now = datetime.now().isoformat()
        cur.execute("""
        UPDATE cards SET 
            alpha_param = ?, beta_param = ?, knowledge_state = ?,
            last_reviewed = ?, last_knowledge_update = ?, review_count = ?
        WHERE card_id = ?
        """, (new_alpha, new_beta, new_knowledge, now, now, review_count + 1, card_id))
        
        # Log review history
        cur.execute("""
        INSERT INTO review_history 
        (card_id, user_id, user_response, response_time, confidence_level,
         pre_review_knowledge_state, post_review_knowledge_state, sampled_theta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (card_id, user_id, int(success), response_time, confidence,
              pre_knowledge, new_knowledge, sampled_theta))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated card {card_id}: knowledge {pre_knowledge:.3f} -> {new_knowledge:.3f}")
    
    def get_learning_analytics(self, user_id: int, days: int = 7) -> dict:
        """Get learning analytics for user"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get recent review history
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        cur.execute("""
        SELECT COUNT(*) as total_reviews,
               AVG(CAST(user_response as FLOAT)) as accuracy_rate,
               AVG(response_time) as avg_response_time,
               AVG(post_review_knowledge_state) as avg_knowledge
        FROM review_history 
        WHERE user_id = ? AND timestamp > ?
        """, (user_id, start_date))
        
        stats = cur.fetchone()
        
        # Get knowledge distribution
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

class SchedulerEvaluator:
    """Evaluates scheduler performance and suggests parameter adjustments"""
    
    def __init__(self, scheduler: HybridScheduler):
        self.scheduler = scheduler
    
    def evaluate_prediction_accuracy(self, user_id: int, days: int = 30) -> float:
        """Evaluate how well the scheduler predicts recall"""
        conn = sqlite3.connect(self.scheduler.db_path)
        cur = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        cur.execute("""
        SELECT sampled_theta, user_response 
        FROM review_history 
        WHERE user_id = ? AND timestamp > ? AND sampled_theta IS NOT NULL
        """, (user_id, start_date))
        
        predictions = []
        actuals = []
        
        for theta, response in cur.fetchall():
            predictions.append(theta)
            actuals.append(float(response))
        
        conn.close()
        
        if len(predictions) < 5:
            return 0.0
        
        # Calculate calibration error
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Bin predictions and calculate calibration
        n_bins = 5
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_prediction = predictions[mask].mean()
                avg_actual = actuals[mask].mean()
                bin_weight = mask.sum() / len(predictions)
                calibration_error += bin_weight * abs(avg_prediction - avg_actual)
        
        return 1.0 - calibration_error  # Higher is better
    
    def suggest_parameter_adjustments(self, user_id: int) -> dict:
        """Suggest parameter adjustments based on performance"""
        analytics = self.scheduler.get_learning_analytics(user_id)
        current_params = self.scheduler.get_user_params(user_id)
        
        suggestions = {}
        
        # Adjust exploration vs exploitation
        if analytics['accuracy_rate'] > 0.85:
            suggestions['exploration_weight'] = min(current_params.exploration_weight + 0.1, 0.9)
            suggestions['knowledge_weight'] = max(current_params.knowledge_weight - 0.1, 0.1)
        elif analytics['accuracy_rate'] < 0.65:
            suggestions['exploration_weight'] = max(current_params.exploration_weight - 0.1, 0.1)
            suggestions['knowledge_weight'] = min(current_params.knowledge_weight + 0.1, 0.9)
        
        # Adjust session length
        if analytics['avg_response_time'] > 10.0:  # Slow responses indicate fatigue
            suggestions['optimal_session_length'] = max(current_params.optimal_session_length - 5, 10)
        elif analytics['accuracy_rate'] > 0.8 and analytics['total_reviews'] > 20:
            suggestions['optimal_session_length'] = min(current_params.optimal_session_length + 5, 50)
        
        return suggestions
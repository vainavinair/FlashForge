# demo_scheduler.py - Demonstration of the Hybrid Scheduler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scheduler.hybrid_scheduler import HybridScheduler, SchedulerEvaluator
from utils.datamodels.db import init_db, get_connection
import numpy as np
from datetime import datetime, timedelta

def setup_demo_data():
    """Create demo user, deck, and cards for testing"""
    print("🔧 Setting up demo data...")
    
    # Initialize database
    init_db()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Create demo user
    cur.execute("INSERT OR IGNORE INTO users (username, password) VALUES ('demo_user', 'demo_pass')")
    cur.execute("SELECT user_id FROM users WHERE username = 'demo_user'")
    user_id = cur.fetchone()[0]
    
    # Create demo deck
    cur.execute("INSERT OR IGNORE INTO decks (user_id, name) VALUES (?, 'Python Programming')", (user_id,))
    cur.execute("SELECT deck_id FROM decks WHERE user_id = ? AND name = 'Python Programming'", (user_id,))
    deck_id = cur.fetchone()[0]
    
    # Create demo flashcards
    demo_cards = [
        ("What keyword is used to define a function in Python?", "def"),
        ("How do you create a list in Python?", "Using square brackets [] or list()"),
        ("What is the difference between == and is in Python?", "== compares values, is compares object identity"),
        ("How do you handle exceptions in Python?", "Using try/except blocks"),
        ("What is a lambda function?", "An anonymous function defined with the lambda keyword"),
        ("How do you import a module in Python?", "Using the import statement"),
        ("What is the difference between a list and a tuple?", "Lists are mutable, tuples are immutable"),
        ("How do you create a dictionary in Python?", "Using curly braces {} or dict()"),
        ("What is list comprehension?", "A concise way to create lists using [expression for item in iterable]"),
        ("How do you read a file in Python?", "Using open() function with 'r' mode"),
        ("What is the purpose of __init__ method?", "Constructor method that initializes object attributes"),
        ("How do you create a class in Python?", "Using the class keyword"),
        ("What is inheritance in Python?", "A way to create new classes based on existing classes"),
        ("What is the difference between append() and extend()?", "append() adds single element, extend() adds multiple elements"),
        ("How do you handle multiple exceptions?", "Using multiple except blocks or tuple of exceptions")
    ]
    
    # Insert cards if they don't exist
    for question, answer in demo_cards:
        cur.execute("""
        INSERT OR IGNORE INTO cards (deck_id, question, answer) 
        VALUES (?, ?, ?)
        """, (deck_id, question, answer))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created demo user (ID: {user_id}) and deck (ID: {deck_id}) with {len(demo_cards)} cards")
    return user_id, deck_id

def demonstrate_thompson_sampling():
    """Demonstrate Thompson Sampling algorithm"""
    print("\n🎯 Thompson Sampling Demonstration")
    print("=" * 50)
    
    scheduler = HybridScheduler("database/flashforge.db")
    
    # Simulate card with different success/failure patterns
    print("Simulating 3 cards with different performance patterns:")
    
    # Card 1: High performer (mostly correct)
    alpha1, beta1 = 1.0, 1.0
    for _ in range(8):  # 8 successes
        alpha1, beta1 = scheduler.thompson_sampler.update_parameters(alpha1, beta1, True)
    for _ in range(2):  # 2 failures
        alpha1, beta1 = scheduler.thompson_sampler.update_parameters(alpha1, beta1, False)
    
    # Card 2: Low performer (mostly incorrect)
    alpha2, beta2 = 1.0, 1.0
    for _ in range(3):  # 3 successes
        alpha2, beta2 = scheduler.thompson_sampler.update_parameters(alpha2, beta2, True)
    for _ in range(7):  # 7 failures
        alpha2, beta2 = scheduler.thompson_sampler.update_parameters(alpha2, beta2, False)
    
    # Card 3: New card (no history)
    alpha3, beta3 = 1.0, 1.0
    
    print(f"\nCard 1 (High performer): α={alpha1}, β={beta1}")
    print(f"Card 2 (Low performer):  α={alpha2}, β={beta2}")
    print(f"Card 3 (New card):       α={alpha3}, β={beta3}")
    
    # Sample from each card multiple times
    print("\nSampling recall probabilities (10 samples each):")
    for i in range(3):
        alpha, beta = [(alpha1, beta1), (alpha2, beta2), (alpha3, beta3)][i]
        samples = [scheduler.thompson_sampler.sample_recall_probability(alpha, beta) 
                  for _ in range(10)]
        avg_sample = np.mean(samples)
        uncertainty = scheduler.thompson_sampler.get_uncertainty(alpha, beta)
        
        print(f"Card {i+1}: Avg sampled θ = {avg_sample:.3f}, Uncertainty = {uncertainty:.4f}")
        print(f"        Samples: {[f'{s:.3f}' for s in samples[:5]]}")

def demonstrate_knowledge_tracing():
    """Demonstrate Bayesian Knowledge Tracing"""
    print("\n🧠 Bayesian Knowledge Tracing Demonstration")
    print("=" * 50)
    
    scheduler = HybridScheduler("database/flashforge.db")
    tracer = scheduler.knowledge_tracer
    
    # Simulate learning progression
    knowledge_state = 0.1  # Start with low knowledge
    learning_rate = 0.3
    slip_prob = 0.1
    guess_prob = 0.2
    
    print("Simulating learning progression over 10 reviews:")
    print("Review | Success | Knowledge State | Recall Probability")
    print("-" * 55)
    
    for review in range(1, 11):
        # Calculate current recall probability
        recall_prob = tracer.get_recall_probability(knowledge_state, slip_prob, guess_prob)
        
        # Simulate user response (higher knowledge = higher chance of success)
        success = np.random.random() < recall_prob
        
        print(f"   {review:2d}  |    {success}    |     {knowledge_state:.3f}      |      {recall_prob:.3f}")
        
        # Update knowledge state
        knowledge_state = tracer.update_knowledge_state(
            knowledge_state, success, learning_rate, slip_prob, guess_prob
        )
    
    # Demonstrate time decay
    print(f"\nFinal knowledge state: {knowledge_state:.3f}")
    print("Demonstrating forgetting over time:")
    
    for days in [1, 7, 30, 90]:
        hours = days * 24
        decayed_knowledge = tracer.apply_time_decay(knowledge_state, hours, 0.1)
        print(f"After {days:2d} days: {decayed_knowledge:.3f}")

def simulate_study_session(user_id, deck_id):
    """Simulate a complete study session"""
    print("\n📚 Simulating ML-Powered Study Session")
    print("=" * 50)
    
    scheduler = HybridScheduler("database/flashforge.db")
    
    # Select cards for review
    selected_cards = scheduler.select_cards_for_review(deck_id, user_id, num_cards=5)
    
    if not selected_cards:
        print("No cards available for review.")
        return
    
    print(f"Selected {len(selected_cards)} cards for review using hybrid algorithm:")
    
    session_results = []
    
    for i, card in enumerate(selected_cards):
        print(f"\n--- Card {i+1} ---")
        print(f"Question: {card.question}")
        print(f"Current knowledge state: {card.knowledge_state:.3f}")
        print(f"Thompson parameters: α={card.alpha_param:.1f}, β={card.beta_param:.1f}")
        
        # Simulate user response (biased by current knowledge state)
        response_prob = min(0.9, max(0.1, card.knowledge_state + np.random.normal(0, 0.2)))
        success = np.random.random() < response_prob
        confidence = np.random.randint(1, 4) if success else np.random.randint(1, 3)
        response_time = np.random.uniform(3, 12)  # 3-12 seconds
        
        print(f"Answer: {card.answer}")
        print(f"User response: {'Correct' if success else 'Incorrect'} (confidence: {confidence}/5)")
        print(f"Response time: {response_time:.1f}s")
        
        # Update card parameters
        scheduler.update_after_review(card.card_id, user_id, success, response_time, confidence)
        
        session_results.append({
            'card_id': card.card_id,
            'success': success,
            'confidence': confidence,
            'response_time': response_time
        })
    
    # Session summary
    accuracy = sum(r['success'] for r in session_results) / len(session_results)
    avg_confidence = np.mean([r['confidence'] for r in session_results])
    avg_time = np.mean([r['response_time'] for r in session_results])
    
    print(f"\n📊 Session Summary:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average confidence: {avg_confidence:.1f}/5")
    print(f"Average response time: {avg_time:.1f}s")

def demonstrate_analytics(user_id):
    """Demonstrate learning analytics"""
    print("\n📊 Learning Analytics Demonstration")
    print("=" * 50)
    
    scheduler = HybridScheduler("database/flashforge.db")
    evaluator = SchedulerEvaluator(scheduler)
    
    # Get analytics
    analytics = scheduler.get_learning_analytics(user_id, days=30)
    
    print("User Learning Analytics (Last 30 days):")
    print(f"• Total reviews: {analytics['total_reviews']}")
    print(f"• Accuracy rate: {analytics['accuracy_rate']:.1%}")
    print(f"• Average response time: {analytics['avg_response_time']:.1f}s")
    print(f"• Average knowledge level: {analytics['avg_deck_knowledge']:.3f}")
    print(f"• Mastered cards: {analytics['mastered_cards']}/{analytics['total_cards']}")
    print(f"• Mastery rate: {analytics['mastery_rate']:.1%}")
    
    # Get parameter suggestions
    suggestions = evaluator.suggest_parameter_adjustments(user_id)
    
    if suggestions:
        print(f"\n🎯 AI Parameter Suggestions:")
        for param, value in suggestions.items():
            print(f"• {param.replace('_', ' ').title()}: {value:.2f}")
    else:
        print(f"\n✅ Current parameters are optimal based on performance!")
    
    # Evaluate prediction accuracy
    accuracy = evaluator.evaluate_prediction_accuracy(user_id)
    print(f"\n🤖 ML Prediction Accuracy: {accuracy:.1%}")

def run_comprehensive_demo():
    """Run complete demonstration of the hybrid scheduler"""
    print("🚀 FlashForge Hybrid Scheduler Demo")
    print("=" * 60)
    
    # Setup
    user_id, deck_id = setup_demo_data()
    
    # Mathematical demonstrations
    demonstrate_thompson_sampling()
    demonstrate_knowledge_tracing()
    
    # Simulate multiple study sessions to generate data
    print(f"\n🔄 Generating training data with 3 simulated sessions...")
    for session in range(3):
        print(f"Session {session + 1}/3...")
        simulate_study_session(user_id, deck_id)
    
    # Show analytics
    demonstrate_analytics(user_id)
    
    print(f"\n✅ Demo completed! Check 'database/flashforge.db' for generated data.")
    print(f"\nKey Features Demonstrated:")
    print(f"• Thompson Sampling for exploration vs exploitation")
    print(f"• Bayesian Knowledge Tracing for learning modeling")
    print(f"• Hybrid priority scoring")
    print(f"• Time-based knowledge decay")
    print(f"• Performance analytics and parameter optimization")
    print(f"• ML-powered card selection")

def interactive_demo():
    """Interactive demo where user can test specific features"""
    print("\n🎮 Interactive Demo Mode")
    print("Choose a feature to explore:")
    print("1. Thompson Sampling")
    print("2. Knowledge Tracing") 
    print("3. Study Session Simulation")
    print("4. Analytics Dashboard")
    print("5. Run Full Demo")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                demonstrate_thompson_sampling()
            elif choice == "2":
                demonstrate_knowledge_tracing()
            elif choice == "3":
                user_id, deck_id = setup_demo_data()
                simulate_study_session(user_id, deck_id)
            elif choice == "4":
                user_id, deck_id = setup_demo_data()
                demonstrate_analytics(user_id)
            elif choice == "5":
                run_comprehensive_demo()
                break
            else:
                print("Invalid choice. Please enter 0-5.")
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("FlashForge Hybrid Scheduler - Demo & Testing")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        run_comprehensive_demo()
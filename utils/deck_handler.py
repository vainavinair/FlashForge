from utils.db import get_connection

def create_deck(user_id, name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO decks (user_id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()

def list_decks(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT deck_id, name FROM decks WHERE user_id=?", (user_id,))
    decks = cur.fetchall()
    conn.close()
    return decks

from utils.datamodels.db import get_connection


def get_deck_cards(deck_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT question, answer FROM cards WHERE deck_id = ?", (deck_id,))
    cards = cur.fetchall()
    conn.close()
    return cards

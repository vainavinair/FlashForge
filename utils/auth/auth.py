import streamlit as st
from utils.datamodels.db import get_connection

def register_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username=? AND password=?", (username, password))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def logout():
    if "user_id" in st.session_state:
        del st.session_state["user_id"]


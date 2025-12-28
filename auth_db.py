"""
Simple Authentication Database
Stores usernames and hashed passwords
"""

import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"

def init_db():
    """Initialize the database with users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    """Add a new user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      (username, password_hash))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

def verify_user(username, password):
    """Verify username and password"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", 
                  (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    return user is not None

def create_default_user():
    """Create default admin user if no users exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Create default admin user
        password_hash = hash_password("admin123")
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      ("admin", password_hash))
        conn.commit()
    
    conn.close()

# Initialize database on import
init_db()
create_default_user()

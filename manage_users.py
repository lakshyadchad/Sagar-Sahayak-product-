"""
Simple User Management Script
Run this to add/list users from command line

Usage:
    py -3.12 manage_users.py list
    py -3.12 manage_users.py add <username> <password>
"""

import sys
import sqlite3
from pathlib import Path
from auth_db import add_user, hash_password, DB_PATH

def list_users():
    """List all users"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, username, created_at FROM users")
    users = cursor.fetchall()
    
    print("\n" + "="*60)
    print("📋 REGISTERED USERS")
    print("="*60)
    
    if users:
        print(f"{'ID':<5} {'Username':<20} {'Created At':<30}")
        print("-"*60)
        for user in users:
            print(f"{user[0]:<5} {user[1]:<20} {user[2]:<30}")
        print(f"\nTotal users: {len(users)}")
    else:
        print("No users found")
    
    print("="*60 + "\n")
    conn.close()

def add_new_user(username, password):
    """Add a new user"""
    if add_user(username, password):
        print(f"✅ User '{username}' added successfully!")
    else:
        print(f"❌ Failed to add user. Username '{username}' may already exist.")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List users:     py manage_users.py list")
        print("  Add user:       py manage_users.py add <username> <password>")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_users()
    elif command == "add":
        if len(sys.argv) < 4:
            print("❌ Please provide username and password")
            print("Usage: py manage_users.py add <username> <password>")
        else:
            username = sys.argv[2]
            password = sys.argv[3]
            add_new_user(username, password)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, add")

if __name__ == "__main__":
    main()

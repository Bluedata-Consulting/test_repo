import sqlite3
import hashlib
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_NAME = 'media_data.db'

@contextmanager
def db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def hash_password(password):
    """Hashes the password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if cursor.fetchone() is None:
                logging.info("Creating 'users' table...")
                cursor.execute('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL
                    )
                ''')
                # Add a demo user
                username = 'admin'
                password = 'admin123'
                password_hash = hash_password(password)
                try:
                    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
                    logging.info(f"Demo user '{username}' created successfully.")
                except sqlite3.IntegrityError:
                    logging.warning(f"Demo user '{username}' already exists.")
                conn.commit()
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")

def authenticate_user(username, password):
    """
    Authenticates a user against the database.
    Returns the user's ID if successful, otherwise None.
    """
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            password_hash = hash_password(password)
            
            cursor.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
            user = cursor.fetchone()
            
            if user:
                logging.info(f"Authentication successful for user: {username}")
                return user[0]
            else:
                logging.warning(f"Authentication failed for user: {username}")
                return None
    except Exception as e:
        logging.error(f"An error occurred during authentication: {e}")
        return None

# Initialize the database when the module is loaded
initialize_db()

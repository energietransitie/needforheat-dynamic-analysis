import os

import mysql.connector

# only use the lines below if you use a .env file instead of ENV set by a container
# from dotenv import load_dotenv
# load_dotenv()


class Database:
    """
    Use this class to connect and disconnect with the database.
    """

    _db: mysql.connector.connection.MySQLConnection

    def _connect(self):
        self._db = mysql.connector.connect(
            host=os.getenv("DATABASE_HOST"),
            port=os.getenv("DATABASE_PORT"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD"),
            database=os.getenv("DATABASE_NAME")
        )
        
    def _close(self):
        self._db.close()

import mysql.connector

DATABASE_CONFIG = {
    "user": "root",
    "password": "",
    "host": "localhost",
    "port": "3306",
    "database": "trashure",
}


def get_db_connection():
    return mysql.connector.connect(**DATABASE_CONFIG)

import mysql.connector
import pandas as pd
def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="actions_language"
        )
        print("Connected to the database")
        return conn
    except mysql.connector.Error as e:
        print("Error connecting to the database:", e)
        return None

def create_table(conn):
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INT AUTO_INCREMENT PRIMARY KEY,
                     name VARCHAR(255),
                     email VARCHAR(255),
                     password VARCHAR(255))''')
        conn.commit()
        print("Table created successfully")
    except mysql.connector.Error as e:
        print("Error creating table:", e)

def insert_record(conn, name, email, password):
    try:
        c = conn.cursor()
        query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
        values = (name, email, password)
        c.execute(query, values)
        conn.commit()
        print("Record inserted successfully")
    except mysql.connector.Error as e:
        print("Error inserting record:", e)

def fetch_all_records(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM users")
        records = c.fetchall()
        return records
    except mysql.connector.Error as e:
        print("Error fetching records:", e)
        return []

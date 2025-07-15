# db.py
import psycopg2

def get_connection():
    return psycopg2.connect(
        host="ep-spring-heart-a112bfbk-pooler.ap-southeast-1.aws.neon.tech",
        dbname="neondb",
        user="neondb_owner",
        password="npg_deywGL4vNQ2z",
        sslmode="require"
    )
# scripts/database.py
import os
import psycopg2
from psycopg2 import sql


def create_database():
    # Kết nối đến PostgreSQL server
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="tranphuong"  # Thay bằng mật khẩu của bạn
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Tạo database nếu chưa tồn tại
    try:
        cursor.execute("CREATE DATABASE child_face_db")
        print("Database created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print("Database already exists")

    cursor.close()
    conn.close()

    # Kết nối đến database mới tạo
    conn = psycopg2.connect(
        host="localhost",
        database="child_face_db",
        user="postgres",
        password="tranphuong"  # Thay bằng mật khẩu của bạn
    )
    cursor = conn.cursor()

    # Tạo bảng images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images(
        id SERIAL PRIMARY KEY,
        file_name VARCHAR(255) NOT NULL UNIQUE,
        age INTEGER,
        gender VARCHAR(10),
        ethnicity INTEGER,
        path VARCHAR(255) NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Tạo bảng features để lưu các vector đặc trưng
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features(
        id SERIAL PRIMARY KEY,
        image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
        feature_type VARCHAR(50) NOT NULL,
        feature_vector BYTEA NOT NULL
    )
    """)
    conn.commit()

    print("Tables created successfully")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    create_database()
import psycopg2

try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="tranphuong",
        host="localhost",
        port="5432"
    )
    print("✅ Kết nối thành công đến PostgreSQL!")
    conn.close()
except Exception as e:
    print("❌ Kết nối thất bại!")
    print(e)

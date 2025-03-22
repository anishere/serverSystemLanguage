import mysql.connector
from mysql.connector import Error

try:
    # Thiết lập kết nối đến MySQL
    connection = mysql.connector.connect(
        host="localhost",          # Địa chỉ máy chủ (localhost vì chạy trên máy bạn)
        user="root",              # Tên người dùng (mặc định là root)
        password="",              # Mật khẩu (trống nếu bạn chưa đặt)
        database="systemlanguaguedb"        # Tên cơ sở dữ liệu (thay bằng tên bạn đã tạo)
    )

    if connection.is_connected():
        # In thông tin kết nối thành công
        db_info = connection.get_server_info()
        print("Kết nối thành công! Version MySQL:", db_info)

        # Tạo một cursor để thực thi câu lệnh SQL
        cursor = connection.cursor()

        # Ví dụ: Thực hiện truy vấn để lấy tất cả dữ liệu từ bảng users
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()

        # In dữ liệu từ bảng
        print("\nDanh sách người dùng:")
        for row in rows:
            print(row)

        # Ví dụ: Thêm một bản ghi mới vào bảng users
        sql_insert = """
            INSERT INTO users (username, email, password, created_at, updated_at, last_login, is_active, credits)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = ("johndoe", "johndoe@example.com", "hashed_password", "2025-03-18 10:00:00", "2025-03-18 10:00:00", None, 1, 50)
        cursor.execute(sql_insert, values)
        connection.commit()
        print("\nThêm bản ghi thành công! ID của bản ghi mới:", cursor.lastrowid)

except Error as e:
    print("Lỗi khi kết nối đến MySQL:", e)

finally:
    # Đóng cursor và kết nối
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("Kết nối đã được đóng.")
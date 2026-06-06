import pymysql


def truncate_tables_and_reset_auto_increment(db_config, table_names):
    try:
        # 打印开始信息
        print("Starting database operations...")

        # 建立数据库连接
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print(f"Connected to database: {db_config['database']}")
        with connection.cursor() as cursor:
            # 禁用外键检查
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
            print("Foreign key checks disabled.")

            # 开始事务
            connection.begin()
            print("Transaction started.")

            try:
                for table in table_names:
                    # 清空表
                    cursor.execute(f"TRUNCATE TABLE `{table}`;")
                    print(f"Truncated table: {table}")

                    # 重置自增计数器
                    cursor.execute(f"ALTER TABLE `{table}` AUTO_INCREMENT = 1;")
                    print(f"Reset auto_increment for table: {table}")

                # 提交事务
                connection.commit()
                print("All operations completed successfully.")

            except Exception as e:
                # 如果出现错误，回滚事务
                connection.rollback()
                print(f"An error occurred during operation: {e}")
                raise

            finally:
                # 恢复外键检查
                cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
                print("Foreign key checks restored.")

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 关闭数据库连接
        if 'connection' in locals():
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    # 数据库配置信息
    db_config = {
        'host': 'localhost',  # 替换为你的数据库主机地址
        'user': 'root',  # 替换为你的数据库用户名
        'password': '123456',  # 替换为你的数据库密码
        'database': 'xxp'  # 替换为你的数据库名称
    }

    # 要清空的表名列表
    table_names = ['tray', 'defection', 'sample']

    # 执行清空表和重置自增计数器的操作
    truncate_tables_and_reset_auto_increment(db_config, table_names)
import os
import pymysql
import configparser

# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
config.read(config_path)

def get_mysql_conn():
    return pymysql.connect(
        host=config["mysql"]["host"],
        port=int(config["mysql"]["port"]),
        user=config["mysql"]["user"],
        password=config["mysql"]["password"],
        database=config["mysql"]["database_name"]
    )

import os

import oracledb
from dotenv import load_dotenv

load_dotenv()


def get_connection() -> oracledb.Connection:
    user = os.getenv("ORA_USER")
    password = os.getenv("ORA_PASS")
    # dsn = os.getenv("ORA_DSN")
    dsn = "oracle-xe:1521/XEPDB1"
    if not user:
        raise ValueError("ORA_USER environment variable not set")
    if not password:
        raise ValueError("ORA_PASS environment variable not set")
    if not dsn:
        raise ValueError("ORA_DSN environment variable not set")

    return oracledb.connect(user=user, password=password, dsn=dsn)

import logging
import os

import oracledb
from dotenv import load_dotenv

from app.core.logging_config import setup_logging

setup_logging(service_name="database")
logger = logging.getLogger(__name__)

load_dotenv()


def get_connection() -> oracledb.Connection:
    user = os.getenv("ORA_USER", "system")
    password = os.getenv("ORA_PASS", "admin")
    dsn = os.getenv("ORA_DSN", "db:1521/xepdb1")

    if not user:
        raise ValueError("ORA_USER environment variable not set")
    if not password:
        raise ValueError("ORA_PASS environment variable not set")
    if not dsn:
        raise ValueError("ORA_DSN environment variable not set")

    try:
        return oracledb.connect(user=user, password=password, dsn=dsn)
    except oracledb.Error as e:
        logger.error(f"Could not connect to Oracle DB ({dsn}): {e}")
        raise

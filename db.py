
from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from dotenv import load_dotenv

SSL_DIR = Path(__file__).resolve().parent / "ssl"
SSL_FILES = {
    "ssl_ca": "ca-cert.pem",
    "ssl_cert": "client-cert.pem",
    "ssl_key": "client-key.pem",
}


def create_engine() -> sa.Engine:
    load_dotenv()
    missing = [name for name in SSL_FILES.values() if not (SSL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {SSL_DIR}")

    url = sa.URL.create(  # URL.create escapes special characters in the password
        "mysql+mysqlconnector",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_SERVER"],
        port=int(os.environ.get("DB_PORT", 3306)),
        database=os.environ["DB_DATABASE"],
    )
    connect_args = {key: str(SSL_DIR / name) for key, name in SSL_FILES.items()}
    # pre_ping + recycle: the dashboard runs for days, so drop connections the server has closed
    return sa.create_engine(
        url, connect_args=connect_args, pool_pre_ping=True, pool_recycle=1800
    )

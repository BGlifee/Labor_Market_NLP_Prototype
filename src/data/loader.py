from sqlalchemy import create_engine, text
import pandas as pd
from src.config import load_config


def get_db_engine():
    cfg = load_config()["database"]
    url = (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )
    return create_engine(url)

def load_jobs(limit: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    jobs_cfg = cfg["jobs"]
    table = jobs_cfg["table_name"]

    sql = f"SELECT * FROM {table}"
    if limit:
        sql += f" LIMIT {limit}"

    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # text null deletion
    text_col = jobs_cfg["text_column"]
    df = df[df[text_col].notna()].reset_index(drop=True)
    return df
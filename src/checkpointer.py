from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import os

load_dotenv()
database_url=os.getenv("DATABASE_URL")
connection_kwargs={
    "autocommit":True,
    "prepare_threshold":0
}

pool=ConnectionPool(
    conninfo=database_url,
    max_size=10,
    kwargs=connection_kwargs,
    open=False
)

def create_checkpointer():
    pool.open(wait=True)
    checkpointer=PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer



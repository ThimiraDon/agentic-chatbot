from src.checkpointer import pool


#Creates the metadata table if necessary
def create_conversation_table():
    with pool.connection() as conn:
        conn.execute(
             """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )


#Saves a new conversation
def create_conversation(thread_id:str,title:str="New Chat"):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (thread_id, title)
            VALUES (%s, %s)
            ON CONFLICT (thread_id) DO NOTHING;
            """,
            (thread_id, title)
        )


#Retrieves conversations, newest activity first
def list_conversation():
    with pool.connection() as conn:
        rows=conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC;
            """
        ).fetchall()

    return [
        {
            "thread_id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat(),
            "updated_at": row[3].isoformat(),
        }
        for row in rows
    ]


def update_conversation(thread_id: str, title: str | None = None):
    with pool.connection() as conn:
        if title is not None:
            conn.execute(
                """
                UPDATE conversations
                SET title = %s, updated_at = NOW()
                WHERE thread_id = %s;
                """,
                (title, thread_id),
            )
        else:
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = NOW()
                WHERE thread_id = %s;
                """,
                (thread_id,),
            )


def delete_conversation(thread_id: str):
    with pool.connection() as conn:
        conn.execute(
            """
            DELETE FROM conversations
            WHERE thread_id = %s;
            """,
            (thread_id,),
        )
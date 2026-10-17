import logging
import os
from pathlib import Path

LOG_FILE_NAME='chatbot.log'

BASE_DIR=Path(r"C:\Users\Thimira\Desktop\Data\ML & AI\Deep Learning\Agentic AI\agentic-chatbot")

LOG_DIR= BASE_DIR /"logs"
LOG_DIR.mkdir(parents=True,exist_ok=True)

LOG_FILE_PATH=LOG_DIR/LOG_FILE_NAME

log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format=log_format
)
from arq.connections import RedisSettings
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.environ.get("API_KEY")

redis_settings = RedisSettings(
    host='redis',
    port=6379,
    database=0
)

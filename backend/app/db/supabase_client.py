import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Always load backend/.env regardless of the current working directory.
BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        f"Supabase environment variables are missing. Checked env file: {ENV_PATH}"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
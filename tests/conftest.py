import os

# No real keys in tests: agents run on pydantic-ai's TestModel and tracing is off.
os.environ.setdefault("USER_API_KEY", "test-key")
os.environ["ENABLE_TRACING"] = "0"

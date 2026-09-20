import os
from dotenv import load_dotenv

# Load values from .env and override any older environment values
load_dotenv(override=True)

# Safe configuration checks
print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("Endpoint:", os.getenv("LANGSMITH_ENDPOINT"))
print("Workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"))
print(
    "API key loaded:",
    bool(os.getenv("LANGSMITH_API_KEY"))
)

from langsmith import traceable


@traceable(name="demografy_test_trace")
def test_lookup(reference_sa2_code):
    return {
        "reference_sa2_code": reference_sa2_code,
        "neighbour_codes": [
            "TEST001",
            "TEST002",
            "TEST003",
        ],
    }


result = test_lookup("TEST_REFERENCE")

print("LangSmith test completed.")
print(result)
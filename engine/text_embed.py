import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def embed_profile(profile_text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=profile_text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return response.embeddings[0].values
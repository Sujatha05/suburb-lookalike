import os
import time
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 768

CACHE_DIR = "cache"

FINAL_CACHE_FILE = os.path.join(
    CACHE_DIR,
    "text_embeddings.npy"
)

PARTIAL_CACHE_FILE = os.path.join(
    CACHE_DIR,
    "text_embeddings_partial.npy"
)

PROGRESS_FILE = os.path.join(
    CACHE_DIR,
    "text_embeddings_progress.txt"
)


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def embed_profiles(
    profiles,
    batch_size=40,
    max_retries=5
):

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    # -------------------------------------------------
    # 1. If final embeddings already exist, use them
    # -------------------------------------------------

    if os.path.exists(FINAL_CACHE_FILE):

        print(
            "Final embeddings found in cache."
        )

        try:

            embeddings = np.load(
                FINAL_CACHE_FILE
            )

            print(
                f"Loaded {len(embeddings)} "
                f"embeddings from disk."
            )

            return embeddings

        except (
            EOFError,
            ValueError,
            OSError
        ) as e:

            print(
                "Final cache file is empty "
                "or corrupted."
            )

            print(
                f"Error: {e}"
            )

            print(
                "Deleting invalid final cache "
                "and continuing..."
            )

            os.remove(
                FINAL_CACHE_FILE
            )


    # -------------------------------------------------
    # 2. Check whether a previous run was interrupted
    # -------------------------------------------------

    if (
        os.path.exists(PARTIAL_CACHE_FILE)
        and
        os.path.exists(PROGRESS_FILE)
    ):

        print(
            "Partial embeddings found."
        )

        all_embeddings = np.load(
            PARTIAL_CACHE_FILE
        ).tolist()

        with open(
            PROGRESS_FILE,
            "r"
        ) as file:

            start_index = int(
                file.read().strip()
            )

        print(
            f"Resuming from profile "
            f"{start_index + 1}"
        )

    else:

        print(
            "No previous embeddings found."
        )

        print(
            "Starting from profile 1."
        )

        all_embeddings = []

        start_index = 0


    # -------------------------------------------------
    # 3. Generate embeddings in batches
    # -------------------------------------------------

    for start in range(
        start_index,
        len(profiles),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(profiles)
        )

        batch = profiles[start:end]


        print(
            f"\nEmbedding profiles "
            f"{start + 1} to {end} "
            f"of {len(profiles)}"
        )


        for attempt in range(
            max_retries
        ):

            try:

                response = client.models.embed_content(
                    model=MODEL_NAME,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=
                        EMBEDDING_DIM
                    )
                )


                batch_embeddings = [
                    embedding.values
                    for embedding
                    in response.embeddings
                ]


                all_embeddings.extend(
                    batch_embeddings
                )


                # -----------------------------------------
                # 4. Save checkpoint after every batch
                # -----------------------------------------

                partial_array = np.array(
                    all_embeddings,
                    dtype=np.float32
                )


                np.save(
                    PARTIAL_CACHE_FILE,
                    partial_array
                )


                with open(
                    PROGRESS_FILE,
                    "w"
                ) as file:

                    file.write(
                        str(end)
                    )


                print(
                    f"Checkpoint saved."
                )

                print(
                    f"{end} profiles completed."
                )


                # Give Gemini some breathing room
                time.sleep(30)


                break


            except Exception as e:

                error_text = str(e)

                print(
                    f"Embedding attempt "
                    f"{attempt + 1} failed:"
                )

                print(e)


                # -----------------------------------------
                # DAILY QUOTA EXCEEDED
                # -----------------------------------------

                if (
                    "PerDay" in error_text
                    or
                    "RequestsPerDay" in error_text
                ):

                    print(
                        "\nDaily Gemini embedding "
                        "quota has been reached."
                    )

                    print(
                        "Your completed embeddings "
                        "have already been saved."
                    )

                    print(
                        "Run the program build_embeddings.py again after "
                        "the quota resets."
                    )

                    print(
                        "The program will resume from "
                        "the last completed profile."
                    )

                    return np.array(
                        all_embeddings,
                        dtype=np.float32
                    )
                    
                    raise SystemExit(
                        "Embedding stopped because "
                        "daily Gemini quota was reached."
                    )


                # -----------------------------------------
                # PER-MINUTE RATE LIMIT
                # -----------------------------------------

                if (
                    "429" in error_text
                    or
                    "RESOURCE_EXHAUSTED" in error_text
                ):

                    wait_time = 65


                else:

                    wait_time = 2 ** attempt


                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )


    # -------------------------------------------------
    # 5. All profiles completed
    # -------------------------------------------------

    embeddings = np.array(
        all_embeddings,
        dtype=np.float32
    )


    np.save(
        FINAL_CACHE_FILE,
        embeddings
    )


    print(
        "\nAll embeddings completed."
    )

    print(
        f"Final embeddings saved to:"
    )

    print(
        FINAL_CACHE_FILE
    )


    # -------------------------------------------------
    # 6. Remove temporary checkpoint files
    # -------------------------------------------------

    if os.path.exists(
        PARTIAL_CACHE_FILE
    ):

        os.remove(
            PARTIAL_CACHE_FILE
        )


    if os.path.exists(
        PROGRESS_FILE
    ):

        os.remove(
            PROGRESS_FILE
        )


    print(
        "Temporary checkpoint files removed."
    )


    return embeddings

def get_or_create_embeddings(profiles):
    return embed_profiles(profiles)
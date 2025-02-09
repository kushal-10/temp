import weaviate
import weaviate.classes.query as wq
import os
from weaviate.classes.init import Auth
from restack_ai.function import function
from pydantic import BaseModel
import logging
import asyncio

log = logging.getLogger(__name__)

# Environment variables
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")

# Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)

# Define Pydantic model for the book
class Book1(BaseModel):
    content: str

# Function to look up books
@function.defn
async def lookup_book(query: str) -> list[Book1]:
    try:
        log.info("lookup_book function started")
        if not query or not isinstance(query, str):
            raise ValueError("Query input must be a non-empty string")
                # Get the collection
        books_collection = client.collections.get("Book")

        # Perform query
        response = books_collection.query.near_text(
            query=query, limit=1000, return_metadata=wq.MetadataQuery(distance=True)
        )

        # Store results in Book1 instances
        items = [Book1(content=o.properties["text"]) for o in response.objects]

        log.info("lookup_book function completed successfully")
        return items  # Return list of Book1 objects

    except Exception as e:
        log.error("lookup_book function failed", exc_info=True)
        raise e

if __name__ == "__main__":
    asyncio.run(lookup_book())

#!/usr/bin/env python
"""
Initialize Qdrant knowledge base with test documents.
"""
import os
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
COLLECTION_NAME = "documents"


def main():
    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

    # Load documents from data directory
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jsonl'):
            with open(os.path.join(DATA_DIR, filename), 'r') as f:
                doc = json.loads(f.read())
                doc_id = doc.get("doc_id", filename)
                content = doc.get("content", doc.get("doc_content", ""))
                title = doc.get("title", doc.get("doc_id", ""))

                documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "content": content
                })

    if not documents:
        print("No documents found in data directory")
        return

    print(f"Found {len(documents)} documents")

    # Initialize embedding model (384 dimensions, matching Qdrant collection)
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create points with embeddings
    points = []
    for i, doc in enumerate(documents):
        embedding = embedding_model.encode(doc["content"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": doc["content"]
                }
            )
        )

    # Upsert to Qdrant
    print(f"Upserting {len(points)} documents...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Done! Knowledge base initialized.")

    # Verify
    result = client.get_collection(COLLECTION_NAME)
    print(f"Collection now has {result.points_count} points")


if __name__ == "__main__":
    main()

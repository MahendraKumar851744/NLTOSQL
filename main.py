import json
import asyncio
from typing import Dict, List, Any
from llm_service_vllm import RecursiveExtractionService


def load_data(path: str = "updated_data.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_results(question: str, matched_items: List[Dict[str, Any]]):
    print(f"\n{'=' * 70}")
    print(f"Question : {question}")
    print(f"{'=' * 70}")

    if not matched_items:
        print("No matching items found.")
        return

    print(json.dumps(matched_items, indent=2))


async def run(question: str, data: List[Dict[str, Any]]):
    service = RecursiveExtractionService()

    print(f"\nStarting hierarchical traversal for question: '{question}'")
    print(f"Top-level entities: {len(data)}")

    matched_items, summary = await service.extract_from_new_data(question, data)

    print_results(question, matched_items)

    return matched_items, summary


def main():
    data = load_data()

    question = input("Enter your question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    asyncio.run(run(question, data))


if __name__ == "__main__":
    main()

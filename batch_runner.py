import json
import random
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any
from llm_service_vllm import RecursiveExtractionService


DATA_PATH      = "updated_data.json"
QUESTIONS_PATH = "questions.txt"
OUTPUT_PATH    = "batch_results.jsonl"


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


async def run_all(questions: List[str], data: List[Dict[str, Any]]):
    service = RecursiveExtractionService()
    total = len(questions)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        for idx, question in enumerate(questions, 1):
            print(f"\n{'#' * 60}")
            print(f"[{idx}/{total}] {question}")
            print(f"{'#' * 60}")

            try:
                result, summary = await service.extract_from_new_data(question, data)
                status = "ok"
                error  = None
            except Exception as e:
                result  = []
                summary = {}
                status  = "error"
                error   = str(e)
                print(f"ERROR: {e}")

            record = {
                "index":     idx,
                "question":  question,
                "status":    status,
                "summary":   summary,
                "result":    result,
                "error":     error,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # write immediately so progress is not lost on crash

    print(f"\n{'=' * 60}")
    print(f"Batch complete. {total} questions processed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for NLToSQL extraction")
    parser.add_argument("n", type=int, help="Number of questions to process")
    args = parser.parse_args()

    data      = load_data(DATA_PATH)
    questions = load_questions(QUESTIONS_PATH)

    if not questions:
        print("No questions found in questions.txt")
        return

    random.shuffle(questions)
    questions = questions[:args.n]

    print(f"Loaded {len(questions)} randomly selected questions (from {QUESTIONS_PATH})")
    asyncio.run(run_all(questions, data))


if __name__ == "__main__":
    main()

import json
import random
import requests
import re
import time

# Terminal Colors for a professional CLI experience
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_random_entry(file_path):
    """Memory-efficiently picks a random line from a large .jsonl file."""
    selected_line = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if random.randrange(i + 1) == 0:
                    selected_line = line
        return json.loads(selected_line) if selected_line else None
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: {file_path} not found.{Colors.ENDC}")
        return None

def format_sql(sql):
    """Beautifies SQL by adding newlines and indentation."""
    keywords = ["SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "INNER JOIN", "GROUP BY", "ORDER BY", "AND", "LIMIT"]
    formatted = sql
    for kw in keywords:
        formatted = re.sub(rf'\b{kw}\b', f'\n{kw}', formatted, flags=re.IGNORECASE)
    return "\n".join([line.strip() for line in formatted.split('\n') if line.strip()])

def format_sql_context(results):
    schema = {}
    for item in results:
        table = item.get("table", "UNKNOWN_TABLE")
        if table not in schema:
            schema[table] = []
        if item["field_type"] == "leaf":
            schema[table].append(f"  - {item['column']} ({item['field']})")

    context_str = "DATABASE SCHEMA:\n"
    for table, columns in schema.items():
        context_str += f"Table: {table}\n"
        context_str += "\n".join(columns) if columns else "  (No specific columns)"
        context_str += "\n\n"
    return context_str

def call_vllm_model(prompt):
    """Calls the vLLM server, prints metadata and RENDERED prompt, then measures time."""
    url = "http://localhost:8000/v1/completions"
    
    # The actual payload sent to the API
    payload = {
        "model": "sqlcoder-7b-2.Q4_K_M.gguf",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0,
        "stop": ["#", ";", "###"]
    }

    # --- NEW NEAT PRINTING LOGIC ---
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*20} API REQUEST START {'='*20}{Colors.ENDC}")
    
    # 1. Print Metadata (everything except the giant prompt)
    metadata = {k: v for k, v in payload.items() if k != 'prompt'}
    print(f"{Colors.CYAN}{Colors.BOLD}METADATA:{Colors.ENDC}")
    print(f"{Colors.CYAN}{json.dumps(metadata, indent=4)}{Colors.ENDC}")
    
    # 2. Print the RENDERED PROMPT (so \n actually breaks the line)
    print(f"\n{Colors.CYAN}{Colors.BOLD}RENDERED PROMPT SENT TO MODEL:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*60}")
    print(f"{Colors.ENDC}{prompt}") # This renders the newlines properly
    print(f"{Colors.BLUE}{'-'*60}{Colors.ENDC}")

    # Measure Time
    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload)
        end_time = time.perf_counter()
        duration = end_time - start_time

        if response.status_code == 200:
            return response.json()['choices'][0]['text'].strip(), duration
        else:
            print(f"{Colors.FAIL}Server Error: {response.status_code}{Colors.ENDC}")
            return None, duration
    except Exception as e:
        print(f"{Colors.FAIL}Connection Error: {e}{Colors.ENDC}")
        return None, 0

if __name__ == "__main__":
    entry = get_random_entry("batch_results.jsonl")
    
    if entry:
        question = entry.get("question", "")
        context = format_sql_context(entry.get("result", []))

        # Build Prompt
        final_prompt = f"### TASK: Generate a MySQL statement.\n\n### SCHEMA:\n{context}\n### QUESTION:\n{question}\n\n### MYSQL STATEMENT:\n"
        
        # Execute Request
        raw_sql, time_taken = call_vllm_model(final_prompt)
        
        if raw_sql:
            formatted_sql = format_sql(raw_sql)
            
            # Final Summary Output
            print(f"\n{Colors.YELLOW}{Colors.BOLD}USER QUESTION:{Colors.ENDC} {question}")
            print(f"\n{Colors.GREEN}{Colors.BOLD}SUCCESS! GENERATED MYSQL:{Colors.ENDC}")
            print(f"{Colors.GREEN}{'='*60}")
            print(formatted_sql)
            print(f"{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}Total Generation Time:{Colors.ENDC} {Colors.GREEN}{time_taken:.4f} seconds{Colors.ENDC}\n")
        else:
            print(f"{Colors.FAIL}Failed to generate SQL.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}No valid entry found in batch_results.jsonl.{Colors.ENDC}")
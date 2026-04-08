import json
import asyncio
import logging
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    chunk_type: str
    extracted_keywords: List[str]
    success: bool
    response_log: Optional[List[str]] = None
    error_message: Optional[str] = None

class RecursiveExtractionService:
    # 1. UPDATED THE API URL TO MATCH vLLM's OPENAI ENDPOINT
    def __init__(self, api_url: str = "http://192.168.168.31:8000/v1/chat/completions", max_concurrent_requests: int = 20, log_file: str = "api_analysis.jsonl", prompts_file: str = "prompts.json"):
        self.api_url = api_url
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.log_file = log_file

        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        self.node_prompt: str = prompts["node"]
        self.leaf_prompt: str = prompts["leaf"]

        # Per-extraction counters
        self._sent: int = 0
        self._selected: int = 0
        self._api_calls: int = 0

    def _chunk_list(self, data_list: List[Any], chunk_size: int = 20):
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i:i + chunk_size]

    def _log_to_file(self, passage: str, chunk_type: str, glossary: dict, extracted: list):
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_type": chunk_type.upper(),
            "passage": passage,
            "glossary_size": len(glossary),
            "glossary": glossary,
            "extracted_count": len(extracted),
            "extracted": extracted
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    async def _process_chunk(self, chunk_type: str, target_text: str, glossary_chunk: Any, system_prompt: str = "") -> ExtractionResult:
        formatted_glossary = {}
        if chunk_type == "leaf":
            for key, val in glossary_chunk:
                formatted_glossary[key] = val
        else:
            formatted_glossary = glossary_chunk

        # 2. FORMAT GLOSSARY FOR THE LLM PROMPT (vLLM expects raw text strings)
        glossary_lines = [f"- {k}: {v}" for k, v in formatted_glossary.items()]
        glossary_str = "\n".join(glossary_lines)

        sys_msg = system_prompt or "You are a helpful assistant. Always respond in valid JSON format only. No markdown, no explanation."
        user_msg = f"Text passage:\n{target_text}\n\nGlossary:\n{glossary_str}"

        # 3. USE STANDARD OPENAI PAYLOAD SCHEMA
        payload = {
            "model": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", # Must exactly match the vLLM server model name
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }

        try:
            async with self.semaphore:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=payload) as response:
                        response.raise_for_status()
                        result_data = await response.json()

                # 4. PARSE vLLM OPENAI RESPONSE
                raw_content = result_data["choices"][0]["message"]["content"].strip()

                # Clean up <think> tags if Qwen uses them
                if "<think>" in raw_content:
                    think_end = raw_content.find("</think>")
                    if think_end != -1:
                        raw_content = raw_content[think_end + len("</think>"):].strip()

                # Clean up markdown code fences (```json ... ```)
                if raw_content.startswith("```"):
                    lines = raw_content.splitlines()
                    raw_content = "\n".join(
                        line for line in lines if not line.strip().startswith("```")
                    ).strip()

                # Parse the string back into a JSON list
                try:
                    extracted_list = json.loads(raw_content)
                    if not isinstance(extracted_list, list):
                        logger.warning(f"Model returned JSON, but not a list: {raw_content}")
                        extracted_list = []
                        return ExtractionResult(chunk_type=chunk_type, extracted_keywords=[], success=False, error_message="Response was not a JSON list.")
                except json.JSONDecodeError:
                    logger.warning(f"Model returned invalid JSON: {raw_content}")
                    extracted_list = []
                    return ExtractionResult(chunk_type=chunk_type, extracted_keywords=[], success=False, error_message="Invalid JSON format.")

                # Filter against valid keys to ensure no hallucinations
                valid_keys = set(formatted_glossary.keys())
                extracted_list = [k for k in extracted_list if k in valid_keys]

                self._api_calls += 1
                self._sent      += len(valid_keys)
                self._selected  += len(extracted_list)

                self._log_to_file(target_text, chunk_type, formatted_glossary, extracted_list)

                return ExtractionResult(
                    chunk_type=chunk_type, 
                    extracted_keywords=extracted_list, 
                    success=True, 
                    response_log=extracted_list
                )

        except Exception as e:
            logger.error(f"Error calling vLLM API: {e}")
            return ExtractionResult(chunk_type=chunk_type, extracted_keywords=[], success=False, error_message=str(e))

    # ... REST OF YOUR EXISTING CODE CONTINUES HERE ...
    # async def _traverse_children(self, question: str, node: Dict[str, Any], depth: int = 0) -> List[Dict[str, Any]]:
    async def _traverse_children(self, question: str, node: Dict[str, Any], depth: int = 0) -> List[Dict[str, Any]]:
        """
        Given a node with a 'children' key (list of groups, each group is a list of items),
        dispatches parallel API requests with different strategies per field_type:
          - "node"  → one individual call per node (so each gets its own focused request)
          - "leaf"  → one batch call per group (all leaves in the group sent together)
        Selected nodes are recursed into; selected leaves are added to results.
        Returns a flat list of matched field metadata dicts.
        """
        children_groups = node.get("children", [])
        indent = "  " * depth

        if not children_groups:
            # Leaf node — return its own metadata
            if node.get("column"):
                return [{
                    "field": node["field"],
                    "table": node["table"],
                    "column": node["column"],
                    "description": node["description"],
                }]
            return []

        tasks = []
        task_meta = []  # parallel list tracking what each task represents
        total_nodes  = 0
        total_leaves = 0

        for group_idx, group in enumerate(children_groups):
            leaves = [item for item in group if item.get("field_type") == "leaf"]
            nodes  = [item for item in group if item.get("field_type") == "node"]

            total_leaves += len(leaves)
            total_nodes  += len(nodes)

            # Leaves — one batch call for all leaves in this group
            if leaves:
                glossary = {item["field"]: item["description"] for item in leaves}
                tasks.append(self._process_chunk(f"leaves_g{group_idx}", question, glossary, self.leaf_prompt))
                task_meta.append({"type": "leaves", "items": leaves})

            # Nodes — one individual call per node
            for item in nodes:
                glossary = {item["field"]: item["description"]}
                tasks.append(self._process_chunk(f"node_{item['field']}", question, glossary, self.node_prompt))
                task_meta.append({"type": "node", "item": item})

        results = await asyncio.gather(*tasks)

        matched_items = []
        recurse_tasks = []
        selected_nodes = []
        selected_leaf_fields = []

        for result, meta in zip(results, task_meta):
            if not result.success:
                continue
            selected = set(result.extracted_keywords)

            if meta["type"] == "leaves":
                for item in meta["items"]:
                    if item["field"] in selected:
                        matched_items.append({
                            "field":       item["field"],
                            "table":       item.get("table", ""),
                            "column":      item.get("column", ""),
                            "field_type":  "leaf",
                            "description": item.get("description", ""),
                        })
                        selected_leaf_fields.append(item["field"])

            elif meta["type"] == "node":
                if meta["item"]["field"] in selected:
                    item = meta["item"]
                    selected_nodes.append(item)
                    # Capture the node itself before recursing into its children
                    matched_items.append({
                        "field":             item["field"],
                        "table":             item.get("table", ""),
                        "column":            item.get("column", ""),
                        "field_type":        "node",
                        "description":       item.get("description", ""),
                        "short_description": item.get("short_description", ""),
                    })
                    recurse_tasks.append(self._traverse_children(question, item, depth + 1))

        # Print compact traversal summary for this node
        label = node.get("field", "root").upper()
        parts = []
        if total_nodes:
            node_names = ", ".join(n["field"] for n in selected_nodes) if selected_nodes else "none"
            parts.append(f"nodes {len(selected_nodes)}/{total_nodes} → {node_names}")
        if total_leaves:
            leaf_names = ", ".join(selected_leaf_fields) if selected_leaf_fields else "none"
            parts.append(f"leaves {len(selected_leaf_fields)}/{total_leaves} → {leaf_names}")
        print(f"{indent}[{label}] {' | '.join(parts) if parts else '— no children'}")

        if recurse_tasks:
            recurse_results = await asyncio.gather(*recurse_tasks)
            for rr in recurse_results:
                matched_items.extend(rr)

        return matched_items

    async def extract_from_new_data(self, question: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for the new updated_data.json structure.

        The top-level array is treated as the children of a virtual root node,
        so the same _traverse_children logic applies at every level uniformly:
        parallel requests per group → select fields → recurse into non-leaves.

        Returns a flat list of matched field metadata dicts:
          [{"field": ..., "table": ..., "column": ..., "description": ...}, ...]
        """
        total_keywords = self._count_keywords(data)
        self._sent = 0
        self._selected = 0
        self._api_calls = 0

        print(f"\nTraversing: '{question}'\n{'-' * 50}")
        root = {
            "field": "root",
            "description": "",
            "table": "",
            "column": "",
            "children": [data],   # top-level entities form one group
        }
        t0 = time.perf_counter()
        result = await self._traverse_children(question, root)
        elapsed = time.perf_counter() - t0

        summary = {
            "total_keywords": total_keywords,
            "api_requests":   self._api_calls,
            "sent_to_api":    self._sent,
            "final_filtered": len(result),
            "time_seconds":   round(elapsed, 2),
        }

        print(f"{'-' * 50}")
        print(f"Total keywords in data : {summary['total_keywords']}")
        print(f"API requests made      : {summary['api_requests']}")
        print(f"Sent to API calls      : {summary['sent_to_api']}")
        print(f"Final filtered         : {summary['final_filtered']}")
        print(f"Time taken             : {summary['time_seconds']}s")
        return result, summary

    @staticmethod
    def _count_keywords(data: List[Dict[str, Any]]) -> int:
        """Recursively counts every named item (nodes + leaves) in the data."""
        def _recurse(node: Dict) -> int:
            count = 1  # the node itself
            for group in node.get("children", []):
                for child in group:
                    count += _recurse(child)
            return count

        return sum(_recurse(entity) for entity in data)

    # ------------------------------------------------------------------ #
    # Legacy methods kept for backwards compatibility                      #
    # ------------------------------------------------------------------ #

    async def extract_tree_recursive(self, node_name: str, node_data: Dict[str, Any], target_text: str) -> Dict[str, List[str]]:

        leaves = node_data.get("LEAVES", [])

        child_nodes = node_data.get("NODES", [])

        complete_glossary = {}

        for leaf_group in leaves:

            if isinstance(leaf_group, dict):

                complete_glossary.update(leaf_group)

        if child_nodes:

            for c in child_nodes:

                complete_glossary[c["NODE"]] = c.get("DESCRIPTION", "")

        tasks = []

        for leaf_group in leaves:

            if isinstance(leaf_group, dict):

                leaf_items = list(leaf_group.items())

                for leaf_chunk in self._chunk_list(leaf_items, chunk_size=20):

                    tasks.append(self._process_chunk("leaf", target_text, leaf_chunk))

        if child_nodes:

            nodes_glossary = {c["NODE"]: c.get("DESCRIPTION", "") for c in child_nodes}

            tasks.append(self._process_chunk("node", target_text, nodes_glossary))

        results = await asyncio.gather(*tasks)

        extracted_leaves, activated_nodes = set(), set()

        for res in results:

            if res.success:

                if res.chunk_type == "leaf":

                    extracted_leaves.update(res.extracted_keywords)

                elif res.chunk_type == "node":

                    activated_nodes.update(res.extracted_keywords)

        final_extraction = {node_name: list(extracted_leaves)} if extracted_leaves else {}

        child_tasks = [

            self.extract_tree_recursive(f"{node_name}->{c['NODE']}", c, target_text)

            for c in child_nodes if c["NODE"] in activated_nodes

        ]

        if child_tasks:

            child_results = await asyncio.gather(*child_tasks)

            for cr in child_results:

                final_extraction.update(cr)

        return final_extraction

    async def extract_from_grouping(self, question: str, grouping_data: List[List[Dict[str, Any]]]) -> Dict[str, Any]:

        tasks = []

        for idx, glossary_object in enumerate(grouping_data):

            glossary = {item["field"]: item["description"] for item in glossary_object}

            tasks.append(self._process_chunk(f"glossary_{idx}", question, glossary))

        results = await asyncio.gather(*tasks)

        all_keywords = []

        for result in results:

            if result.success:

                all_keywords.extend(result.extracted_keywords)

        return {
            "extracted_keywords": list(dict.fromkeys(all_keywords)),
            "total_objects_processed": len(grouping_data),
            "successful_requests": sum(1 for r in results if r.success),
            "details": [
                {
                    "object_index": i,
                    "success": r.success,
                    "extracted": r.extracted_keywords,
                    "error": r.error_message,
                }
                for i, r in enumerate(results)
            ],
        }

    async def generate_mysql_command(self, question: str, metadata: List[Dict[str, Any]]) -> str:

        raise NotImplementedError("SQL Generation endpoint not yet configured on the local server.")

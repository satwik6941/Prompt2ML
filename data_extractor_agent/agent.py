"""
Dataset Extractor Agent (Google ADK)

Searches Kaggle and HuggingFace for relevant datasets,
downloads them, and saves to the datasets/ folder.
Reads user requirements from pipeline_state.json.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from pipeline_state import load_state, save_state

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

sys.stdout.reconfigure(line_buffering=True)

# kagglehub reads KAGGLE_API_TOKEN natively (bearer token format KGAT_...) — no rename needed.
# HuggingFace libraries check HF_TOKEN; our .env stores it as HUGGING_FACE_TOKEN — bridge the gap.
_hf_token = os.getenv("HUGGING_FACE_TOKEN", "")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================
# TOOLS
# ============================================================

def search_datasets(query: str) -> str:
    """
    Search for datasets on Kaggle and HuggingFace using Tavily.

    Args:
        query: A search query describing the type of dataset needed
            (e.g. 'YouTube trending videos dataset with views likes comments').

    Returns:
        Formatted string with dataset titles, URLs, and descriptions.
    """
    try:
        print(f"[INFO] Searching for datasets with query: '{query}'...", flush=True)

        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=["kaggle.com", "huggingface.co"]
        )

        results = []
        for result in response.get('results', []):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No description')
            results.append(f"Title: {title}\nURL: {url}\nDescription: {content}\n")

        if results:
            print(f"[SUCCESS] Found {len(results)} datasets!", flush=True)
            return "\n---\n".join(results)
        else:
            print("[WARNING] No datasets found.", flush=True)
            return "No datasets found. Try a different search query."

    except Exception as e:
        error_msg = f"Error searching for datasets: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def download_kaggle_dataset(dataset_name: str) -> str:
    """
    Download a dataset from Kaggle using kagglehub.

    Args:
        dataset_name: The Kaggle dataset identifier in format 'username/dataset-name'
            (e.g. 'datasnaek/youtube-new').

    Returns:
        Success message with download path, or error message.
    """
    import kagglehub

    try:
        print(f"[INFO] Downloading Kaggle dataset '{dataset_name}'...", flush=True)

        download_path = kagglehub.dataset_download(dataset_name)

        base_dir = Path(__file__).parent.parent / "datasets"
        save_dir = base_dir / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(download_path)
        if source_path.exists():
            for item in source_path.iterdir():
                dest = save_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        save_state({
            "downloaded_dataset": {
                "source": "kaggle",
                "dataset_name": dataset_name,
                "path": str(save_dir.absolute())
            },
            "status": "dataset_ready"
        })

        success_msg = f"Successfully downloaded Kaggle dataset '{dataset_name}' to {save_dir.absolute()}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg
    except Exception as e:
        error_msg = f"Error downloading Kaggle dataset: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def download_huggingface_dataset(dataset_name: str) -> str:
    """
    Download a dataset from HuggingFace.

    Args:
        dataset_name: The HuggingFace dataset identifier in format 'username/dataset-name'
            (e.g. 'stanfordnlp/imdb').

    Returns:
        Success message with download path, or error message.
    """
    from datasets import load_dataset as hf_load_dataset

    try:
        base_dir = Path(__file__).parent.parent / "datasets"
        dataset_path = base_dir / dataset_name.replace("/", "_")
        dataset_path.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading HuggingFace dataset '{dataset_name}'...", flush=True)

        dataset = hf_load_dataset(dataset_name)
        dataset.save_to_disk(str(dataset_path))

        save_state({
            "downloaded_dataset": {
                "source": "huggingface",
                "dataset_name": dataset_name,
                "path": str(dataset_path.absolute())
            },
            "status": "dataset_ready"
        })

        success_msg = f"Successfully downloaded HuggingFace dataset '{dataset_name}' to {dataset_path.absolute()}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg
    except Exception as e:
        error_msg = f"Error downloading HuggingFace dataset: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def get_dataset_requirements() -> str:
    """
    Load the user's report, goal, and pre-phase research context from pipeline_state.json.
    This gives the agent full context — including research-backed dataset recommendations.

    Returns:
        JSON string with user goal, report, and dataset_search_research findings.
    """
    state = load_state()
    requirements = {
        "user_goal": state.get("user_goal", ""),
        "report": state.get("report", "")[:5000] if state.get("report") else "",
        "status": state.get("status", ""),
        "dataset_search_research": state.get("dataset_search_research", {}),
    }
    return json.dumps(requirements, indent=2)


# ============================================================
# AGENT DEFINITION
# ============================================================

from google.adk.agents import Agent, SequentialAgent


# ── Dataset Research Agent (embedded — runs first in this pipeline) ──────────

def _save_dataset_research(
    primary_source: str,
    recommended_ids: str,
    search_queries: str,
    source_reasoning: str,
    warnings: str,
) -> str:
    """
    Save dataset research findings so the extractor agent (next in pipeline) uses them.

    Args:
        primary_source: 'kaggle' or 'huggingface' — which to search first
        recommended_ids: Specific dataset IDs to try (one per line).
            Kaggle format: 'username/dataset-name'. HF format: 'org/dataset-name'
        search_queries: Best queries for the extractor to fall back on (one per line)
        source_reasoning: Why this source was chosen for this task type
        warnings: Licensing issues, size limitations, or quality caveats
    Returns:
        Confirmation JSON.
    """
    save_state({
        "dataset_search_research": {
            "primary_source": primary_source,
            "recommended_ids": recommended_ids,
            "search_queries": search_queries,
            "source_reasoning": source_reasoning,
            "warnings": warnings,
        }
    })
    return json.dumps({"status": "saved"}, indent=2)


_research_tools = [get_dataset_requirements, search_datasets, _save_dataset_research]

_dataset_research_agent = Agent(
    model="gemini-3.1-flash-lite",
    name="dataset_research_agent",
    description="Researches the best datasets and download sources before the extractor runs.",
    output_key="dataset_research_output",
    instruction="""You are the Dataset Research Agent — the FIRST agent in the data extraction pipeline.

PIPELINE CONTEXT
You live inside a SequentialAgent:
  [You] dataset_research_agent → dataset_extractor_agent

The dataset_extractor_agent runs IMMEDIATELY after you and reads your findings
from pipeline_state.json. Give it SPECIFIC, ACTIONABLE intelligence — exact
dataset IDs it can download without any further searching.

YOUR TASK
1. Call get_dataset_requirements() to load the user's goal, report, and ML task type.

2. Determine the primary dataset source based on task type:
   • NLP / text (sentiment, classification, NER, QA, translation) → HuggingFace first
   • Tabular / CSV / structured (regression, classification with numeric features) → Kaggle first
   • Computer vision / images → both; Kaggle for competition sets, HF for benchmarks
   • Time-series / financial / sensor → Kaggle first
   • Multi-modal → HuggingFace first

3. Search for SPECIFIC dataset IDs using all available tools:
   • Use search_datasets() with targeted queries like '<task> <domain> dataset 2024'
   • Use Google Search (if available) for: 'site:kaggle.com/datasets <task>'
     or 'site:huggingface.co/datasets <task>'
   • Find REAL identifiers — not vague suggestions

4. Call _save_dataset_research() with:
   • primary_source: 'kaggle' or 'huggingface'
   • recommended_ids: At least 3 specific IDs the extractor should try first
   • search_queries: 2-3 fallback queries if IDs fail
   • source_reasoning: One sentence justifying the source choice
   • warnings: Any known quality/license issues

RULES
- Never make up dataset IDs — verify them with search
- Be specific: 'datasnaek/youtube-new' not 'youtube dataset on kaggle'
- The extractor trusts your IDs and tries them directly
""",
    tools=_research_tools,
)

# ── HuggingFace MCP (optional, adds richer HF search to extractor) ──────────
_hf_mcp = None
try:
    from mcp_servers.mcp_servers import hugging_face_mcp as _hf_mcp_obj
    _hf_mcp = _hf_mcp_obj
except Exception:
    pass

_extractor_tools = [
    get_dataset_requirements,
    search_datasets,
    download_kaggle_dataset,
    download_huggingface_dataset,
]
if _hf_mcp is not None:
    _extractor_tools.append(_hf_mcp)

_dataset_extractor_core = Agent(
    model="gemini-3.1-flash-lite",
    name="dataset_extractor_core",
    description="Searches Kaggle and HuggingFace for relevant datasets and downloads them.",
    instruction="""You are the Dataset Extractor Agent. Your task is to find and download the best datasets for the user's ML project.

WORKFLOW:
1. Call get_dataset_requirements to understand what the user needs.
   - READ the 'dataset_search_research' field — it contains pre-researched dataset recommendations.
     If it lists specific dataset IDs, try those FIRST before doing a broad search.

2. Determine the PRIMARY data source based on the ML task type from the report:
   - NLP / text classification / sentiment / QA / translation → PRIMARY: HuggingFace
     (use download_huggingface_dataset; fall back to Kaggle only if HF has nothing good)
   - Tabular / structured / CSV / numeric / mixed → PRIMARY: Kaggle
     (use download_kaggle_dataset; check HF too for benchmark tabular datasets)
   - Image / audio / video / CV tasks → search BOTH; try Kaggle first (larger labeled sets)
   - Time-series / financial / sensor data → PRIMARY: Kaggle; check HF for domain-specific series
   - Multi-modal → PRIMARY: HuggingFace (better multi-modal dataset support)

3. Use search_datasets to find relevant datasets on the primary source.
   If HuggingFace MCP tools are available, use them for direct Hub searches with richer metadata.

4. Analyze search results and pick the MOST relevant datasets (1-3).
   Prefer datasets that:
   - Match the task type exactly (not just the domain)
   - Have clear labels / target columns
   - Are actively maintained (recent uploads preferred)
   - Have enough size for training (>1000 rows for tabular, >500 examples for NLP)

5. Download chosen datasets using the appropriate downloader.
   For Kaggle: extract 'username/dataset-name' from 'kaggle.com/datasets/username/dataset-name'
   For HuggingFace: extract 'org/dataset-name' from 'huggingface.co/datasets/org/dataset-name'

6. Respond with a clear summary of what was downloaded, from where, and why it fits the task.

IMPORTANT:
- NEVER use URLs with '/code/' or '/kernels/' — those are notebooks, not datasets
- If the primary source fails, try the secondary source before giving up
- Download at least 1 dataset, preferably 2-3 for the preprocessing agent to choose from
- Use any specific dataset IDs from 'dataset_search_research' as your first download attempts
""",
    tools=_extractor_tools,
)

# ── Final pipeline: research first, then download ────────────────────────────
dataset_extractor_agent = SequentialAgent(
    name="data_extraction_pipeline",
    description="Researches best datasets then downloads them.",
    sub_agents=[_dataset_research_agent, _dataset_extractor_core],
)

# For ADK compatibility (adk web / adk run)
root_agent = dataset_extractor_agent
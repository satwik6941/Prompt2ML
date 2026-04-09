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

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

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
    Load the user's report and goal from pipeline_state.json.
    This gives the agent context for what datasets to search for.

    Returns:
        JSON string with the user's goal and report.
    """
    state = load_state()
    requirements = {
        "user_goal": state.get("user_goal", ""),
        "report": state.get("report", "")[:5000] if state.get("report") else "",
        "status": state.get("status", ""),
    }
    return json.dumps(requirements, indent=2)


# ============================================================
# AGENT DEFINITION
# ============================================================

from google.adk.agents import Agent

# MCP toolsets — commented out for now, uncomment when MCP servers are ready
# sys.path.append(str(Path(__file__).parent.parent))
# from mcp_servers.mcp_servers import tavily_mcp, hugging_face_mcp

dataset_extractor_agent = Agent(
    model="gemini-3.1-flash-lite-preview",
    name="dataset_extractor_agent",
    description="Searches Kaggle and HuggingFace for relevant datasets and downloads them.",
    instruction="""You are the Dataset Extractor Agent. Your task is to find and download the best datasets for the user's ML project.

WORKFLOW:
1. Call get_dataset_requirements to understand what the user needs.
2. Use search_datasets to find relevant datasets on Kaggle or HuggingFace.
3. Analyze the search results — pick the MOST relevant datasets (prefer Kaggle for tabular data).
4. Download 1-3 of the best datasets using download_kaggle_dataset or download_huggingface_dataset.
5. Respond with a summary of what was downloaded and where.

IMPORTANT:
- Actually call the download functions with the exact dataset identifiers.
- Extract dataset identifiers from URLs:
  * For Kaggle: ONLY use URLs containing '/datasets/' (NOT '/code/')
  * From 'kaggle.com/datasets/username/dataset-name' extract 'username/dataset-name'
  * For HuggingFace: From 'huggingface.co/datasets/username/dataset-name' extract 'username/dataset-name'
- Skip any results that are code notebooks or kernels (URLs with '/code/' or '/kernels/')
- If no good datasets are found, try different search queries
- Download at least 1 dataset, preferably 2-3 for Agent 1 to choose from
""",
    tools=[
        get_dataset_requirements,
        search_datasets,
        download_kaggle_dataset,
        download_huggingface_dataset,
        # tavily_mcp,
        # hugging_face_mcp,
    ],
)

# For ADK compatibility (adk web / adk run)
root_agent = dataset_extractor_agent

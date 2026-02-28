import asyncio
from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from datasets import load_dataset
from dotenv import load_dotenv
import kagglehub
from pathlib import Path
import shutil
import os
import sys
import json
from tavily import TavilyClient
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_state import load_state, save_state

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_datasets(query: str) -> str:
    try:
        print(f"[INFO] Searching for datasets with query: '{query}'...", flush=True)

        # Search for datasets using Tavily
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=["kaggle.com", "huggingface.co"]
        )

        # Format the results
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
    try:
        print(f"[INFO] Downloading Kaggle dataset '{dataset_name}'...", flush=True)

        # Download dataset - kagglehub downloads to cache and returns the path
        download_path = kagglehub.dataset_download(dataset_name)

        # Create datasets directory with dataset-specific folder
        base_dir = Path(__file__).parent.parent / "datasets"
        save_dir = base_dir / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Copy from cache to datasets folder
        source_path = Path(download_path)
        if source_path.exists():
            # Copy all files from download path to save_dir
            for item in source_path.iterdir():
                dest = save_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        # Save dataset info back to shared pipeline state
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
    try:
        # Create datasets directory with dataset-specific folder
        base_dir = Path(__file__).parent.parent / "datasets"
        dataset_path = base_dir / dataset_name.replace("/", "_")
        dataset_path.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading HuggingFace dataset '{dataset_name}'...", flush=True)

        # Download dataset
        dataset = load_dataset(dataset_name)

        # Save dataset to disk
        dataset.save_to_disk(str(dataset_path))

        # Save dataset info back to shared pipeline state
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

def read_file(file_path: str) -> str:
    file_path = Path(__file__).parent.parent / "outputs" / file_path
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[INFO] Successfully read file: {file_path}", flush=True)
        return content
    except Exception as e:
        error_msg = f"Error reading file '{file_path}': {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg

async def main():
    root_agent = Agent(
        model=LiteLlm(model="openai/gpt-5-mini-2025-08-07"),
        name="dataset_creator_agent",
        description='An expert agent who finds and downloads datasets based on user requirements',
        instruction="""
        You are an expert dataset creator. Your task is to fetch high-quality datasets for the user.

        If the dataset is present, skip searching and fetching steps.

        If the dataset is NOT present, follow these steps:
        1. Use search_datasets to find relevant datasets on Kaggle or HuggingFace for the user's problem
        2. Analyze the search results and identify the most relevant dataset
        3. Use download_kaggle_dataset for Kaggle datasets (format: 'username/dataset-name')
            OR download_huggingface_dataset for HuggingFace datasets (format: 'username/dataset-name')
        4. Provide a summary of the downloaded dataset and its location

        IMPORTANT:
        - Actually call the download functions with the exact dataset names from the search results
        - Extract the dataset identifier from URLs:
          * For Kaggle: ONLY use URLs containing '/datasets/' (NOT '/code/')
          * From 'kaggle.com/datasets/username/dataset-name' extract 'username/dataset-name'
          * For HuggingFace: From 'huggingface.co/datasets/username/dataset-name' extract 'username/dataset-name'
        - Skip any results that are code notebooks or kernels (URLs containing '/code/' or '/kernels/')
        """,
        tools=[search_datasets, download_kaggle_dataset, download_huggingface_dataset]
    )

    app_name = 'my_agent_app'
    user_id = 'user1'
    session_id = 'session1'

    session_service = InMemorySessionService()

    # Use async versions of session methods
    existing_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    if existing_session is None:
        print(f"[INFO] Creating new session and running agent.", flush=True)
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
    else:
        print(f"[INFO] Session already exists. Reusing session.", flush=True)

    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        session_service=session_service,
    )

    # Load dataset requirements from shared pipeline state
    state = load_state()
    user_input = state.get("report") or state.get("user_goal")
    if not user_input:
        print("[ERROR] No data found in pipeline_state.json. Run main.py first.")
        return
    print(f"[INFO] Loaded requirements from pipeline state (status: {state.get('status')})", flush=True)

    print(f"[INFO] Dataset requirements: {user_input}", flush=True)

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=user_input)],
    )

    # Use async runner
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        if event.is_final_response and event.content and event.content.parts:
            print(f"Agent: {event.content.parts[0].text}")


if __name__ == "__main__":
    asyncio.run(main())

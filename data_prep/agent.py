import asyncio
from google.adk.agents import Agent
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from datasets import load_dataset
from dotenv import load_dotenv
import kagglehub
from pathlib import Path
import os
import sys
from tavily import TavilyClient

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

        # Create datasets directory and copy there
        save_dir = Path.cwd() / "datasets" / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Copy from cache to datasets folder
        import shutil
        source_path = Path(download_path)
        if source_path.exists():
            # Copy all files from download path to save_dir
            for item in source_path.iterdir():
                dest = save_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        success_msg = f"Successfully downloaded Kaggle dataset '{dataset_name}' to {save_dir.absolute()}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg
    except Exception as e:
        error_msg = f"Error downloading Kaggle dataset: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg


def download_huggingface_dataset(dataset_name: str) -> str:
    try:
        # Create datasets directory in current working directory
        save_dir = Path.cwd() / "datasets"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading HuggingFace dataset '{dataset_name}'...", flush=True)

        # Download dataset
        dataset = load_dataset(dataset_name)

        # Save dataset to disk
        dataset_path = save_dir / dataset_name.replace("/", "_")
        dataset.save_to_disk(str(dataset_path))

        success_msg = f"Successfully downloaded HuggingFace dataset '{dataset_name}' to {dataset_path.absolute()}"
        print(f"[SUCCESS] {success_msg}", flush=True)
        return success_msg
    except Exception as e:
        error_msg = f"Error downloading HuggingFace dataset: {str(e)}"
        print(f"[ERROR] {error_msg}", flush=True)
        return error_msg

async def main():
    user_query = input("Please enter your problem statement: ")

    root_agent = Agent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name="dataset_creator_agent",
        description='An expert agent who finds and downloads datasets based on user requirements',
        instruction=f"""
        User Query: {user_query}

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
        - Extract the dataset identifier from URLs (e.g., from 'kaggle.com/datasets/username/dataset-name' extract 'username/dataset-name')
        """,
        tools=[search_datasets, download_kaggle_dataset, download_huggingface_dataset]
    )

    app_name = 'my_agent_app'
    user_id = 'user1'

    runner = InMemoryRunner(
        agent=root_agent,
        app_name=app_name,
    )

    my_session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )

    content = types.Content(
            role='user', parts=[types.Part.from_text(text=user_query)]
        )
    print('** User says:', content.model_dump(exclude_none=True))
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=my_session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            print(f'** {event.author}: {event.content.parts[0].text}')

if __name__ == "__main__":
    asyncio.run(main())

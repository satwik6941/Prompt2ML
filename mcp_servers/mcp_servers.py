import os
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

hugging_face_mcp = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@llmindset/hf-mcp-server",
            ],
            env={
                "HF_TOKEN": HUGGING_FACE_TOKEN,
            }
        ),
        timeout=120,
    ),
)

tavily_mcp = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "tavily-mcp@0.1.3",
            ],
            env={
                "TAVILY_API_KEY": TAVILY_API_KEY,
            }
        ),
        timeout=120,
    ),
)
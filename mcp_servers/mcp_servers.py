from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

HUGGING_FACE_TOKEN = "YOUR_HUGGING_FACE_TOKEN"

hugging_face_mcp = McpToolset(
            connection_params=StdioConnectionParams(
                server_params = StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "@llmindset/hf-mcp-server",
                    ],
                    env={
                        "HF_TOKEN": HUGGING_FACE_TOKEN,
                    }
                ),
                timeout=30,
            ),
        )


# Connect Verity MCP to Claude Desktop

Add the following entry to Claude Desktop's MCP configuration file, then restart Claude Desktop. On Windows, open **Settings > Developer > Edit Config** and merge this object into `mcpServers`.

```json
{
  "mcpServers": {
    "verity-mcp": {
      "command": "C:\\Users\\Aniket\\AppData\\Local\\hermes\\bin\\uv.exe",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\Aniket\\Downloads\\mcpresearch\\verity-mcp",
        "python",
        "-m",
        "src.mcp_server"
      ]
    }
  }
}
```

The server uses stdio. Add any required provider keys to a `.env` file in the project directory before calling the evidence or originality tools. The `verify_source` tool can operate without LLM keys.

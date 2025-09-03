@echo off
echo ============================================================
echo DeFi Risk Analysis System Startup
echo ============================================================
echo.

echo Starting MCP server in background...
start "MCP Server" python servers/mcpServer.py

echo Waiting for server to start...
timeout /t 3 /nobreak >nul

echo Starting risk analysis agent...
python agent.py

echo.
echo Stopping MCP server...
taskkill /f /im python.exe /fi "WINDOWTITLE eq MCP Server*" >nul 2>&1

echo Done!
pause

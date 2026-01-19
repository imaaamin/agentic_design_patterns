"""
A2A Protocol Common Utilities

Shared utilities and types for A2A-compliant agents using the official a2a-sdk.

Based on: https://github.com/a2aproject/A2A
SDK Reference: https://a2a-protocol.org/latest/sdk/python/

Install: pip install a2a-sdk
"""

import json
import logging
from pathlib import Path
from typing import Optional
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_agent_card_from_file(path: Path) -> dict:
    """Load an agent card from a JSON file."""
    with open(path) as f:
        return json.load(f)


async def discover_agent(base_url: str, httpx_client: Optional[httpx.AsyncClient] = None) -> dict:
    """
    Discover an agent by fetching its Agent Card from the well-known endpoint.
    
    Args:
        base_url: Base URL of the agent (e.g., http://localhost:8002)
        httpx_client: Optional httpx client to reuse
    
    Returns:
        The agent's Agent Card as a dict
    """
    card_url = f"{base_url.rstrip('/')}/.well-known/agent.json"
    logger.info(f"🔍 Discovering agent at {card_url}")
    
    if httpx_client:
        response = await httpx_client.get(card_url, timeout=10)
    else:
        async with httpx.AsyncClient() as client:
            response = await client.get(card_url, timeout=10)
    
    response.raise_for_status()
    card = response.json()
    
    logger.info(f"   ✅ Discovered: {card.get('name', 'Unknown')}")
    return card


async def send_task_to_agent(
    agent_url: str,
    message_text: str,
    message_data: Optional[dict] = None,
    httpx_client: Optional[httpx.AsyncClient] = None
) -> dict:
    """
    Send a task to an A2A agent using JSON-RPC 2.0.
    
    Args:
        agent_url: Base URL of the target agent
        message_text: Text message to send
        message_data: Optional structured data
        httpx_client: Optional httpx client to reuse
    
    Returns:
        The task result from the agent
    """
    import uuid
    
    message_id = str(uuid.uuid4())
    
    parts = [{"kind": "text", "text": message_text}]
    if message_data:
        parts.append({"kind": "data", "data": message_data})
    
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": parts
            },
            "configuration": {
                "acceptedOutputModes": ["text"]
            }
        }
    }
    
    logger.info(f"📤 Sending task to {agent_url}")
    
    if httpx_client:
        response = await httpx_client.post(
            agent_url.rstrip('/'),
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent_url.rstrip('/'),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
    
    response.raise_for_status()
    result = response.json()
    
    if "error" in result:
        logger.error(f"   ❌ Error: {result['error']}")
        raise RuntimeError(f"A2A Error: {result['error']}")
    
    logger.info(f"   ✅ Task completed")
    return result.get("result", {})

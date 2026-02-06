# ============================================================================
# FILE: core/execution.py
# PURPOSE: Execute tools with error handling, retry logic, timeout
# ============================================================================

from typing import List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# WHY THIS SEPARATE MODULE?
# ============================================================================
# Tool execution is complex and error-prone:
# - LLM might not return valid tool calls
# - Tools might fail
# - Network might timeout
# - We want retry logic and error messages
#
# This module encapsulates all that complexity.
# ============================================================================


def execute_tool_calls(
    messages: List,
    llm_with_tools,
    tools: Dict[str, Tool],
    max_iterations: int = 3,
    timeout: Optional[float] = 60.0,
    retry_attempts: int = 2
) -> AIMessage:
    """
    Execute LLM with tool calling loop.
    
    FLOW:
    1. Send messages to LLM
    2. LLM decides: "I need tool X with args Y"
    3. Execute tool X
    4. Add result to messages
    5. Repeat until LLM says "done"
    
    Args:
        messages: List of chat messages
        llm_with_tools: LLM instance with tools bound
        tools: Dict mapping tool names to Tool objects
        max_iterations: Max tool calls before giving up
        timeout: Max seconds for entire execution
        retry_attempts: How many times to retry failed LLM calls
    
    Returns:
        Final AIMessage from LLM
    
    Example:
        messages = [
            SystemMessage(content="You are helpful..."),
            HumanMessage(content="What's 2+2?")
        ]
        response = execute_tool_calls(
            messages,
            llm_with_tools=model.bind_tools([add_tool]),
            tools={"add": add_tool}
        )
        print(response.content)  # "The answer is 4"
    """
    
    tool_iterations = 0
    final_response = None
    start_time = time.time()
    
    while tool_iterations < max_iterations:
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            logger.warning(f"Timeout after {timeout}s")
            break
        
        # Retry LLM call with exponential backoff
        response = None
        for attempt in range(retry_attempts + 1):
            try:
                response = llm_with_tools.invoke(messages)
                break
            except Exception as e:
                if attempt < retry_attempts:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {retry_attempts+1} attempts failed: {e}")
                    return AIMessage(content=f"Error: {str(e)}")
        
        if not response or not hasattr(response, "content"):
            logger.warning("Empty response from LLM")
            break
        
        # Check if LLM wants to call tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            messages.append(response)  # Add LLM response
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                
                if tool_name in tools:
                    try:
                        # Tool takes 'args' dict and converts to kwargs
                        tool_args = tool_call.get("args", {})
                        if not isinstance(tool_args, dict):
                            tool_args = {"query": str(tool_args)}
                        
                        tool_result = tools[tool_name].invoke(tool_args)
                        
                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        ))
                    except Exception as e:
                        logger.error(f"Tool {tool_name} failed: {e}")
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        ))
                else:
                    logger.warning(f"Unknown tool: {tool_name}")
                    messages.append(ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call["id"]
                    ))
            
            tool_iterations += 1
        else:
            # LLM didn't call a tool - it's done!
            final_response = response
            break
    
    if final_response is None:
        # Try to return last response if available
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_response = msg
                break
        if final_response is None:
            final_response = AIMessage(content="Tool execution incomplete")
    
    return final_response


def detect_errors_llm(validation_result: str) -> bool:
    """
    Use LLM to detect if validation found errors.
    More reliable than keyword matching.
    """
    # (Implementation from previous code - omitted for brevity)
    pass
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
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                
                if tool_name in tools:
                    try:
                        tool_args = tool_call.get("args", {})
                        if not isinstance(tool_args, dict):
                            tool_args = {"query": str(tool_args)}
                        
                        tool_result = tools[tool_name].invoke(tool_args)
                        
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
            final_response = response
            break
    
    if final_response is None:
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
    """
    from core.tools import LLMManager
    
    try:
        llm_manager = LLMManager()
        mini_llm = llm_manager.get_mini_llm()
        
        # Smart truncation
        max_length = 1000
        if len(validation_result) > max_length:
            first_part = validation_result[:700]
            last_part = validation_result[-300:]
            truncated = f"{first_part}\n\n[... truncated ...]\n\n{last_part}"
        else:
            truncated = validation_result
        
        error_detection_prompt = f"""
Analyze this validation result and determine if it indicates any errors or issues.

Validation Result:
{truncated}

Respond with ONLY the word "YES" if there are errors, or ONLY the word "NO" if everything is valid.
        """
        
        response = mini_llm.invoke([SystemMessage(content=error_detection_prompt)])
        result_text = getattr(response, "content", "").strip().upper()
        
        if result_text.startswith("YES"):
            return True
        elif result_text.startswith("NO"):
            return False
        else:
            # Fallback to keyword matching
            strong_indicators = ["error", "incorrect", "invalid", "misconfiguration", "wrong", "needs fix"]
            weak_indicators = ["problem", "should be", "issue", "fix", "improve"]
            
            strong_count = sum(1 for kw in strong_indicators if kw in validation_result.lower())
            weak_count = sum(1 for kw in weak_indicators if kw in validation_result.lower())
            
            return strong_count >= 1 or weak_count >= 2
    except Exception as e:
        logger.warning(f"Error detection failed, returning True (assume error): {e}")
        return True
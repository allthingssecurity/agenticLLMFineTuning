#!/usr/bin/env python3
"""
Structured Action System for Agent Training

Defines standardized action formats that can be parsed by clients and 
supports multi-turn conversation trajectories.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class ActionType(Enum):
    """Standardized action types for agent operations."""
    # File operations
    CREATE_FILE = "create_file"
    READ_FILE = "read_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    
    # API operations
    API_CALL = "api_call"
    HTTP_REQUEST = "http_request"
    
    # System operations
    BASH_COMMAND = "bash_command"
    SEARCH = "search"
    
    # Communication
    SEND_MESSAGE = "send_message"
    ASK_QUESTION = "ask_question"
    PROVIDE_RESPONSE = "provide_response"
    
    # Task management
    CREATE_TASK = "create_task"
    UPDATE_TASK = "update_task"
    COMPLETE_TASK = "complete_task"
    
    # Special actions
    THINK = "think"
    WAIT = "wait"
    ERROR = "error"


@dataclass
class StructuredAction:
    """Standardized action format for client parsing."""
    action_type: ActionType
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "metadata": self.metadata or {}
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredAction':
        """Create from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            parameters=data["parameters"],
            metadata=data.get("metadata")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StructuredAction':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ActionResult:
    """Standardized result from action execution."""
    success: bool
    result_data: Any
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ConversationTurn:
    """Single turn in a multi-turn agent conversation."""
    turn_id: int
    user_input: str
    agent_thought: Optional[str]
    agent_action: StructuredAction
    action_result: ActionResult
    agent_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "agent_thought": self.agent_thought,
            "agent_action": self.agent_action.to_dict(),
            "action_result": self.action_result.to_dict(),
            "agent_response": self.agent_response
        }


@dataclass
class MultiTurnTrajectory:
    """Complete multi-turn conversation trajectory."""
    trajectory_id: str
    initial_task: str
    turns: List[ConversationTurn]
    final_success: bool
    total_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for training."""
        return {
            "trajectory_id": self.trajectory_id,
            "initial_task": self.initial_task,
            "turns": [turn.to_dict() for turn in self.turns],
            "final_success": self.final_success,
            "total_reward": self.total_reward
        }
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format suitable for model training."""
        messages = []
        
        # System prompt
        messages.append({
            "role": "system",
            "content": "You are a helpful AI agent that can perform actions to complete tasks. Always format your actions as structured JSON and provide clear responses."
        })
        
        # Conversation turns
        for turn in self.turns:
            # User message
            messages.append({
                "role": "user", 
                "content": turn.user_input
            })
            
            # Agent response with structured action
            agent_content = ""
            
            # Add thinking if present
            if turn.agent_thought:
                agent_content += f"Let me think about this: {turn.agent_thought}\n\n"
            
            # Add structured action
            agent_content += f"I'll take this action:\n```json\n{turn.agent_action.to_json()}\n```\n\n"
            
            # Add result and response
            agent_content += f"Result: {turn.action_result.result_data}\n\n{turn.agent_response}"
            
            messages.append({
                "role": "assistant",
                "content": agent_content
            })
        
        return {
            "messages": messages,
            "trajectory_metadata": {
                "task": self.initial_task,
                "turns": len(self.turns),
                "success": self.final_success,
                "reward": self.total_reward
            }
        }


class ActionParser:
    """Parser for extracting structured actions from agent responses."""
    
    @staticmethod
    def extract_action_from_text(text: str) -> Optional[StructuredAction]:
        """Extract structured action from agent text response."""
        try:
            # Look for JSON code blocks
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                action_data = json.loads(matches[0])
                return StructuredAction.from_dict(action_data)
            
            # Look for inline JSON
            json_pattern = r'\{[^{}]*"action_type"[^{}]*\}'
            matches = re.findall(json_pattern, text)
            
            if matches:
                action_data = json.loads(matches[0])
                return StructuredAction.from_dict(action_data)
            
            return None
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    @staticmethod
    def parse_agent_response(response: str) -> Dict[str, Any]:
        """Parse complete agent response into components."""
        components = {
            "thinking": None,
            "action": None,
            "response_text": response
        }
        
        # Extract thinking
        thinking_pattern = r'(?:Let me think|I need to think|Thinking):\s*([^\n]+)'
        thinking_match = re.search(thinking_pattern, response, re.IGNORECASE)
        if thinking_match:
            components["thinking"] = thinking_match.group(1).strip()
        
        # Extract action
        action = ActionParser.extract_action_from_text(response)
        if action:
            components["action"] = action
        
        return components


# Factory functions for common actions
class ActionFactory:
    """Factory for creating common structured actions."""
    
    @staticmethod
    def create_file(filename: str, content: str, encoding: str = "utf-8") -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.CREATE_FILE,
            parameters={
                "filename": filename,
                "content": content,
                "encoding": encoding
            }
        )
    
    @staticmethod
    def api_call(url: str, method: str = "GET", headers: Optional[Dict] = None, 
                 data: Optional[Dict] = None) -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.API_CALL,
            parameters={
                "url": url,
                "method": method,
                "headers": headers or {},
                "data": data
            }
        )
    
    @staticmethod
    def send_message(recipient: str, message: str, 
                     message_type: str = "text") -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.SEND_MESSAGE,
            parameters={
                "recipient": recipient,
                "message": message,
                "message_type": message_type
            }
        )
    
    @staticmethod
    def bash_command(command: str, working_dir: Optional[str] = None,
                     timeout: int = 30) -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.BASH_COMMAND,
            parameters={
                "command": command,
                "working_directory": working_dir,
                "timeout_seconds": timeout
            }
        )
    
    @staticmethod
    def think(thought: str) -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.THINK,
            parameters={
                "thought": thought
            }
        )
    
    @staticmethod
    def complete_task(task_id: str, summary: str, 
                      success: bool = True) -> StructuredAction:
        return StructuredAction(
            action_type=ActionType.COMPLETE_TASK,
            parameters={
                "task_id": task_id,
                "summary": summary,
                "success": success
            }
        )


# Example usage and demonstration
def demonstrate_structured_actions():
    """Demonstrate structured action system."""
    
    print("🔧 STRUCTURED ACTION SYSTEM DEMO")
    print("=" * 50)
    
    # Create sample actions
    actions = [
        ActionFactory.create_file(
            "config.json", 
            '{"api_url": "https://api.example.com", "timeout": 30}'
        ),
        ActionFactory.api_call(
            "https://api.example.com/users",
            "GET",
            {"Authorization": "Bearer token123"}
        ),
        ActionFactory.send_message(
            "user@example.com",
            "Your API integration is ready for testing!"
        ),
        ActionFactory.complete_task(
            "task_001",
            "Successfully set up API integration with authentication"
        )
    ]
    
    print("📋 Sample Structured Actions:")
    for i, action in enumerate(actions, 1):
        print(f"\n{i}. {action.action_type.value}:")
        print(f"   Parameters: {json.dumps(action.parameters, indent=6)}")
    
    # Demonstrate JSON serialization
    print(f"\n📤 JSON Serialization Example:")
    sample_action = actions[0]
    print(sample_action.to_json())
    
    # Demonstrate parsing
    print(f"\n📥 Parsing from JSON:")
    parsed_action = StructuredAction.from_json(sample_action.to_json())
    print(f"   Parsed action type: {parsed_action.action_type.value}")
    print(f"   Parsed parameters: {parsed_action.parameters}")


def create_sample_multi_turn_trajectory() -> MultiTurnTrajectory:
    """Create a sample multi-turn conversation trajectory."""
    
    turns = []
    
    # Turn 1: User requests API setup
    turn1 = ConversationTurn(
        turn_id=1,
        user_input="I need to set up API authentication for my web application. Can you help me create the configuration and test it?",
        agent_thought="The user needs API authentication setup. I should create a config file first, then test the connection.",
        agent_action=ActionFactory.create_file(
            "api_config.json",
            '{\n  "api_base_url": "https://api.myservice.com",\n  "api_version": "v1",\n  "authentication": {\n    "type": "bearer_token",\n    "token_endpoint": "/auth/token"\n  },\n  "timeout": 30,\n  "retry_attempts": 3\n}'
        ),
        action_result=ActionResult(
            success=True,
            result_data="Created api_config.json with 156 bytes"
        ),
        agent_response="I've created an API configuration file for you. This includes the base URL, authentication settings, and retry configuration. Let me now test the connection to make sure everything works."
    )
    turns.append(turn1)
    
    # Turn 2: Agent tests API connection
    turn2 = ConversationTurn(
        turn_id=2,
        user_input="Great! Can you test the API connection now?",
        agent_thought="I should make a test API call to verify the configuration works correctly.",
        agent_action=ActionFactory.api_call(
            "https://api.myservice.com/v1/health",
            "GET",
            {"Authorization": "Bearer test_token_123", "Content-Type": "application/json"}
        ),
        action_result=ActionResult(
            success=True,
            result_data={"status": "healthy", "version": "1.2.3", "uptime": "99.9%"}
        ),
        agent_response="Perfect! The API connection test was successful. The service is healthy and running version 1.2.3 with 99.9% uptime. Your authentication is working correctly."
    )
    turns.append(turn2)
    
    # Turn 3: User asks for error handling
    turn3 = ConversationTurn(
        turn_id=3,
        user_input="What about error handling? Can you show me how to handle API failures gracefully?",
        agent_thought="I should create a comprehensive error handling example with different failure scenarios.",
        agent_action=ActionFactory.create_file(
            "api_error_handler.py",
            '''import requests
import time
from typing import Dict, Any, Optional

class APIClient:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.base_url = self.config['api_base_url']
        self.timeout = self.config.get('timeout', 30)
        self.retry_attempts = self.config.get('retry_attempts', 3)
    
    def make_request(self, endpoint: str, method: str = 'GET', 
                    data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request with comprehensive error handling."""
        
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_auth_headers()
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                elif response.status_code == 401:
                    return {"success": False, "error": "Authentication failed", "code": 401}
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {"success": False, "error": "Server error", "code": response.status_code}
                else:
                    return {"success": False, "error": f"Unexpected status: {response.status_code}"}
                    
            except requests.exceptions.ConnectionError:
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"success": False, "error": "Connection failed"}
            except requests.exceptions.Timeout:
                return {"success": False, "error": "Request timeout"}
            except Exception as e:
                return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on config."""
        auth_config = self.config.get('authentication', {})
        if auth_config.get('type') == 'bearer_token':
            return {"Authorization": f"Bearer {auth_config.get('token')}"}
        return {}

# Usage example
if __name__ == "__main__":
    client = APIClient('api_config.json')
    result = client.make_request('users/profile')
    
    if result['success']:
        print(f"Success: {result['data']}")
    else:
        print(f"Error: {result['error']}")
'''
        ),
        action_result=ActionResult(
            success=True,
            result_data="Created api_error_handler.py with 2847 bytes"
        ),
        agent_response="I've created a comprehensive error handling system for your API client. It includes retry logic with exponential backoff, handling for different HTTP status codes (401 auth errors, 429 rate limits, 5xx server errors), and connection timeout handling. This will make your API integration much more robust in production."
    )
    turns.append(turn3)
    
    # Turn 4: Task completion
    turn4 = ConversationTurn(
        turn_id=4,
        user_input="This looks perfect! I think we're all set.",
        agent_thought="The user is satisfied with the API setup. I should complete the task and summarize what we accomplished.",
        agent_action=ActionFactory.complete_task(
            "api_integration_setup",
            "Successfully created API configuration, tested connection, and implemented comprehensive error handling system"
        ),
        action_result=ActionResult(
            success=True,
            result_data="Task completed successfully"
        ),
        agent_response="Excellent! I've completed the API integration setup for you. Here's what we accomplished:\n\n✅ Created api_config.json with authentication settings\n✅ Tested API connection successfully\n✅ Built comprehensive error handling with retries\n✅ Included rate limiting and timeout protection\n\nYour API integration is now production-ready with robust error handling. You can use the APIClient class in your application to make reliable API calls."
    )
    turns.append(turn4)
    
    return MultiTurnTrajectory(
        trajectory_id="traj_001_api_setup",
        initial_task="Set up API authentication for web application with error handling",
        turns=turns,
        final_success=True,
        total_reward=8.7
    )


if __name__ == "__main__":
    # Demonstrate structured actions
    demonstrate_structured_actions()
    
    print(f"\n\n🔄 MULTI-TURN TRAJECTORY DEMO")
    print("=" * 50)
    
    # Create sample trajectory
    trajectory = create_sample_multi_turn_trajectory()
    
    print(f"📋 Trajectory Overview:")
    print(f"   Task: {trajectory.initial_task}")
    print(f"   Turns: {len(trajectory.turns)}")
    print(f"   Success: {'✅' if trajectory.final_success else '❌'}")
    print(f"   Reward: {trajectory.total_reward}")
    
    print(f"\n💬 Conversation Flow:")
    for turn in trajectory.turns:
        print(f"\n   Turn {turn.turn_id}:")
        print(f"   👤 User: {turn.user_input[:60]}...")
        print(f"   🤖 Action: {turn.agent_action.action_type.value}")
        print(f"   📊 Result: {'✅' if turn.action_result.success else '❌'}")
        print(f"   💬 Response: {turn.agent_response[:60]}...")
    
    print(f"\n📤 Training Format Preview:")
    training_data = trajectory.to_training_format()
    print(f"   Messages: {len(training_data['messages'])}")
    print(f"   Metadata: {training_data['trajectory_metadata']}")
    
    print(f"\n🎯 Benefits of This System:")
    print(f"   ✅ Structured actions for client parsing")
    print(f"   ✅ Multi-turn conversation support") 
    print(f"   ✅ Rich action metadata and parameters")
    print(f"   ✅ Training-ready format generation")
    print(f"   ✅ Real-world conversation patterns")
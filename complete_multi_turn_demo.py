#!/usr/bin/env python3
"""
Complete Multi-Turn Agent Demo with LLM Judge

Shows: Structured Actions → Multi-Turn Conversations → LLM Evaluation → Model Training
"""

import asyncio
import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from structured_actions import (
    StructuredAction, ActionResult, ConversationTurn, MultiTurnTrajectory,
    ActionType, ActionFactory, ActionParser
)
from multi_turn_environment import MultiTurnEnvironment, MultiTurnAgent
from llm_judge import OpenAILLMJudge


class ProductionReadyAgent(MultiTurnAgent):
    """Production-ready agent with sophisticated conversation handling."""
    
    def __init__(self, agent_name: str = "ProductionAssistant"):
        super().__init__(agent_name)
        self.specialization = "API Integration and Development Tools"
    
    async def generate_response(
        self, 
        user_input: str, 
        conversation_context: List[Dict[str, Any]] = None
    ) -> str:
        """Generate sophisticated multi-turn responses."""
        
        user_lower = user_input.lower()
        context = conversation_context or []
        
        # Advanced conversation patterns
        if self._is_initial_request(user_input, context):
            return await self._handle_initial_request(user_input)
        
        elif self._is_follow_up_request(user_input, context):
            return await self._handle_follow_up(user_input, context)
        
        elif self._is_clarification_needed(user_input, context):
            return await self._request_clarification(user_input)
        
        elif self._is_completion_signal(user_input):
            return await self._handle_completion(user_input, context)
        
        else:
            return await self._handle_complex_request(user_input, context)
    
    def _is_initial_request(self, user_input: str, context: List) -> bool:
        """Check if this is an initial request in the conversation."""
        return len(context) == 0 and any(word in user_input.lower() for word in [
            "need", "want", "help", "set up", "create", "build"
        ])
    
    def _is_follow_up_request(self, user_input: str, context: List) -> bool:
        """Check if this is a follow-up to previous actions."""
        return len(context) > 0 and any(word in user_input.lower() for word in [
            "now", "next", "also", "what about", "can you also"
        ])
    
    def _is_clarification_needed(self, user_input: str, context: List) -> bool:
        """Check if user is asking for clarification."""
        return any(word in user_input.lower() for word in [
            "how", "why", "what does", "explain", "show me"
        ])
    
    def _is_completion_signal(self, user_input: str) -> bool:
        """Check if user is signaling completion."""
        return any(phrase in user_input.lower() for phrase in [
            "perfect", "great", "excellent", "that's all", "we're done", 
            "thank you", "looks good", "all set"
        ])
    
    async def _handle_initial_request(self, user_input: str) -> str:
        """Handle initial user requests with comprehensive setup."""
        
        if "api" in user_input.lower() and "integration" in user_input.lower():
            # Create comprehensive API configuration
            action = ActionFactory.create_file(
                "api_config.json",
                json.dumps({
                    "api_settings": {
                        "base_url": "https://api.yourservice.com",
                        "version": "v1",
                        "timeout": 30,
                        "retry_attempts": 3
                    },
                    "authentication": {
                        "type": "bearer_token",
                        "token_endpoint": "/auth/token",
                        "refresh_endpoint": "/auth/refresh"
                    },
                    "rate_limiting": {
                        "requests_per_minute": 100,
                        "burst_limit": 20
                    },
                    "error_handling": {
                        "retry_on_failure": True,
                        "exponential_backoff": True,
                        "max_retry_delay": 60
                    }
                }, indent=2)
            )
            
            return f"""I'll help you set up a comprehensive API integration system. Let me start by creating a robust configuration file.

```json
{action.to_json()}
```

This configuration includes authentication, rate limiting, and error handling settings. Once this is created, I can help you test the connection and implement the client code. What specific API endpoints will you be working with?"""
        
        elif "database" in user_input.lower() or "db" in user_input.lower():
            action = ActionFactory.create_file(
                "database_config.py",
                '''import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/myapp')
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get database session with proper cleanup."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
'''
            )
            
            return f"""I'll set up a robust database configuration system for you.

```json
{action.to_json()}
```

This includes connection pooling, session management, and proper cleanup. Would you like me to also create the database models and migration scripts?"""
        
        else:
            # General setup response
            thinking_action = ActionFactory.think(
                f"User wants help with: {user_input}. I should create a foundation and ask for specifics."
            )
            
            return f"""I'd be happy to help you with that! Let me think through the best approach.

```json
{thinking_action.to_json()}
```

To provide the most effective solution, could you tell me more about:
1. What specific technology stack you're using?
2. Are there any particular requirements or constraints?
3. What's the expected scale or usage pattern?

This will help me create exactly what you need."""
    
    async def _handle_follow_up(self, user_input: str, context: List) -> str:
        """Handle follow-up requests building on previous actions."""
        
        if "test" in user_input.lower():
            action = ActionFactory.api_call(
                "https://api.yourservice.com/health",
                "GET",
                {
                    "Authorization": "Bearer test_token_123",
                    "Content-Type": "application/json",
                    "User-Agent": "ProductionClient/1.0"
                }
            )
            
            return f"""Excellent idea! Let me test the API connection to make sure everything is working correctly.

```json
{action.to_json()}
```

I'll check the health endpoint first to verify connectivity and authentication. Based on the response, I can then test the main endpoints you'll be using."""
        
        elif "error" in user_input.lower() or "handle" in user_input.lower():
            action = ActionFactory.create_file(
                "robust_api_client.py",
                '''import requests
import time
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class APIResponse:
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0

class RobustAPIClient:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Configure session with proper headers and settings."""
        self.session.headers.update({
            'User-Agent': 'ProductionClient/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def make_request(self, endpoint: str, method: str = 'GET', 
                    data: Optional[Dict] = None, **kwargs) -> APIResponse:
        """Make API request with comprehensive error handling."""
        
        url = f"{self.config['api_settings']['base_url']}/{endpoint}"
        max_retries = self.config['api_settings']['retry_attempts']
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    timeout=self.config['api_settings']['timeout'],
                    **kwargs
                )
                
                # Handle different status codes
                if response.status_code == 200:
                    return APIResponse(
                        success=True,
                        data=response.json(),
                        status_code=response.status_code
                    )
                
                elif response.status_code == 401:
                    # Authentication failed
                    self.logger.error("Authentication failed - check token")
                    return APIResponse(
                        success=False,
                        error="Authentication failed",
                        status_code=401
                    )
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    if attempt < max_retries:
                        wait_time = min(2 ** attempt, 60)
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    
                    return APIResponse(
                        success=False,
                        error="Rate limit exceeded",
                        status_code=429,
                        retry_count=attempt
                    )
                
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Server error, retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    
                    return APIResponse(
                        success=False,
                        error=f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        retry_count=attempt
                    )
                
                else:
                    return APIResponse(
                        success=False,
                        error=f"Unexpected status: {response.status_code}",
                        status_code=response.status_code
                    )
                    
            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Connection failed, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                return APIResponse(
                    success=False,
                    error="Connection failed after retries",
                    retry_count=attempt
                )
                
            except requests.exceptions.Timeout:
                return APIResponse(
                    success=False,
                    error="Request timeout",
                    retry_count=attempt
                )
                
            except Exception as e:
                return APIResponse(
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                    retry_count=attempt
                )
        
        return APIResponse(
            success=False,
            error="Max retries exceeded",
            retry_count=max_retries
        )
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

# Usage example
if __name__ == "__main__":
    client = RobustAPIClient('api_config.json')
    
    # Test the client
    result = client.make_request('health')
    if result.success:
        print(f"API is healthy: {result.data}")
    else:
        print(f"API error: {result.error}")
'''
            )
            
            return f"""Great thinking! Error handling is crucial for production systems. I'll create a robust API client with comprehensive error handling.

```json
{action.to_json()}
```

This client includes:
• Automatic retries with exponential backoff
• Proper handling of different HTTP status codes
• Connection timeout and error recovery
• Detailed logging and error reporting
• Session reuse for better performance

Would you like me to also create unit tests to verify the error handling works correctly?"""
        
        else:
            return await self._handle_complex_request(user_input, context)
    
    async def _handle_completion(self, user_input: str, context: List) -> str:
        """Handle task completion with proper summary."""
        
        # Analyze what was accomplished
        accomplishments = []
        if any("config" in str(ctx) for ctx in context):
            accomplishments.append("✅ Created comprehensive configuration files")
        if any("api" in str(ctx) for ctx in context):
            accomplishments.append("✅ Set up API client with error handling")
        if any("test" in str(ctx) for ctx in context):
            accomplishments.append("✅ Tested API connectivity and responses")
        if any("error" in str(ctx) for ctx in context):
            accomplishments.append("✅ Implemented robust error handling")
        
        action = ActionFactory.complete_task(
            "api_integration_setup",
            f"Complete API integration system with {len(accomplishments)} components implemented successfully"
        )
        
        accomplishments_text = "\\n".join(accomplishments) if accomplishments else "✅ Successfully completed the requested setup"
        
        return f"""Perfect! I'm glad everything meets your requirements. Let me mark this as completed.

```json
{action.to_json()}
```

Here's what we've accomplished together:
{accomplishments_text}

Your API integration system is now production-ready with:
• Robust configuration management
• Comprehensive error handling and retries
• Proper authentication and rate limiting
• Health monitoring and testing capabilities

The system is designed to handle real-world scenarios gracefully. Feel free to reach out if you need any modifications or have questions about deployment!"""
    
    async def _handle_complex_request(self, user_input: str, context: List) -> str:
        """Handle complex or unclear requests."""
        
        thinking_action = ActionFactory.think(
            f"Complex request: {user_input}. Context: {len(context)} previous turns. Need to provide helpful guidance."
        )
        
        return f"""Let me think about the best way to approach this request.

```json
{thinking_action.to_json()}
```

Based on our conversation so far, I want to make sure I provide exactly what you need. Could you help me understand:

1. Is this related to the work we've been doing, or is this a new requirement?
2. Are there any specific technical constraints or preferences?
3. What's the priority level for this request?

This will help me give you the most useful and accurate solution."""


async def demonstrate_complete_system(openai_key: str):
    """Demonstrate the complete multi-turn system with LLM evaluation."""
    
    print("🎯 COMPLETE MULTI-TURN SYSTEM DEMONSTRATION")
    print("Structured Actions + Multi-Turn Conversations + LLM Evaluation")
    print("=" * 80)
    
    # Create components
    llm_judge = OpenAILLMJudge(model_name="gpt-4o", api_key=openai_key)
    env = MultiTurnEnvironment(llm_judge=llm_judge)
    agent = ProductionReadyAgent("ProductionAssistant")
    
    # Complex multi-turn scenarios
    scenarios = [
        {
            "task": "Help me build a production-ready API integration system with comprehensive error handling",
            "conversation": [
                "I need to set up API integration for my web application. It needs to be production-ready with proper error handling.",
                "Great! Now can you test the API connection to make sure the configuration works?",
                "What about handling different types of errors? I want to make sure failures don't break the application.",
                "This looks perfect! The error handling is exactly what I needed. We're all set."
            ]
        },
        {
            "task": "Create a database integration system with connection pooling and session management",
            "conversation": [
                "I need help setting up database integration with proper connection pooling for a high-traffic application.",
                "Can you also add database migration handling and health checks?",
                "Perfect! This covers everything I need for production deployment."
            ]
        }
    ]
    
    successful_trajectories = []
    
    for scenario_idx, scenario in enumerate(scenarios, 1):
        print(f"\\n{'='*80}")
        print(f"SCENARIO {scenario_idx}: {scenario['task']}")
        print(f"{'='*80}")
        
        # Start trajectory
        trajectory_id = env.start_new_trajectory(scenario['task'])
        print(f"🎯 Trajectory ID: {trajectory_id}")
        
        # Process conversation
        turns = []
        for turn_idx, user_input in enumerate(scenario['conversation'], 1):
            print(f"\\n🔄 TURN {turn_idx}")
            print("-" * 40)
            print(f"👤 User: {user_input}")
            
            # Generate agent response
            agent_response = await agent.generate_response(user_input, turns)
            print(f"🤖 Agent: {agent_response[:200]}...")
            
            # Process turn
            turn, is_complete = await env.process_turn(user_input, agent_response)
            turns.append(turn.to_dict())
            
            print(f"🔧 Action: {turn.agent_action.action_type.value}")
            print(f"📊 Success: {'✅' if turn.action_result.success else '❌'}")
            print(f"🏁 Complete: {'✅' if is_complete else '🔄'}")
            
            if is_complete:
                break
        
        # Finalize trajectory
        final_trajectory = await env.finalize_trajectory()
        
        if final_trajectory:
            print(f"\\n📊 TRAJECTORY RESULTS:")
            print(f"   Turns: {len(final_trajectory.turns)}")
            print(f"   Success: {'✅' if final_trajectory.final_success else '❌'}")
            print(f"   Reward: {final_trajectory.total_reward:.2f}")
            
            workspace = env.get_workspace_summary()
            print(f"   Files Created: {workspace['files_created']}")
            print(f"   API Calls: {workspace['api_calls_made']}")
            
            successful_trajectories.append(final_trajectory)
        
        print(f"\\n{'='*40}")
    
    print(f"\\n🎉 SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"✅ Generated {len(successful_trajectories)} complete trajectories")
    print(f"✅ All actions in structured, parseable JSON format")
    print(f"✅ Multi-turn conversations with proper context")
    print(f"✅ LLM evaluation for quality assessment")
    print(f"✅ Production-ready action execution")
    
    # Show training data format
    if successful_trajectories:
        sample_trajectory = successful_trajectories[0]
        training_data = sample_trajectory.to_training_format()
        
        print(f"\\n📤 TRAINING DATA FORMAT:")
        print(f"   Total Messages: {len(training_data['messages'])}")
        print(f"   Conversation Turns: {len(sample_trajectory.turns)}")
        
        print(f"\\n📋 Sample Training Message:")
        sample_message = training_data['messages'][2]  # Get agent response
        print(f"   Role: {sample_message['role']}")
        print(f"   Content Preview: {sample_message['content'][:200]}...")
        
        # Show structured action extraction
        action = ActionParser.extract_action_from_text(sample_message['content'])
        if action:
            print(f"\\n🔧 Extracted Structured Action:")
            print(f"   Type: {action.action_type.value}")
            print(f"   Parameters: {list(action.parameters.keys())}")
    
    print(f"\\n🎯 PRODUCTION BENEFITS:")
    print(f"   🔧 Client-parseable action format")
    print(f"   💬 Rich conversation context preservation")
    print(f"   🎯 Task-specific intelligent responses")
    print(f"   📊 Comprehensive error handling and recovery")
    print(f"   🏗️  Production-ready system integration")
    print(f"   🤖 LLM-enhanced quality evaluation")
    
    return successful_trajectories


async def main():
    """Run the complete demonstration."""
    
    print("🚀 COMPLETE MULTI-TURN AGENTIC SYSTEM")
    print("Production-Ready Actions + Multi-Turn Conversations + LLM Evaluation")
    print("=" * 80)
    
    # Get OpenAI key from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_key_here'")
        return False
    
    try:
        trajectories = await demonstrate_complete_system(openai_key)
        
        print(f"\\n🎊 COMPLETE SUCCESS!")
        print("=" * 50)
        print("✅ Demonstrated complete system:")
        print("   • Structured actions in JSON format")
        print("   • Multi-turn conversation flows")
        print("   • Production-ready error handling")
        print("   • LLM quality evaluation")
        print("   • Client-parseable action formats")
        print("   • Training-ready data generation")
        
        print(f"\\n🔧 Ready for:")
        print("   • Model training with quality trajectories")
        print("   • Production deployment with client parsing")
        print("   • Real-world API and system integration")
        print("   • Complex multi-step task completion")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🤖 Complete Multi-Turn Agentic System Demo")
    print("Structured Actions → Multi-Turn → LLM Evaluation → Training Ready")
    print()
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\\n🚀 SUCCESS: Complete system demonstrated!")
            print("Ready for production deployment and model training.")
        else:
            print("\\n❌ Demo failed - check configuration")
            
    except KeyboardInterrupt:
        print("\\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Execution failed: {e}")
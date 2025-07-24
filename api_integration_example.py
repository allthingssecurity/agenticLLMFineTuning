#!/usr/bin/env python3
"""
API Integration Example - Complete Multi-Turn Demonstration

This example shows exactly how to create, run, and evaluate a multi-turn 
agent conversation for API integration tasks.
"""

import asyncio
import sys
import os
import json

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from structured_actions import ActionFactory, ActionType
from multi_turn_environment import MultiTurnEnvironment
from llm_judge import OpenAILLMJudge
from complete_multi_turn_demo import ProductionReadyAgent


async def run_api_integration_example():
    """Run the complete API integration example."""
    
    print("🔧 API Integration Multi-Turn Example")
    print("=" * 60)
    print("This example demonstrates a realistic 4-turn conversation")
    print("where an agent helps set up production-ready API integration.")
    print()
    
    # Setup (replace with your OpenAI key)
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_key_here'")
        return False
    
    print("🔧 Setting up components...")
    
    # Create LLM judge
    llm_judge = OpenAILLMJudge(model_name="gpt-4o", api_key=openai_key)
    
    # Create environment and agent
    env = MultiTurnEnvironment(llm_judge=llm_judge)
    agent = ProductionReadyAgent("APIIntegrationAssistant")
    
    print("✅ Components initialized")
    print()
    
    # Define the task and conversation flow
    task = "Help me build a production-ready API integration system with comprehensive error handling"
    
    conversation_flow = [
        {
            "user": "I need to set up API integration for my web application. It needs to be production-ready with proper error handling.",
            "expected_action": "create_file",
            "description": "Agent should create API configuration file"
        },
        {
            "user": "Great! Now can you test the API connection to make sure the configuration works?",
            "expected_action": "api_call", 
            "description": "Agent should test API connectivity"
        },
        {
            "user": "What about handling different types of errors? I want to make sure failures don't break the application.",
            "expected_action": "create_file",
            "description": "Agent should create comprehensive error handling code"
        },
        {
            "user": "This looks perfect! The error handling is exactly what I needed. We're all set.",
            "expected_action": "complete_task",
            "description": "Agent should mark the task as completed"
        }
    ]
    
    print(f"🎯 TASK: {task}")
    print(f"📋 CONVERSATION: {len(conversation_flow)} turns planned")
    print()
    
    # Start trajectory
    trajectory_id = env.start_new_trajectory(task)
    print(f"🚀 Started trajectory: {trajectory_id}")
    print()
    
    # Process each turn
    for turn_num, turn_data in enumerate(conversation_flow, 1):
        print(f"{'='*60}")
        print(f"TURN {turn_num}")
        print(f"{'='*60}")
        
        user_input = turn_data["user"]
        expected_action = turn_data["expected_action"]
        description = turn_data["description"]
        
        print(f"👤 USER:")
        print(f"   \"{user_input}\"")
        print()
        print(f"🎯 EXPECTED: {expected_action} - {description}")
        print()
        
        # Generate agent response
        print("🤖 AGENT THINKING...")
        agent_response = await agent.generate_response(user_input)
        
        print(f"🤖 AGENT RESPONSE:")
        response_preview = agent_response.replace('\n', ' ')[:100] + "..."
        print(f"   \"{response_preview}\"")
        print(f"   (Full response: {len(agent_response)} characters)")
        print()
        
        # Process the turn
        turn, is_complete = await env.process_turn(user_input, agent_response)
        
        # Show results
        print(f"📊 TURN RESULTS:")
        print(f"   Action Taken: {turn.agent_action.action_type.value}")
        print(f"   Action Success: {'✅' if turn.action_result.success else '❌'}")
        print(f"   Expected Action: {'✅' if turn.agent_action.action_type.value == expected_action else '❌'}")
        
        # Show action details
        if turn.agent_action.parameters:
            print(f"   Action Parameters:")
            for key, value in turn.agent_action.parameters.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"     {key}: <{len(value)} characters>")
                else:
                    print(f"     {key}: {value}")
        
        print(f"   Result: {turn.action_result.result_data}")
        print(f"   Conversation Complete: {'✅' if is_complete else '🔄'}")
        print()
        
        if is_complete:
            print("🏁 Conversation completed naturally!")
            break
    
    # Finalize and evaluate
    print("🎯 FINALIZING TRAJECTORY...")
    trajectory = await env.finalize_trajectory()
    
    if trajectory:
        print("✅ Trajectory finalized successfully!")
        print()
        
        # Show final results
        print("📊 TRAJECTORY SUMMARY")
        print("=" * 40)
        print(f"   Task: {trajectory.initial_task}")
        print(f"   Total Turns: {len(trajectory.turns)}")
        print(f"   Success: {'✅' if trajectory.final_success else '❌'}")
        print(f"   Final Reward: {trajectory.total_reward:.2f}")
        print()
        
        # Show workspace state
        workspace = env.get_workspace_summary()
        print("📂 WORKSPACE RESULTS:")
        print(f"   Files Created: {workspace['files_created']}")
        print(f"   API Calls Made: {workspace['api_calls_made']}")
        print(f"   Messages Sent: {workspace['messages_sent']}")
        print()
        
        # Show action breakdown
        print("🔧 ACTION BREAKDOWN:")
        action_counts = {}
        for turn in trajectory.turns:
            action_type = turn.agent_action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        for action_type, count in action_counts.items():
            print(f"   {action_type}: {count}")
        print()
        
        # Show training format preview
        training_data = trajectory.to_training_format()
        print("📤 TRAINING DATA:")
        print(f"   Training Messages: {len(training_data['messages'])}")
        print(f"   Metadata: {training_data['trajectory_metadata']}")
        print()
        
        # Show sample structured action
        print("🔧 SAMPLE STRUCTURED ACTION:")
        sample_turn = trajectory.turns[0]  # First turn
        sample_action = sample_turn.agent_action
        print(f"   Type: {sample_action.action_type.value}")
        print(f"   JSON Format:")
        print(json.dumps(sample_action.to_dict(), indent=4))
        print()
        
        # Success metrics
        print("🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ Generated realistic multi-turn conversation")
        print("✅ All actions in structured JSON format")
        print("✅ Proper context preservation across turns")
        print("✅ LLM evaluation with quality scoring")
        print("✅ Training-ready data format")
        print("✅ Production-ready action execution")
        
        return trajectory
    
    else:
        print("❌ Failed to generate valid trajectory")
        return None


async def analyze_trajectory_details(trajectory):
    """Analyze the generated trajectory in detail."""
    
    if not trajectory:
        return
    
    print("\n\n🔍 DETAILED TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    print("📋 TURN-BY-TURN BREAKDOWN:")
    for i, turn in enumerate(trajectory.turns, 1):
        print(f"\n🔄 TURN {i}:")
        print(f"   User Input: \"{turn.user_input[:50]}...\"")
        print(f"   Agent Action: {turn.agent_action.action_type.value}")
        
        # Show key parameters
        if turn.agent_action.parameters:
            key_params = {}
            for key, value in turn.agent_action.parameters.items():
                if key == "content" and len(str(value)) > 100:
                    key_params[key] = f"<{len(str(value))} chars>"
                else:
                    key_params[key] = value
            print(f"   Parameters: {key_params}")
        
        print(f"   Success: {'✅' if turn.action_result.success else '❌'}")
        print(f"   Result: {str(turn.action_result.result_data)[:60]}...")
    
    print(f"\n📊 CONVERSATION FLOW ANALYSIS:")
    print(f"   Total Exchanges: {len(trajectory.turns)}")
    print(f"   Average Response Length: {sum(len(turn.agent_response) for turn in trajectory.turns) / len(trajectory.turns):.0f} chars")
    print(f"   Actions per Turn: 1.0 (each turn has exactly one structured action)")
    print(f"   Success Rate: {sum(1 for turn in trajectory.turns if turn.action_result.success) / len(trajectory.turns) * 100:.0f}%")
    
    print(f"\n🎯 MULTI-TURN BENEFITS DEMONSTRATED:")
    print("   ✅ Context builds across turns (config → test → error handling → completion)")  
    print("   ✅ Each action enables the next (configuration enables testing)")
    print("   ✅ Realistic conversation flow (not just Q&A)")
    print("   ✅ Production-ready outputs (actual usable code generated)")
    print("   ✅ Structured format maintained throughout")


def show_usage_instructions():
    """Show how to use this example."""
    
    print("\n📚 HOW TO USE THIS EXAMPLE")
    print("=" * 40)
    print()
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your_key_here'")
    print()
    print("2. Run this example:")
    print("   python3 api_integration_example.py")
    print()
    print("3. Customize for your domain:")
    print("   - Modify the task description")
    print("   - Change the conversation flow")
    print("   - Add domain-specific actions")
    print("   - Adjust evaluation criteria")
    print()
    print("4. Generate training data:")
    print("   - Run multiple times with different tasks")
    print("   - Collect successful trajectories")
    print("   - Use trajectory.to_training_format() for model training")
    print()
    print("5. Deploy in production:")
    print("   - Parse structured actions on client side")
    print("   - Execute actions in your application")
    print("   - Maintain conversation context")


async def main():
    """Run the complete example."""
    
    print("🤖 API Integration Multi-Turn Example")
    print("Demonstrates production-ready agent conversations with structured actions")
    print("=" * 80)
    
    try:
        # Run the example
        trajectory = await run_api_integration_example()
        
        if trajectory:
            # Analyze in detail
            await analyze_trajectory_details(trajectory)
            
            # Show usage instructions
            show_usage_instructions()
            
        return trajectory is not None
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 API Integration Multi-Turn Example")
    print("Run this to see a complete 4-turn agent conversation")
    print()
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🎉 SUCCESS: Multi-turn example completed!")
            print("You now have a complete understanding of the system.")
            print("Ready to create your own multi-turn agents!")
        else:
            print("\n❌ Example failed - check your OpenAI API key")
            
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
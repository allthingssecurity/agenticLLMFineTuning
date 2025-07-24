#!/usr/bin/env python3
"""
Show Detailed Trajectory Analysis

Displays the exact trajectory generated and how it was rewarded by the LLM judge.
"""

import asyncio
import sys
import os
import json
from typing import Dict, List, Any

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from structured_actions import (
    StructuredAction, ActionResult, ConversationTurn, MultiTurnTrajectory,
    ActionType, ActionFactory, ActionParser
)
from multi_turn_environment import MultiTurnEnvironment, MultiTurnAgent
from llm_judge import OpenAILLMJudge
from complete_multi_turn_demo import ProductionReadyAgent


async def analyze_single_trajectory(openai_key: str):
    """Generate and analyze a single trajectory in detail."""
    
    print("🔍 DETAILED TRAJECTORY ANALYSIS")
    print("=" * 80)
    
    # Create components
    llm_judge = OpenAILLMJudge(model_name="gpt-4o", api_key=openai_key)
    env = MultiTurnEnvironment(llm_judge=llm_judge)
    agent = ProductionReadyAgent("DetailedAssistant")
    
    # Complex task for analysis
    task = "Help me build a production-ready API integration system with comprehensive error handling"
    
    print(f"🎯 TASK: {task}")
    print(f"🤖 AGENT: {agent.agent_name}")
    print()
    
    # Start trajectory
    trajectory_id = env.start_new_trajectory(task)
    
    # Detailed conversation
    conversation = [
        "I need to set up API integration for my web application. It needs to be production-ready with proper error handling.",
        "Great! Now can you test the API connection to make sure the configuration works?", 
        "What about handling different types of errors? I want to make sure failures don't break the application.",
        "This looks perfect! The error handling is exactly what I needed. We're all set."
    ]
    
    print("📋 STEP-BY-STEP TRAJECTORY GENERATION")
    print("=" * 60)
    
    turns = []
    for turn_idx, user_input in enumerate(conversation, 1):
        print(f"\n🔄 TURN {turn_idx}")
        print("=" * 40)
        
        print(f"👤 USER INPUT:")
        print(f"   \"{user_input}\"")
        
        # Generate agent response
        agent_response = await agent.generate_response(user_input, turns)
        
        print(f"\n🤖 AGENT RESPONSE:")
        print(f"   Response Length: {len(agent_response)} characters")
        print(f"   Preview: \"{agent_response[:100]}...\"")
        
        # Extract structured action
        parsed = ActionParser.parse_agent_response(agent_response)
        action = parsed.get("action")
        thinking = parsed.get("thinking")
        
        if thinking:
            print(f"\n💭 AGENT THINKING:")
            print(f"   \"{thinking}\"")
        
        if action:
            print(f"\n🔧 STRUCTURED ACTION:")
            print(f"   Type: {action.action_type.value}")
            print(f"   Parameters: {list(action.parameters.keys())}")
            
            # Show key parameters
            for key, value in action.parameters.items():
                if key == "content" and len(str(value)) > 100:
                    print(f"     {key}: {str(value)[:100]}... ({len(str(value))} chars)")
                else:
                    print(f"     {key}: {value}")
        
        # Process turn and execute action
        turn, is_complete = await env.process_turn(user_input, agent_response)
        turns.append(turn.to_dict())
        
        print(f"\n📊 ACTION EXECUTION:")
        print(f"   Success: {'✅' if turn.action_result.success else '❌'}")
        print(f"   Result: {turn.action_result.result_data}")
        if turn.action_result.metadata:
            print(f"   Metadata: {turn.action_result.metadata}")
        
        print(f"\n🏁 CONVERSATION STATUS:")
        print(f"   Complete: {'✅ YES' if is_complete else '🔄 CONTINUING'}")
        
        if is_complete:
            break
    
    print(f"\n\n🎯 TRAJECTORY FINALIZATION")
    print("=" * 50)
    
    # Finalize trajectory
    final_trajectory = await env.finalize_trajectory()
    
    if final_trajectory:
        print(f"📊 BASIC TRAJECTORY METRICS:")
        print(f"   Trajectory ID: {final_trajectory.trajectory_id}")
        print(f"   Total Turns: {len(final_trajectory.turns)}")
        print(f"   Basic Success: {'✅' if final_trajectory.final_success else '❌'}")
        print(f"   Environment Reward: {final_trajectory.total_reward:.2f}")
        
        # Show workspace state
        workspace = env.get_workspace_summary()
        print(f"\n📂 WORKSPACE STATE:")
        print(f"   Files Created: {workspace['files_created']}")
        print(f"   API Calls Made: {workspace['api_calls_made']}")
        print(f"   Messages Sent: {workspace['messages_sent']}")
        
        # Detailed turn analysis
        print(f"\n📋 DETAILED TURN-BY-TURN BREAKDOWN:")
        print("=" * 60)
        
        for i, turn in enumerate(final_trajectory.turns, 1):
            print(f"\n🔄 TURN {i} DETAILS:")
            print(f"   User: \"{turn.user_input[:60]}...\"")
            print(f"   Action: {turn.agent_action.action_type.value}")
            print(f"   Success: {'✅' if turn.action_result.success else '❌'}")
            print(f"   Agent Response: \"{turn.agent_response[:80]}...\"")
            
            # Show action parameters summary
            if turn.agent_action.parameters:
                param_summary = []
                for key, value in turn.agent_action.parameters.items():
                    if isinstance(value, str) and len(value) > 50:
                        param_summary.append(f"{key}=<{len(value)} chars>")
                    else:
                        param_summary.append(f"{key}={value}")
                print(f"   Parameters: {', '.join(param_summary)}")
        
        # LLM Judge Evaluation (if available)
        if hasattr(env, 'llm_judge') and env.llm_judge:
            print(f"\n\n🧠 LLM JUDGE EVALUATION (GPT-4o)")
            print("=" * 60)
            
            # Convert trajectory for LLM evaluation
            trajectory_data = []
            for turn in final_trajectory.turns:
                trajectory_data.append({
                    "action": turn.agent_action.action_type.value,
                    "action_params": json.dumps(turn.agent_action.parameters),
                    "next_observation": str(turn.action_result.result_data),
                    "reward": 0.5,  # Placeholder
                    "info": {"action_successful": turn.action_result.success}
                })
            
            try:
                print("📡 Sending trajectory to GPT-4o for evaluation...")
                llm_evaluation = await env.llm_judge.evaluate_trajectory(task, trajectory_data)
                
                print(f"\n📊 LLM EVALUATION SCORES:")
                print(f"   Task Completion: {llm_evaluation.task_completion_score:.1f}/10")
                print(f"   Action Quality: {llm_evaluation.action_quality_score:.1f}/10")
                print(f"   Efficiency: {llm_evaluation.efficiency_score:.1f}/10")
                print(f"   Creativity: {llm_evaluation.creativity_score:.1f}/10")
                
                print(f"\n🎯 LLM REWARDS:")
                print(f"   Intermediate Rewards: {llm_evaluation.intermediate_rewards}")
                print(f"   Trajectory-End Reward: {llm_evaluation.trajectory_end_reward:.2f}")
                print(f"   Total LLM Reward: {llm_evaluation.total_reward:.2f}")
                
                print(f"\n✅ LLM SUCCESS DETERMINATION:")
                print(f"   Success: {'✅ YES' if llm_evaluation.success else '❌ NO'}")
                print(f"   Confidence: {llm_evaluation.evaluation_confidence:.2f}")
                
                print(f"\n💭 LLM REASONING:")
                print(f"   \"{llm_evaluation.reasoning}\"")
                
                print(f"\n📝 PER-ACTION FEEDBACK:")
                for i, feedback in enumerate(llm_evaluation.action_feedback, 1):
                    print(f"   Turn {i}: \"{feedback}\"")
                
                print(f"\n⚖️ REWARD COMPARISON:")
                print(f"   Environment Reward: {final_trajectory.total_reward:.2f} (basic file/API checks)")
                print(f"   LLM Reward: {llm_evaluation.total_reward:.2f} (quality evaluation)")
                print(f"   Improvement: {llm_evaluation.total_reward - final_trajectory.total_reward:.2f} points")
                
                # Update trajectory with LLM evaluation
                final_trajectory.total_reward = llm_evaluation.total_reward
                final_trajectory.final_success = llm_evaluation.success
                
            except Exception as e:
                print(f"❌ LLM evaluation failed: {e}")
        
        # Training format
        print(f"\n\n📤 TRAINING DATA FORMAT")
        print("=" * 50)
        
        training_data = final_trajectory.to_training_format()
        print(f"📋 Training Messages: {len(training_data['messages'])}")
        
        # Show sample training message
        for i, message in enumerate(training_data['messages']):
            if message['role'] == 'assistant' and '```json' in message['content']:
                print(f"\n📝 SAMPLE TRAINING MESSAGE (Assistant Turn {i//2 + 1}):")
                print(f"   Role: {message['role']}")
                print(f"   Content Length: {len(message['content'])} characters")
                print(f"   Content Preview:")
                
                # Show structured parts
                lines = message['content'].split('\n')
                for j, line in enumerate(lines[:10]):  # Show first 10 lines
                    print(f"     {line}")
                if len(lines) > 10:
                    print(f"     ... ({len(lines) - 10} more lines)")
                break
        
        print(f"\n🎯 TRAJECTORY SUMMARY:")
        print(f"   Task: {final_trajectory.initial_task}")
        print(f"   Turns: {len(final_trajectory.turns)}")
        print(f"   Final Success: {'✅' if final_trajectory.final_success else '❌'}")
        print(f"   Final Reward: {final_trajectory.total_reward:.2f}")
        print(f"   Ready for Training: ✅")
        
        return final_trajectory
    
    return None


async def main():
    """Run detailed trajectory analysis."""
    
    print("🔬 TRAJECTORY DEEP DIVE ANALYSIS")
    print("Shows exactly how trajectories are generated and rewarded")
    print("=" * 80)
    
    # Get OpenAI key from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_key_here'")
        return False
    
    try:
        trajectory = await analyze_single_trajectory(openai_key)
        
        if trajectory:
            print(f"\n\n🎉 ANALYSIS COMPLETE!")
            print("=" * 50)
            print("✅ Detailed trajectory generated and analyzed")
            print("✅ LLM evaluation with GPT-4o completed")
            print("✅ Training format demonstrated")
            print("✅ Reward system explained")
            
            print(f"\n🔑 KEY INSIGHTS:")
            print(f"   • Each turn builds on previous actions")
            print(f"   • Structured actions enable client parsing")
            print(f"   • LLM judge evaluates quality beyond completion")
            print(f"   • Multi-turn conversations create rich training data")
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Detailed Trajectory Analysis")
    print("Deep dive into trajectory generation and LLM evaluation")
    print()
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🚀 SUCCESS: Complete trajectory analysis demonstrated!")
        else:
            print("\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
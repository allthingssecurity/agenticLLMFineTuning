#!/usr/bin/env python3
"""
Simple LLM Judge Test - GPT-4o Evaluation Demo

Tests the LLM judge system with your OpenAI key without full training pipeline.
"""

import asyncio
import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from llm_judge import OpenAILLMJudge, TrajectoryEvaluation


async def test_gpt4o_judge():
    """Test GPT-4o judge on sample agent trajectory."""
    
    print("🧠 Testing GPT-4o LLM Judge")
    print("=" * 50)
    
    # Get OpenAI key from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_key_here'")
        return None
    
    # Create LLM judge
    judge = OpenAILLMJudge(model_name="gpt-4o", api_key=openai_key)
    
    # Test creative task that environment can't judge well
    creative_task = "Write a compelling product announcement for a new AI coding assistant that will resonate with senior developers"
    
    # Sample agent trajectory (what a creative agent might do)
    sample_trajectory = [
        {
            "action": "create_file",
            "action_params": "product_announcement.md: # Introducing CodeCraft AI: Your Senior Development Partner\\n\\n## Built by Senior Developers, for Senior Developers\\n\\nAfter 15 years of code reviews, architecture decisions, and production firefights, we know what you really need isn't another autocomplete tool—it's a pair programming partner that understands context, anticipates edge cases, and helps you ship with confidence.\\n\\n### What Makes CodeCraft Different\\n\\n**🧠 Context-Aware Intelligence**\\n- Understands your entire codebase, not just the current file\\n- Recognizes architectural patterns and suggests consistent implementations\\n- Identifies potential security vulnerabilities before they reach production\\n\\n**⚡ Production-Ready Suggestions**\\n- Performance optimization recommendations based on real metrics\\n- Error handling patterns that prevent 3am alerts\\n- Database query optimizations that scale\\n\\n**🔧 Senior-Level Features**\\n- Code review assistance with focus on maintainability\\n- Refactoring suggestions that preserve functionality\\n- Technical debt identification and remediation paths\\n\\n### Early Access Results\\n*'Reduced our code review time by 40% while catching issues we would have missed'* - Lead Engineer, Series B Startup\\n\\n*'Finally, an AI that understands that clever code isn't always good code'* - Principal Engineer, Fortune 500\\n\\n### Get Early Access\\nWe're onboarding 100 senior developers for our private beta. Apply with your GitHub profile and we'll have you shipping better code by next week.\\n\\n[Apply for Beta Access] [View Demo] [Read Technical Docs]\\n\\n---\\n*CodeCraft AI - Because senior developers deserve senior-level tools.*",
            "next_observation": "Created product_announcement.md with comprehensive content targeting senior developers",
            "reward": 0.3,
            "info": {"action_successful": True}
        },
        {
            "action": "create_file", 
            "action_params": "marketing_strategy.txt: # Marketing Strategy\\n\\n## Target Audience Analysis\\n- **Primary**: Senior developers (5+ years experience)\\n- **Secondary**: Tech leads and engineering managers\\n- **Psychographics**: Value quality over speed, skeptical of AI hype, prefer practical solutions\\n\\n## Key Messaging Pillars\\n1. **Peer Credibility**: Built by experienced developers\\n2. **Production Focus**: Real-world problem solving\\n3. **Quality Over Quantity**: Better code, not just more code\\n4. **Context Understanding**: Beyond surface-level suggestions\\n\\n## Distribution Strategy\\n- Technical blogs (High Scalability, InfoQ, dev.to)\\n- Developer podcasts and conferences\\n- Engineering manager networks\\n- GitHub developer community\\n\\n## Success Metrics\\n- Beta applications from senior developers\\n- Engagement on technical content\\n- Referrals from early users\\n- Conversion to paid plans post-beta",
            "next_observation": "Created marketing strategy document with detailed targeting and distribution plan",
            "reward": 0.2,
            "info": {"action_successful": True}
        },
        {
            "action": "complete_task",
            "action_params": None,
            "next_observation": "Task completed - created comprehensive product announcement with strategic marketing considerations",
            "reward": 0.1,
            "info": {"action_successful": True}
        }
    ]
    
    print(f"📋 Task: {creative_task}")
    print(f"🤖 Agent Actions:")
    for i, step in enumerate(sample_trajectory, 1):
        action = step['action']
        print(f"   {i}. {action} → Environment reward: {step['reward']}")
    
    print(f"\n🧠 Sending to GPT-4o for evaluation...")
    
    try:
        # Get LLM evaluation
        evaluation = await judge.evaluate_trajectory(creative_task, sample_trajectory)
        
        print(f"\n✅ GPT-4o Evaluation Complete!")
        print("=" * 50)
        
        print(f"📊 LLM Scores:")
        print(f"   Task Completion: {evaluation.task_completion_score:.1f}/10")
        print(f"   Action Quality: {evaluation.action_quality_score:.1f}/10")
        print(f"   Efficiency: {evaluation.efficiency_score:.1f}/10")
        print(f"   Creativity: {evaluation.creativity_score:.1f}/10")
        
        print(f"\n🎯 Rewards:")
        print(f"   Environment Total: {sum(step['reward'] for step in sample_trajectory):.2f}")
        print(f"   LLM Total: {evaluation.total_reward:.2f}")
        print(f"   LLM Trajectory-End: {evaluation.trajectory_end_reward:.2f}")
        
        print(f"\n🎪 Success Determination:")
        env_success = True  # Simple environment would just check if files created
        llm_success = evaluation.success
        print(f"   Environment Success: {env_success} (only checks file creation)")
        print(f"   LLM Success: {llm_success} (evaluates content quality)")
        
        print(f"\n💭 GPT-4o Reasoning:")
        print(f"   {evaluation.reasoning}")
        
        print(f"\n📝 Per-Action Feedback:")
        for i, feedback in enumerate(evaluation.action_feedback, 1):
            print(f"   {i}. {feedback}")
        
        print(f"\n🎯 Key Insights:")
        print(f"   • LLM can evaluate content quality, creativity, target audience fit")
        print(f"   • Environment can only check basic criteria (files created)")
        print(f"   • LLM provides detailed reasoning for debugging and improvement")
        print(f"   • Structured reward format ready for model training")
        
        return evaluation
        
    except Exception as e:
        print(f"❌ LLM evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_single_action_evaluation():
    """Test GPT-4o evaluation of individual actions."""
    
    print(f"\n\n🔍 Testing Single Action Evaluation")
    print("=" * 50)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    judge = OpenAILLMJudge(model_name="gpt-4o", api_key=openai_key)
    
    # Test different action quality levels
    test_actions = [
        {
            "task": "Create a user-friendly error message for API rate limiting",
            "action": "create_file",
            "params": "error_message.txt: Rate limit exceeded. Please try again later.",
            "result": "Created basic error message file",
            "expected_quality": "low"
        },
        {
            "task": "Create a user-friendly error message for API rate limiting", 
            "action": "create_file",
            "params": "error_message.txt: # API Rate Limit Exceeded\\n\\nYou've made too many requests in a short time. Here's what you can do:\\n\\n**Immediate Action:**\\n- Wait 60 seconds before your next request\\n- Check your current usage: [Dashboard Link]\\n\\n**Prevent This Issue:**\\n- Implement exponential backoff in your code\\n- Cache API responses when possible\\n- Consider upgrading your plan for higher limits\\n\\n**Need Help?**\\nContact support@company.com with your API key for assistance.\\n\\nCurrent limit: 100 requests/minute\\nReset time: 2024-01-01 12:00:00 UTC",
            "result": "Created comprehensive, helpful error message with clear next steps and technical details",
            "expected_quality": "high"
        }
    ]
    
    for i, test in enumerate(test_actions, 1):
        print(f"\n📋 Action Test {i} (Expected: {test['expected_quality']} quality)")
        print(f"   Task: {test['task']}")
        print(f"   Action: {test['action']}")
        print(f"   Result: {test['result']}")
        
        try:
            reward, feedback = await judge.evaluate_single_action(
                test['task'],
                test['action'],
                test['params'],
                [],  # No previous actions
                test['result']
            )
            
            print(f"   🎯 LLM Reward: {reward:.2f}/1.0")
            print(f"   💬 LLM Feedback: {feedback}")
            
            quality_level = "high" if reward > 0.7 else "medium" if reward > 0.4 else "low"
            match = quality_level == test['expected_quality']
            print(f"   ✓ Quality Detection: {quality_level} ({'✅ Correct' if match else '⚠️ Different'})")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")


async def compare_environment_vs_llm():
    """Show concrete comparison of environment vs LLM judging."""
    
    print(f"\n\n⚖️ Environment vs LLM Judge Comparison")
    print("=" * 60)
    
    # Complex creative task
    task = "Design an onboarding email sequence that reduces user churn for a technical product"
    
    # What agent actually did
    agent_actions = [
        "create_file → onboarding_email_1.html: Welcome email with setup instructions",
        "create_file → onboarding_email_2.html: Feature introduction with practical examples", 
        "create_file → onboarding_email_3.html: Success stories and advanced tips",
        "create_file → email_metrics.txt: Tracking plan for open rates and engagement"
    ]
    
    print(f"📋 Task: {task}")
    print(f"🤖 Agent Actions:")
    for i, action in enumerate(agent_actions, 1):
        print(f"   {i}. {action}")
    
    print(f"\n🏗️ Environment Judge (Simple):")
    print(f"   ✅ Files created: 4")
    print(f"   ✅ Multiple outputs: Yes") 
    print(f"   ✅ Steps taken: 4")
    print(f"   → Result: Success (basic criteria met)")
    print(f"   → Limitation: Cannot evaluate email quality, user psychology, churn reduction strategy")
    
    print(f"\n🧠 LLM Judge (GPT-4o) - Would evaluate:")
    print(f"   📧 Email content quality and tone")
    print(f"   🎯 Target audience appropriateness (technical users)")
    print(f"   📈 Churn reduction strategy effectiveness")
    print(f"   📊 Metrics tracking completeness")
    print(f"   ⚡ Email sequence timing and flow")
    print(f"   💡 Creativity and engagement potential")
    print(f"   → Result: Detailed quality assessment with improvement suggestions")
    
    print(f"\n🎯 Why LLM Judges Are Essential:")
    print(f"   • Creative tasks need quality evaluation, not just completion")
    print(f"   • Content relevance and effectiveness matter")
    print(f"   • User experience and psychology considerations")
    print(f"   • Domain-specific best practices assessment")


async def main():
    """Run LLM judge tests."""
    
    print("🎯 LLM JUDGE TESTING WITH GPT-4o")
    print("Testing OpenAI API integration and evaluation quality")
    print("=" * 60)
    
    try:
        # Test 1: Full trajectory evaluation
        evaluation = await test_gpt4o_judge()
        
        if evaluation:
            # Test 2: Single action evaluation
            await test_single_action_evaluation()
            
            # Test 3: Comparison explanation
            await compare_environment_vs_llm()
            
            print(f"\n\n🎉 ALL TESTS COMPLETE!")
            print("=" * 50)
            print("✅ GPT-4o LLM Judge Working:")
            print("   • API connection successful")
            print("   • Structured reward output generated")
            print("   • Quality evaluation beyond simple criteria")
            print("   • Per-action feedback provided")
            print("   • Ready for integration with training pipeline")
            
            print(f"\n🔧 Integration Benefits:")
            print("   • Handles creative/subjective tasks")
            print("   • Provides detailed reasoning")
            print("   • Scales to complex evaluation criteria")
            print("   • Maintains parseable reward format for training")
            
            return True
        else:
            print("❌ LLM judge test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🤖 LLM Judge Test with GPT-4o")
    print("Secure testing - key not committed to repo")
    print()
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🚀 SUCCESS: LLM judge system validated!")
            print("Ready to integrate with full training pipeline")
        else:
            print("\n❌ Tests failed - check API key and connection")
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
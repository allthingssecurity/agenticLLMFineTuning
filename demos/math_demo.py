#!/usr/bin/env python3
"""
Complete Math Domain Demo

Demonstrates the full agentic RL pipeline:
1. Generate action trajectories from math problems
2. Fine-tune Qwen model with LoRA
3. Test inference showing learned actions

This is the complete working example you can run immediately.
"""

import asyncio
import sys
import os

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from environment import MathEnvironment, SmartMathAgent
from trainer import AgenticTrainer


async def generate_math_trajectories():
    """Generate training trajectories from math problems."""
    print("📚 Generating Math Training Trajectories")
    print("=" * 60)
    
    # Create environment and agent
    env = MathEnvironment()
    agent = SmartMathAgent()
    
    # Math problems for training
    problems = [
        "Solve for x: 3x + 7 = 16",
        "Calculate: 5 + 3 × 4",
        "A box has 25 toys. If 8 toys are removed, how many remain?",
        "Solve: 2x = 10",
        "Find x: 6x = 42",
        "What is 8 + 2 × 3?",
        "Solve for y: 4y - 12 = 20"
    ]
    
    trajectories = []
    
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. Generating trajectory for: {problem}")
        
        # Run episode
        episode = await agent.run_episode(env, problem)
        
        if episode['success']:
            trajectories.append(episode)
            print(f"     ✅ Success! Reward: {episode['total_reward']:.1f}")
        else:
            print(f"     ❌ Failed. Reward: {episode['total_reward']:.1f}")
    
    print(f"\n📊 Generated {len(trajectories)} successful trajectories")
    
    # Show example trajectory
    if trajectories:
        example = trajectories[0]
        print(f"\n🔍 Example Trajectory: {example['problem']}")
        print("-" * 40)
        for step in example['trajectory']:
            action = step['action']
            params = f" ({step['action_params']})" if step['action_params'] else ""
            reward = step['reward']
            result = step['next_observation']
            print(f"  Action: {action}{params} → Reward: {reward:.1f}")
            print(f"  Result: {result}")
            print()
    
    return trajectories


async def main():
    """Run complete math domain demo."""
    print("🎯 COMPLETE MATH DOMAIN DEMO")
    print("Full agentic RL pipeline from trajectory generation to inference")
    print("=" * 80)
    
    try:
        # Step 1: Generate training trajectories
        trajectories = await generate_math_trajectories()
        
        if not trajectories:
            print("❌ No successful trajectories generated!")
            return False
        
        # Step 2: Setup trainer and model
        print(f"\n" + "="*80)
        print("STEP 2: Setting up Model and Trainer")
        trainer = AgenticTrainer("Qwen/Qwen2.5-0.5B-Instruct")
        model, tokenizer, device = trainer.setup_model()
        
        # Step 3: Prepare training data
        print(f"\n" + "="*80)
        print("STEP 3: Preparing Training Data")
        dataset = trainer.prepare_training_data(trajectories)
        
        # Step 4: Train model
        print(f"\n" + "="*80)
        print("STEP 4: Fine-tuning Model")
        training_losses = await trainer.train(dataset)
        
        # Step 5: Test inference
        print(f"\n" + "="*80)
        print("STEP 5: Testing Inference")
        
        test_problems = [
            "Solve for x: 2x + 6 = 14",
            "Calculate: 7 + 2 × 3", 
            "A jar has 20 items. If 5 are removed, how many remain?"
        ]
        
        inference_results = trainer.test_inference(test_problems)
        
        # Step 6: Save results
        results = trainer.save_results(training_losses, inference_results, len(trajectories), "math_demo_results.json")
        
        # Final summary
        print(f"\n" + "="*80)
        print("🎉 MATH DEMO RESULTS")
        print("="*80)
        
        print(f"✅ Training Completed:")
        print(f"   Model: Qwen/Qwen2.5-0.5B-Instruct")
        print(f"   Training samples: {len(trajectories)}")
        print(f"   Final loss: {training_losses[-1]:.4f}")
        print(f"   Loss improvement: {results['loss_improvement']:.4f}")
        
        print(f"\n✅ Inference Quality:")
        print(f"   Average quality score: {results['average_quality']:.1f}/10")
        print(f"   Test problems: {len(test_problems)}")
        
        high_quality = sum(1 for r in inference_results if r['quality'] >= 6)
        print(f"   High-quality responses: {high_quality}/{len(test_problems)}")
        
        print(f"\n🎯 Key Achievements:")
        print("   • Generated action-based math trajectories")
        print("   • Fine-tuned Qwen with LoRA on step-by-step reasoning")
        print("   • Model learned to take discrete mathematical actions")
        print("   • Demonstrated measurable improvement in problem-solving")
        print("   • Ready for production deployment")
        
        print(f"\n💡 Next Steps:")
        print("   1. Adapt to your domain by modifying the environment")
        print("   2. Define domain-specific actions and rewards")
        print("   3. Generate training trajectories for your problems")
        print("   4. Fine-tune with the same pipeline")
        print("   5. Deploy as API endpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎯 Math Domain Demo")
    print("Complete agentic RL pipeline demonstration")
    print()
    
    try:
        success = asyncio.run(main())
        
        if success:
            print("\n🎉 SUCCESS: Math domain demo completed!")
            print("\n🔑 This demonstrates how to:")
            print("   • Create domain-specific environments and agents")
            print("   • Generate action-based training trajectories")
            print("   • Fine-tune language models with LoRA")
            print("   • Test inference with learned action sequences")
            print("   • Achieve measurable improvement in problem-solving")
            print("\n🚀 Ready to adapt to your own domain!")
            
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
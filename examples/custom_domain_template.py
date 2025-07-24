#!/usr/bin/env python3
"""
Custom Domain Template

This template shows how to create your own domain-specific agentic RL system.
Follow the TODOs to adapt this to your specific use case.

Example domains:
- Code generation/debugging
- Creative writing
- Business decision making
- Scientific reasoning
- Game playing
- Task planning
"""

import asyncio
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from environment import BaseEnvironment, BaseAgent, ActionResult
from trainer import AgenticTrainer


# TODO: Replace this with your domain-specific environment
class CustomEnvironment(BaseEnvironment):
    """
    Template for your custom domain environment.
    
    TODO: Adapt this class for your specific domain:
    1. Define your problem space
    2. Set up domain-specific state representation
    3. Define valid actions for your domain
    4. Implement reward logic
    5. Add domain-specific validation
    """
    
    def __init__(self):
        super().__init__()
        # TODO: Add domain-specific initialization
        self.domain_state = {}
        self.solution_quality = 0.0
    
    def reset(self, problem: str) -> str:
        """
        TODO: Reset environment for your domain.
        
        Examples:
        - Code: Set up code structure, requirements
        - Writing: Set up topic, style, constraints
        - Business: Set up scenario, objectives, constraints
        """
        self.problem = problem
        self.current_state = f"Starting problem: {problem}"
        self.steps_taken = 0
        self.domain_state = {"initialized": True}
        self.trajectory = []
        
        # TODO: Add domain-specific reset logic
        
        return self.current_state
    
    def get_valid_actions(self) -> List[str]:
        """
        TODO: Define your domain-specific actions.
        
        Examples:
        - Code: ["analyze_requirements", "write_function", "add_logic", "test_code", "debug", "optimize"]
        - Writing: ["brainstorm_ideas", "create_outline", "write_paragraph", "edit_content", "add_details"]
        - Business: ["analyze_situation", "identify_options", "evaluate_risks", "make_decision", "create_plan"]
        """
        # TODO: Replace with your actions
        base_actions = [
            "analyze_problem",
            "plan_approach", 
            "take_action",
            "evaluate_result",
            "refine_solution",
            "finalize_answer"
        ]
        
        # TODO: Add logic to filter actions based on current state
        return base_actions
    
    def step(self, action: str, action_params: Optional[str] = None) -> ActionResult:
        """
        TODO: Implement your domain-specific action logic.
        
        For each action:
        1. Update environment state
        2. Calculate domain-specific reward
        3. Check if task is complete
        4. Return observation and feedback
        """
        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {"action_successful": False}
        
        # TODO: Replace with your domain logic
        if action == "analyze_problem":
            observation = f"Analyzed: {self.problem}"
            reward = 1.0
            info["action_successful"] = True
            
        elif action == "plan_approach":
            observation = "Created solution approach"
            reward = 1.5
            info["action_successful"] = True
            
        elif action == "take_action":
            observation = f"Executed action: {action_params or 'default action'}"
            reward = 2.0
            info["action_successful"] = True
            
        elif action == "evaluate_result":
            observation = "Evaluated current progress"
            reward = 1.0
            info["action_successful"] = True
            
        elif action == "refine_solution":
            observation = "Refined the solution"
            reward = 1.5
            info["action_successful"] = True
            
        elif action == "finalize_answer":
            observation = f"Final solution: {action_params or 'solution provided'}"
            reward = 3.0
            done = True
            info["action_successful"] = True
            
        else:
            observation = f"Unknown action: {action}"
            reward = -0.5
        
        # TODO: Add domain-specific reward logic
        # Examples:
        # - Code: Reward for correctness, efficiency, style
        # - Writing: Reward for coherence, creativity, grammar
        # - Business: Reward for feasibility, ROI, risk management
        
        self.current_state = observation
        return ActionResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )


class CustomAgent(BaseAgent):
    """
    TODO: Implement your domain-specific agent logic.
    
    This agent should understand your domain and make intelligent
    action choices based on the current problem state.
    """
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[str]) -> Tuple[str, Optional[str]]:
        """
        TODO: Implement intelligent action selection for your domain.
        
        Consider:
        1. Current problem state
        2. Steps already taken
        3. Domain-specific heuristics
        4. Exploration vs exploitation
        """
        current_state = state["current_state"]
        steps_taken = state["steps_taken"]
        
        # TODO: Replace with your domain logic
        
        # Early steps: analyze and plan
        if steps_taken == 0:
            return "analyze_problem", None
        elif steps_taken == 1:
            return "plan_approach", None
        
        # Middle steps: take action and evaluate
        elif steps_taken < 4:
            if "analyze" in current_state.lower():
                return "take_action", "implementing solution"
            else:
                return "evaluate_result", None
        
        # Later steps: refine and finalize
        elif steps_taken < 6:
            return "refine_solution", None
        else:
            return "finalize_answer", "final result"


async def generate_custom_trajectories():
    """
    TODO: Generate training trajectories for your domain.
    
    Create diverse, representative problems that cover
    the range of scenarios your agent should handle.
    """
    print("📚 Generating Custom Domain Trajectories")
    print("=" * 60)
    
    env = CustomEnvironment()
    agent = CustomAgent()
    
    # TODO: Replace with your domain-specific problems
    problems = [
        "Sample problem 1 for your domain",
        "Sample problem 2 with different complexity",
        "Edge case problem for robustness",
        "Complex problem requiring multiple steps",
        "Creative problem for generalization"
    ]
    
    trajectories = []
    
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. Generating trajectory for: {problem}")
        
        episode = await agent.run_episode(env, problem)
        
        if episode['success']:
            trajectories.append(episode)
            print(f"     ✅ Success! Reward: {episode['total_reward']:.1f}")
        else:
            print(f"     ❌ Failed. Reward: {episode['total_reward']:.1f}")
    
    print(f"\n📊 Generated {len(trajectories)} successful trajectories")
    return trajectories


def create_custom_system_prompt() -> str:
    """
    TODO: Create a system prompt for your domain.
    
    This should:
    1. Explain the agent's role
    2. List available actions
    3. Provide domain-specific guidance
    4. Set expectations for output format
    """
    # TODO: Customize this for your domain
    return """You are an expert problem-solving agent for [YOUR DOMAIN].

Available actions:
- analyze_problem: Understand the problem requirements
- plan_approach: Create a solution strategy
- take_action: Execute specific solution steps
- evaluate_result: Assess current progress
- refine_solution: Improve the current solution
- finalize_answer: Provide the final result

Format each step as:
Action: [action_name]
Result: [what happens]

Use systematic reasoning to reach high-quality solutions."""


async def run_custom_domain_demo():
    """
    TODO: Adapt this demo for your domain.
    
    This function shows the complete pipeline:
    1. Generate domain-specific trajectories
    2. Fine-tune model on your data
    3. Test inference on new problems
    """
    print("🎯 CUSTOM DOMAIN DEMO")
    print("Adapt this template for your specific use case")
    print("=" * 80)
    
    try:
        # Generate training data
        trajectories = await generate_custom_trajectories()
        
        if not trajectories:
            print("❌ No successful trajectories! Check your environment and agent logic.")
            return False
        
        # Setup trainer
        trainer = AgenticTrainer("Qwen/Qwen2.5-0.5B-Instruct")
        model, tokenizer, device = trainer.setup_model()
        
        # Prepare training data
        dataset = trainer.prepare_training_data(trajectories)
        
        # Train model
        training_losses = await trainer.train(dataset)
        
        # Test inference
        # TODO: Replace with your domain-specific test problems
        test_problems = [
            "Test problem 1 for your domain",
            "Test problem 2 with different characteristics",
            "Challenging test problem for evaluation"
        ]
        
        inference_results = trainer.test_inference(test_problems)
        
        # Save results
        results = trainer.save_results(training_losses, inference_results, len(trajectories), "custom_domain_results.json")
        
        print(f"\n" + "="*80)
        print("🎉 CUSTOM DOMAIN RESULTS")
        print("="*80)
        print(f"✅ Training completed successfully")
        print(f"✅ Loss improvement: {results['loss_improvement']:.4f}")
        print(f"✅ Average inference quality: {results['average_quality']:.1f}/10")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


# TODO: Add domain-specific helper functions
def validate_domain_solution(solution: str) -> float:
    """
    TODO: Implement domain-specific solution validation.
    
    Return a quality score (0-10) based on:
    - Correctness
    - Completeness
    - Efficiency
    - Best practices
    """
    # Placeholder implementation
    return 5.0


def generate_domain_problems(count: int = 10) -> List[str]:
    """
    TODO: Generate diverse problems for your domain.
    
    Consider:
    - Different difficulty levels
    - Various problem types
    - Edge cases
    - Real-world scenarios
    """
    # Placeholder implementation
    return [f"Generated problem {i+1}" for i in range(count)]


if __name__ == "__main__":
    print("🎯 Custom Domain Template")
    print("Follow the TODOs to adapt this to your domain")
    print()
    
    print("📋 TODO Checklist:")
    print("  1. ✏️  Define your domain actions in CustomEnvironment.get_valid_actions()")
    print("  2. ✏️  Implement action logic in CustomEnvironment.step()")
    print("  3. ✏️  Add domain-specific rewards and validation")
    print("  4. ✏️  Create intelligent agent in CustomAgent.select_action()")
    print("  5. ✏️  Generate diverse training problems")
    print("  6. ✏️  Customize system prompt for your domain")
    print("  7. ✏️  Add domain-specific test cases")
    print("  8. ✏️  Run and iterate on your implementation")
    print()
    
    try:
        success = asyncio.run(run_custom_domain_demo())
        
        if success:
            print("\n🎉 SUCCESS: Custom domain template completed!")
            print("\n💡 Remember to:")
            print("   • Replace placeholder logic with your domain expertise")
            print("   • Test with diverse, challenging problems")
            print("   • Iterate on rewards and action definitions")
            print("   • Validate results with domain experts")
            print("\n🚀 Ready to build production agentic systems!")
        else:
            print("\n📝 Follow the TODOs to complete your implementation")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check the TODOs and implement the required functions")
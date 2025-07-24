#!/usr/bin/env python3
"""
Enhanced Environment with LLM Judge Integration

Seamlessly integrates LLM-based judging with existing environment framework.
Backwards compatible with existing environments.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from .environment import BaseEnvironment, BaseAgent, ActionResult
from .llm_judge import BaseLLMJudge, TrajectoryEvaluation, JudgeType, create_judge


class EnhancedEnvironment(BaseEnvironment):
    """
    Enhanced environment that supports both traditional and LLM-based judging.
    
    Drop-in replacement for BaseEnvironment with additional LLM judging capabilities.
    """
    
    def __init__(self, judge_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Setup judge
        self.judge_config = judge_config or {"judge_type": "environment"}
        self.llm_judge = None
        self.use_llm_judge = False
        
        if judge_config and judge_config.get("judge_type") in ["llm", "hybrid"]:
            self.use_llm_judge = True
            self.llm_judge = create_judge(**judge_config)
    
    def enable_llm_judging(self, judge_config: Dict[str, Any]):
        """Enable LLM judging with given configuration."""
        self.judge_config = judge_config
        self.llm_judge = create_judge(**judge_config)
        self.use_llm_judge = True
    
    def disable_llm_judging(self):
        """Disable LLM judging, fall back to environment-based."""
        self.use_llm_judge = False
        self.llm_judge = None
    
    async def evaluate_trajectory_with_llm(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvaluation:
        """Evaluate complete trajectory using LLM judge."""
        if not self.use_llm_judge or not self.llm_judge:
            raise ValueError("LLM judge not configured. Use enable_llm_judging() first.")
        
        return await self.llm_judge.evaluate_trajectory(task, trajectory, context)
    
    async def get_enhanced_action_feedback(
        self,
        task: str,
        action: str,
        action_params: Optional[str],
        previous_actions: List[Dict[str, Any]],
        result_observation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get detailed action feedback from LLM judge."""
        if not self.use_llm_judge or not self.llm_judge:
            return {
                "reward": 0.0,
                "feedback": "LLM judge not configured",
                "source": "error"
            }
        
        reward, feedback = await self.llm_judge.evaluate_single_action(
            task, action, action_params, previous_actions, result_observation, context
        )
        
        return {
            "reward": reward,
            "feedback": feedback,
            "source": "llm_judge"
        }


class EnhancedAgent(BaseAgent):
    """
    Enhanced agent that can work with both traditional and LLM-judged environments.
    """
    
    def __init__(self, use_llm_feedback: bool = False):
        super().__init__()
        self.use_llm_feedback = use_llm_feedback
        self.llm_feedback_history = []
    
    async def run_enhanced_episode(
        self, 
        env: EnhancedEnvironment, 
        problem: str,
        use_llm_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Run episode with optional LLM evaluation.
        
        Returns enhanced episode data with LLM evaluations.
        """
        # Reset environment
        initial_obs = env.reset(problem)
        
        trajectory = []
        total_reward = 0.0
        done = False
        llm_feedback_log = []
        
        print(f"🤖 Running Enhanced Episode")
        print(f"   Task: {problem}")
        print(f"   LLM Judging: {'Enabled' if env.use_llm_judge else 'Disabled'}")
        print(f"   LLM Feedback: {'Enabled' if self.use_llm_feedback else 'Disabled'}")
        print()
        
        while not done and env.steps_taken < env.max_steps:
            # Get current state and valid actions
            state = env.get_state()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Select action
            action, action_params = self.select_action(state, valid_actions)
            
            # Take action in environment
            result = env.step(action, action_params)
            
            # Get LLM feedback if enabled
            llm_feedback = None
            if self.use_llm_feedback and env.use_llm_judge:
                llm_feedback = await env.get_enhanced_action_feedback(
                    problem, action, action_params, trajectory, result.observation
                )
                llm_feedback_log.append(llm_feedback)
                
                print(f"   Step {env.steps_taken}: {action}")
                print(f"   → Environment reward: {result.reward:.2f}")
                print(f"   → LLM reward: {llm_feedback['reward']:.2f}")
                print(f"   → LLM feedback: {llm_feedback['feedback']}")
                print()
            
            # Record step
            step_data = {
                "action": action,
                "action_params": action_params,
                "reward": result.reward,
                "next_observation": result.observation,
                "info": result.info
            }
            
            if llm_feedback:
                step_data["llm_feedback"] = llm_feedback
            
            trajectory.append(step_data)
            total_reward += result.reward
            done = result.done
        
        # Get final LLM evaluation if enabled
        llm_evaluation = None
        if use_llm_evaluation and env.use_llm_judge:
            print(f"🧠 Getting LLM evaluation of complete trajectory...")
            llm_evaluation = await env.evaluate_trajectory_with_llm(problem, trajectory)
            
            print(f"   Task Completion: {llm_evaluation.task_completion_score:.1f}/10")
            print(f"   Action Quality: {llm_evaluation.action_quality_score:.1f}/10")
            print(f"   Efficiency: {llm_evaluation.efficiency_score:.1f}/10")
            print(f"   Creativity: {llm_evaluation.creativity_score:.1f}/10")
            print(f"   LLM Success: {llm_evaluation.success}")
            print(f"   LLM Total Reward: {llm_evaluation.total_reward:.2f}")
            print(f"   Confidence: {llm_evaluation.evaluation_confidence:.2f}")
            print()
        
        # Determine final success and reward
        env_success = getattr(env, 'task_completed', len(trajectory) > 0)
        final_success = env_success
        final_reward = total_reward
        
        # Use LLM evaluation if available and more confident
        if llm_evaluation and llm_evaluation.evaluation_confidence > 0.7:
            final_success = llm_evaluation.success
            final_reward = llm_evaluation.total_reward
        
        episode_result = {
            "problem": problem,
            "trajectory": trajectory,
            "total_reward": final_reward,
            "success": final_success,
            "steps_taken": len(trajectory),
            "environment_reward": total_reward,
            "environment_success": env_success
        }
        
        # Add LLM data if available
        if llm_evaluation:
            episode_result.update({
                "llm_evaluation": llm_evaluation,
                "llm_feedback_log": llm_feedback_log,
                "judge_type": "llm" if env.use_llm_judge else "environment"
            })
        
        return episode_result


# Convenience functions for easy setup
def create_enhanced_environment_with_judge(
    base_env_class, 
    judge_type: str = "llm",
    **judge_kwargs
) -> EnhancedEnvironment:
    """Create enhanced environment with LLM judge."""
    
    # Create judge configuration
    judge_config = {"judge_type": judge_type, **judge_kwargs}
    
    # Create enhanced environment
    class EnhancedSpecificEnvironment(EnhancedEnvironment, base_env_class):
        def __init__(self):
            base_env_class.__init__(self)
            EnhancedEnvironment.__init__(self, judge_config)
    
    return EnhancedSpecificEnvironment()


async def compare_judging_methods(
    env_class,
    agent_class, 
    task: str,
    judge_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare different judging methods on the same task."""
    
    results = {}
    
    for config in judge_configs:
        judge_name = config.get("name", str(config.get("judge_type", "unknown")))
        
        print(f"\n🔍 Testing {judge_name.upper()} Judge")
        print("=" * 50)
        
        # Create environment with this judge
        env = create_enhanced_environment_with_judge(env_class, **config)
        agent = agent_class()
        
        # Run episode
        episode = await EnhancedAgent(use_llm_feedback=True).run_enhanced_episode(env, task)
        
        results[judge_name] = {
            "success": episode["success"],
            "total_reward": episode["total_reward"],
            "steps_taken": episode["steps_taken"],
            "trajectory": episode["trajectory"]
        }
        
        if "llm_evaluation" in episode:
            results[judge_name]["llm_scores"] = {
                "task_completion": episode["llm_evaluation"].task_completion_score,
                "action_quality": episode["llm_evaluation"].action_quality_score,
                "efficiency": episode["llm_evaluation"].efficiency_score,
                "creativity": episode["llm_evaluation"].creativity_score
            }
        
        print(f"   Result: {'✅ Success' if episode['success'] else '❌ Failed'}")
        print(f"   Reward: {episode['total_reward']:.2f}")
        print(f"   Steps: {episode['steps_taken']}")
    
    return results


# Example usage configurations
JUDGE_CONFIGS = {
    "environment_only": {
        "judge_type": "environment",
        "name": "environment"
    },
    "gpt4_judge": {
        "judge_type": "llm",
        "llm_type": "openai", 
        "model_name": "gpt-4",
        "name": "gpt4"
    },
    "gpt3_judge": {
        "judge_type": "llm",
        "llm_type": "openai",
        "model_name": "gpt-3.5-turbo", 
        "name": "gpt3.5"
    },
    "local_judge": {
        "judge_type": "llm",
        "llm_type": "local",
        "model_name": "microsoft/DialoGPT-medium",
        "name": "local"
    },
    "hybrid_judge": {
        "judge_type": "hybrid",
        "llm_type": "openai",
        "model_name": "gpt-4",
        "blend_ratio": 0.6,  # 60% environment, 40% LLM
        "name": "hybrid"
    }
}
#!/usr/bin/env python3
"""
LLM Judge Extension for Agentic RL

Provides seamless integration of LLM-based trajectory evaluation and reward assignment
for cases where environment-based judging is insufficient or impossible.
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class JudgeType(Enum):
    """Types of judges available."""
    ENVIRONMENT = "environment"  # Built-in environment criteria
    LLM = "llm"                 # LLM-based evaluation
    HYBRID = "hybrid"           # Combination of both


@dataclass
class TrajectoryEvaluation:
    """Structured result from trajectory evaluation."""
    task_completion_score: float      # 0.0-10.0 how well task was completed
    action_quality_score: float       # 0.0-10.0 quality of individual actions
    efficiency_score: float           # 0.0-10.0 efficiency (fewer steps = better)
    creativity_score: float           # 0.0-10.0 novel/creative problem solving
    
    # Final computed rewards
    intermediate_rewards: List[float]  # Per-step rewards
    trajectory_end_reward: float       # Final completion reward
    total_reward: float               # Sum of all rewards
    
    # Explanations for debugging
    reasoning: str                    # Why this evaluation was given
    action_feedback: List[str]        # Per-step feedback
    success: bool                     # Overall success/failure
    
    # Metadata
    judge_type: JudgeType
    evaluation_confidence: float      # 0.0-1.0 how confident the judge is


class BaseLLMJudge(ABC):
    """Base class for LLM-based trajectory judges."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.judge_type = JudgeType.LLM
        
    @abstractmethod
    async def evaluate_trajectory(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvaluation:
        """Evaluate a complete trajectory and assign rewards."""
        pass
    
    @abstractmethod
    async def evaluate_single_action(
        self,
        task: str,
        action: str,
        action_params: Optional[str],
        previous_actions: List[Dict[str, Any]],
        result_observation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """Evaluate a single action and return (reward, feedback)."""
        pass


class OpenAILLMJudge(BaseLLMJudge):
    """OpenAI GPT-based trajectory judge."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key
        self._setup_client()
    
    def _setup_client(self):
        """Setup OpenAI client."""
        try:
            import openai
            if self.api_key:
                openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    async def evaluate_trajectory(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvaluation:
        """Evaluate complete trajectory using GPT."""
        
        # Build evaluation prompt
        prompt = self._build_trajectory_evaluation_prompt(task, trajectory, context)
        
        try:
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse structured response
            evaluation = self._parse_evaluation_response(response, trajectory)
            evaluation.judge_type = JudgeType.LLM
            
            return evaluation
            
        except Exception as e:
            # Fallback evaluation on error
            return self._create_fallback_evaluation(task, trajectory, str(e))
    
    async def evaluate_single_action(
        self,
        task: str,
        action: str,
        action_params: Optional[str],
        previous_actions: List[Dict[str, Any]],
        result_observation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """Evaluate single action using GPT."""
        
        prompt = self._build_action_evaluation_prompt(
            task, action, action_params, previous_actions, result_observation, context
        )
        
        try:
            response = await self._call_llm(prompt)
            reward, feedback = self._parse_action_response(response)
            return reward, feedback
            
        except Exception as e:
            return 0.0, f"Evaluation error: {str(e)}"
    
    def _build_trajectory_evaluation_prompt(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build structured prompt for trajectory evaluation."""
        
        context_info = ""
        if context:
            context_info = f"\n\nContext Information:\n{json.dumps(context, indent=2)}"
        
        # Build action sequence
        action_sequence = ""
        for i, step in enumerate(trajectory, 1):
            action = step.get('action', 'unknown')
            params = step.get('action_params', '')
            observation = step.get('next_observation', '')
            
            action_sequence += f"{i}. Action: {action}"
            if params:
                action_sequence += f" ({params})"
            action_sequence += f"\n   Result: {observation}\n\n"
        
        prompt = f"""You are an expert AI agent evaluator. Evaluate how well this agent completed the given task.

TASK: {task}

AGENT'S ACTIONS:
{action_sequence}

{context_info}

Evaluate the agent's performance and provide a structured response in JSON format:

{{
    "task_completion_score": <float 0.0-10.0>,
    "action_quality_score": <float 0.0-10.0>, 
    "efficiency_score": <float 0.0-10.0>,
    "creativity_score": <float 0.0-10.0>,
    "intermediate_rewards": [<float per action>],
    "trajectory_end_reward": <float 0.0-10.0>,
    "reasoning": "<detailed explanation>",
    "action_feedback": ["<feedback for each action>"],
    "success": <true/false>,
    "evaluation_confidence": <float 0.0-1.0>
}}

EVALUATION CRITERIA:
- task_completion_score: How completely was the task accomplished?
- action_quality_score: Were the actions appropriate and well-executed?
- efficiency_score: Was the solution efficient (fewer, better actions)?
- creativity_score: Was the approach creative or novel?
- intermediate_rewards: Small rewards (0.0-0.5) for each action step
- trajectory_end_reward: Large reward (0.0-10.0) for overall completion
- success: true if task was substantially completed, false otherwise
- evaluation_confidence: How confident are you in this evaluation?

Provide honest, constructive evaluation focusing on concrete task completion."""
        
        return prompt
    
    def _build_action_evaluation_prompt(
        self,
        task: str,
        action: str, 
        action_params: Optional[str],
        previous_actions: List[Dict[str, Any]],
        result_observation: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for single action evaluation."""
        
        previous_context = ""
        if previous_actions:
            previous_context = "Previous actions:\n"
            for i, prev_action in enumerate(previous_actions[-3:], 1):  # Last 3 actions
                prev_context += f"{i}. {prev_action.get('action')} → {prev_action.get('next_observation', '')}\n"
            previous_context += "\n"
        
        context_info = ""
        if context:
            context_info = f"Context: {json.dumps(context, indent=2)}\n\n"
        
        prompt = f"""Evaluate this single action taken by an AI agent.

TASK: {task}

{previous_context}{context_info}CURRENT ACTION: {action}
PARAMETERS: {action_params or 'None'}
RESULT: {result_observation}

Provide evaluation in JSON format:
{{
    "reward": <float 0.0-1.0>,
    "feedback": "<brief constructive feedback>"
}}

Give higher rewards (0.7-1.0) for actions that clearly advance task completion.
Give medium rewards (0.3-0.6) for reasonable but not optimal actions.
Give low rewards (0.0-0.2) for irrelevant or counterproductive actions."""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to OpenAI."""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert AI agent evaluator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for consistent evaluation
        )
        
        return response.choices[0].message.content
    
    def _parse_evaluation_response(self, response: str, trajectory: List[Dict[str, Any]]) -> TrajectoryEvaluation:
        """Parse LLM response into TrajectoryEvaluation."""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            data = json.loads(response)
            
            # Calculate total reward
            intermediate_sum = sum(data.get('intermediate_rewards', []))
            trajectory_end = data.get('trajectory_end_reward', 0.0)
            total_reward = intermediate_sum + trajectory_end
            
            return TrajectoryEvaluation(
                task_completion_score=float(data.get('task_completion_score', 0.0)),
                action_quality_score=float(data.get('action_quality_score', 0.0)),
                efficiency_score=float(data.get('efficiency_score', 0.0)),
                creativity_score=float(data.get('creativity_score', 0.0)),
                intermediate_rewards=data.get('intermediate_rewards', [0.0] * len(trajectory)),
                trajectory_end_reward=float(trajectory_end),
                total_reward=float(total_reward),
                reasoning=data.get('reasoning', 'No reasoning provided'),
                action_feedback=data.get('action_feedback', ['No feedback'] * len(trajectory)),
                success=bool(data.get('success', False)),
                judge_type=JudgeType.LLM,
                evaluation_confidence=float(data.get('evaluation_confidence', 0.5))
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return self._create_fallback_evaluation(
                "parsing_error", trajectory, f"Failed to parse LLM response: {str(e)}"
            )
    
    def _parse_action_response(self, response: str) -> Tuple[float, str]:
        """Parse single action evaluation response."""
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            data = json.loads(response)
            reward = float(data.get('reward', 0.0))
            feedback = data.get('feedback', 'No feedback provided')
            
            return reward, feedback
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0, f"Failed to parse action evaluation: {response[:100]}"
    
    def _create_fallback_evaluation(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        error: str
    ) -> TrajectoryEvaluation:
        """Create fallback evaluation when LLM fails."""
        return TrajectoryEvaluation(
            task_completion_score=0.0,
            action_quality_score=0.0,
            efficiency_score=0.0,
            creativity_score=0.0,
            intermediate_rewards=[0.0] * len(trajectory),
            trajectory_end_reward=0.0,
            total_reward=0.0,
            reasoning=f"LLM evaluation failed: {error}",
            action_feedback=[f"Error: {error}"] * len(trajectory),
            success=False,
            judge_type=JudgeType.LLM,
            evaluation_confidence=0.0
        )


class LocalLLMJudge(BaseLLMJudge):
    """Local LLM judge using transformers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(model_name)
        self._setup_model()
    
    def _setup_model(self):
        """Setup local model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("Transformers library not installed. Run: pip install transformers")
    
    async def evaluate_trajectory(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvaluation:
        """Evaluate trajectory using local model."""
        # Simplified evaluation for local models
        
        # Basic scoring based on trajectory length and action types
        num_actions = len(trajectory)
        action_diversity = len(set(step.get('action', '') for step in trajectory))
        
        # Simple heuristic scoring
        task_completion_score = min(8.0, num_actions * 1.5) if num_actions > 0 else 0.0
        action_quality_score = min(8.0, action_diversity * 2.0)
        efficiency_score = max(2.0, 10.0 - num_actions * 0.5) if num_actions > 0 else 2.0
        creativity_score = min(7.0, action_diversity * 1.8)
        
        # Generate rewards
        intermediate_rewards = [0.2] * len(trajectory)  # Small uniform rewards
        trajectory_end_reward = task_completion_score * 0.8
        total_reward = sum(intermediate_rewards) + trajectory_end_reward
        
        return TrajectoryEvaluation(
            task_completion_score=task_completion_score,
            action_quality_score=action_quality_score,
            efficiency_score=efficiency_score,
            creativity_score=creativity_score,
            intermediate_rewards=intermediate_rewards,
            trajectory_end_reward=trajectory_end_reward,
            total_reward=total_reward,
            reasoning=f"Local model heuristic evaluation based on {num_actions} actions with {action_diversity} unique action types",
            action_feedback=[f"Action {i+1}: Reasonable step" for i in range(len(trajectory))],
            success=task_completion_score >= 5.0,
            judge_type=JudgeType.LLM,
            evaluation_confidence=0.6  # Medium confidence for heuristic
        )
    
    async def evaluate_single_action(
        self,
        task: str,
        action: str,
        action_params: Optional[str],
        previous_actions: List[Dict[str, Any]],
        result_observation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """Evaluate single action using local heuristics."""
        
        # Simple scoring based on action type and context
        action_rewards = {
            'create_file': 0.4,
            'api_call': 0.5,
            'search_files': 0.3,
            'bash_command': 0.3,
            'complete_task': 0.8
        }
        
        base_reward = action_rewards.get(action, 0.2)
        
        # Bonus for task relevance (simple keyword matching)
        task_lower = task.lower()
        action_lower = action.lower()
        
        if any(keyword in task_lower for keyword in ['create', 'file'] if keyword in action_lower):
            base_reward += 0.2
        if any(keyword in task_lower for keyword in ['api', 'call'] if keyword in action_lower):
            base_reward += 0.2
        if any(keyword in task_lower for keyword in ['search'] if keyword in action_lower):
            base_reward += 0.2
        
        reward = min(1.0, base_reward)
        feedback = f"Action '{action}' seems {'highly ' if reward > 0.6 else 'moderately ' if reward > 0.3 else ''}relevant to task"
        
        return reward, feedback


class HybridJudge:
    """Combines environment and LLM-based judging."""
    
    def __init__(self, environment_judge, llm_judge: BaseLLMJudge, blend_ratio: float = 0.7):
        """
        Args:
            environment_judge: Environment's built-in judging method
            llm_judge: LLM judge instance
            blend_ratio: How much to weight environment vs LLM (0.0=all LLM, 1.0=all env)
        """
        self.environment_judge = environment_judge
        self.llm_judge = llm_judge
        self.blend_ratio = blend_ratio
        self.judge_type = JudgeType.HYBRID
    
    async def evaluate_trajectory(
        self, 
        task: str, 
        trajectory: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvaluation:
        """Evaluate using both environment and LLM, then blend results."""
        
        # Get environment evaluation (if available)
        env_evaluation = None
        if hasattr(self.environment_judge, 'evaluate_trajectory'):
            env_evaluation = await self.environment_judge.evaluate_trajectory(task, trajectory, context)
        
        # Get LLM evaluation
        llm_evaluation = await self.llm_judge.evaluate_trajectory(task, trajectory, context)
        
        # Blend evaluations
        if env_evaluation:
            blended_evaluation = self._blend_evaluations(env_evaluation, llm_evaluation)
            blended_evaluation.judge_type = JudgeType.HYBRID
            return blended_evaluation
        else:
            # Fall back to LLM only
            llm_evaluation.judge_type = JudgeType.HYBRID
            return llm_evaluation
    
    def _blend_evaluations(
        self, 
        env_eval: TrajectoryEvaluation, 
        llm_eval: TrajectoryEvaluation
    ) -> TrajectoryEvaluation:
        """Blend two evaluations based on blend_ratio."""
        
        w_env = self.blend_ratio
        w_llm = 1.0 - self.blend_ratio
        
        return TrajectoryEvaluation(
            task_completion_score=w_env * env_eval.task_completion_score + w_llm * llm_eval.task_completion_score,
            action_quality_score=w_env * env_eval.action_quality_score + w_llm * llm_eval.action_quality_score,
            efficiency_score=w_env * env_eval.efficiency_score + w_llm * llm_eval.efficiency_score,
            creativity_score=w_env * env_eval.creativity_score + w_llm * llm_eval.creativity_score,
            intermediate_rewards=[
                w_env * e + w_llm * l for e, l in zip(env_eval.intermediate_rewards, llm_eval.intermediate_rewards)
            ],
            trajectory_end_reward=w_env * env_eval.trajectory_end_reward + w_llm * llm_eval.trajectory_end_reward,
            total_reward=w_env * env_eval.total_reward + w_llm * llm_eval.total_reward,
            reasoning=f"Hybrid evaluation (env: {w_env:.1f}, llm: {w_llm:.1f})\nEnv: {env_eval.reasoning}\nLLM: {llm_eval.reasoning}",
            action_feedback=llm_eval.action_feedback,  # Use LLM feedback (more detailed)
            success=env_eval.success or llm_eval.success,  # Success if either judge says so
            judge_type=JudgeType.HYBRID,
            evaluation_confidence=(env_eval.evaluation_confidence + llm_eval.evaluation_confidence) / 2
        )


# Factory function for easy judge creation
def create_judge(
    judge_type: Union[JudgeType, str] = JudgeType.LLM,
    **kwargs
) -> Union[BaseLLMJudge, HybridJudge]:
    """Factory function to create judges easily."""
    
    if isinstance(judge_type, str):
        judge_type = JudgeType(judge_type)
    
    if judge_type == JudgeType.LLM:
        llm_type = kwargs.get('llm_type', 'openai')
        if llm_type == 'openai':
            return OpenAILLMJudge(
                model_name=kwargs.get('model_name', 'gpt-4'),
                api_key=kwargs.get('api_key')
            )
        elif llm_type == 'local':
            return LocalLLMJudge(
                model_name=kwargs.get('model_name', 'microsoft/DialoGPT-medium')
            )
    
    elif judge_type == JudgeType.HYBRID:
        llm_judge = create_judge(JudgeType.LLM, **kwargs)
        return HybridJudge(
            environment_judge=kwargs.get('environment_judge'),
            llm_judge=llm_judge,
            blend_ratio=kwargs.get('blend_ratio', 0.7)
        )
    
    raise ValueError(f"Unknown judge type: {judge_type}")
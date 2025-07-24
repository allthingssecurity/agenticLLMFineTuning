#!/usr/bin/env python3
"""
Core Training Framework for Agentic RL

Handles model loading, LoRA application, training data preparation,
and fine-tuning for any domain-specific environment.
"""

import asyncio
import json
import logging
import torch
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class TrajectoryDataset(Dataset):
    """Dataset for training on action trajectories."""
    
    def __init__(self, trajectories: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._process_trajectories()
    
    def _process_trajectories(self) -> List[Dict[str, Any]]:
        """Convert trajectories to tokenized training data."""
        processed = []
        
        for traj in self.trajectories:
            # Convert to conversation format
            text = self._format_conversation(traj['messages'])
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            processed.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze(),
                'reward': torch.tensor(traj['reward'], dtype=torch.float32)
            })
        
        return processed
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as conversation string."""
        conversation = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == "system":
                conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                conversation += f"<|im_start|>assistant\n{content}<|im_end|>"
        
        return conversation
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


class AgenticTrainer:
    """
    Main trainer class for agentic RL models.
    
    Handles:
    - Model loading from HuggingFace
    - LoRA configuration and application
    - Training data preparation
    - Fine-tuning with reward weighting
    - Inference testing
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.lora_config = None
        
        # Training configuration
        self.training_config = {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "epochs": 3,
            "max_length": 512,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1
        }
    
    def setup_model(self) -> Tuple[Any, Any, torch.device]:
        """Load and configure model with LoRA."""
        print(f"🤖 Loading Model: {self.model_name}")
        print("=" * 60)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Load tokenizer
            print("   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            print("   Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"   ✅ Base model loaded: {self.model.num_parameters():,} parameters")
            
            # Configure LoRA
            print("   Applying LoRA...")
            self.lora_config = LoraConfig(
                r=self.training_config["lora_rank"],
                lora_alpha=self.training_config["lora_alpha"],
                target_modules=self._get_target_modules(),
                lora_dropout=self.training_config["lora_dropout"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Calculate efficiency
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            efficiency = 100 * trainable / total
            
            self.device = next(self.model.parameters()).device
            
            print(f"   ✅ LoRA applied successfully!")
            print(f"      Trainable parameters: {trainable:,}")
            print(f"      Total parameters: {total:,}")
            print(f"      Training efficiency: {efficiency:.2f}%")
            print(f"      Device: {self.device}")
            
            return self.model, self.tokenizer, self.device
            
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            raise
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model architecture."""
        if "qwen" in self.model_name.lower():
            return ["q_proj", "v_proj"]
        elif "llama" in self.model_name.lower():
            return ["q_proj", "v_proj"]
        else:
            # Generic transformer modules
            return ["q_proj", "v_proj"]
    
    def prepare_training_data(self, trajectories: List[Dict[str, Any]]) -> TrajectoryDataset:
        """Prepare trajectories for training."""
        print(f"\n📋 Preparing Training Data...")
        print("=" * 60)
        
        print(f"   Converting {len(trajectories)} trajectories...")
        
        # Convert episodes to training format
        training_data = []
        for traj in trajectories:
            training_item = self._convert_trajectory_to_training(traj)
            training_data.append(training_item)
        
        # Create dataset
        dataset = TrajectoryDataset(training_data, self.tokenizer, self.training_config["max_length"])
        
        print(f"   ✅ Dataset created: {len(dataset)} samples")
        print(f"      Max sequence length: {self.training_config['max_length']}")
        
        return dataset
    
    def _convert_trajectory_to_training(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Convert episode to training format."""
        messages = []
        
        # System prompt (domain-agnostic)
        system_prompt = f"""You are an expert problem-solving agent. Solve problems step-by-step using specific actions.

Format each step as:
Action: [action_name]
Result: [what happens]

Take actions in logical sequence to reach the solution."""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Problem
        messages.append({"role": "user", "content": f"Solve step by step: {episode['problem']}"})
        
        # Solution as action sequence
        action_steps = []
        for step in episode['trajectory']:
            action = step['action']
            params = f" ({step['action_params']})" if step['action_params'] else ""
            result = step['next_observation']
            
            action_line = f"Action: {action}{params}"
            action_line += f"\nResult: {result}"
            action_steps.append(action_line)
        
        solution = "\n\n".join(action_steps)
        if 'final_state' in episode:
            solution += f"\n\nFinal Answer: {episode['final_state']}"
        
        messages.append({"role": "assistant", "content": solution})
        
        # Calculate reward
        base_reward = min(episode['total_reward'] / 10.0, 1.0)
        success_bonus = 0.3 if episode['success'] else 0.0
        efficiency_bonus = 0.2 if len(episode['trajectory']) <= 5 else 0.0
        
        final_reward = base_reward + success_bonus + efficiency_bonus
        
        return {
            "messages": messages,
            "reward": final_reward,
            "metadata": {
                "original_reward": episode['total_reward'],
                "success": episode['success'],
                "steps": len(episode['trajectory'])
            }
        }
    
    async def train(self, dataset: TrajectoryDataset) -> List[float]:
        """Fine-tune the model on trajectory data."""
        print(f"\n🏋️ Training Model...")
        print("=" * 60)
        
        try:
            # Training setup
            batch_size = self.training_config["batch_size"]
            learning_rate = self.training_config["learning_rate"]
            epochs = self.training_config["epochs"]
            
            print(f"   Configuration:")
            print(f"      Batch size: {batch_size}")
            print(f"      Learning rate: {learning_rate}")
            print(f"      Epochs: {epochs}")
            print(f"      Training samples: {len(dataset)}")
            
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            training_losses = []
            
            print(f"\n   🚀 Starting training...")
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                print(f"\n   Epoch {epoch + 1}/{epochs}:")
                
                for batch_idx, batch in enumerate(dataloader):
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    rewards = batch['reward'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Reward-weighted loss
                    loss = outputs.loss
                    weighted_loss = loss * rewards.mean()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += weighted_loss.item()
                    num_batches += 1
                    
                    print(f"      Batch {batch_idx + 1}: Loss = {weighted_loss.item():.4f}")
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                training_losses.append(avg_loss)
                print(f"      Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            
            improvement = training_losses[0] - training_losses[-1] if len(training_losses) > 1 else 0
            print(f"\n   ✅ Training completed!")
            print(f"      Final loss: {training_losses[-1]:.4f}")
            print(f"      Total improvement: {improvement:.4f}")
            
            return training_losses
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def test_inference(self, test_problems: List[str]) -> List[Dict[str, Any]]:
        """Test the trained model on new problems."""
        print(f"\n🎯 Testing Inference...")
        print("=" * 60)
        
        self.model.eval()
        results = []
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n   Test {i}: {problem}")
            print("   " + "-"*50)
            
            try:
                # Create prompt
                prompt = f"""<|im_start|>system
You are an expert problem-solving agent. Solve problems step-by-step using specific actions.

Format each step as:
Action: [action_name]
Result: [what happens]

Take actions in logical sequence to reach the solution.<|im_end|>
<|im_start|>user
Solve step by step: {problem}<|im_end|>
<|im_start|>assistant
"""
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_response = response[len(prompt):].strip()
                
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0].strip()
                
                print(f"   Response: {assistant_response[:200]}...")
                
                # Analyze quality
                action_count = assistant_response.lower().count('action:')
                result_count = assistant_response.lower().count('result:')
                has_answer = 'answer' in assistant_response.lower()
                
                quality = action_count * 2 + result_count + (2 if has_answer else 0)
                quality = min(quality, 10)
                
                print(f"   Quality: {quality}/10 ({action_count} actions, {result_count} results)")
                
                results.append({
                    'problem': problem,
                    'response': assistant_response,
                    'quality': quality,
                    'action_count': action_count
                })
                
            except Exception as e:
                print(f"   Error: {e}")
                results.append({
                    'problem': problem,
                    'response': f"Error: {str(e)}",
                    'quality': 0,
                    'action_count': 0
                })
        
        avg_quality = sum(r['quality'] for r in results) / len(results)
        print(f"\n   📊 Average quality: {avg_quality:.1f}/10")
        
        return results
    
    def save_results(self, training_losses: List[float], inference_results: List[Dict[str, Any]], 
                    trajectories_count: int, filename: str = "training_results.json"):
        """Save training and inference results."""
        results = {
            "model_name": self.model_name,
            "training_completed": True,
            "training_samples": trajectories_count,
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else None,
            "loss_improvement": training_losses[0] - training_losses[-1] if len(training_losses) > 1 else 0,
            "inference_results": inference_results,
            "average_quality": sum(r['quality'] for r in inference_results) / len(inference_results),
            "lora_config": {
                "rank": self.training_config["lora_rank"],
                "alpha": self.training_config["lora_alpha"],
                "target_modules": self._get_target_modules()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return results
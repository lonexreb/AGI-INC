"""Online GRPO (Group Relative Policy Optimization) Trainer.

This module implements Online RL training for the HALO Agent using GRPO.

Key concepts:
- Online: Collect trajectories and update policy in real-time
- GRPO: Compute advantages relative to group of sampled actions
- dr_grpo: Mean-centering only, no std normalization (handles low variance)

Training loop:
1. Observe current state (screenshot)
2. Sample N actions from policy (VLM)
3. Execute best action in environment
4. Compute dense rewards
5. Compute group-relative advantages: A_i = r_i - mean(r)
6. Update LoRA weights using policy gradient
7. Optionally hot-reload weights to vLLM

Architecture:
- VLLMPolicyClient: Inference (sampling actions)
- PyTorch: Gradient computation and weight updates
- LoRA: Parameter-efficient fine-tuning

References:
- GRPO: https://arxiv.org/abs/2402.03300
- Agent Q: https://arxiv.org/abs/2408.07199
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..constants import DEFAULT_MODEL
from ..policy import VLLMPolicyClient
from .progress import DenseRewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single transition in an episode."""
    step: int
    screenshot: bytes
    obs: Dict[str, Any]
    goal: str
    sampled_actions: List[Dict[str, Any]]  # All sampled actions with rewards
    selected_action: str
    reward: float
    reward_components: Dict[str, float]
    done: bool
    success: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Complete episode trajectory."""
    task_name: str
    goal: str
    transitions: List[Transition]
    total_reward: float
    success: bool
    num_steps: int
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOConfig:
    """Configuration for Online GRPO training."""
    # Sampling
    num_generations: int = 8  # Number of actions to sample per step (n in GRPO)
    temperature: float = 0.7  # Sampling temperature

    # Reward
    progress_weight: float = 1.0
    novelty_bonus: float = 0.2
    loop_penalty: float = -0.5
    action_error_penalty: float = -0.2
    success_bonus: float = 1.0

    # GRPO
    loss_type: str = "dr_grpo"  # "grpo" or "dr_grpo" (mean-centering only)
    beta: float = 0.1  # KL penalty coefficient (if using KL regularization)
    clip_range: float = 0.2  # PPO-style clipping (optional)

    # Training
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    warmup_steps: int = 10

    # Episode
    max_steps: int = 70

    # Checkpointing
    checkpoint_dir: str = "checkpoints/qwen3vl_grpo_lora"
    save_every: int = 10  # Save checkpoint every N episodes
    log_every: int = 1  # Log metrics every N episodes

    # vLLM
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = DEFAULT_MODEL


class OnlineGRPOTrainer:
    """Online GRPO trainer for HALO Agent.

    This trainer implements online reinforcement learning using GRPO.
    It collects trajectories by interacting with the environment and
    updates the policy using group-relative advantages.

    Key features:
    - Dense rewards for learning signal
    - dr_grpo advantages (mean-centering, no std normalization)
    - LoRA parameter updates
    - Support for vLLM hot-reload

    Usage:
        trainer = OnlineGRPOTrainer(config)
        trainer.train(env, num_episodes=100)
    """

    def __init__(self, config: GRPOConfig = None):
        """Initialize GRPO trainer.

        Args:
            config: Training configuration
        """
        self.config = config or GRPOConfig()

        # Initialize policy client for inference
        self.policy_client = VLLMPolicyClient(
            base_url=self.config.vllm_base_url,
            model=self.config.model_name,
        )

        # Initialize reward calculator
        self.reward_calculator = DenseRewardCalculator(
            progress_weight=self.config.progress_weight,
            novelty_bonus=self.config.novelty_bonus,
            loop_penalty=self.config.loop_penalty,
            action_error_penalty=self.config.action_error_penalty,
            success_bonus=self.config.success_bonus,
        )

        # Training state
        self.episodes: List[Episode] = []
        self.total_steps = 0
        self.total_episodes = 0

        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []

        # LoRA model (lazily initialized)
        self._lora_model = None
        self._optimizer = None

        logger.info(f"OnlineGRPOTrainer initialized with config: {self.config}")

    def collect_episode(
        self,
        env: Any,
        task_name: str,
        task_seed: Optional[int] = None,
    ) -> Episode:
        """Collect a single episode by interacting with the environment.

        Args:
            env: Browser environment (REAL harness)
            task_name: Name of the task
            task_seed: Optional seed for task

        Returns:
            Completed episode with all transitions
        """
        start_time = time.time()
        self.reward_calculator.reset()

        # Reset environment
        obs, info = env.reset(seed=task_seed)
        goal = self._extract_goal(obs)

        transitions = []
        done = False
        success = False
        step = 0

        while not done and step < self.config.max_steps:
            step += 1
            self.total_steps += 1

            # Get screenshot
            screenshot = self._get_screenshot(obs)

            # Get action history and valid bids
            action_history = [t.selected_action for t in transitions]
            valid_bids = self._extract_valid_bids(obs)
            last_action_error = obs.get("last_action_error", "")

            # Sample N actions from policy
            sampled_actions = self.policy_client.sample_actions(
                screenshot=screenshot,
                goal=goal,
                n=self.config.num_generations,
                temperature=self.config.temperature,
                action_history=action_history,
                valid_bids=valid_bids,
                last_action_error=last_action_error,
            )

            # Select action (for online training, we execute the first/best action)
            # In full GRPO, we would evaluate all actions and compute advantages
            selected_action = sampled_actions[0]["action"]

            # Execute action
            next_obs, env_reward, terminated, truncated, info = env.step(selected_action)
            done = terminated or truncated
            success = info.get("success", False)

            # Compute dense reward
            reward_components = self.reward_calculator.compute_reward(
                obs=obs,
                action=selected_action,
                next_obs=next_obs,
                done=done,
                success=success,
                last_action_error=last_action_error,
            )

            # Store transition
            transition = Transition(
                step=step,
                screenshot=screenshot,
                obs=obs.copy() if isinstance(obs, dict) else {},
                goal=goal,
                sampled_actions=sampled_actions,
                selected_action=selected_action,
                reward=reward_components["total"],
                reward_components=reward_components,
                done=done,
                success=success,
                info=info.copy() if isinstance(info, dict) else {},
            )
            transitions.append(transition)

            # Update for next iteration
            obs = next_obs

            # Log step
            if step % 10 == 0:
                logger.debug(
                    f"Step {step}: action={selected_action[:50]}, "
                    f"reward={reward_components['total']:.3f}, "
                    f"progress={reward_components['progress']:.3f}"
                )

        end_time = time.time()

        # Create episode
        episode = Episode(
            task_name=task_name,
            goal=goal,
            transitions=transitions,
            total_reward=sum(t.reward for t in transitions),
            success=success,
            num_steps=len(transitions),
            start_time=start_time,
            end_time=end_time,
            metadata={
                "task_seed": task_seed,
                "reward_stats": self.reward_calculator.get_episode_stats(),
            }
        )

        self.episodes.append(episode)
        self.total_episodes += 1

        return episode

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages (dr_grpo style).

        For dr_grpo (default): A_i = r_i - mean(r)
        For standard grpo: A_i = (r_i - mean(r)) / std(r)

        Args:
            rewards: List of rewards for sampled actions

        Returns:
            List of advantages
        """
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array)

        if self.config.loss_type == "dr_grpo":
            # Mean-centering only (no std normalization)
            advantages = rewards_array - mean_reward
        else:
            # Standard GRPO with std normalization
            std_reward = np.std(rewards_array)
            if std_reward < 1e-8:
                # Handle zero variance case
                advantages = np.zeros_like(rewards_array)
            else:
                advantages = (rewards_array - mean_reward) / std_reward

        return advantages.tolist()

    def compute_grpo_loss(
        self,
        transitions: List[Transition],
    ) -> Dict[str, Any]:
        """Compute GRPO loss for a batch of transitions.

        For each transition, we have N sampled actions with rewards.
        We compute advantages and use them to weight the log probabilities.

        Loss = -E[A(s,a) * log π(a|s)]

        Args:
            transitions: List of transitions with sampled actions

        Returns:
            Dict with loss and metrics
        """
        total_loss = 0.0
        num_samples = 0
        all_advantages = []
        all_rewards = []

        for transition in transitions:
            # Get rewards for all sampled actions at this step
            # Note: In online setting, we only execute one action,
            # but we can estimate rewards for others using a value function
            rewards = [transition.reward]  # Only have reward for selected action

            # For full GRPO, we would need rewards for all sampled actions
            # This requires either:
            # 1. Executing all actions (expensive)
            # 2. Using a learned value function to estimate rewards
            # 3. Using the dense reward calculator with counterfactual states

            # For now, use single-action advantage (reduces to REINFORCE)
            advantages = self.compute_advantages(rewards)
            all_advantages.extend(advantages)
            all_rewards.extend(rewards)

            # Loss computation would go here (requires PyTorch model)
            # total_loss += -advantages[0] * log_prob
            num_samples += 1

        metrics = {
            "mean_advantage": np.mean(all_advantages) if all_advantages else 0.0,
            "std_advantage": np.std(all_advantages) if all_advantages else 0.0,
            "mean_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "std_reward": np.std(all_rewards) if all_rewards else 0.0,
            "num_samples": num_samples,
        }

        return {"loss": total_loss, "metrics": metrics}

    def update_policy(self, episode: Episode) -> Dict[str, Any]:
        """Update policy using GRPO on episode data.

        This is a placeholder for the actual gradient update.
        Full implementation would:
        1. Load LoRA weights into PyTorch
        2. Compute log probabilities of actions
        3. Compute GRPO loss
        4. Backpropagate and update weights
        5. Save updated LoRA weights
        6. Optionally hot-reload to vLLM

        Args:
            episode: Episode with transitions

        Returns:
            Dict with update metrics
        """
        # Compute loss and metrics
        loss_result = self.compute_grpo_loss(episode.transitions)

        # TODO: Implement actual gradient update
        # This requires:
        # 1. Loading the base model with LoRA
        # 2. Computing log probabilities
        # 3. Gradient descent

        update_metrics = {
            "loss": loss_result["loss"],
            "episode_reward": episode.total_reward,
            "episode_steps": episode.num_steps,
            "success": episode.success,
            **loss_result["metrics"],
        }

        return update_metrics

    def train(
        self,
        env: Any,
        task_names: List[str],
        num_episodes: int = 100,
        task_seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            env: Browser environment
            task_names: List of task names to train on
            num_episodes: Total episodes to collect
            task_seeds: Optional list of seeds for reproducibility

        Returns:
            Training summary with metrics
        """
        logger.info(f"Starting GRPO training for {num_episodes} episodes")
        logger.info(f"Tasks: {task_names}")

        start_time = time.time()
        successes = 0
        total_rewards = []

        for ep_idx in range(num_episodes):
            # Select task (round-robin or random)
            task_name = task_names[ep_idx % len(task_names)]
            task_seed = task_seeds[ep_idx] if task_seeds else None

            # Collect episode
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {ep_idx + 1}/{num_episodes}: {task_name}")
            logger.info(f"{'='*60}")

            try:
                episode = self.collect_episode(env, task_name, task_seed)
                total_rewards.append(episode.total_reward)

                if episode.success:
                    successes += 1
                    logger.info(f"SUCCESS in {episode.num_steps} steps")
                else:
                    logger.info(f"FAILED after {episode.num_steps} steps")

                # Update policy
                update_metrics = self.update_policy(episode)

                # Log metrics
                if (ep_idx + 1) % self.config.log_every == 0:
                    self._log_metrics(ep_idx + 1, episode, update_metrics)

                # Save checkpoint
                if (ep_idx + 1) % self.config.save_every == 0:
                    self._save_checkpoint(ep_idx + 1)

            except Exception as e:
                logger.error(f"Episode {ep_idx + 1} failed: {e}", exc_info=True)
                continue

        # Training summary
        end_time = time.time()
        summary = {
            "total_episodes": num_episodes,
            "successful_episodes": successes,
            "success_rate": successes / num_episodes if num_episodes > 0 else 0.0,
            "mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "std_reward": np.std(total_rewards) if total_rewards else 0.0,
            "total_steps": self.total_steps,
            "training_time": end_time - start_time,
        }

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"Mean reward: {summary['mean_reward']:.3f}")
        logger.info(f"Total steps: {summary['total_steps']}")
        logger.info(f"Training time: {summary['training_time']:.1f}s")

        return summary

    def _extract_goal(self, obs: Dict[str, Any]) -> str:
        """Extract goal from observation."""
        goal_obj = obs.get("goal_object", [])
        if isinstance(goal_obj, list):
            return " ".join(
                item.get("text", "")
                for item in goal_obj
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(goal_obj)

    def _get_screenshot(self, obs: Dict[str, Any]) -> bytes:
        """Get screenshot bytes from observation."""
        screenshot = obs.get("screenshot")
        if screenshot is not None:
            if isinstance(screenshot, bytes):
                return screenshot
            # Handle numpy array
            try:
                from PIL import Image
                import io

                if hasattr(screenshot, 'tobytes'):
                    # Numpy array
                    img = Image.fromarray(screenshot)
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    return buffer.getvalue()
            except Exception as e:
                logger.warning(f"Failed to convert screenshot: {e}")

        # Return empty bytes if no screenshot
        return b""

    def _extract_valid_bids(self, obs: Dict[str, Any]) -> set:
        """Extract valid element IDs from observation."""
        from ..obs import extract_actionable_nodes

        axtree = obs.get("axtree_object")
        if axtree:
            elements = extract_actionable_nodes(axtree, top_k=50)
            return {str(e["bid"]) for e in elements}
        return set()

    def _log_metrics(
        self,
        episode_num: int,
        episode: Episode,
        update_metrics: Dict[str, Any],
    ):
        """Log training metrics."""
        metrics = {
            "episode": episode_num,
            "task": episode.task_name,
            "success": episode.success,
            "steps": episode.num_steps,
            "total_reward": episode.total_reward,
            "duration": episode.end_time - episode.start_time,
            **update_metrics,
        }

        self.metrics_history.append(metrics)

        logger.info(
            f"[{episode_num}] {episode.task_name}: "
            f"{'SUCCESS' if episode.success else 'FAIL'} | "
            f"steps={episode.num_steps} | "
            f"reward={episode.total_reward:.3f} | "
            f"time={metrics['duration']:.1f}s"
        )

    def _save_checkpoint(self, episode_num: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = checkpoint_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

        # Save training state
        state_file = checkpoint_dir / f"state_ep{episode_num}.json"
        state = {
            "episode_num": episode_num,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_generations": self.config.num_generations,
                "temperature": self.config.temperature,
                "learning_rate": self.config.learning_rate,
                "loss_type": self.config.loss_type,
            },
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved at episode {episode_num}")

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.episodes:
            return {"total_episodes": 0}

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "success_rate": sum(1 for e in self.episodes if e.success) / len(self.episodes),
            "mean_episode_reward": np.mean([e.total_reward for e in self.episodes]),
            "mean_episode_steps": np.mean([e.num_steps for e in self.episodes]),
            "policy_stats": self.policy_client.get_stats(),
        }

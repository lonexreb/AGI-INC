"""Qwen-based Worker policy for HALO Agent.

Supports two backends:
1. Local Transformers (for quick local tests)
2. OpenAI-compatible endpoint (vLLM) via base_url

This enables training with Unsloth and deployment with vLLM.
"""

import json
import logging
import re
import os
from typing import Dict, Optional, Any, Set

logger = logging.getLogger(__name__)

# Action grammar reference (same as OpenAI worker)
ACTION_GRAMMAR = """
Available actions (use exact syntax):
- click("bid") - Click element with browser ID
- fill("bid", "text") - Fill text input with value
- select_option("bid", "option") - Select dropdown option
- scroll(x, y) - Scroll page by x,y pixels
- go_back() - Navigate back
- go_forward() - Navigate forward
- goto("url") - Navigate to URL
- send_msg_to_user("message") - Send message to complete task
- noop() - Do nothing this step
"""


class QwenWorkerPolicy:
    """Qwen-based worker policy with LoRA adapter support.
    
    Supports:
    - Zero-shot inference (no finetuning)
    - BC finetuned (LoRA adapter)
    - DPO finetuned (LoRA adapter)
    
    Backends:
    - local: Uses transformers directly (slow but works anywhere)
    - vllm: Uses OpenAI-compatible API (fast, requires vLLM server)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        backend: str = "vllm",
        base_url: str = "http://localhost:8000/v1",
        adapter_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 300
    ):
        """Initialize Qwen worker policy.
        
        Args:
            model_name: HuggingFace model name or path
            backend: 'local' for transformers, 'vllm' for OpenAI-compatible API
            base_url: API base URL for vllm backend
            adapter_path: Path to LoRA adapter (for BC/DPO finetuned models)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.backend = backend
        self.base_url = base_url
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._model = None
        self._tokenizer = None
        
        self.invalid_id_count = 0
        self.total_action_count = 0

    def _init_vllm_client(self):
        """Initialize OpenAI-compatible client for vLLM."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="dummy"  # vLLM doesn't require real key
            )
        return self._client

    def _init_local_model(self):
        """Initialize local transformers model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                logger.info(f"Loading model: {self.model_name}")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Load LoRA adapter if specified
                if self.adapter_path and os.path.exists(self.adapter_path):
                    logger.info(f"Loading LoRA adapter: {self.adapter_path}")
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        self.adapter_path
                    )
                
                logger.info("Model loaded successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import required libraries: {e}")
                raise RuntimeError(
                    "Local backend requires: pip install transformers torch peft"
                )
        
        return self._model, self._tokenizer

    def get_action(
        self,
        obs_summary: str,
        goal: str,
        action_history: list = None,
        manager_guidance: Optional[Dict] = None,
        valid_bids: Optional[Set[str]] = None,
        last_action_error: str = ""
    ) -> Dict[str, Any]:
        """Get action from Qwen worker policy.
        
        Args:
            obs_summary: Summarized observation text
            goal: Task goal
            action_history: List of previous actions
            manager_guidance: Optional guidance from manager
            valid_bids: Set of valid element IDs
            last_action_error: Error from last action
            
        Returns:
            Dict with keys: action, rationale, confidence, raw_response
        """
        action_history = action_history or []
        valid_bids = valid_bids or set()
        self.total_action_count += 1
        
        # Build prompt
        prompt = self._build_prompt(
            obs_summary, goal, action_history,
            manager_guidance, valid_bids, last_action_error
        )
        
        try:
            if self.backend == "vllm":
                response = self._call_vllm(prompt)
            else:
                response = self._call_local(prompt)
            
            # Parse response
            result = self._parse_response(response)
            action = result.get("action", "noop()")
            
            # Validate and repair
            action, was_repaired = self._validate_and_repair_action(action, valid_bids)
            if was_repaired:
                self.invalid_id_count += 1
            
            return {
                "action": action,
                "rationale": result.get("rationale", ""),
                "confidence": result.get("confidence", 0.5),
                "raw_response": response,
                "was_repaired": was_repaired
            }
            
        except Exception as e:
            logger.error(f"Qwen worker error: {e}")
            return {
                "action": "noop()",
                "rationale": f"Error: {e}",
                "confidence": 0.0,
                "raw_response": "",
                "was_repaired": True
            }

    def _build_prompt(
        self,
        obs_summary: str,
        goal: str,
        action_history: list,
        manager_guidance: Optional[Dict],
        valid_bids: Set[str],
        last_action_error: str
    ) -> str:
        """Build the prompt for Qwen."""
        system_prompt = f"""You are a browser automation agent. Your goal is to complete the given task.

{ACTION_GRAMMAR}

CRITICAL RULES:
1. You MUST ONLY use element IDs (bid values) from the "Actionable Elements" list.
2. NEVER invent element IDs. If no suitable element, use scroll(0, 300) or go_back().
3. Do NOT call send_msg_to_user unless you see clear confirmation.
4. If the last action had an error, try a different approach.

Output ONLY valid JSON:
{{"action": "your_action_here", "rationale": "brief explanation", "confidence": 0.0-1.0}}
"""
        
        user_parts = [f"# Goal\n{goal}"]
        
        if last_action_error:
            user_parts.append(f"# ⚠️ LAST ACTION ERROR\n{last_action_error}")
        
        user_parts.append(obs_summary)
        
        if action_history:
            recent = action_history[-5:]
            user_parts.append(f"# Recent Actions\n" + "\n".join(recent))
        
        if manager_guidance:
            user_parts.append(
                f"# Manager Guidance\n"
                f"Subgoal: {manager_guidance.get('subgoal', 'N/A')}\n"
                f"Skill: {manager_guidance.get('skill', 'N/A')}"
            )
        
        if valid_bids:
            user_parts.append(f"# VALID ELEMENT IDs\n{', '.join(sorted(valid_bids)[:50])}")
        
        user_parts.append('# Your Next Action\nRespond with JSON only.')
        
        user_content = "\n\n".join(user_parts)
        
        # Format for Qwen chat template
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    def _call_vllm(self, prompt: str) -> str:
        """Call vLLM OpenAI-compatible API."""
        client = self._init_vllm_client()
        
        response = client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["<|im_end|>"]
        )
        
        return response.choices[0].text.strip()

    def _call_local(self, prompt: str) -> str:
        """Call local transformers model."""
        model, tokenizer = self._init_local_model()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract action directly
            action_match = re.search(r'(click|fill|scroll|go_back|noop|send_msg_to_user)\([^)]*\)', response)
            if action_match:
                return {"action": action_match.group(), "rationale": "Extracted from response"}
            return {"action": "noop()", "rationale": "Failed to parse response"}

    def _extract_bid_from_action(self, action: str) -> Optional[str]:
        """Extract element ID from action string."""
        match = re.match(r'^(?:click|fill|select_option|hover|press|focus)\("([^"]+)"', action)
        if match:
            return match.group(1)
        return None

    def _validate_and_repair_action(self, action: str, valid_bids: Set[str]) -> tuple:
        """Validate action and repair if needed."""
        action = action.strip()
        
        valid_prefixes = [
            'click(', 'fill(', 'select_option(', 'scroll(',
            'go_back()', 'go_forward()', 'goto(', 'send_msg_to_user(',
            'noop()', 'hover(', 'press(', 'focus('
        ]
        
        has_valid_prefix = any(action.startswith(p) for p in valid_prefixes)
        if not has_valid_prefix:
            return "noop()", True
        
        bid = self._extract_bid_from_action(action)
        if bid and valid_bids and bid not in valid_bids:
            return "scroll(0, 300)", True
        
        return action, False

    def get_invalid_id_rate(self) -> float:
        """Get the rate of invalid ID usage."""
        if self.total_action_count == 0:
            return 0.0
        return self.invalid_id_count / self.total_action_count


def create_qwen_worker_policy(
    mode: str = "zero",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    backend: str = "vllm",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
) -> QwenWorkerPolicy:
    """Factory function for Qwen worker policy.
    
    Args:
        mode: 'zero' (zero-shot), 'bc' (BC finetuned), 'dpo' (DPO finetuned)
        model_name: HuggingFace model name
        backend: 'local' or 'vllm'
        base_url: API URL for vllm backend
        temperature: Sampling temperature (0 for deterministic)
        
    Returns:
        Configured QwenWorkerPolicy
    """
    adapter_path = None
    
    if mode == "bc":
        adapter_path = "checkpoints/qwen_bc_lora"
    elif mode == "dpo":
        adapter_path = "checkpoints/qwen_dpo_lora"
    elif mode == "grpo":
        adapter_path = "checkpoints/qwen_grpo_lora"
    
    return QwenWorkerPolicy(
        model_name=model_name,
        backend=backend,
        base_url=base_url,
        adapter_path=adapter_path,
        temperature=temperature,
    )

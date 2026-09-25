"""Load a causal LM from Hugging Face, generate, then release the GPU.

Attacker, classifier, safe-response teacher, and defender are never resident
together. Each caller loads one model, uses it, and calls ``unload``.
"""

from __future__ import annotations

from .utils import free_gpu, get_logger

LOGGER = get_logger(__name__)


class LocalCausalLM:
    def __init__(
        self,
        model_id: str,
        *,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        max_seq_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self._load_in_4bit = load_in_4bit
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = None
        if self._load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        LOGGER.info("Loading %s from Hugging Face", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self._trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": self._trust_remote_code,
        }
        if not self._load_in_4bit:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        free_gpu()

    def complete_chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 0.95,
    ) -> str:
        """Chat-template generation. The template already includes special tokens."""
        self.load()
        tokenizer = self._tokenizer
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        return self._generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

    def _generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_length,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

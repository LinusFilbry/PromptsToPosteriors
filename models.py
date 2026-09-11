import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from google import genai
from google.genai import types
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig


# Abstract interface for all LLM provider adapters
class Model(ABC):
    # Generate n completions (strings) for the provided messages at the given temperature
    def __init__(self):
        self.is_thinking = False

    @abstractmethod
    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1, *,
                      disable_thinking: bool = False, seed: int | None = None, addon_assistant: str = "") -> list[str]:
        raise NotImplementedError


    # for each conversation, get the probability of each of the responses occurring (relative to each other)
    @abstractmethod
    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int, disable_thinking: bool = False) -> list[dict[str, float]]:
        raise NotImplementedError


class VLLMAdapter(Model):
    def __init__(self, seed: int, model_name: str = "Qwen/Qwen2.5-72B", tensor_parallel_size: int | None = None,
                 dtype: str = "float16", max_model_len: int = 1000, max_tokens: int = 1500,
                 gpu_memory_util: float = 0.82, max_num_seqs: int | None = 50, **llm_kwargs):
        self.seed = seed
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.is_thinking = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs

        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_util,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            **llm_kwargs
        )


    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1, *,
                      disable_thinking: bool = False, seed: int | None = None, addon_assistant: str = "") -> list[str]:
        prompt = self._messages_to_prompt(messages)
        prompt += addon_assistant

        sampling_params = SamplingParams(
            seed=self.seed if seed is None else seed,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_tokens=self.max_tokens,
            n=num_responses,
        )

        outputs = self.llm.generate([prompt], sampling_params)

        completions = [
            addon_assistant + o.text.lstrip()
            for o in outputs[0].outputs
        ]

        return completions


    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int, disable_thinking: bool = False) -> list[dict[str, float]]:
        # there are either many conversations or many responses - this way, parallelization happens over the more effective one
        if len(responses) > len(messages):
            result = []
            for conv in messages:
                result.append(self.chat_probabilities_single_message_parallel_responses(conv, responses, temperature, block_size))

            return result

        prompts = [self._messages_to_prompt(m) for m in messages]

        sampling_params = SamplingParams(
            temperature=0.0, # no seed necessary (generation is irrelevant + temp 0.0)
            max_tokens=1,
            prompt_logprobs=1,
        )

        tokenized_prompts = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in prompts
        ]

        base_lens = [len(toks) for toks in tokenized_prompts]

        results: list[dict[str, float]] = []

        for block_start in range(0, len(prompts), block_size):
            block_prompts = prompts[block_start:block_start + block_size]
            base_token_lens = base_lens[block_start:block_start + block_size]

            # Collect log-probs per response, per conversation
            # shape: [num_responses][block_size]
            response_logprobs = []

            for response in responses:
                full_prompts = [
                    p + response for p in block_prompts
                ]

                outputs = self.llm.generate(
                    full_prompts,
                    sampling_params=sampling_params,
                )

                block_logprobs = [0.0] * len(outputs)

                for i, output in enumerate(outputs):
                    prompt_logprobs = output.prompt_logprobs
                    start = base_token_lens[i]

                    total_lp = 0.0
                    # exactly one token for 'Yes'/'No'
                    for j in range(start, len(prompt_logprobs)):
                        token_lp_dict = prompt_logprobs[j]
                        # first item is always the 'fixed' response
                        total_lp += next(iter(token_lp_dict.values())).logprob

                    block_logprobs[i] = total_lp

                response_logprobs.append(block_logprobs)

            # Convert log-probs to probabilities PER CONVERSATION
            # transpose to [block_size][num_responses]
            for convo_idx in range(len(block_prompts)):
                lps = [
                    response_logprobs[r][convo_idx]
                    for r in range(len(responses))
                ]

                # calculate softmax
                scaled = [lp / temperature for lp in lps]
                max_lp = max(scaled)
                exps = [math.exp(lp - max_lp) for lp in scaled]
                norm = sum(exps)

                probs = [e / norm for e in exps]

                results.append(dict(zip(responses, probs)))

        return results


    def chat_probabilities_single_message_parallel_responses(self, messages: list[dict[str, str]], responses: list[str], temperature: float, block_size: int) -> dict[str, float]:
        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=0.0, # no seed necessary (generation is irrelevant + temp 0.0)
            max_tokens=1,
            prompt_logprobs=1,
        )

        base_tokens = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        base_len = len(base_tokens)

        # prevent CUDA OOM: the typical yes/no batched calls in 20Q are 120 tokens in length,
        # this calculation forces similar memory usage
        max_batch_size = max(1, (120 * block_size) // base_len)

        response_logprobs = [0.0] * len(responses)

        for batch_start in range(0, len(responses), max_batch_size):
            batch_responses = responses[batch_start : batch_start + max_batch_size]

            full_prompts = [
                prompt + response
                for response in batch_responses
            ]

            outputs = self.llm.generate(
                full_prompts,
                sampling_params=sampling_params,
            )

            for i, output in enumerate(outputs):
                prompt_logprobs = output.prompt_logprobs
                total_lp = 0.0

                # Sum logprobs for response tokens only
                for j in range(base_len, len(prompt_logprobs)):
                    token_lp_dict = prompt_logprobs[j]
                    total_lp += next(iter(token_lp_dict.values())).logprob

                response_logprobs[batch_start + i] = total_lp

        # Softmax over responses
        scaled = [lp / temperature for lp in response_logprobs]
        max_lp = max(scaled)
        exps = [math.exp(lp - max_lp) for lp in scaled]
        norm = sum(exps)

        probs = [e / norm for e in exps]

        return dict(zip(responses, probs))


class GeminiAdapter(Model):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", thinking_budget: int = -1, thinking_level: str = "low", allow_code: bool = False):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget # for series 2.5: https://ai.google.dev/gemini-api/docs/thinking#set-budget --> 0 for no thinking, -1 for dynamic
        self.thinking_level = thinking_level # for series 3: minimal, low, medium, high. for gemma-4: minimal = Off, high = On.
        self.allow_code = allow_code
        self.is_thinking = thinking_budget != 0


    @staticmethod
    def _transform_messages(messages: list[dict[str, str]]):
        system_instr = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_instr = content
            else:
                genai_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=genai_role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )
        return system_instr, contents


    def _get_thinking_config(self):
        if "gemini-3" in self.model_name or "gemma-4" in self.model_name:
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=self.thinking_level
            )

        return types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=self.thinking_budget
        )

    def _get_http_options(self):
        retry_options = types.HttpRetryOptions(
            attempts=10,
            initial_delay=2.0,
            max_delay=60.0,
            exp_base=2.0,
            jitter=0.5,
            http_status_codes=[429, 503]
        )
        return types.HttpOptions(retry_options=retry_options)

    def chat_complete(self, messages, temperature=1.0, num_responses=1, *, disable_thinking: bool = False,
                      seed: int | None = None, addon_assistant: str = ""):
        if num_responses != 1:
            raise ValueError("num_responses > 1 is not implemented by me for Gemini API")

        system_instruction, contents = self._transform_messages(messages)
        thinking_cfg = self._get_thinking_config()
        http_options = self._get_http_options()
        tools = [types.Tool(code_execution=types.ToolCodeExecution)] if self.allow_code else []

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            thinking_config=thinking_cfg,
            http_options=http_options,
            tools=tools,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        first_candidate = response.candidates[0]
        final_text_parts = []
        # Counters for better logging
        thought_idx = 1
        code_idx = 1

        for part in first_candidate.content.parts:
            if part.thought:
                print(f"\n--- GEMINI THOUGHT PROCESS (Part {thought_idx}) ---")
                print(part.text)
                print("------------------------------------------\n")
                thought_idx += 1

            if part.executable_code:
                print(f">>> EXECUTABLE CODE ({code_idx}):")
                print(part.executable_code.code)

            if part.code_execution_result:
                print(f">>> EXECUTION RESULT ({code_idx}):")
                print(part.code_execution_result.output)
                print("-" * 30)
                code_idx += 1

            if part.text and not part.thought:
                final_text_parts.append(part.text)

        return ["\n".join(final_text_parts)]


    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int, disable_thinking: bool = False) -> list[dict[str, float]]:
        raise NotImplementedError()


class ThinkingVLLMAdapter(VLLMAdapter):
    LOG_LABEL = "MODEL"

    def __init__(self, seed: int, model_name: str, tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16", thinking: bool = True, max_model_len: int = 1500,
                 max_model_len_thinking: int = 5000, gpu_memory_util: float = 0.5,
                 max_num_seqs: int | None = 150, thinking_token_budget: int = 3000):
        llm_kwargs = self._reasoning_llm_kwargs() if thinking else {}

        super().__init__(seed, model_name, tensor_parallel_size, dtype,
                         max_model_len=max_model_len_thinking if thinking else max_model_len,
                         max_tokens=max_model_len_thinking if thinking else max_model_len,
                         gpu_memory_util=gpu_memory_util, max_num_seqs=max_num_seqs,
                         **llm_kwargs)

        self.is_thinking = thinking
        self.max_tokens_thinking = max_model_len_thinking
        self.max_tokens_no_thinking = max_model_len
        self.thinking_token_budget = thinking_token_budget

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "audio": 0}
        )

    # Reasoning-config kwargs to pass to the vLLM engine; only called when thinking is enabled.
    def _reasoning_llm_kwargs(self) -> dict:
        raise NotImplementedError

    def _process_reasoning(self, output, prefix: str) -> tuple[str, str | None]:
        raise NotImplementedError

    def _messages_to_prompt(self, messages: list[dict[str, str]], thinking_enabled: bool | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_thinking if thinking_enabled is None else thinking_enabled,
        )

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1, *,
                      disable_thinking: bool = False, seed: int | None = None, addon_assistant: str = "") -> list[str]:
        effective_thinking = self.is_thinking and not disable_thinking

        # addon_assistant is accepted for interface compatibility but not applied here: only the
        # base VLLMAdapter prefills the assistant turn (this mirrors the pre-merge behaviour).
        prompt = self._messages_to_prompt(messages, thinking_enabled=effective_thinking)

        llm_kwargs = {}
        if effective_thinking:
            llm_kwargs["thinking_token_budget"] = self.thinking_token_budget

        sampling_params = SamplingParams(
            seed=self.seed if seed is None else seed,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            # Use the higher token limit only if thinking is effectively active
            max_tokens=self.max_tokens_thinking if effective_thinking else self.max_tokens_no_thinking,
            n=num_responses,
            **llm_kwargs,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        completions = []

        # Process each response in the batch
        for i, o in enumerate(outputs[0].outputs):
            final_text, thinking_trace = self._process_reasoning(o, prompt)

            # Use effective_thinking to decide if we should log thoughts
            if effective_thinking and thinking_trace:
                print(f"\n--- {self.LOG_LABEL} THOUGHT PROCESS (Response {i+1}) ---")
                print(thinking_trace)
                print("------------------------------------------\n")

            completions.append(final_text)

        return completions

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int, disable_thinking: bool = False) -> list[dict[str, float]]:
        effective_thinking = self.is_thinking and not disable_thinking

        if not effective_thinking:
            return super().chat_probabilities_messages_batched(messages, responses, temperature, block_size)

        prompts = [self._messages_to_prompt(message, effective_thinking) for message in messages]

        sampling_params = SamplingParams(
            seed=self.seed,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_tokens=self.max_tokens_thinking,
            thinking_token_budget=self.thinking_token_budget,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        batch_completions = []

        for prompt_idx, output in enumerate(outputs):
            for i, o in enumerate(output.outputs):
                final_text, thinking_trace = self._process_reasoning(o, prompts[prompt_idx])

                if self.is_thinking and thinking_trace:
                    print(f"\n--- {self.LOG_LABEL} THOUGHT PROCESS (Prompt {prompt_idx+1}, Response {i+1}) ---")
                    print(thinking_trace)
                    print("------------------------------------------\n")

                try:
                    # Attempt to parse the LLM output
                    completion_dict = {
                        k.strip(): float(v.strip())
                        for k, v in (item.split(":") for item in final_text.split(","))
                    }

                    if "Yes" not in completion_dict and "No" not in completion_dict:
                        print("\n\n\nError when parsing response")
                        print(final_text)
                        print("\n\n\n")
                        completion_dict = {"Yes": 0.5, "No": 0.5}
                    else:
                        # If only one key is missing, calculate the complement
                        if "Yes" not in completion_dict:
                            completion_dict["Yes"] = 1.0 - completion_dict["No"]
                        if "No" not in completion_dict:
                            completion_dict["No"] = 1.0 - completion_dict["Yes"]

                except (ValueError, IndexError):
                    # Fallback if splitting fails or float conversion blows up
                    completion_dict = {"Yes": 0.5, "No": 0.5}
                    print("\n\n\nError when parsing response")
                    print(final_text)
                    print("\n\n\n")

                batch_completions.append(completion_dict)

        return batch_completions


class GemmaVLLMAdapter(ThinkingVLLMAdapter):
    LOG_LABEL = "GEMMA"

    def __init__(self, seed: int, model_name: str, tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16", thinking: bool = True, max_model_len: int = 1500,
                 max_model_len_thinking: int = 5000, gpu_memory_util: float = 0.5,
                 max_num_seqs: int | None = 150, thinking_token_budget: int = 3000):
        super().__init__(seed, model_name, tensor_parallel_size, dtype, thinking,
                         max_model_len, max_model_len_thinking, gpu_memory_util,
                         max_num_seqs, thinking_token_budget)

    def _reasoning_llm_kwargs(self) -> dict:
        return {
            "reasoning_config": ReasoningConfig(
                reasoning_start_str="<|channel>thought",
                reasoning_end_str="\n\nI have to give the solution based on my reasoning immediately after this.<channel|>",
            ),
        }

    def _process_reasoning(self, output, prefix: str) -> tuple[str, str | None]:
        parse_response = getattr(self.tokenizer, "parse_response", None)
        token_ids = getattr(output, "token_ids", None)

        if not parse_response or not token_ids:
            return output.text.lstrip(), None

        parsed = parse_response(token_ids, prefix=prefix)

        if isinstance(parsed, dict):
            thinking = parsed.get("thinking", "").strip()
            content = parsed.get("content", "").strip()

            if content:
                return content, thinking if thinking else None

        return output.text.lstrip(), None


class QwenThinkingVLLMAdapter(ThinkingVLLMAdapter):
    LOG_LABEL = "QWEN 3.5"

    def __init__(self, seed: int, model_name: str = "Qwen/Qwen3.5-35B-A3B", tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16", thinking: bool = True, max_model_len: int = 1500,
                 max_model_len_thinking: int = 5000, gpu_memory_util: float = 0.9,
                 max_num_seqs: int | None = 50, thinking_token_budget: int = 3500):
        super().__init__(seed, model_name, tensor_parallel_size, dtype, thinking,
                         max_model_len, max_model_len_thinking, gpu_memory_util,
                         max_num_seqs, thinking_token_budget)

    # Qwen uses standard <think> tags for its reasoning blocks
    def _reasoning_llm_kwargs(self) -> dict:
        return {
            "reasoning_parser": "qwen3",
            "reasoning_config": ReasoningConfig(
                reasoning_parser="qwen3",
                reasoning_start_str="<think>",
                reasoning_end_str="\nI have to give the solution based on my reasoning immediately after this.</think>",
            ),
        }

    def _process_reasoning(self, output, prefix: str) -> tuple[str, str | None]:
        reasoning_text = getattr(output, "reasoning_text", None)
        if reasoning_text is not None:
            return output.text.lstrip(), reasoning_text.strip()

        text = output.text.lstrip()
        think_start_str = "<think>"
        think_end_str = "</think>"

        if think_start_str in text:
            start_idx = text.find(think_start_str)
            end_idx = text.find(think_end_str, start_idx)

            if end_idx != -1:
                thinking_trace = text[start_idx + len(think_start_str):end_idx].strip()
                final_text = text[end_idx + len(think_end_str):].lstrip()
                return final_text, thinking_trace
            else:
                # Edge case: The model hit the max_tokens limit before finishing its thought
                thinking_trace = text[start_idx + len(think_start_str):].strip()
                return "", thinking_trace

        if think_end_str in text:
            end_idx = text.find(think_end_str)
            thinking_trace = text[:end_idx].strip()
            final_text = text[end_idx + len(think_end_str):].lstrip()
            return final_text, thinking_trace

        return text, None


# --- Model registry -----------------------------------------------------------------------------

# One catalogue entry: which weights, which provider, which adapter, thinking on or off.
# Capacity is deliberately absent - that comes from the task's model_runtime block.
@dataclass
class ModelInfo:
    model_name: str
    provider: str
    family: str = "default"
    thinking: bool = False
    kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"


def load_model_registry(path: str | Path | None = None) -> dict[str, ModelInfo]:
    path = Path(path) if path is not None else MODEL_REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model registry '{path}' not found.")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        key: ModelInfo(
            model_name=data["model_name"],
            provider=data["provider"],
            family=data.get("family", "default"),
            thinking=data.get("thinking", False),
            kwargs=data.get("kwargs", {}),
        )
        for key, data in raw.items()
    }


# --- Model construction -------------------------------------------------------------------------

_THINKING_RUNTIME_KEYS = ("max_model_len", "max_model_len_thinking", "gpu_memory_util",
                          "max_num_seqs", "thinking_token_budget")

RUNTIME_KEYS = {
    VLLMAdapter: ("max_model_len", "gpu_memory_util", "max_num_seqs"),
    GemmaVLLMAdapter: _THINKING_RUNTIME_KEYS,
    QwenThinkingVLLMAdapter: _THINKING_RUNTIME_KEYS,
}

THINKING_ADAPTERS = (GemmaVLLMAdapter, QwenThinkingVLLMAdapter)


def get_adapter_class(model_info):
    if model_info.provider == "gemini":
        return GeminiAdapter

    if model_info.provider == "local":
        if model_info.family == "gemma":
            return GemmaVLLMAdapter
        if model_info.family == "qwen":
            return QwenThinkingVLLMAdapter
        return VLLMAdapter

    raise ValueError(f"Unknown provider: {model_info.provider}")


def _runtime_kwargs(config, adapter_class) -> dict:
    runtime = config.model_runtime
    kwargs = {key: runtime[key] for key in RUNTIME_KEYS[adapter_class] if key in runtime}

    if adapter_class is VLLMAdapter and "max_model_len" in runtime:
        kwargs["max_tokens"] = runtime["max_model_len"]

    return kwargs


def build_model(config, info: ModelInfo) -> Model:
    if info.provider == "gemini" and config.task == "twenty_questions":
        raise ValueError(
            f"Gemini is not supported for 20Q."
        )

    AdapterClass = get_adapter_class(info)

    init_kwargs = {"model_name": info.model_name}
    init_kwargs.update(info.kwargs)

    if info.provider == "gemini":
        init_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")
        init_kwargs["thinking_budget"] = config.model_runtime.get("thinking_token_budget", -1) if info.thinking else 0
    elif info.provider == "local":
        init_kwargs["seed"] = config.seed
        if issubclass(AdapterClass, THINKING_ADAPTERS):
            init_kwargs["thinking"] = info.thinking
        elif info.thinking:
            raise ValueError(
                f"Model '{info.model_name}' asks for thinking, but family '{info.family}' has no "
                f"thinking adapter. Use one of: gemma, qwen."
            )
        init_kwargs.update(_runtime_kwargs(config, AdapterClass))

    return AdapterClass(**init_kwargs)

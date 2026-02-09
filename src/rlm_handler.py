import os
import rlm
import time
from typing import Optional
from rlm.clients import get_client
from rlm.utils.prompts import RLM_SYSTEM_PROMPT


class _InMemoryRLMLogger:
    """Collect RLM iterations in memory so we can compute runtime metrics."""

    def __init__(self) -> None:
        self.iterations = []

    def log_metadata(self, metadata) -> None:
        _ = metadata

    def log(self, iteration) -> None:
        self.iterations.append(iteration)

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)


class RLMHandler:
    def __init__(
        self,
        backend: str = "openrouter",
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.backend = backend
        self.api_key = api_key
        self.verbose = verbose
        self.last_metrics: dict = {}

        # Ensure API key is set in environment for the backend to pick it up
        if api_key and backend == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif api_key and backend == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key

    @staticmethod
    def _compose_single_pass_prompt(question: str, context: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {question}"

    def query(
        self,
        prompt: str,
        context: str,
        model: Optional[str] = None,
        use_subagents: bool = False,
        system_prompt: Optional[str] = None,
        subagent_backend: Optional[str] = None,
        subagent_model: Optional[str] = None,
    ) -> str:
        """
        Query the model with prompt + context.

        Args:
            prompt: The user's question or instruction.
            context: The document text to be used as context.
            model: The specific model to use (e.g., 'gpt-4o', 'google/gemini-2.0-flash-001').
            use_subagents: Whether to enable RLM recursion (max_depth=1).
                When False, runs a direct LM completion (no RLM loop).
            system_prompt: Optional system prompt to guide the RLM.
            subagent_backend: Optional backend for recursive sub-calls.
            subagent_model: Optional model name for recursive sub-calls.

        Returns:
            The model's response.
        """
        backend_kwargs = {}
        if model:
            backend_kwargs["model_name"] = model

        try:
            if not use_subagents:
                client = get_client(self.backend, backend_kwargs)
                payload: str | list[dict[str, str]] = self._compose_single_pass_prompt(
                    prompt, context
                )
                if system_prompt:
                    payload = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload},
                    ]
                start_time = time.perf_counter()
                completion_text = str(client.completion(payload)).strip()
                end_time = time.perf_counter()
                self.last_metrics = self._build_direct_metrics(
                    model_name=getattr(client, "model_name", model or "unknown"),
                    model_usage=getattr(client, "get_last_usage", lambda: None)(),
                    execution_time=end_time - start_time,
                )
                return completion_text

            tracker = _InMemoryRLMLogger()
            rlm_system_prompt = self._compose_rlm_system_prompt(
                user_system_prompt=system_prompt,
                subagent_model=subagent_model,
            )
            agent = rlm.RLM(
                **self._build_rlm_kwargs(
                    backend_kwargs=backend_kwargs,
                    system_prompt=rlm_system_prompt,
                    subagent_backend=subagent_backend,
                    subagent_model=subagent_model,
                    logger=tracker,
                )
            )

            completions = []
            completion = agent.completion(prompt=context, root_prompt=prompt)
            completions.append(completion)
            response_text = self._extract_completion_text(completion)

            retry_attempted = False
            fallback_used = False
            unresolved_final_var_error = self._looks_like_unresolved_final_var_error(
                response_text
            )

            if unresolved_final_var_error:
                retry_attempted = True
                retry_prompt = self._build_final_var_retry_prompt(prompt, response_text)
                retry_completion = agent.completion(prompt=context, root_prompt=retry_prompt)
                completions.append(retry_completion)
                retry_response = self._extract_completion_text(retry_completion)
                if not self._looks_like_unresolved_final_var_error(retry_response):
                    response_text = retry_response
                    unresolved_final_var_error = False

            fallback_metrics: dict | None = None
            if unresolved_final_var_error:
                fallback_used = True
                response_text, fallback_metrics = self._fallback_after_rlm_final_var_error(
                    prompt=prompt,
                    context=context,
                    backend_kwargs=backend_kwargs,
                    system_prompt=system_prompt,
                )

            self.last_metrics = self._build_subagent_metrics(
                completions=completions,
                tracker=tracker,
                configured_root_model=backend_kwargs.get("model_name"),
                configured_subagent_model=subagent_model,
                retry_attempted=retry_attempted,
                fallback_used=fallback_used,
                unresolved_final_var_error=unresolved_final_var_error,
                fallback_metrics=fallback_metrics,
            )
            return response_text

        except Exception as e:
            self.last_metrics = {"error": str(e)}
            return f"Error querying RLM: {e}"

    @staticmethod
    def _extract_completion_text(completion) -> str:
        if hasattr(completion, "response"):
            return str(completion.response).strip()
        return str(completion).strip()

    @staticmethod
    def _looks_like_unresolved_final_var_error(response_text: str) -> bool:
        lower = response_text.lower()
        return (
            "error: variable" in lower
            and "not found" in lower
            and "final_var" in lower
        )

    @staticmethod
    def _build_final_var_retry_prompt(original_prompt: str, previous_response: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous attempt failed with a FINAL_VAR variable-not-found error:\n"
            f"{previous_response}\n\n"
            "Retry from scratch. Do not call FINAL_VAR unless you first create the variable in "
            "a repl block and confirm it exists (use SHOW_VARS()). If unsure, use FINAL(...) "
            "with a direct answer."
        )

    def _build_rlm_kwargs(
        self,
        backend_kwargs: dict[str, str],
        system_prompt: Optional[str],
        subagent_backend: Optional[str],
        subagent_model: Optional[str],
        logger,
    ) -> dict:
        rlm_kwargs = {
            "backend": self.backend,
            "backend_kwargs": backend_kwargs,
            "max_depth": 1,
            "custom_system_prompt": system_prompt,
            "verbose": self.verbose,
            "logger": logger,
        }

        if bool(subagent_backend) != bool(subagent_model):
            raise ValueError(
                "Both subagent_backend and subagent_model must be set together."
            )

        if subagent_backend and subagent_model:
            rlm_kwargs["other_backends"] = [subagent_backend]
            rlm_kwargs["other_backend_kwargs"] = [{"model_name": subagent_model}]

        return rlm_kwargs

    def _compose_rlm_system_prompt(
        self,
        user_system_prompt: Optional[str],
        subagent_model: Optional[str],
    ) -> str:
        prompt_parts = [RLM_SYSTEM_PROMPT]
        if user_system_prompt:
            prompt_parts.append(
                "Additional user instructions (must be followed while preserving RLM workflow):\n"
                + user_system_prompt.strip()
            )
        if subagent_model:
            prompt_parts.append(
                "Subagent routing: llm_query calls are routed to the configured subagent model "
                f"'{subagent_model}' by default."
            )
        else:
            prompt_parts.append(
                "Subagent routing: llm_query calls use the root model by default when no "
                "alternate subagent backend/model is configured."
            )
        prompt_parts.append(
            "Before FINAL, make at least one llm_query or llm_query_batched call unless the "
            "task is impossible."
        )
        prompt_parts.append(
            "Never call FINAL_VAR(variable_name) unless that variable is already created in the "
            "REPL. If uncertain, return FINAL(...) instead."
        )
        return "\n\n".join(part for part in prompt_parts if part)

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _model_usage_to_dict(self, usage) -> dict:
        if usage is None:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        return {
            "total_calls": self._safe_int(getattr(usage, "total_calls", 0)),
            "total_input_tokens": self._safe_int(
                getattr(usage, "total_input_tokens", 0)
            ),
            "total_output_tokens": self._safe_int(
                getattr(usage, "total_output_tokens", 0)
            ),
        }

    def _usage_summary_to_dict(self, usage_summary) -> dict:
        if not usage_summary:
            return {}

        model_usage = getattr(usage_summary, "model_usage_summaries", {}) or {}
        converted = {}
        for model_name, usage in model_usage.items():
            converted[model_name] = self._model_usage_to_dict(usage)
        return converted

    def _totals_from_model_usage(self, model_usage: dict) -> tuple[int, int]:
        total_input = sum(
            self._safe_int(item.get("total_input_tokens", 0))
            for item in model_usage.values()
        )
        total_output = sum(
            self._safe_int(item.get("total_output_tokens", 0))
            for item in model_usage.values()
        )
        return total_input, total_output

    def _build_direct_metrics(
        self,
        model_name: str,
        model_usage,
        execution_time: float | None,
    ) -> dict:
        usage = self._model_usage_to_dict(model_usage)
        total_input, total_output = self._totals_from_model_usage({model_name: usage})
        return {
            "mode": "direct_lm",
            "execution_time": execution_time,
            "iterations": 0,
            "subagent_calls": 0,
            "model_usage": {model_name: usage},
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    def _direct_completion_with_metrics(
        self,
        prompt: str,
        context: str,
        backend_kwargs: dict[str, str],
        system_prompt: Optional[str],
    ) -> tuple[str, dict]:
        client = get_client(self.backend, backend_kwargs)
        payload: str | list[dict[str, str]] = self._compose_single_pass_prompt(
            prompt, context
        )
        if system_prompt:
            payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ]
        start_time = time.perf_counter()
        completion_text = str(client.completion(payload)).strip()
        end_time = time.perf_counter()
        metrics = self._build_direct_metrics(
            model_name=getattr(client, "model_name", backend_kwargs.get("model_name", "unknown")),
            model_usage=getattr(client, "get_last_usage", lambda: None)(),
            execution_time=end_time - start_time,
        )
        return completion_text, metrics

    def _fallback_after_rlm_final_var_error(
        self,
        prompt: str,
        context: str,
        backend_kwargs: dict[str, str],
        system_prompt: Optional[str],
    ) -> tuple[str, dict]:
        text, metrics = self._direct_completion_with_metrics(
            prompt=prompt,
            context=context,
            backend_kwargs=backend_kwargs,
            system_prompt=system_prompt,
        )
        prefixed = (
            "[Recovered after RLM FINAL_VAR failure using direct completion]\n\n"
            + text
        )
        return prefixed, metrics

    def _merge_model_usage(self, base: dict, incoming: dict) -> dict:
        merged = dict(base)
        for model_name, usage in incoming.items():
            current = merged.get(
                model_name,
                {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                },
            )
            merged[model_name] = {
                "total_calls": self._safe_int(current.get("total_calls", 0))
                + self._safe_int(usage.get("total_calls", 0)),
                "total_input_tokens": self._safe_int(
                    current.get("total_input_tokens", 0)
                )
                + self._safe_int(usage.get("total_input_tokens", 0)),
                "total_output_tokens": self._safe_int(
                    current.get("total_output_tokens", 0)
                )
                + self._safe_int(usage.get("total_output_tokens", 0)),
            }
        return merged

    def _build_subagent_metrics(
        self,
        completions: list,
        tracker: _InMemoryRLMLogger,
        configured_root_model: Optional[str],
        configured_subagent_model: Optional[str],
        retry_attempted: bool,
        fallback_used: bool,
        unresolved_final_var_error: bool,
        fallback_metrics: Optional[dict],
    ) -> dict:
        model_usage = {}
        execution_time = 0.0
        execution_time_present = False
        for completion in completions:
            usage = self._usage_summary_to_dict(getattr(completion, "usage_summary", None))
            model_usage = self._merge_model_usage(model_usage, usage)
            completion_time = getattr(completion, "execution_time", None)
            if completion_time is not None:
                execution_time += float(completion_time)
                execution_time_present = True

        if configured_root_model and configured_root_model not in model_usage:
            model_usage[configured_root_model] = {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        if configured_subagent_model and configured_subagent_model not in model_usage:
            model_usage[configured_subagent_model] = {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        if fallback_metrics:
            model_usage = self._merge_model_usage(
                model_usage, fallback_metrics.get("model_usage", {})
            )
            fallback_time = fallback_metrics.get("execution_time")
            if fallback_time is not None:
                execution_time += float(fallback_time)
                execution_time_present = True

        total_input, total_output = self._totals_from_model_usage(model_usage)
        subagent_calls = 0
        for iteration in tracker.iterations:
            for code_block in getattr(iteration, "code_blocks", []):
                result = getattr(code_block, "result", None)
                if not result:
                    continue
                subagent_calls += len(getattr(result, "rlm_calls", []) or [])

        return {
            "mode": "rlm_subagents",
            "execution_time": execution_time if execution_time_present else None,
            "iterations": tracker.iteration_count,
            "subagent_calls": subagent_calls,
            "configured_root_model": configured_root_model,
            "configured_subagent_model": configured_subagent_model,
            "retry_attempted": retry_attempted,
            "fallback_used": fallback_used,
            "unresolved_final_var_error": unresolved_final_var_error,
            "model_usage": model_usage,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

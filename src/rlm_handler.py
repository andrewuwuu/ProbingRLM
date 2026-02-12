import os
import time
from typing import Any, Callable, Optional

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None

API_KEY_ENV_BY_BACKEND = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "portkey": "PORTKEY_API_KEY",
    "vercel": "AI_GATEWAY_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}
STRICT_API_KEY_BACKENDS = {"anthropic", "portkey", "gemini", "azure_openai", "litellm"}


def _read_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


if dspy is not None:
    class _TracingLMProxy(dspy.BaseLM):
        def __init__(self, lm: Any, label: str):
            # Initialize BaseLM so DSPy internals that type/attribute-check BaseLM
            # can safely consume this proxy instance.
            super().__init__(
                model=getattr(lm, "model", "unknown"),
                model_type=getattr(lm, "model_type", "chat"),
                temperature=getattr(lm, "temperature", 0.0),
                max_tokens=getattr(lm, "max_tokens", 1000),
                cache=getattr(lm, "cache", True),
            )
            self._lm = lm
            self._label = label
            if hasattr(lm, "history"):
                self.history = getattr(lm, "history")
            if hasattr(lm, "kwargs"):
                self.kwargs = dict(getattr(lm, "kwargs") or {})

        def __getattr__(self, name: str) -> Any:
            return getattr(self._lm, name)

        @staticmethod
        def _usage_from_entry(entry: dict) -> tuple[int, int]:
            usage = entry.get("usage") or {}
            if not isinstance(usage, dict):
                return 0, 0
            input_tokens = (
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or usage.get("total_input_tokens")
                or usage.get("prompt_token_count")
                or 0
            )
            output_tokens = (
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or usage.get("total_output_tokens")
                or usage.get("candidates_token_count")
                or 0
            )
            try:
                return int(input_tokens), int(output_tokens)
            except Exception:
                return 0, 0

        def _log_end(self, start_hist: int, start_time: float) -> None:
            elapsed = time.perf_counter() - start_time
            new_entries = list((getattr(self._lm, "history", []) or [])[start_hist:])
            in_tokens = 0
            out_tokens = 0
            for entry in new_entries:
                in_t, out_t = self._usage_from_entry(entry)
                in_tokens += in_t
                out_tokens += out_t
            print(
                f"[{self._label}] call finished in {elapsed:.1f}s"
                f" (calls={len(new_entries)}, in={in_tokens}, out={out_tokens})",
                flush=True,
            )

        def __call__(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
            start_hist = len(getattr(self._lm, "history", []) or [])
            start_time = time.perf_counter()
            print(f"[{self._label}] call started", flush=True)
            if messages is not None and prompt is None:
                result = self._lm(messages=messages, **kwargs)
            elif prompt is not None and messages is None:
                result = self._lm(prompt=prompt, **kwargs)
            else:
                result = self._lm(prompt=prompt, messages=messages, **kwargs)
            self._log_end(start_hist, start_time)
            return result

        async def acall(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
            start_hist = len(getattr(self._lm, "history", []) or [])
            start_time = time.perf_counter()
            print(f"[{self._label}] call started", flush=True)
            if hasattr(self._lm, "acall"):
                if messages is not None and prompt is None:
                    result = await self._lm.acall(messages=messages, **kwargs)
                elif prompt is not None and messages is None:
                    result = await self._lm.acall(prompt=prompt, **kwargs)
                else:
                    result = await self._lm.acall(prompt=prompt, messages=messages, **kwargs)
            else:
                result = self.__call__(prompt=prompt, messages=messages, **kwargs)
            self._log_end(start_hist, start_time)
            return result

        def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
            if hasattr(self._lm, "forward"):
                if messages is not None and prompt is None:
                    return self._lm.forward(messages=messages, **kwargs)
                if prompt is not None and messages is None:
                    return self._lm.forward(prompt=prompt, **kwargs)
                return self._lm.forward(prompt=prompt, messages=messages, **kwargs)
            return self.__call__(prompt=prompt, messages=messages, **kwargs)

        async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
            if hasattr(self._lm, "aforward"):
                if messages is not None and prompt is None:
                    return await self._lm.aforward(messages=messages, **kwargs)
                if prompt is not None and messages is None:
                    return await self._lm.aforward(prompt=prompt, **kwargs)
                return await self._lm.aforward(prompt=prompt, messages=messages, **kwargs)
            return await self.acall(prompt=prompt, messages=messages, **kwargs)
else:
    class _TracingLMProxy:  # pragma: no cover
        def __init__(self, lm: Any, label: str):
            self._lm = lm
            self._label = label

        def __getattr__(self, name: str) -> Any:
            return getattr(self._lm, name)

        def __call__(self, *args, **kwargs):
            return self._lm(*args, **kwargs)


class DSPyRLMHandler:
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

        key_env_var = API_KEY_ENV_BY_BACKEND.get(backend)
        if api_key and key_env_var:
            os.environ[key_env_var] = api_key

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _compose_single_pass_prompt(question: str, context: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {question}"

    @staticmethod
    def _compose_system_prompt(
        system_prompt: Optional[str],
        custom_prompt: Optional[str],
    ) -> str | None:
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        if custom_prompt:
            parts.append(custom_prompt.strip())
        if not parts:
            return None
        return "\n\n".join(parts)

    @staticmethod
    def _compose_rlm_guidance(
        system_prompt: Optional[str],
        custom_prompt: Optional[str],
        max_depth: Optional[int],
        subagent_model: Optional[str],
    ) -> str:
        parts = [
            "Answer only from the provided context and tool outputs.",
            "Use llm_query or llm_query_batched when decomposition is helpful.",
            "Return the final answer with SUBMIT(...).",
            "Execution safety: avoid unbounded loops (no while True). Use bounded loops only "
            "(for example, for i in range(n) with a small explicit n) and finish quickly.",
        ]
        if max_depth is not None:
            parts.append(
                "Depth policy: keep recursive decomposition depth at or below "
                f"{max_depth}."
            )
        if subagent_model:
            parts.append(
                "Subagent routing: llm_query calls route to configured subagent model "
                f"'{subagent_model}'."
            )
        if system_prompt:
            parts.append("System instructions:\n" + system_prompt.strip())
        if custom_prompt:
            parts.append("Custom instructions:\n" + custom_prompt.strip())
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_rlm_signature(signature: str) -> str:
        """
        DSPy reserves the attribute name `instructions` on Signature.
        If users provide that as an input field, rewrite it to `guidance`.
        """
        stripped = signature.strip()
        if "->" not in stripped:
            return stripped

        left, right = stripped.split("->", 1)
        fields = [field.strip() for field in left.split(",") if field.strip()]
        normalized: list[str] = []
        for field in fields:
            name, *rest = field.split(":", 1)
            mapped = "guidance" if name.strip() == "instructions" else name.strip()
            if rest:
                normalized.append(f"{mapped}:{rest[0].strip()}")
            else:
                normalized.append(mapped)
        return f"{', '.join(normalized)} -> {right.strip()}"

    @staticmethod
    def _extract_lm_text(completion_output) -> str:
        if isinstance(completion_output, list) and completion_output:
            first = completion_output[0]
            if isinstance(first, dict):
                if "content" in first:
                    return str(first["content"]).strip()
                if "text" in first:
                    return str(first["text"]).strip()
            return str(first).strip()
        return str(completion_output).strip()

    @staticmethod
    def _extract_prediction_text(prediction) -> str:
        for key in ("answer", "output", "response", "result"):
            if hasattr(prediction, key):
                return str(getattr(prediction, key)).strip()
        if isinstance(prediction, dict):
            for key in ("answer", "output", "response", "result"):
                if key in prediction:
                    return str(prediction[key]).strip()
        if hasattr(prediction, "_store"):
            store = getattr(prediction, "_store", {}) or {}
            for key in ("answer", "output", "response", "result"):
                if key in store:
                    return str(store[key]).strip()
        return str(prediction).strip()

    @staticmethod
    def _usage_from_history_entry(entry: dict) -> tuple[int, int]:
        usage = entry.get("usage") or {}
        if not isinstance(usage, dict):
            return 0, 0

        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("total_input_tokens")
            or usage.get("prompt_token_count")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("total_output_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        return int(input_tokens), int(output_tokens)

    def _history_to_model_usage(self, history_entries: list[dict]) -> dict:
        model_usage: dict[str, dict[str, int]] = {}
        for entry in history_entries:
            model_name = str(entry.get("model") or "unknown")
            current = model_usage.get(
                model_name,
                {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                },
            )
            in_tokens, out_tokens = self._usage_from_history_entry(entry)
            current["total_calls"] = self._safe_int(current["total_calls"]) + 1
            current["total_input_tokens"] = (
                self._safe_int(current["total_input_tokens"]) + in_tokens
            )
            current["total_output_tokens"] = (
                self._safe_int(current["total_output_tokens"]) + out_tokens
            )
            model_usage[model_name] = current
        return model_usage

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

    @staticmethod
    def canonical_model_for_backend(backend: str, model: Optional[str]) -> str:
        raw = (model or "").strip()

        if backend == "openrouter":
            candidate = raw or "openai/gpt-4.1-mini"
            if candidate == "free":
                return "openrouter/free"
            if candidate.startswith("openrouter/"):
                return candidate
            if "/" not in candidate:
                candidate = f"openai/{candidate}"
            return f"openrouter/{candidate}"

        if backend == "azure_openai":
            candidate = raw or "gpt-4.1-mini"
            if "/" in candidate and not candidate.startswith("azure/"):
                return f"azure/{candidate}"
            if candidate.startswith("azure/"):
                return candidate
            deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or candidate).strip()
            return f"azure/{deployment}"

        if raw and "/" in raw:
            return raw

        default_model = raw or "gpt-4.1-mini"
        prefix_by_backend = {
            "openai": "openai",
            "anthropic": "anthropic",
            "gemini": "gemini",
        }
        prefix = prefix_by_backend.get(backend, "openai")
        return f"{prefix}/{default_model}"

    def _resolve_model_name(self, backend: str, model: Optional[str]) -> str:
        return self.canonical_model_for_backend(backend, model)

    @staticmethod
    def _adapt_model_for_litellm_backend(backend: str, model_name: str) -> str:
        """
        LiteLLM strips one provider prefix (e.g. `openrouter/...` -> `...`) before
        dispatching to provider adapters. For OpenRouter router IDs like
        `openrouter/free`, that becomes `free` and OpenRouter rejects it.
        Prefix once more so the stripped value remains `openrouter/free`.
        """
        if backend != "openrouter":
            return model_name
        if not model_name.startswith("openrouter/"):
            return model_name

        suffix = model_name.split("/", 1)[1]
        if "/" in suffix:
            return model_name

        return f"openrouter/{model_name}"

    def _inject_backend_config(
        self,
        backend: str,
        lm_kwargs: dict[str, Any],
        explicit_api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        kwargs = dict(lm_kwargs)

        api_key = explicit_api_key or kwargs.get("api_key")
        if not api_key:
            key_env_var = API_KEY_ENV_BY_BACKEND.get(backend)
            if key_env_var:
                api_key = _read_env(key_env_var) or None

        if backend in STRICT_API_KEY_BACKENDS.union({"openai", "openrouter", "vercel"}):
            if api_key:
                kwargs.setdefault("api_key", api_key)

        if backend == "openrouter":
            kwargs.setdefault(
                "api_base",
                _read_env("DSPY_OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1",
            )

        if backend == "vllm":
            base_url = (
                kwargs.get("api_base")
                or _read_env("DSPY_VLLM_BASE_URL")
            )
            if not base_url:
                raise ValueError(
                    "Backend 'vllm' requires base_url. Set DSPY_VLLM_BASE_URL "
                    "in .env."
                )
            kwargs["api_base"] = base_url

        if backend == "litellm":
            api_base = (
                kwargs.get("api_base")
                or _read_env("DSPY_LITELLM_API_BASE")
            )
            if api_base:
                kwargs["api_base"] = api_base
            lite_api_key = _read_env("DSPY_LITELLM_API_KEY")
            if lite_api_key:
                kwargs.setdefault("api_key", lite_api_key)

        if backend == "vercel":
            vercel_api_base = _read_env("DSPY_VERCEL_API_BASE")
            if vercel_api_base:
                kwargs.setdefault("api_base", vercel_api_base)

        if backend == "portkey":
            portkey_api_base = _read_env("DSPY_PORTKEY_API_BASE")
            if portkey_api_base:
                kwargs.setdefault("api_base", portkey_api_base)

        if backend == "azure_openai":
            azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
            if azure_endpoint:
                kwargs.setdefault("api_base", azure_endpoint.rstrip("/"))
            api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
            if api_version:
                kwargs.setdefault("api_version", api_version)

        if backend in STRICT_API_KEY_BACKENDS and not kwargs.get("api_key"):
            key_env_var = API_KEY_ENV_BY_BACKEND.get(backend, "API key env var")
            raise ValueError(f"Backend '{backend}' requires API key ({key_env_var}).")

        return kwargs

    def _build_lm(
        self,
        backend: str,
        model: Optional[str],
        explicit_api_key: Optional[str],
        lm_kwargs: Optional[dict[str, Any]] = None,
    ):
        if dspy is None:
            raise RuntimeError(
                "dspy is not installed. Install project dependencies with `uv sync`."
            )

        resolved_model = self._resolve_model_name(backend, model)
        resolved_model = self._adapt_model_for_litellm_backend(backend, resolved_model)
        final_kwargs = self._inject_backend_config(
            backend=backend,
            lm_kwargs=lm_kwargs or {},
            explicit_api_key=explicit_api_key,
        )
        return dspy.LM(resolved_model, **final_kwargs)

    def _build_direct_metrics(
        self,
        model_usage: dict,
        execution_time: float | None,
    ) -> dict:
        total_input, total_output = self._totals_from_model_usage(model_usage)
        return {
            "mode": "direct_lm",
            "execution_time": execution_time,
            "iterations": 0,
            "subagent_calls": 0,
            "model_usage": model_usage,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    @staticmethod
    def _build_signature(
        signature_text: str,
        guidance: str,
    ):
        if dspy is None:
            raise RuntimeError(
                "dspy is not installed. Install project dependencies with `uv sync`."
            )
        return dspy.Signature(signature_text, guidance)

    @staticmethod
    def _build_rlm_call_inputs(
        signature_obj,
        context: str,
        query: str,
        guidance: str,
    ) -> dict[str, str]:
        context_aliases = {"context", "document", "document_context", "passage"}
        query_aliases = {"query", "question", "prompt", "user_query"}
        guidance_aliases = {"guidance", "instructions", "system_prompt", "task_instructions"}

        call_inputs: dict[str, str] = {}
        input_fields = list(getattr(signature_obj, "input_fields", {}).keys())
        for field_name in input_fields:
            lowered = field_name.lower()
            if lowered in context_aliases:
                call_inputs[field_name] = context
            elif lowered in query_aliases:
                call_inputs[field_name] = query
            elif lowered in guidance_aliases:
                call_inputs[field_name] = guidance

        missing = [name for name in input_fields if name not in call_inputs]
        if missing:
            raise ValueError(
                "Unsupported RLM signature inputs. This app can auto-populate only "
                "context/query/guidance-style fields. Missing values for: "
                + ", ".join(missing)
            )
        return call_inputs

    def query(
        self,
        prompt: str,
        context: str,
        model: Optional[str] = None,
        use_subagents: bool = False,
        system_prompt: Optional[str] = None,
        subagent_backend: Optional[str] = None,
        subagent_model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_subagent_calls: Optional[int] = None,
        max_llm_calls: Optional[int] = None,
        max_output_chars: Optional[int] = None,
        max_depth: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        rlm_signature: Optional[str] = None,
        root_lm_kwargs: Optional[dict[str, Any]] = None,
        subagent_lm_kwargs: Optional[dict[str, Any]] = None,
        rlm_tools: Optional[list[Callable]] = None,
        rlm_interpreter: Any = None,
        require_subagent_call: bool = False,
        subagent_prefetch_calls: int = 0,
        trace_lm_calls: bool = False,
        disable_json_adapter_fallback: bool = False,
        force_subagent_bootstrap: bool = False,
    ) -> str:
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0 when provided.")
        if max_output_chars is not None and max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than 0 when provided.")
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when provided.")
        if subagent_prefetch_calls < 0:
            raise ValueError("subagent_prefetch_calls must be >= 0.")

        effective_max_llm_calls = max_llm_calls
        if effective_max_llm_calls is None and max_subagent_calls is not None:
            effective_max_llm_calls = max_subagent_calls
        if effective_max_llm_calls is None and max_depth is not None:
            effective_max_llm_calls = max_depth
        if effective_max_llm_calls is not None and effective_max_llm_calls <= 0:
            raise ValueError("max_llm_calls must be greater than 0 when provided.")

        if bool(subagent_backend) != bool(subagent_model):
            raise ValueError("Both subagent_backend and subagent_model must be set together.")

        try:
            root_lm = self._build_lm(
                backend=self.backend,
                model=model,
                explicit_api_key=self.api_key,
                lm_kwargs=root_lm_kwargs,
            )
        except Exception as error:
            self.last_metrics = {"error": str(error)}
            return f"Error querying DSPy: {error}"

        try:
            root_runtime_lm = (
                _TracingLMProxy(root_lm, f"root:{getattr(root_lm, 'model', 'unknown')}")
                if trace_lm_calls
                else root_lm
            )

            if not use_subagents:
                payload = self._compose_single_pass_prompt(prompt, context)
                messages = [{"role": "user", "content": payload}]
                merged_system_prompt = self._compose_system_prompt(system_prompt, custom_prompt)
                if merged_system_prompt:
                    messages.insert(0, {"role": "system", "content": merged_system_prompt})

                start_hist = len(getattr(root_lm, "history", []) or [])
                start_time = time.perf_counter()
                completion_output = root_runtime_lm(messages=messages)
                end_time = time.perf_counter()

                response_text = self._extract_lm_text(completion_output)
                history_entries = list((getattr(root_lm, "history", []) or [])[start_hist:])
                model_usage = self._history_to_model_usage(history_entries)
                if not model_usage:
                    model_usage = {
                        getattr(root_lm, "model", model or "unknown"): {
                            "total_calls": 1,
                            "total_input_tokens": 0,
                            "total_output_tokens": 0,
                        }
                    }

                self.last_metrics = self._build_direct_metrics(
                    model_usage=model_usage,
                    execution_time=end_time - start_time,
                )
                return response_text

            sub_lm = None
            if use_subagents:
                # Always bind llm_query/llm_query_batched to an explicit sub_lm.
                # If no dedicated subagent config is provided, mirror root backend/model
                # with a separate LM instance so routing is deterministic.
                effective_sub_backend = subagent_backend or self.backend
                effective_sub_model = subagent_model or model or getattr(root_lm, "model", None)
                effective_sub_lm_kwargs = subagent_lm_kwargs
                effective_sub_api_key: Optional[str] = None

                if subagent_backend is None and subagent_model is None:
                    effective_sub_api_key = self.api_key
                    if effective_sub_lm_kwargs is None:
                        effective_sub_lm_kwargs = root_lm_kwargs

                sub_lm = self._build_lm(
                    backend=effective_sub_backend,
                    model=effective_sub_model,
                    explicit_api_key=effective_sub_api_key,
                    lm_kwargs=effective_sub_lm_kwargs,
                )

            sub_runtime_lm = (
                _TracingLMProxy(sub_lm, f"sub:{getattr(sub_lm, 'model', 'unknown')}")
                if trace_lm_calls and sub_lm is not None
                else sub_lm
            )
            configured_subagent_model = (
                getattr(sub_lm, "model", subagent_model) if sub_lm else None
            )
            sub_start_hist = len(getattr(sub_lm, "history", []) or []) if sub_lm else 0

            prefetch_blocks: list[str] = []
            if sub_runtime_lm and force_subagent_bootstrap:
                bootstrap_context = context[:12000]
                bootstrap_messages = [
                    {
                        "role": "user",
                        "content": (
                            "Subagent bootstrap task. Extract 3-5 highest-signal evidence bullets "
                            "for this question. Keep it brief and include citations if available.\n\n"
                            f"Question:\n{prompt}\n\nContext snippet:\n{bootstrap_context}"
                        ),
                    }
                ]
                bootstrap_output = sub_runtime_lm(messages=bootstrap_messages)
                bootstrap_text = self._extract_lm_text(bootstrap_output)
                if bootstrap_text:
                    prefetch_blocks.append(
                        "Bootstrap evidence memo:\n" + bootstrap_text.strip()
                    )
            if sub_runtime_lm and subagent_prefetch_calls > 0:
                prefetch_messages = [
                    {
                        "role": "user",
                        "content": (
                            "Research pre-analysis task. Extract the most relevant evidence for the "
                            "question from the context. Return concise bullets with short quotes and "
                            "citations if available.\n\n"
                            f"Question:\n{prompt}\n\nContext:\n{context}"
                        ),
                    }
                ]
                for _ in range(subagent_prefetch_calls):
                    prefetch_output = sub_runtime_lm(messages=prefetch_messages)
                    prefetch_text = self._extract_lm_text(prefetch_output)
                    if prefetch_text:
                        prefetch_blocks.append(prefetch_text.strip())

            guidance = self._compose_rlm_guidance(
                system_prompt=system_prompt,
                custom_prompt=custom_prompt,
                max_depth=max_depth,
                subagent_model=configured_subagent_model,
            )
            if prefetch_blocks:
                guidance = (
                    guidance
                    + "\n\nSubagent pre-analysis:\n"
                    + "\n\n".join(prefetch_blocks)
                )
            raw_signature = (rlm_signature or "context, query -> answer").strip()
            signature_text = self._normalize_rlm_signature(raw_signature)
            signature_obj = self._build_signature(signature_text, guidance)
            call_inputs = self._build_rlm_call_inputs(
                signature_obj=signature_obj,
                context=context,
                query=prompt,
                guidance=guidance,
            )

            root_start_hist = len(getattr(root_lm, "history", []) or [])
            start_time = time.perf_counter()
            adapter = None
            if disable_json_adapter_fallback and hasattr(dspy, "ChatAdapter"):
                adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
            context_kwargs = {"lm": root_runtime_lm}
            if adapter is not None:
                context_kwargs["adapter"] = adapter
            with dspy.context(**context_kwargs):
                agent = dspy.RLM(
                    signature=signature_obj,
                    max_iterations=max_iterations or 20,
                    max_llm_calls=effective_max_llm_calls or 50,
                    max_output_chars=max_output_chars or 10000,
                    verbose=self.verbose,
                    tools=rlm_tools,
                    sub_lm=sub_runtime_lm,
                    interpreter=rlm_interpreter,
                )
                result = agent(**call_inputs)
            end_time = time.perf_counter()

            response_text = self._extract_prediction_text(result)
            root_entries = list((getattr(root_lm, "history", []) or [])[root_start_hist:])
            sub_entries = (
                list((getattr(sub_lm, "history", []) or [])[sub_start_hist:])
                if sub_lm
                else []
            )

            model_usage = self._merge_model_usage(
                self._history_to_model_usage(root_entries),
                self._history_to_model_usage(sub_entries),
            )
            configured_root_model = getattr(root_lm, "model", model)
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

            total_input, total_output = self._totals_from_model_usage(model_usage)
            trajectory = getattr(result, "trajectory", None)
            iterations = len(trajectory) if trajectory is not None else 0
            subagent_calls = len(sub_entries)

            base_metrics = {
                "mode": "dspy_rlm",
                "execution_time": end_time - start_time,
                "iterations": iterations,
                "subagent_calls": subagent_calls,
                "configured_root_model": configured_root_model,
                "configured_subagent_model": configured_subagent_model,
                "max_iterations": max_iterations or 20,
                "max_llm_calls": effective_max_llm_calls or 50,
                "max_output_chars": max_output_chars or 10000,
                "max_depth": max_depth,
                "rlm_signature": signature_text,
                "model_usage": model_usage,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
            }
            if require_subagent_call and sub_lm is not None and subagent_calls == 0:
                base_metrics["error"] = "required_subagent_call_missing"
                self.last_metrics = base_metrics
                return (
                    "Error querying DSPy: strict subagent-call mode is enabled but no "
                    "llm_query/llm_query_batched calls were made."
                )

            self.last_metrics = base_metrics
            return response_text

        except Exception as error:
            self.last_metrics = {"error": str(error)}
            return f"Error querying DSPy: {error}"


# Backward-compatible alias used across the current project.
RLMHandler = DSPyRLMHandler

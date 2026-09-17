from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import uuid
from typing import Any


class WorkerStartError(RuntimeError):
    pass


class HardRequestTimeout(TimeoutError):
    pass


def _worker_loop(
    model: dict[str, Any], device: dict[str, Any], mode: str,
    generation: dict[str, Any], warmup: list[dict[str, str]],
    requests: Any, responses: Any,
) -> None:
    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(device["id"]) if device["type"] == "Vulkan" else ""
    llm = None
    started = time.perf_counter()
    try:
        import utils
        from .runner import generate_case

        llm = utils.load_llm(
            model,
            {
                "model_path": model["path"],
                "n_gpu_layers": int(generation.get("gpu_layers", -1)) if device["type"] == "Vulkan" else 0,
                "n_ctx": int(generation["context_size"]),
                "verbose": False,
            },
            warmup_inputs=warmup,
            custom_jinja=True,
            reasoning=mode,
            log=True,
        )
        responses.put({"kind": "ready", "load_seconds": time.perf_counter() - started})
        while True:
            request = requests.get()
            if request is None:
                return
            request_id = request["request_id"]
            try:
                llm.reset()
                generated, turns, transcript, generated_raw_turns = generate_case(
                    llm, request["case"], mode, generation, request["seed"]
                )
                responses.put({
                    "kind": "result", "request_id": request_id,
                    "generation": generated.as_dict(), "turns": turns,
                    "transcript": transcript, "generated_raw_turns": generated_raw_turns,
                })
            except Exception as exc:
                responses.put({
                    "kind": "error", "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    except Exception as exc:
        responses.put({"kind": "start_error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if llm is not None:
            llm.close()


class IsolatedModelSession:
    """Persistent model subprocess with a parent-enforced timeout per case."""

    def __init__(
        self, model: dict[str, Any], device: dict[str, Any], mode: str,
        generation: dict[str, Any], warmup: list[dict[str, str]],
    ) -> None:
        self.model = model
        self.device = device
        self.mode = mode
        self.generation = generation
        self.warmup = warmup
        self._context = mp.get_context("spawn")
        self._requests: Any = None
        self._responses: Any = None
        self._process: Any = None
        self.load_seconds = -1.0

    def start(self, timeout_seconds: float = 600) -> None:
        self.close()
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_loop,
            args=(self.model, self.device, self.mode, self.generation, self.warmup, self._requests, self._responses),
            daemon=True,
        )
        self._process.start()
        try:
            message = self._responses.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self.close(force=True)
            raise WorkerStartError(f"Model worker did not start within {timeout_seconds:g}s") from exc
        if message["kind"] != "ready":
            self.close(force=True)
            raise WorkerStartError(message.get("error", "Model worker failed during startup"))
        self.load_seconds = float(message["load_seconds"])

    def generate(self, case: dict[str, Any], seed: int, timeout_seconds: float) -> dict[str, Any]:
        if self._process is None or not self._process.is_alive():
            raise WorkerStartError("Model worker is not running")
        request_id = uuid.uuid4().hex
        self._requests.put({"request_id": request_id, "case": case, "seed": seed})
        try:
            message = self._responses.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self.close(force=True)
            raise HardRequestTimeout(f"Hard request timeout exceeded after {timeout_seconds:g}s") from exc
        if message.get("request_id") != request_id:
            self.close(force=True)
            raise RuntimeError("Model worker protocol desynchronized")
        if message["kind"] == "error":
            raise RuntimeError(message["error"])
        return message

    def close(self, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive() and not force:
            try:
                self._requests.put(None)
                process.join(timeout=5)
            except (OSError, ValueError):
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2)
        for channel in (self._requests, self._responses):
            if channel is not None:
                if force:
                    channel.cancel_join_thread()
                channel.close()
                if not force:
                    channel.join_thread()
        self._process = self._requests = self._responses = None

    def __enter__(self) -> "IsolatedModelSession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

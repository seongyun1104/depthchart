from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from speculative.adapter import SpecConfig


@dataclass(frozen=True)
class ServerHandle:
    pid: int
    base_url: str
    log_path: Path


class VLLMServer:
    def __init__(
        self,
        hf_id: str,
        port: int,
        lmcache_config: Path | None,
        spec: SpecConfig,
        quantization: str = "fp8",
        max_num_batched_tokens: int = 8192,
        gpu_memory_utilization: float = 0.85,
        tp_size: int = 1,
        max_model_len: int | None = None,
        reasoning_parser: str | None = None,
        tool_call_parser: str | None = None,
        kv_cache_dtype: str | None = None,
        log_dir: Path = Path("runs"),
    ):
        self.hf_id = hf_id
        self.port = port
        self.lmcache_config = lmcache_config
        self.spec = spec
        self.quantization = quantization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.reasoning_parser = reasoning_parser
        self.tool_call_parser = tool_call_parser
        self.kv_cache_dtype = kv_cache_dtype
        self.log_dir = log_dir
        self._proc: subprocess.Popen | None = None
        self._log_fp = None

    def _build_cmd(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.hf_id,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tp_size),
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
        ]
        if self.quantization and self.quantization != "none":
            cmd += ["--quantization", self.quantization]
        if self.kv_cache_dtype:
            cmd += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.max_model_len is not None:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.reasoning_parser:
            cmd += ["--reasoning-parser", self.reasoning_parser]
        if self.tool_call_parser:
            cmd += ["--enable-auto-tool-choice",
                    "--tool-call-parser", self.tool_call_parser]

        spec_cfg = self.spec.to_vllm_speculative_config()
        if spec_cfg is not None:
            cmd += ["--speculative-config", json.dumps(spec_cfg)]

        if self.lmcache_config is not None:
            kv_cfg = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
            cmd += ["--kv-transfer-config", json.dumps(kv_cfg)]

        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.lmcache_config is not None:
            env["LMCACHE_CONFIG_FILE"] = str(self.lmcache_config.resolve())
        return env

    async def start(self, ready_timeout_s: float = 600) -> ServerHandle:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"vllm_{self.port}_{os.getpid()}.log"
        self._log_fp = log_path.open("w")
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            env=self._build_env(),
            start_new_session=True,
        )
        await self._wait_ready(ready_timeout_s)
        return ServerHandle(
            pid=self._proc.pid,
            base_url=f"http://127.0.0.1:{self.port}",
            log_path=log_path,
        )

    async def _wait_ready(self, timeout_s: float) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        async with httpx.AsyncClient(timeout=2.0) as client:
            for _ in range(int(timeout_s / 2)):
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited early (rc={self._proc.returncode}); "
                        f"see log at {self._log_fp.name if self._log_fp else '?'}"
                    )
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError(f"vLLM server on :{self.port} not ready in {timeout_s}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        pid = self._proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            self._proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            if self._log_fp:
                self._log_fp.close()
            self._proc = None

    async def __aenter__(self) -> ServerHandle:
        return await self.start()

    async def __aexit__(self, *_exc) -> None:
        self.stop()

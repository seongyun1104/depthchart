from __future__ import annotations

import asyncio
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


class SGLangServer:
    """Owns the lifecycle of one SGLang server process configured for a
    single (model, spec, lmcache) combination. One server per sweep cell.
    """

    def __init__(
        self,
        hf_id: str,
        port: int,
        lmcache_config: Path,
        spec: SpecConfig,
        quantization: str = "fp8",
        chunked_prefill_size: int = 8192,
        mem_fraction_static: float = 0.81,
        tp_size: int = 1,
        max_model_len: int | None = None,
        reasoning_parser: str | None = None,
        tool_call_parser: str | None = None,
        log_dir: Path = Path("runs"),
    ):
        self.hf_id = hf_id
        self.port = port
        self.lmcache_config = lmcache_config
        self.spec = spec
        self.quantization = quantization
        self.chunked_prefill_size = chunked_prefill_size
        self.mem_fraction_static = mem_fraction_static
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.reasoning_parser = reasoning_parser
        self.tool_call_parser = tool_call_parser
        self.log_dir = log_dir
        self._proc: subprocess.Popen | None = None
        self._log_fp = None

    def _build_cmd(self) -> list[str]:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.hf_id,
            "--port", str(self.port),
            "--enable-lmcache",
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--tp-size", str(self.tp_size),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
        ]
        if self.quantization and self.quantization != "none":
            cmd += ["--quantization", self.quantization]
        if self.max_model_len is not None:
            cmd += ["--context-length", str(self.max_model_len)]
        if self.reasoning_parser:
            cmd += ["--reasoning-parser", self.reasoning_parser]
        if self.tool_call_parser:
            cmd += ["--tool-call-parser", self.tool_call_parser]
        cmd += self.spec.to_sglang_args()
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LMCACHE_CONFIG_FILE"] = str(self.lmcache_config.resolve())
        return env

    async def start(self, ready_timeout_s: float = 600) -> ServerHandle:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"sglang_{self.port}_{os.getpid()}.log"
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
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError(f"SGLang server on :{self.port} not ready in {timeout_s}s")

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

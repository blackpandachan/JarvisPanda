"""System monitor tool — CPU, memory, disk, and optional GPU stats.

Uses psutil (always available) and pynvml (optional, NVIDIA only).
Safe to call in any environment; degrades gracefully if sensors are missing.
"""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _gpu_stats() -> list[str]:
    """Return per-GPU lines via pynvml, or empty list if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        lines = []
        for i in range(count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            used_gb  = mem.used  / 1024**3
            total_gb = mem.total / 1024**3
            lines.append(
                f"GPU {i} ({name}): {util.gpu}% util · "
                f"{used_gb:.1f}/{total_gb:.1f} GB VRAM · {temp_c}°C"
            )
        pynvml.nvmlShutdown()
        return lines
    except Exception:
        return []


@ToolRegistry.register("system_monitor")
class SystemMonitorTool(BaseTool):
    """Report CPU, RAM, disk, and GPU usage of the host machine."""

    tool_id = "system_monitor"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_monitor",
            description=(
                "Get current system resource usage: CPU%, memory, disk, and GPU "
                "(if available). Useful for checking whether the host has capacity "
                "for new workloads or diagnosing performance issues."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "Level of detail (default: summary).",
                        "default": "summary",
                    },
                },
                "required": [],
            },
            category="utility",
        )

    def execute(self, **params: Any) -> ToolResult:
        detail: str = params.get("detail", "summary")

        try:
            import psutil
        except ImportError:
            return ToolResult(
                tool_name="system_monitor",
                content="psutil is not installed in this environment.",
                success=False,
            )

        # CPU
        cpu_pct = psutil.cpu_percent(interval=0.5)
        cpu_count_l = psutil.cpu_count(logical=True)
        cpu_count_p = psutil.cpu_count(logical=False) or cpu_count_l

        # Memory
        vm      = psutil.virtual_memory()
        ram_used  = vm.used  / 1024**3
        ram_total = vm.total / 1024**3
        ram_pct   = vm.percent

        # Disk (root)
        try:
            disk    = psutil.disk_usage("/")
            disk_used  = disk.used  / 1024**3
            disk_total = disk.total / 1024**3
            disk_pct   = disk.percent
            disk_str = f"{disk_used:.0f}/{disk_total:.0f} GB ({disk_pct:.0f}%)"
        except Exception:
            disk_str = "unavailable"

        lines = [
            "**System Status**",
            f"CPU   : {cpu_pct:.0f}%  ({cpu_count_p}P/{cpu_count_l}L cores)",
            f"RAM   : {ram_used:.1f}/{ram_total:.1f} GB ({ram_pct:.0f}%)",
            f"Disk  : {disk_str}",
        ]

        # GPU
        gpu_lines = _gpu_stats()
        if gpu_lines:
            lines.extend(gpu_lines)
        else:
            lines.append("GPU   : not detected (pynvml unavailable or no NVIDIA GPU)")

        if detail == "full":
            # Per-CPU usage
            per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)
            lines.append("\nPer-core CPU: " + "  ".join(f"C{i}:{p:.0f}%" for i, p in enumerate(per_cpu)))

            # Network I/O
            try:
                net = psutil.net_io_counters()
                sent_gb = net.bytes_sent / 1024**3
                recv_gb = net.bytes_recv / 1024**3
                lines.append(f"Net   : ↑ {sent_gb:.2f} GB sent · ↓ {recv_gb:.2f} GB recv (since boot)")
            except Exception:
                pass

            # Swap
            try:
                sw = psutil.swap_memory()
                if sw.total > 0:
                    lines.append(
                        f"Swap  : {sw.used/1024**3:.1f}/{sw.total/1024**3:.1f} GB ({sw.percent:.0f}%)"
                    )
            except Exception:
                pass

        return ToolResult(
            tool_name="system_monitor",
            content="\n".join(lines),
            success=True,
            metadata={
                "cpu_pct": cpu_pct,
                "ram_pct": ram_pct,
                "gpu_count": len(gpu_lines),
            },
        )


__all__ = ["SystemMonitorTool"]

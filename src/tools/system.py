"""System tools: live host-machine resource stats and process listing. Use
these for questions about THIS machine's hardware/performance, never for
general knowledge (that's research.py's web_search).

get_system_stats' GPU metric is the one platform-sensitive part of this
module (everything else goes through psutil, which is already
cross-platform): it dispatches by OS/hardware rather than assuming macOS,
since this agent also runs on Linux hosts and Jetson Orin boards (see
scripts/setup_jetson.sh) where 'system_profiler' doesn't exist."""

import json
import os
import platform
import shutil
import subprocess
import time

import psutil

from src.tools.registry import registry

_SUBPROCESS_TIMEOUT_SECONDS = 5
_JETSON_MARKER = "/etc/nv_tegra_release"  # same marker scripts/setup_jetson.sh checks for
_JETSON_GPU_LOAD_PATH = "/sys/devices/gpu.0/load"  # Tegra integrated GPU busy%, 0-1000 scale


def _run_command(args: list[str]) -> str:
    return subprocess.check_output(
        args, stderr=subprocess.DEVNULL, timeout=_SUBPROCESS_TIMEOUT_SECONDS
    ).decode("utf-8", errors="replace")


def _macos_gpu_info():
    out = _run_command(["system_profiler", "SPDisplaysDataType"])
    gpu_lines = [
        line.strip() for line in out.splitlines()
        if any(x in line for x in ["Chipset Model", "Type", "Total Number of Cores", "Vendor", "Metal Support", "VRAM"])
    ]
    return gpu_lines if gpu_lines else "Apple Silicon GPU (native)"


def _jetson_gpu_info():
    # Jetson's integrated Tegra GPU doesn't show up in nvidia-smi the way a
    # discrete NVIDIA GPU does — read the sysfs load file tegrastats itself
    # reads from, which needs no elevated permissions or extra tooling.
    with open(_JETSON_GPU_LOAD_PATH) as f:
        raw = f.read().strip()
    return f"{int(raw) / 10:.1f}%"  # file reports busy% * 10


def _nvidia_smi_gpu_info():
    out = _run_command([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    lines = []
    for row in out.strip().splitlines():
        name, util, mem_used, mem_total = [p.strip() for p in row.split(",")]
        lines.append(f"{name}: {util}% util, {mem_used}/{mem_total} MB")
    return lines if lines else "nvidia-smi returned no GPUs."


def _get_gpu_info() -> str | list:
    """Best-effort GPU info across the platforms this agent actually runs
    on. Tries the mechanism matching the detected platform first, falls back
    to whatever else is available, and only reports a real failure if
    nothing applicable is found — never a raw 'command not found' leaked to
    the model."""
    system = platform.system()
    attempts = []

    if system == "Darwin":
        attempts.append(("system_profiler", _macos_gpu_info))
    elif system == "Linux":
        if os.path.exists(_JETSON_MARKER):
            attempts.append(("Jetson sysfs", _jetson_gpu_info))
        if shutil.which("nvidia-smi"):
            attempts.append(("nvidia-smi", _nvidia_smi_gpu_info))

    if not attempts:
        return f"GPU stats are not available on this platform ({system})."

    errors = []
    for label, fn in attempts:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{label}: {e}")

    return f"GPU stats unavailable — tried {', '.join(errors)}."


@registry.register(
    name="get_system_stats",
    description="Get this machine's own live hardware/performance numbers right now — CPU load, RAM, GPU, disk space, battery, network throughput, uptime, or OS version. Only for THIS host machine's real-time status, never for general knowledge about hardware brands/products (use web_search for that).",
    parameters={"metric": {"type": "string", "description": "Which metric to check: 'cpu', 'memory', 'gpu', 'disk', 'battery', 'network', 'uptime', 'os', or 'all' for everything."}},
    required=["metric"],
    group="system",
)
def get_system_stats(metric: str) -> str:
    try:
        metric_lower = metric.lower()
        stats = {}

        if metric_lower in ["cpu", "all"]:
            stats["cpu_percentage"] = f"{psutil.cpu_percent(interval=0.1)}%"
        if metric_lower in ["memory", "all"]:
            mem = psutil.virtual_memory()
            stats["memory_used_gb"] = f"{mem.used / (1024**3):.2f} GB"
            stats["memory_total_gb"] = f"{mem.total / (1024**3):.2f} GB"
            stats["memory_percentage"] = f"{mem.percent}%"
        if metric_lower in ["gpu", "all"]:
            stats["gpu_info"] = _get_gpu_info()
        if metric_lower in ["disk", "all"]:
            disk = psutil.disk_usage('/')
            stats["disk_used_gb"] = f"{disk.used / (1024**3):.2f} GB"
            stats["disk_total_gb"] = f"{disk.total / (1024**3):.2f} GB"
            stats["disk_percentage"] = f"{disk.percent}%"
        if metric_lower in ["battery", "all"]:
            if hasattr(psutil, "sensors_battery"):
                bat = psutil.sensors_battery()
                if bat:
                    stats["battery_percentage"] = f"{bat.percent}%"
                    stats["battery_power_plugged"] = bat.power_plugged
                else:
                    stats["battery_info"] = "No battery detected (e.g. desktop Mac)"
        if metric_lower in ["network", "all"]:
            net = psutil.net_io_counters()
            stats["network_bytes_sent"] = f"{net.bytes_sent / (1024**2):.2f} MB"
            stats["network_bytes_recv"] = f"{net.bytes_recv / (1024**2):.2f} MB"
        if metric_lower in ["uptime", "all"]:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = uptime_seconds / 3600
            stats["uptime_hours"] = f"{uptime_hours:.2f} hours"
        if metric_lower in ["os", "system", "all"]:
            stats["os_platform"] = platform.system()
            stats["os_release"] = platform.release()
            stats["os_version"] = platform.version()
            stats["os_architecture"] = platform.machine()

        return json.dumps(stats, indent=2) if stats else "Error: Invalid metric requested. Choose 'cpu', 'memory', 'gpu', 'disk', 'battery', 'network', 'uptime', 'os' or 'all'."
    except Exception as e:
        return f"Failed to gather system metrics: {str(e)}"


_CPU_SAMPLE_INTERVAL_SECONDS = 0.3  # psutil needs two samples over time to compute real per-process CPU%


@registry.register(
    name="top_processes",
    description="List the processes currently using the most CPU or RAM on this machine right now. Use this for questions like 'what's using my RAM' or 'what's eating my CPU'.",
    parameters={
        "sort_by": {"type": "string", "description": "Either 'cpu' or 'memory' — which metric to rank processes by. Defaults to 'cpu'."}
    },
    required=[],
    group="system",
)
def top_processes(sort_by: str = "cpu") -> str:
    try:
        sort_key = "memory_percent" if sort_by.strip().lower() == "memory" else "cpu_percent"

        procs = []
        for p in psutil.process_iter(["pid", "name", "memory_percent"]):
            try:
                p.cpu_percent(None)  # primes the internal sample; first read is always 0.0
                procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if sort_key == "cpu_percent":
            # A real per-process CPU% needs a second sample separated by a
            # wall-clock interval — without this every process reads back
            # 0.0%/None and "top by CPU" is effectively unsorted.
            time.sleep(_CPU_SAMPLE_INTERVAL_SECONDS)

        rows = []
        for p in procs:
            try:
                cpu = p.cpu_percent(None) if sort_key == "cpu_percent" else 0.0
                rows.append({
                    "pid": p.pid,
                    "name": p.info.get("name") or "?",
                    "cpu_percent": cpu,
                    "memory_percent": p.info.get("memory_percent") or 0.0,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        rows.sort(key=lambda r: r[sort_key], reverse=True)
        top = rows[:8]

        lines = [f"PID {r['pid']:>6} | CPU {r['cpu_percent']:>5.1f}% | MEM {r['memory_percent']:>5.1f}% | {r['name']}" for r in top]
        return "\n".join(lines) if lines else "No process data available."
    except Exception as e:
        return f"Failed to list processes: {str(e)}"

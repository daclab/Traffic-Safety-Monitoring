"""Linux procfs/sysfs helpers: core load, clocks, temperature, thread CPU time.

Every function degrades to None/empty off Linux so the runner can be smoke
tested on a PC.
"""

import os
import threading
import time
from pathlib import Path


CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
IS_LINUX = Path("/proc/self/task").is_dir()


def parse_cpus(text):
    """'0,1,2' or '0-2' -> {0, 1, 2}; empty/None -> None."""
    if not text:
        return None
    cpus = set()
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            cpus.update(range(int(low), int(high) + 1))
        else:
            cpus.add(int(part))
    return cpus


def pin_current_thread(cpus, nice=0):
    """Pin the calling thread (Linux applies sched_setaffinity(0) per thread)."""
    if cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU pinning requested but unsupported on this OS")
        os.sched_setaffinity(0, cpus)
    if nice:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)


def process_memory_mb():
    """Current (VmRSS) and peak (VmHWM) resident memory of this process, in MiB."""
    if not IS_LINUX:
        return {"rss_mb": None, "peak_rss_mb": None}
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            values[key] = int(rest.split()[0]) / 1024.0
    return {"rss_mb": values.get("VmRSS"), "peak_rss_mb": values.get("VmHWM")}

def proc_stat_cores():
    """Per-core (busy, total) jiffies from /proc/stat."""
    if not IS_LINUX:
        return {}
    cores = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            name, *values = line.split()
            values = [int(v) for v in values]
            idle = values[3] + values[4]  # idle + iowait
            cores[name] = (sum(values) - idle, sum(values))
    return cores


def core_busy_percent(before, after):
    result = {}
    for name, (busy_after, total_after) in after.items():
        busy_before, total_before = before.get(name, (0, 0))
        total = total_after - total_before
        result[name] = 100.0 * (busy_after - busy_before) / total if total else None
    return result


def thread_cpu_seconds():
    """{tid: (comm, cpu_seconds, cpus_allowed_list)} for every thread."""
    if not IS_LINUX:
        return {}
    threads = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            stat = (task / "stat").read_text()
            status = (task / "status").read_text()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        cpu = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS  # utime + stime
        allowed = next(
            (l.split(":", 1)[1].strip() for l in status.splitlines()
             if l.startswith("Cpus_allowed_list")), None)
        threads[int(task.name)] = (comm, cpu, allowed)
    return threads


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def cpu_frequencies_mhz():
    base = Path("/sys/devices/system/cpu")
    freqs = {}
    for cpu in sorted(base.glob("cpu[0-9]*")):
        value = _read_int(cpu / "cpufreq" / "scaling_cur_freq")
        if value is not None:
            freqs[cpu.name] = value / 1000.0
    return freqs


def temperatures_c():
    temps = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        value = _read_int(zone / "temp")
        if value is None:
            continue
        try:
            kind = (zone / "type").read_text().strip()
        except OSError:
            kind = zone.name
        temps[f"{zone.name}:{kind}"] = value / 1000.0
    return temps


def cpufreq_policy():
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    def read(name):
        try:
            return (base / name).read_text().strip()
        except OSError:
            return None
    return {
        "governor": read("scaling_governor"),
        "min_mhz": (_read_int(base / "scaling_min_freq") or 0) / 1000.0 or None,
        "max_mhz": (_read_int(base / "scaling_max_freq") or 0) / 1000.0 or None,
        "hw_max_mhz": (_read_int(base / "cpuinfo_max_freq") or 0) / 1000.0 or None,
    }


class Sampler(threading.Thread):
    """Samples clocks and temperatures at a low rate on the auxiliary core."""

    def __init__(self, interval, cpus=None):
        super().__init__(name="monitor", daemon=True)
        self.interval = interval
        self.cpus = cpus
        self.samples = []
        self._stop_event = threading.Event()

    def run(self):
        pin_current_thread(self.cpus)
        while not self._stop_event.wait(self.interval):
            self.samples.append(
                (time.monotonic(), cpu_frequencies_mhz(), temperatures_c())
            )

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    def summary(self):
        def stats(index):
            series = {}
            for sample in self.samples:
                for key, value in sample[index].items():
                    series.setdefault(key, []).append(value)
            return {
                key: {"min": min(v), "mean": sum(v) / len(v), "max": max(v)}
                for key, v in series.items()
            }
        return {
            "samples": len(self.samples),
            "cpu_mhz": stats(1),
            "temperature_c": stats(2),
        }

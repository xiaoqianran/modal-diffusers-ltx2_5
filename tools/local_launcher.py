from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
ROUTER_URL = "http://127.0.0.1:48125/api/health"
APP_URL = "http://127.0.0.1:5187"

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def load_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key] = value

    env.setdefault("LTX25_LOCAL_KEEP_GPU_WARM", "1")
    env.setdefault("LTX25_MODAL_GPU_IDLE_SECONDS", "600")
    env.setdefault("LTX25_MEDIA_BACKEND", "volume")
    env["VITE_API_TARGET"] = "http://127.0.0.1:48125"
    return env


def make_kill_on_close_job() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def assign(job: int, process: subprocess.Popen) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ok = kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def wait_http(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def main() -> int:
    if os.name != "nt":
        print("[LTX-2.5] This launcher currently targets Windows.")
        return 2

    env = load_env()
    job = make_kill_on_close_job()
    children: list[subprocess.Popen] = []

    try:
        print("[LTX-2.5] Starting local Router + Frontend under one Windows Job Object.")
        print("[LTX-2.5] Close THIS terminal and Windows will terminate both services.")
        print("[Router  ] http://127.0.0.1:48125")
        print("[Frontend] http://127.0.0.1:5187")
        print()

        router = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "ltx25.api:app", "--host", "127.0.0.1", "--port", "48125"],
            cwd=ROOT,
            env=env,
        )
        assign(job, router)
        children.append(router)

        if not wait_http(ROUTER_URL, 20):
            print("[LTX-2.5] Router failed to become healthy.")
            return 1

        frontend = subprocess.Popen(
            ["cmd.exe", "/d", "/s", "/c", "npm run dev"],
            cwd=FRONTEND,
            env=env,
        )
        assign(job, frontend)
        children.append(frontend)

        if not wait_http(APP_URL, 15):
            print("[LTX-2.5] Frontend failed to become healthy.")
            return 1

        webbrowser.open(APP_URL)
        print("[LTX-2.5] Running. Close this terminal to free ports 48125 and 5187.")
        print("[LTX-2.5] Ctrl+C also shuts everything down.")

        while True:
            for child, name in ((router, "Router"), (frontend, "Frontend")):
                code = child.poll()
                if code is not None:
                    print(f"[LTX-2.5] {name} exited unexpectedly with code {code}.")
                    return code or 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[LTX-2.5] Stopping...")
        return 0
    finally:
        # Closing the Job Object handle enforces KILL_ON_JOB_CLOSE for every
        # assigned process and descendants that remain in the job.
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


if __name__ == "__main__":
    raise SystemExit(main())

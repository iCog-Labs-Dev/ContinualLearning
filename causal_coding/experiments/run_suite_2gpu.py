#!/usr/bin/env python3
"""Concurrent Multi-GPU Runner for Meta Causal Coding Suite.

Schedules the 5 required experiment runs across available GPUs (e.g. 2x T4 on Kaggle),
streaming real-time tagged logs to stdout while preserving full per-experiment logs to disk.
"""

import argparse
from datetime import datetime
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Default list of the 5 required runs
EXPERIMENT_SUITE = [
    {
        "name": "meta_causal_coding_seed0",
        "description": "Baseline Full Meta-CC (Seed 0)",
        "args": ["--seed", "0", "--name", "meta_causal_coding_seed0"],
    },
    {
        "name": "meta_causal_coding_seed1",
        "description": "Full Meta-CC (Seed 1)",
        "args": ["--seed", "1", "--name", "meta_causal_coding_seed1"],
    },
    {
        "name": "meta_causal_coding_seed2",
        "description": "Full Meta-CC (Seed 2)",
        "args": ["--seed", "2", "--name", "meta_causal_coding_seed2"],
    },
    {
        "name": "meta_fixed_ctrl_seed0",
        "description": "Ablation: Fixed Controller (Seed 0)",
        "args": ["--seed", "0", "--fixed-controller", "--name", "meta_fixed_ctrl_seed0"],
    },
    {
        "name": "meta_no_exp_seed0",
        "description": "Ablation: Exploration-Disabled (Seed 0)",
        "args": ["--seed", "0", "--disable-exploration", "--name", "meta_no_exp_seed0"],
    },
]

print_lock = threading.Lock()


def log_line(msg: str):
    with print_lock:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


def run_experiment(exp_cfg: dict, gpu_id: int, script_path: str, dry_run: bool = False) -> dict:
    name = exp_cfg["name"]
    desc = exp_cfg["description"]
    args = exp_cfg["args"]

    results_dir = os.path.join("results", name)
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "run.log")

    start_time = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_line(f"\n{'='*75}\n[START] {name} on GPU {gpu_id}\nDescription: {desc}\nTime: {start_str}\nLog file: {log_file_path}\n{'='*75}\n")

    cmd = [sys.executable, "-u", script_path] + args

    if dry_run:
        log_line(f"[DRY RUN] Would execute on GPU {gpu_id}: {' '.join(cmd)}")
        time.sleep(1.0)
        return {
            "name": name,
            "gpu": gpu_id,
            "status": "DRY_RUN",
            "duration": "0s",
            "log": log_file_path,
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    status = "SUCCESS"
    returncode = 0

    try:
        with open(log_file_path, "w", encoding="utf-8") as f_log:
            header = f"=== Run: {name} (GPU {gpu_id}) Started at {start_str} ===\nCommand: {' '.join(cmd)}\n\n"
            f_log.write(header)
            f_log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            for raw_line in iter(process.stdout.readline, ""):
                f_log.write(raw_line)
                f_log.flush()

                timestamp = datetime.now().strftime("%H:%M:%S")
                tag = f"[{timestamp} | GPU {gpu_id} | {name}] "
                with print_lock:
                    sys.stdout.write(f"{tag}{raw_line}")
                    sys.stdout.flush()

            process.stdout.close()
            returncode = process.wait()

            if returncode != 0:
                status = f"FAILED (exit code {returncode})"
    except Exception as exc:
        status = f"ERROR ({exc})"
        log_line(f"[ERROR] Exception running {name} on GPU {gpu_id}: {exc}")

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    duration_str = f"{mins}m {secs}s"

    log_line(f"\n{'-'*75}\n[FINISH] {name} on GPU {gpu_id} -> {status} (Duration: {duration_str})\n{'-'*75}\n")

    return {
        "name": name,
        "gpu": gpu_id,
        "status": status,
        "duration": duration_str,
        "log": log_file_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 5 Meta Causal Coding experiments in parallel across GPUs.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU indices to use (default: '0,1')",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="causal_coding/experiments/run_meta_causal_coding.py",
        help="Path to run_meta_causal_coding.py script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing actual training",
    )
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        gpu_list = [0]

    num_workers = len(gpu_list)
    print(f"\n{'='*75}")
    print(f"🚀 Meta Causal Coding Multi-GPU Runner")
    print(f"GPUs available: {gpu_list} ({num_workers} parallel workers)")
    print(f"Total experiments: {len(EXPERIMENT_SUITE)}")
    print(f"{'='*75}\n")

    for i, exp in enumerate(EXPERIMENT_SUITE, 1):
        print(f"  {i}. {exp['name']:<26} -> {exp['description']}")
    print()

    gpu_queue = queue.Queue()
    for g in gpu_list:
        gpu_queue.put(g)

    results = []
    total_start = time.time()

    def worker_task(exp_cfg):
        gpu_id = gpu_queue.get()
        try:
            return run_experiment(exp_cfg, gpu_id, args.script, dry_run=args.dry_run)
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, exp) for exp in EXPERIMENT_SUITE]
        for f in futures:
            results.append(f.result())

    total_elapsed = time.time() - total_start
    total_m, total_s = divmod(int(total_elapsed), 60)

    print(f"\n{'='*75}")
    print(f"📊 SUMMARY OF ALL RUNS (Total Time: {total_m}m {total_s}s)")
    print(f"{'='*75}")
    print(f"{'Experiment Name':<28} {'GPU':<6} {'Status':<20} {'Duration':<10} Log Path")
    print("-" * 75)
    for res in results:
        print(f"{res['name']:<28} {res['gpu']:<6} {res['status']:<20} {res['duration']:<10} {res['log']}")
    print("=" * 75)


if __name__ == "__main__":
    main()

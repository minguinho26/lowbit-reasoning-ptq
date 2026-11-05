"""실제 자식 프로세스로 큐 종료 후 엔진 잔류와 타임아웃 정리를 검사한다."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_matrix import run_queue_command
from common import enumerate_runs, load_config


class QueueProcessTest(unittest.TestCase):
    def test_active_scope(self):
        """4B를 제외한 매트릭스와 AWQ 셀 수를 확인한다."""
        cfg = load_config()
        self.assertEqual(set(cfg["models"]), {"qwen3-8b", "qwen3-1.7b"})
        runs = enumerate_runs(cfg)
        self.assertEqual(len(runs), 35)
        self.assertEqual(sum(r["fmt"] != "bf16" for r in runs), 33)
        self.assertEqual(sum(r["algo"] == "awq" for r in runs), 12)
        self.assertEqual(sum(len(b["seeds"]) for b in cfg["benchmarks"].values()) * len(runs), 210)

    @unittest.skipUnless(shutil.which("flock"), "queue_main requires Linux flock")
    def test_model_phase_order(self):
        """실제 shell 실행으로 8B 진단·속도가 1.7B 평가보다 먼저인지 확인한다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(Path(__file__).resolve().parents[1] / "queue_main.sh", root / "queue_main.sh")
            (root / "src").mkdir()
            runner = (
                "import os,sys\n"
                "with open(os.environ['PHASE_TRACE'],'a') as f: f.write(' '.join(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1]=='queue': print('queue finished. not done: 0')\n"
            )
            (root / "src/run_matrix.py").write_text(runner)
            (root / "src/analyze.py").write_text(
                "import os\nwith open(os.environ['PHASE_TRACE'],'a') as f: f.write('analysis\\n')\n"
            )
            py = root / "venv_serve/bin/python"
            py.parent.mkdir(parents=True)
            py.symlink_to(sys.executable)
            fakebin = root / "bin"
            fakebin.mkdir()
            gpu = fakebin / "nvidia-smi"
            gpu.write_text("#!/bin/sh\nexit 0\n")
            gpu.chmod(0o755)
            trace = root / "phases.txt"
            env = dict(os.environ, WS=tmp, GPUS="0,1", AUTO_TERMINATE="0", WAIT_FOR_IDLE="0",
                       QUEUE_LOCK=str(root / "queue.lock"), PHASE_TRACE=str(trace),
                       PATH=str(fakebin) + os.pathsep + os.environ["PATH"])
            result = subprocess.run(["bash", str(root / "queue_main.sh")], env=env,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            phases = trace.read_text().splitlines()
            self.assertEqual([line.split()[0] for line in phases],
                             ["queue", "diagnose", "speed", "analysis"] * 2)
            self.assertTrue(all("qwen3-8b" in line for line in phases[:3]))
            self.assertTrue(all("qwen3-1.7b" in line for line in phases[4:7]))

    def test_return_codes(self):
        """일반 성공·실패 상태를 스케줄러에 그대로 전달한다."""
        for code in (0, 7):
            with self.subTest(code=code), open(os.devnull, "w") as log:
                self.assertEqual(run_queue_command(
                    [sys.executable, "-c", f"raise SystemExit({code})"],
                    env=os.environ.copy(), stdout=log, timeout=5,
                ), code)

    def test_descendant_cleanup(self):
        """부모 timeout·실패·성공 모두 자식이 계속 실행되지 않는지 확인한다."""
        for mode in ("timeout", "failure", "success"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                pidfile = Path(tmp) / "child.pid"
                child = "import time; time.sleep(60)"
                parent = (
                    "import subprocess,sys,time; from pathlib import Path; "
                    f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                    f"Path({str(pidfile)!r}).write_text(str(p.pid)); "
                    + ("time.sleep(60)" if mode == "timeout" else f"raise SystemExit({7 if mode == 'failure' else 0})")
                )
                child_pid = None
                try:
                    with open(os.devnull, "w") as log:
                        rc = run_queue_command([sys.executable, "-c", parent],
                                               env=os.environ.copy(), stdout=log,
                                               timeout=1 if mode == "timeout" else 5)
                    self.assertEqual(rc, "timeout" if mode == "timeout" else (7 if mode == "failure" else 0))
                    child_pid = int(pidfile.read_text())
                    for _ in range(50):
                        # Linux의 고아 zombie는 실행하거나 GPU를 점유하지 않는다.
                        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)],
                                               capture_output=True, text=True).stdout.strip()
                        if not state or state.startswith("Z"):
                            break
                        time.sleep(.05)
                    self.assertTrue(not state or state.startswith("Z"), f"child still running: {state}")
                finally:
                    if child_pid is None and pidfile.exists():
                        child_pid = int(pidfile.read_text())
                    if child_pid is not None:
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()

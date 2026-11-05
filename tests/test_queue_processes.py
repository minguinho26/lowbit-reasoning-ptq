"""실제 자식 프로세스로 큐 종료 후 엔진 잔류와 타임아웃 정리를 검사한다."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_matrix import run_queue_command


class QueueProcessTest(unittest.TestCase):
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

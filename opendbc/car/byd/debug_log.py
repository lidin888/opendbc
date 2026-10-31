"""BYD 调试期专用异步文件日志。

调用方只做 queue.put_nowait（微秒级，永不阻塞），后台 daemon 线程批量写盘。
磁盘 writeback 抖动 / pty 卡死时只丢日志，不会拖慢 card 的 100Hz 控制循环。
调试完成后本文件连同所有调用点（card.py / carcontroller.py / carstate.py /
radar_interface.py）一起删除。
"""
import os
import queue
import threading
import time

_QUEUE_MAX = 20000                   # 有界：磁盘卡死时丢弃而非堆积内存
_FLUSH_INTERVAL_S = 1.0              # 攒批最长等待
_BATCH_LINES = 256                   # 攒批行数阈值
_MAX_FILE_BYTES = 64 * 1024 * 1024   # 单文件上限，超出重命名为 .1（只留一档）


class AsyncFileWriter:
  def __init__(self, path: str):
    self.path = path
    self.pid = os.getpid()
    self.disabled = False
    self._queue = queue.Queue(maxsize=_QUEUE_MAX)
    self._thread = threading.Thread(target=self._run, args=(self._queue,), daemon=True)
    self._thread.start()

  def write(self, line: str) -> None:
    if self.disabled:
      return
    try:
      self._queue.put_nowait(line)
    except queue.Full:
      pass  # 背压时丢日志，绝不阻塞调用方

  def _run(self, q) -> None:
    try:
      f = open(self.path, "a", buffering=64 * 1024)
    except OSError:
      self.disabled = True  # 无权限/路径不存在：静默禁用（等效原 _log_enabled=False）
      return
    buf = []
    last_flush = time.monotonic()
    while True:
      try:
        buf.append(q.get(timeout=_FLUSH_INTERVAL_S))
      except queue.Empty:
        pass
      now = time.monotonic()
      if not buf or (len(buf) < _BATCH_LINES and now - last_flush < _FLUSH_INTERVAL_S):
        continue
      try:
        f.write("".join(buf))
        f.flush()
        if f.tell() > _MAX_FILE_BYTES:
          f.close()
          os.replace(self.path, self.path + ".1")
          f = open(self.path, "a", buffering=64 * 1024)
      except OSError:
        self.disabled = True  # 磁盘故障：禁用，write 变 no-op
        try:
          f.close()
        except OSError:
          pass
        return
      buf.clear()
      last_flush = now


_writers = {}
_writers_lock = threading.Lock()


def get_writer(path: str) -> AsyncFileWriter:
  """按路径共享 writer 单例（card 与 carcontroller 同进程写同一文件，避免双线程交错）。"""
  pid = os.getpid()
  with _writers_lock:
    w = _writers.get(path)
    if w is None or w.pid != pid:  # fork 后重建：旧 queue/线程留在父进程，不 join
      w = AsyncFileWriter(path)
      _writers[path] = w
    return w


def acc_debug_path() -> str:
  """byd_acc_debug.log 路径：优先 LOG_ROOT（launch_pc.sh 设为 $OP_DIR/.comma），兜底项目根/.comma。"""
  log_root = os.environ.get("LOG_ROOT")
  if not log_root:
    log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".comma"))
  return os.path.join(log_root, "byd_acc_debug.log")


def acc_debug_write(line: str) -> None:
  get_writer(acc_debug_path()).write(line)

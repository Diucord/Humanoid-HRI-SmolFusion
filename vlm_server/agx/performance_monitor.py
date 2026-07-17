# performance_monitor.py
import os
import time
import csv
import threading
import datetime
import subprocess
import shutil
import ctypes
import re

import psutil
import pandas as pd

# ===== NVML 초기화 시도 =====
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    pynvml = None
    GPU_AVAILABLE = False

# ===== CUDA 런타임 로딩 (VRAM 유사 지표용) =====
_cudart = None
_cuda_mem_get_info = None
_cuda_free = None
_cuda_loaded = False

def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    cand = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.12.0",
        "libcudart.so.11",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
        "libcudart.so.10.1",
    ]
    for name in cand:
        try:
            _cudart = ctypes.CDLL(name)
            return _cudart
        except Exception:
            continue
    return None

def _ensure_cuda_runtime():
    global _cudart, _cuda_mem_get_info, _cuda_free, _cuda_loaded
    if _cuda_loaded:
        return True
    _cudart = _load_cudart()
    if _cudart is None:
        return False
    # cudaMemGetInfo 심볼 이름 매칭
    for name in ("cudaMemGetInfo_v2", "cudaMemGetInfo"):
        if hasattr(_cudart, name):
            _cuda_mem_get_info = getattr(_cudart, name)
            break
    if _cuda_mem_get_info is None:
        return False
    _cuda_mem_get_info.restype = ctypes.c_int
    _cuda_mem_get_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    # cudaFree(0)로 런타임 초기화 유도
    if hasattr(_cudart, "cudaFree"):
        _cuda_free = _cudart.cudaFree
        _cuda_free.restype = ctypes.c_int
        _cuda_free.argtypes = [ctypes.c_void_p]
    _cuda_loaded = True
    return True

def _cuda_mem_used_mb():
    """CUDA 컨텍스트가 없으면 cudaFree(0)로 초기화 시도 후 used 메모리(MB) 반환."""
    if not _ensure_cuda_runtime():
        return -1.0
    try:
        if _cuda_free is not None:
            try:
                _cuda_free(0)
            except Exception:
                pass
        free_b = ctypes.c_size_t(0)
        total_b = ctypes.c_size_t(0)
        ret = _cuda_mem_get_info(ctypes.byref(free_b), ctypes.byref(total_b))
        if ret != 0:
            return -1.0
        used_mb = (total_b.value - free_b.value) / (1024 ** 2)
        return float(used_mb)
    except Exception:
        return -1.0

# ===== tegrastats 지원 여부/1회 실행 =====
def _tegrastats_available():
    return shutil.which("tegrastats") is not None

def _tegrastats_once(interval_ms=500, timeout=2.0):
    """tegrastats 실행 후 (vram_MB, util_pct) 반환. Jetson은 VRAM -1."""
    if not _tegrastats_available():
        return -1.0, -1.0
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms), "--count", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, _ = proc.communicate(timeout=timeout)
        if not out:
            return -1.0, -1.0
        line = out.splitlines()[0]
        # GR3D util 파싱
        util = _parse_gr3d_util(line)
        return -1.0, util
    except Exception:
        return -1.0, -1.0

# ===== tegrastats 스트리밍 수집기 (안정) =====
_ts_lock = threading.Lock()
_ts_last_util = -1.0
_ts_last_ts = 0.0
_ts_proc = None
_ts_thread = None

def _parse_gr3d_util(line: str) -> float:
    """
    JetPack/모델별 다양한 출력 포맷 대응:
    - GR3D_FREQ 23%
    - GR3D_FREQ 23%@918
    - GR3D 23%
    - GR3D_BUSY 23%
    """
    try:
        m = re.search(r"(GR3D(?:_FREQ|_BUSY)?|GR3D)\s+(\d+)\s*%", line)
        if m:
            return float(m.group(2))
    except Exception:
        pass
    return -1.0

def _ts_reader(proc):
    global _ts_last_util, _ts_last_ts
    for line in iter(proc.stdout.readline, ''):
        util = _parse_gr3d_util(line)
        if util >= 0:
            with _ts_lock:
                _ts_last_util = util
                _ts_last_ts = time.time()

def _ensure_tegrastats_stream(interval_ms=500):
    """상시 실행되는 tegrastats 프로세스를 보장"""
    global _ts_proc, _ts_thread
    if _ts_proc is not None and _ts_proc.poll() is None:
        return True
    if shutil.which("tegrastats") is None:
        return False
    try:
        cmd = ["tegrastats", "--interval", str(interval_ms)]
        _ts_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr 케이스도 흡수
            text=True,
            bufsize=1
        )
        _ts_thread = threading.Thread(target=_ts_reader, args=(_ts_proc,), daemon=True)
        _ts_thread.start()
        return True
    except Exception:
        _ts_proc = None
        _ts_thread = None
        return False

def _get_tegrastats_util_from_stream(max_age_sec=2.0):
    """최근 샘플이 max_age_sec 이내면 그 값을, 아니면 -1 반환"""
    with _ts_lock:
        if _ts_last_ts and (time.time() - _ts_last_ts) <= max_age_sec:
            return _ts_last_util
    return -1.0

def stop_tegrastats_stream():
    """백그라운드 tegrastats 프로세스를 정리합니다."""
    global _ts_proc
    try:
        if _ts_proc is not None and _ts_proc.poll() is None:
            _ts_proc.terminate()
            try:
                _ts_proc.wait(timeout=1.0)
            except Exception:
                _ts_proc.kill()
    finally:
        _ts_proc = None

# ===== 측정 함수 =====
def get_vram_usage():
    # 1) NVML
    if GPU_AVAILABLE and pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return meminfo.used / (1024 ** 2)  # MB
        except Exception:
            pass
    # 2) CUDA 런타임 폴백 (Jetson에서 유사 VRAM 지표)
    used = _cuda_mem_used_mb()
    if used >= 0:
        return used
    # 3) 최종 실패
    return -1.0

def get_gpu_util():
    # 1) NVML 시도
    if GPU_AVAILABLE and pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            pass

    # 2) tegrastats 스트리밍(우선)
    if _ensure_tegrastats_stream(interval_ms=500):
        util = _get_tegrastats_util_from_stream(max_age_sec=2.0)
        if util >= 0:
            return util

    # 3) 최후의 수단: 1회 실행(낮은 신뢰도)
    _, util = _tegrastats_once()
    return util

def get_cpu_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss  # bytes
    return mem / (1024 ** 2)  # MB

# ===== 간단 출력/타이머 =====
def print_system_metrics():
    print(f"[GPU VRAM 사용량]: {get_vram_usage():.1f}MB")
    print(f"[GPU 사용률]: {get_gpu_util():.1f}%")
    print(f"[CPU 메모리 사용량]: {get_cpu_memory_usage():.1f}MB")

def measure_execution_time(func, *args, label=""):
    start = time.time()
    result = func(*args)
    end = time.time()
    print(f"[⏱ 실행 시간 - {label}]: {end - start:.2f}초")
    return result

# ===== CSV 모니터링 루프 =====
def monitor_loop(log_path, interval=0.5, stop_event=None):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "elapsed", "gpu_vram_MB",
            "gpu_util_pct", "cpu_mem_MB"
        ])

        start = time.time()
        while not stop_event.is_set():
            now = time.time()
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(now - start, 2),
                round(get_vram_usage(), 1),
                round(get_gpu_util(), 1),
                round(get_cpu_memory_usage(), 1)
            ])
            f.flush()
            time.sleep(interval)

# ===== 제어 함수 =====
def start_monitoring_thread(log_path="performance_log_agx.csv", interval=0.5):
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_loop, args=(log_path, interval, stop_event), daemon=True)
    thread.start()
    return stop_event

# ===== 요약/버스트 분석 =====
def _summ(values):
    if len(values) == 0:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(sum(values) / len(values)),
    }

def _pct(values, q):
    """선형 보간 퍼센타일"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)

def _find_bursts(elapsed_series, util_series, threshold=80.0, min_samples=1):
    """
    연속 샘플에서 GPU 사용률이 threshold 이상인 구간을 (start_elapsed, end_elapsed)로 반환.
    """
    bursts = []
    start_idx = None
    for i, util in enumerate(util_series):
        if util >= threshold:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                if i - start_idx >= min_samples:
                    bursts.append((elapsed_series[start_idx], elapsed_series[i - 1]))
                start_idx = None
    if start_idx is not None:
        if len(util_series) - start_idx >= min_samples:
            bursts.append((elapsed_series[start_idx], elapsed_series[-1]))
    return bursts

def stop_monitoring_and_summarize(log_path, total_time=None, burst_threshold=80.0, burst_min_samples=1):
    print("\n[📊 성능 요약 결과]")
    try:
        df = pd.read_csv(log_path)

        vram_valid = df["gpu_vram_MB"] >= 0
        util_valid = df["gpu_util_pct"] >= 0

        vram_vals = df.loc[vram_valid, "gpu_vram_MB"].tolist()
        util_vals = df.loc[util_valid, "gpu_util_pct"].tolist()
        cpu_vals  = df["cpu_mem_MB"].tolist()

        vram_sum = _summ(vram_vals)
        util_sum = _summ(util_vals)
        cpu_sum  = _summ(cpu_vals)

        if total_time is not None:
            print(f"총 실행 시간: {round(total_time, 2)}초")
        print(f"샘플 수: {len(df)}")

        if vram_vals:
            print(f"VRAM(MB): 초기 {vram_vals[0]:.1f}, 최대 {vram_sum['max']:.1f}, 평균 {vram_sum['avg']:.1f}")
        else:
            print(f"VRAM(MB): 측정 불가(통합 메모리 또는 NVML/CUDA 미가용)")

        if util_vals:
            print(f"GPU 사용률(%): 최대 {util_sum['max']:.1f}, 평균 {util_sum['avg']:.1f}")
        else:
            print(f"GPU 사용률(%): 측정 불가")

        print(f"프로세스 메모리(MB): 최대 {cpu_sum['max']:.1f}, 평균 {cpu_sum['avg']:.1f}")

        if util_vals:
            bursts = _find_bursts(
                elapsed_series=df["elapsed"].tolist(),
                util_series=df["gpu_util_pct"].tolist(),
                threshold=burst_threshold,
                min_samples=burst_min_samples,
            )
            if bursts:
                burst_str = ", ".join([f"{s:.2f}–{e:.2f}s" for s, e in bursts])
                print(f"GPU 활성 구간(≥{burst_threshold:.0f}%): {burst_str}")
            else:
                print(f"GPU 활성 구간(≥{burst_threshold:.0f}%): 없음")
        else:
            print(f"GPU 활성 구간: 계산 불가 (GPU 사용률 없음)")

        print(f"로그 파일: {os.path.abspath(log_path)}")

    except Exception as e:
        print(f"[⚠️ 요약 실패]: {e}")
    finally:
        # tegrastats 백그라운드 수집기 종료
        try:
            stop_tegrastats_stream()
        except Exception:
            pass

# ===== API 타이밍 계측 유틸 =====
_api_lock = threading.Lock()
_api_data = {"/analyze": [], "/chat": []}  # (elapsed_sec, status, ts)

def record_api_call(endpoint: str, elapsed_sec: float, status: int = 200):
    """엔드포인트 호출 시간을 기록합니다."""
    with _api_lock:
        if endpoint not in _api_data:
            _api_data[endpoint] = []
        _api_data[endpoint].append((float(elapsed_sec), int(status), time.time()))

def summarize_api_calls(csv_path: str = "api_metrics_agx.csv"):
    """기록된 API 호출 시간을 요약해서 콘솔과 CSV로 출력합니다."""
    with _api_lock:
        snapshot = {ep: list(vals) for ep, vals in _api_data.items()}

    print("\n[API 응답 시간 요약]")
    rows = []
    header = ["endpoint", "count", "min_s", "p50_s", "avg_s", "p95_s", "p99_s", "max_s", "success_rate_pct"]

    for ep, items in snapshot.items():
        if not items:
            print(f"- {ep}: N/A")
            rows.append([ep, 0, None, None, None, None, None, None, 0.0])
            continue

        elapseds = [x[0] for x in items]
        statuses = [x[1] for x in items]
        cnt = len(items)

        mn  = min(elapseds)
        p50 = _pct(elapseds, 50)
        avg = sum(elapseds) / cnt
        p95 = _pct(elapseds, 95)
        p99 = _pct(elapseds, 99)
        mx  = max(elapseds)
        success = sum(1 for s in statuses if 200 <= s < 400)
        success_rate = round(success * 100.0 / cnt, 2)

        print(f"- {ep}: 호출 {cnt}회, min {mn:.2f}s / p50 {p50:.2f}s / avg {avg:.2f}s / p95 {p95:.2f}s / p99 {p99:.2f}s / max {mx:.2f}s, 성공률 {success_rate}%")
        rows.append([ep, cnt, f"{mn:.3f}", f"{p50:.3f}", f"{avg:.3f}", f"{p95:.3f}", f"{p99:.3f}", f"{mx:.3f}", f"{success_rate}"])

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"API 지표 CSV: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"[⚠️ API 지표 CSV 저장 실패]: {e}")

# test_performance_agx.py
import asyncio
import time
import os

from test import main_loop  # async def main_loop()
from performance_monitor import (
    start_monitoring_thread,
    stop_monitoring_and_summarize,
    summarize_api_calls,
)

# === 측정 파라미터 ===
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_LOG = os.path.join(LOG_DIR, "performance_log_agx.csv")
API_LOG  = os.path.join(LOG_DIR, "api_metrics_agx.csv")

SAMPLE_INTERVAL_SEC = 0.2     # 리소스 샘플링 간격 (s)
BURST_THRESHOLD_PCT = 40.0     # 버스트 임계값 (%)
BURST_MIN_SAMPLES   = 1        # 버스트 최소 연속 샘플 수


async def run_with_monitoring():
    print("[🟢 AGX 성능 측정 - 시작]")
    stop_event = start_monitoring_thread(log_path=PERF_LOG, interval=SAMPLE_INTERVAL_SEC)

    total_time = None
    try:
        t0 = time.time()
        await main_loop()
        total_time = time.time() - t0
    except Exception as e:
        print(f"[❌ 예외 발생]: {e}")
    finally:
        stop_event.set()
        # 리소스 요약
        stop_monitoring_and_summarize(
            log_path=PERF_LOG,
            total_time=total_time,
            burst_threshold=BURST_THRESHOLD_PCT,
            burst_min_samples=BURST_MIN_SAMPLES,
        )
        # API 타이밍 요약
        summarize_api_calls(csv_path=API_LOG)
        print("[🔴 AGX 성능 측정 - 종료]")


if __name__ == "__main__":
    # Python 3.7+ 우선
    try:
        asyncio.run(run_with_monitoring())
    except AttributeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_with_monitoring())
        loop.close()

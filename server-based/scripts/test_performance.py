# test_performance.py
import asyncio, time, os
from test import main_loop  # async def main_loop()
from performance_monitor import (
    start_monitoring_thread,
    stop_monitoring_and_summarize,
    summarize_api_calls,
)

# === Measurement Parameters ===
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_LOG = os.path.join(LOG_DIR, "performance_log.csv")
API_LOG  = os.path.join(LOG_DIR, "api_metrics.csv")

SAMPLE_INTERVAL_SEC = 0.2     # Resource sampling interval (seconds)
BURST_THRESHOLD_PCT = 40.0    # Burst threshold (%)
BURST_MIN_SAMPLES   = 1       # Minimum consecutive samples for a burst


async def run_with_monitoring():
    print("[Performance Test - START]")
    stop_event = start_monitoring_thread(log_path=PERF_LOG, interval=SAMPLE_INTERVAL_SEC)

    total_time = None
    try:
        t0 = time.time()
        await main_loop()
        total_time = time.time() - t0
    except Exception as e:
        print(f"[❌ Exception occurred]: {e}")
    finally:
        stop_event.set()
        # Summarize resource usage
        stop_monitoring_and_summarize(
            log_path=PERF_LOG,
            total_time=total_time,
            burst_threshold=BURST_THRESHOLD_PCT,
            burst_min_samples=BURST_MIN_SAMPLES,
        )
        # Summarize API timing metrics
        summarize_api_calls(csv_path=API_LOG)
        print("[AGX Performance Test - END]")


if __name__ == "__main__":
    # Prefer Python 3.7+ (asyncio.run available)
    try:
        asyncio.run(run_with_monitoring())
    except AttributeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_with_monitoring())
        loop.close()
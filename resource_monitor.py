import os
import time
import threading
import psutil
import pandas as pd


class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.process = psutil.Process(os.getpid())

        self.records = []
        self.running = False
        self.thread = None

        self.start_time = None
        self.end_time = None

    def _monitor(self):        
        self.process.cpu_percent(None)
        while self.running:
            timestamp = time.time() - self.start_time

            cpu = self.process.cpu_percent(interval=None)
            #cpu = cpu_raw / psutil.cpu_count(logical=True)
            mem = self.process.memory_info().rss / (1024 ** 2)

            self.records.append({
                "time_sec": timestamp,
                "cpu_percent": cpu,
                "memory_mb": mem
            })
            time.sleep(self.interval)

    def start(self):

        self.start_time = time.time()
        self.running = True

        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):

        self.running = False
        self.thread.join()
        self.end_time = time.time()

    def save(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)        
        df.to_csv(
            os.path.join(output_dir, "resource_timeseries.csv"),
            index=False
        )        
        summary = pd.DataFrame([{

            "training_time_sec":
                self.end_time - self.start_time,

            "avg_cpu_percent":
                df["cpu_percent"].mean(),

            "max_cpu_percent":
                df["cpu_percent"].max(),

            "avg_memory_mb":
                df["memory_mb"].mean(),

            "max_memory_mb":
                df["memory_mb"].max(),

            "samples":
                len(df)
        }])

        summary.to_csv(
            os.path.join(output_dir, "resource_summary.csv"),
            index=False
        )

        return summary
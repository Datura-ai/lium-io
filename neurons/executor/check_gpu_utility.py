import time
import pynvml
import csv

def scrape_gpu_metrics(output_file=None, interval=1):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    if output_file:
        # Open a CSV file to save metrics
        csv_file = open(output_file, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "GPU Index", "GPU Utilization (%)", "Memory Usage (MB)", "Total Memory (MB)"])

    print(f"Scraping metrics for {device_count} GPUs...")
    try:
        while True:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Handle potential bytes or str return for name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util = utilization.gpu
                mem_used = memory.used / 1024**2
                mem_total = memory.total / 1024**2
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                print(f"{timestamp} | GPU {i} ({name}): GPU {gpu_util}% | Memory {mem_used:.2f}/{mem_total:.2f} MB")

                # Write to file if needed
                if output_file:
                    writer.writerow([timestamp, i, gpu_util, mem_used, mem_total])
            
            time.sleep(interval)
            signature = "signature"
            payload = {
                "gpu_utilization": gpu_util,
                "memory_usage": mem_used,
                "memory_total": mem_total,
                "timestamp": timestamp,
                "signature": signature,
            }
            
            # send payload to the backend-app using websocket
    
            
    except KeyboardInterrupt:
        print("Stopping GPU scraping...")
    finally:
        pynvml.nvmlShutdown()
        if output_file:
            csv_file.close()

# Example: Save metrics to a file
scrape_gpu_metrics(output_file="gpu_metrics.csv", interval=10)

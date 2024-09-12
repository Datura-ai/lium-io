import time
import random

start_time = time.time()

wait_time = random.uniform(10, 30)
time.sleep(wait_time)

# print("Job finished")
print(time.time() - start_time)
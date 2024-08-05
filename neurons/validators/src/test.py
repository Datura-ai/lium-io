import time

print("Job starts")
start_time = time.time()

time.sleep(15)

print("Job finished")
print("--- %s seconds ---" % (time.time() - start_time))
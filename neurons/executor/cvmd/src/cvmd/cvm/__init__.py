"""The CVM launch path — DAH-2576.

`manager.py` is the entry point; everything else is one step of it. The order in `manager` is
the safety argument: guard, resolve, prepare, **measure**, then launch. Nothing reaches QEMU
until the artifacts on disk hash to the triple the caller asked for.
"""

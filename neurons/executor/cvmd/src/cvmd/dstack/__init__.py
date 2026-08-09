"""The seam onto `dstacktee/scripts/dstack.py`.

cvmd does not reimplement any part of the launcher. dstack.py shapes what the CVM measures,
so a second implementation would be a second set of measurements — the whole point of CVM v2
is that the validator can predict them. This package imports the real file and calls the same
functions its CLI calls.
"""

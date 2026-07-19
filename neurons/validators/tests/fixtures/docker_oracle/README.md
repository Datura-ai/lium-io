# Docker oracle golden files

Golden snapshot files for the docker-service characterization oracle
(`tests/docker_oracle/`, DAH-2382). A test writes its golden on first run (and
fails, asking for a re-run), asserts byte-equality against it afterwards, and
rewrites it only when pytest is invoked with `--update-snapshot` — the repo's
own snapshot flag from `tests/conftest.py`. Review every diff under this
directory as a behavior change of `docker_service.py`, never as noise.

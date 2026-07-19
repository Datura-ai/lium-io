# E7 differential gate (DAH-2382, decision D-B): mandatory LOCAL run against a
# real docker daemon before each extraction PR. See tests/integration/README_E7.md.

PYTEST = pdm run pytest
E7_TESTS = tests/integration/test_docker_service_differential.py

.PHONY: e7 e7-record

e7:
	RUN_RENTAL_DOCKER_SDK_INTEGRATION=1 $(PYTEST) $(E7_TESTS) -q

e7-record:
	RUN_RENTAL_DOCKER_SDK_INTEGRATION=1 $(PYTEST) $(E7_TESTS) -q --update-snapshot

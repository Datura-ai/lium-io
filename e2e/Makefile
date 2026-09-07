# make e2e            build every image from this checkout, bring the stack up, seed the miner, run every suite, tear down
# make e2e-full       the merge gate (gate.sh): the same with a hard timeout on every step, every suite run even after a
#                     failure, artifacts/summary.md + junit files always — what CI's e2e-gate job runs
# make up / down      keep the stack around while writing a test (README.md)
# Docker-only (dind is privileged): a CI runner, a dev box or a Lium DinD pod — never a laptop that cannot spare it.

SHELL := /bin/bash
.DEFAULT_GOAL := e2e

include stack.env
export

ARTIFACTS := $(CURDIR)/artifacts
WAIT_TIMEOUT ?= 300
ifeq ($(E2E_GPU),)
  COMPOSE := docker compose --env-file stack.env -f docker-compose.e2e.yml
  # the executor host's dockerd (rental containers land there); the pod image is built onto it in `seed`
  EXECUTOR_DOCKER := docker -H tcp://127.0.0.1:23750
else
  # GPU host: the executor runs on THIS machine's dockerd (docker-compose.gpu.yml) and is reached at the network gateway
  COMPOSE := docker compose --env-file stack.env -f docker-compose.e2e.yml -f docker-compose.gpu.yml
  EXECUTOR_DOCKER := docker
  export E2E_EXECUTOR_IP := 172.30.0.1
endif

.PHONY: e2e e2e-full build up seed test logs down

e2e:
	$(MAKE) build
	$(MAKE) up
	$(MAKE) test || { $(MAKE) logs; $(MAKE) down; exit 1; }
	$(MAKE) down

# a new suite = a tests/<name> directory plus a test-<name> target here; gate.sh picks it up by name
e2e-full:
	./gate.sh

build:
	$(COMPOSE) build
	$(COMPOSE) --profile tools build tester

up:
	$(COMPOSE) up -d --wait --wait-timeout $(WAIT_TIMEOUT)
	$(MAKE) seed

# ready = every healthcheck green (compose --wait above) + the miner knows the executor + the rental image is on the
# executor host's daemon. `seed` is idempotent (ON CONFLICT DO NOTHING; build is cached).
seed:
	$(COMPOSE) exec -T miner-db psql -qU postgres -d miner -v ON_ERROR_STOP=1 \
	  -v live_uuid=$(E2E_EXECUTOR_UUID) -v live_ip=$(E2E_EXECUTOR_IP) -v dead_uuid=$(E2E_DEAD_EXECUTOR_UUID) -v dead_ip=$(E2E_DEAD_EXECUTOR_IP) \
	  -v port=$(E2E_EXECUTOR_PORT) -v validator=$(E2E_VALIDATOR_HOTKEY) < miner/seed.sql
	$(EXECUTOR_DOCKER) build -q -t $(E2E_POD_IMAGE) pod >/dev/null
	@echo "seeded: 2 executor rows for $(E2E_VALIDATOR_HOTKEY); $(E2E_POD_IMAGE) on the executor host"

test: test-protocol test-cycle test-rental

# each suite is one `docker compose run` of the tester (validator image + pytest); junit lands in artifacts/
test-%:
	mkdir -p $(ARTIFACTS)
	$(COMPOSE) --profile tools run --rm -T tester pytest tests/$* -v --junitxml=/e2e/artifacts/$*-junit.xml

logs:
	mkdir -p $(ARTIFACTS)
	$(COMPOSE) ps -a > $(ARTIFACTS)/compose-ps.txt || true
	$(COMPOSE) logs --no-color --timestamps > $(ARTIFACTS)/compose.log 2>&1 || true
	$(EXECUTOR_DOCKER) ps -a > $(ARTIFACTS)/executor-docker-ps.txt 2>&1 || true

down:
	$(COMPOSE) --profile tools down -v --remove-orphans

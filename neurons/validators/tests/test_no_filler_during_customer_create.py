"""A filler create stands down while a customer create ran on its executor since the filler started (the customer's sweep
would remove the filler); only customer creates that end after the filler started count."""

from __future__ import annotations

import contextlib

from payload_models.payloads import WorkloadKind
from services.docker_service import customer_creates
from test_deploy_optimizations import _payload


@contextlib.contextmanager
def _customer_create_that_reached_create_container(customer):
    with customer_creates.track(customer):
        customer_creates.reached_create_container(customer)
        yield


def test_filler_is_refused_by_a_customer_create_ending_after_its_start_not_by_older_ones() -> None:
    customer = _payload()
    filler = _payload(workload_kind=WorkloadKind.FILLER, executor_id=customer.executor_id)
    with _customer_create_that_reached_create_container(customer):
        pass
    customer_running = contextlib.ExitStack()
    customer_running.enter_context(_customer_create_that_reached_create_container(customer))

    with customer_creates.track(filler):
        customer_running.close()
        refused_after_customer_ended = customer_creates.ran_since_filler_started(filler)
    with customer_creates.track(filler):
        refused_after_older_customer_creates = customer_creates.ran_since_filler_started(filler)
        with _customer_create_that_reached_create_container(customer):
            pass
        with customer_creates.track(filler.model_copy()):
            pass
        refused_beside_a_retry_of_its_pod = customer_creates.ran_since_filler_started(filler)

    assert refused_after_customer_ended is True
    assert refused_after_older_customer_creates is False
    assert refused_beside_a_retry_of_its_pod is True
    assert id(filler) not in customer_creates._ended_seen_by_filler_create

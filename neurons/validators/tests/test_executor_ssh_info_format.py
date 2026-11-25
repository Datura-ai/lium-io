import pydantic
import pytest
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "Ab+/" * 11


def _info(**fields) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        **{
            "uuid": "node",
            "address": "203.0.113.7",
            "port": 8001,
            "ssh_username": "root",
            "ssh_port": 2200,
            "python_path": "/root/app/.venv/bin/python",
            "root_dir": "/root/app",
            "ssh_host_key": f"{KEY} root@0123456789ab",
            **fields,
        }
    )


@pytest.mark.parametrize(
    "fields",
    [
        # the shapes the node fleet answers with
        {},
        {"python_path": "/usr/bin/python3", "root_dir": "/app"},
        {"port_range": "40400-40500"},
        {"port_range": "20300, 20311"},
        {"port_range": "30000,30001,30002"},
        {"port_range": ""},
        {"port_range": None, "port_mappings": None},
        {"port_mappings": "[[30000,40001],[30001,40002]]"},
        {"port_mappings": "[[60006, 60006]]"},
        {"port_mappings": ""},
        # older executors send a port as a digit string
        {"port_mappings": '[[22, "30001"]]'},
        {"ssh_host_key": KEY},
        {"ssh_host_key": f"{KEY} root@node-host_1"},
        {"ssh_host_key": None},
        {"ssh_username": "exec-user"},
        # the validator's own rows for a node that gave no answer
        {"ssh_username": "", "python_path": "", "root_dir": ""},
        # the validator reads a malformed range itself and tells the provider
        {"port_range": "40000:65535"},
        {"ssh_host_key": f"  {KEY}  "},
        {"ssh_host_key": "   "},
    ],
)
def test_the_fields_nodes_answer_with_pass(fields):
    _info(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"ssh_username": "-oProxyCommand=x"},
        {"ssh_username": "root x"},
        {"python_path": "/bin/true; id"},
        {"python_path": "python3"},
        {"python_path": "-oProxyCommand=x"},
        {"python_path": "/usr/bin/python3 -c x"},
        {"python_path": "/root/$(id)"},
        {"root_dir": "/root/app;id"},
        {"root_dir": "/root/app\nid"},
        {"root_dir": "/root/app dir"},
        {"port_range": "1-2;id"},
        {"port_range": "-oProxyCommand=x"},
        {"port_range": "1-2 $(id)"},
        {"port_mappings": "[[1, 2], [3]]"},
        {"port_mappings": '[["1;id", 2]]'},
        {"port_mappings": '[["-1", 2]]'},
        {"port_mappings": '[["123456", 2]]'},
        {"port_mappings": "[[1, 70000]]"},
        {"port_mappings": "[[true, 2]]"},
        {"port_mappings": "{"},
        {"ssh_host_key": f"{KEY} x;id"},
        {"ssh_host_key": "-oProxyCommand=x"},
        {"ssh_host_key": f"{KEY}\n{KEY}"},
        {"ssh_host_key": "ssh-dss AAAA"},
    ],
)
def test_a_field_outside_its_format_is_refused(fields):
    with pytest.raises(pydantic.ValidationError):
        _info(**fields)


def test_the_validator_leaves_out_a_node_answered_with_such_a_field_and_keeps_its_siblings():
    # the validator parses the miner's answer with the same model, so a miner that skips the check gains nothing;
    # one node out of format does not cost the healthy nodes of the same answer their checks
    node = _info().model_dump(mode="json")
    bad = {**node, "uuid": "bad", "python_path": "/bin/true; id"}
    answer = AcceptSSHKeyRequest.model_validate({"executors": [node, bad], "known_executor_ids": ["node", "bad"]})
    assert [e.uuid for e in answer.executors] == ["node"]
    assert answer.known_executor_ids == ["node", "bad"]
    with pytest.raises(pydantic.ValidationError):
        AcceptSSHKeyRequest.model_validate({"executors": "not a list"})

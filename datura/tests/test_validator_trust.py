import pytest

from datura.validator_trust import (
    RELEASE_SUBNET_OWNER_COLDKEY,
    subnet_owner_from_reply,
    subnet_owner_request,
    subnet_owner_storage_key,
)


def test_storage_key_is_the_identity_hashed_netuid():
    assert subnet_owner_storage_key(51).endswith("3300")
    assert subnet_owner_storage_key(51).startswith("0x658faa385070e074c85bf6b568cf0555")


def test_reads_and_encodes_the_owner():
    from scalecodec.utils.ss58 import ss58_decode

    assert subnet_owner_request(51)["params"] == [subnet_owner_storage_key(51)]
    reply = {"result": "0x" + ss58_decode(RELEASE_SUBNET_OWNER_COLDKEY)}
    assert subnet_owner_from_reply(reply) == RELEASE_SUBNET_OWNER_COLDKEY


@pytest.mark.parametrize("reply", [{}, {"result": None}, {"result": "0x00"}, ["x"]])
def test_refuses_any_other_reply(reply):
    with pytest.raises(ValueError):
        subnet_owner_from_reply(reply)

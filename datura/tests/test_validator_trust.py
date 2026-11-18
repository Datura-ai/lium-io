import io
import json

from datura import validator_trust
from datura.validator_trust import RELEASE_SUBNET_OWNER_HOTKEY, read_subnet_owner_hotkey, subnet_owner_storage_key



def test_storage_key_is_the_identity_hashed_netuid():
    assert subnet_owner_storage_key(51).endswith("3300")
    assert subnet_owner_storage_key(51).startswith("0x658faa385070e074c85bf6b568cf0555")


def test_reads_and_encodes_the_owner(monkeypatch):
    from scalecodec.utils.ss58 import ss58_decode

    sent = {}

    def urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return io.BytesIO(json.dumps({"result": "0x" + ss58_decode(RELEASE_SUBNET_OWNER_HOTKEY)}).encode())

    monkeypatch.setattr(validator_trust.urllib.request, "urlopen", urlopen)
    assert read_subnet_owner_hotkey("https://node.invalid") == RELEASE_SUBNET_OWNER_HOTKEY
    assert sent["method"] == "state_getStorage" and sent["params"] == [subnet_owner_storage_key(51)]

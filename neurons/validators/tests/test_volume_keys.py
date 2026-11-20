
from services.volume_keys import (
    VolumeKeyDeriver,
    derive_volume_passphrase,
)

_MASTER_32 = "test-master-secret-32-chars-long!!"


def test_derive_volume_passphrase_is_stable_per_pod():
    master = _MASTER_32
    a = derive_volume_passphrase(master, "pod_a")
    b = derive_volume_passphrase(master, "pod_a")
    c = derive_volume_passphrase(master, "pod_b")
    assert a == b
    assert a != c
    assert " " not in a


def test_material_derives_per_pod():
    deriver = VolumeKeyDeriver(master_secret=_MASTER_32)
    material = deriver.material("pod-1")
    assert material.passphrase == derive_volume_passphrase(_MASTER_32, "pod-1")
    assert material.key_id == "pod-1"



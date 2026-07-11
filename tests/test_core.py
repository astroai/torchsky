from torchsky.core import ChecksumVerifier


def test_core_exposes_only_the_torchfits_checksum_boundary():
    assert ChecksumVerifier.__module__ == "torchfits.core"

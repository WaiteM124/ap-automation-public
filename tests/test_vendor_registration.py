"""History import registers vendors so imported names resolve at full MDM confidence."""

from ap_automation.core import mdm


def test_register_vendor_makes_a_new_name_resolve_exactly(tmp_path):
    path = str(tmp_path / "v.csv")
    assert mdm.resolve_vendor("Northwind Traders")[1] < 0.85          # unknown: cold start
    vid = mdm.register_vendor("Northwind Traders", path)
    assert vid.startswith("IMP_")
    assert mdm.resolve_vendor("Northwind Traders") == (vid, 1.0)


def test_register_vendor_is_deterministic_and_does_not_merge_distinct_vendors(tmp_path):
    a = mdm.register_vendor("Acme Supply Group", None)
    b = mdm.register_vendor("Acme Supply Partners International", None)
    assert a != b                                                    # no fuzzy joining at import
    mdm.VENDOR_MAP.clear()
    assert mdm.register_vendor("Acme Supply Group", None) == a       # same id after a restart


def test_known_vendor_keeps_its_existing_id():
    existing = next(iter(mdm.VENDOR_MAP.items()))
    assert mdm.register_vendor(existing[0], None) == existing[1]


def test_registrations_persist_and_reload(tmp_path):
    path = str(tmp_path / "v.csv")
    vid = mdm.register_vendor("Contoso Ltd", path)
    mdm.VENDOR_MAP.clear()
    assert mdm.load_client_vendors(path) == 1
    assert mdm.resolve_vendor("Contoso Ltd") == (vid, 1.0)

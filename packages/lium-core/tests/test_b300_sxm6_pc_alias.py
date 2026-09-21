"""`NVIDIA B300 SXM6 PC` is the B300 SXM6 AC module under the second name its driver reports.

Every consumer of this table (the validator's price pin, the backend's shared config, the miner's
deposit CLI) treats the two names as one class, so every table here that carries the AC name carries
the PC name at the same value. A future edit to one name fails here until the other name follows.
"""

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

AC = "NVIDIA B300 SXM6 AC"
PC = "NVIDIA B300 SXM6 PC"
UNLISTED = "NVIDIA B300 SXM6 LC"  # a third spelling nobody has added: the negative control


def _tables_naming_the_ac_card() -> dict[str, dict]:
    return {
        field: table
        for field, table in DEFAULT_SHARED_CONFIG.model_dump().items()
        if isinstance(table, dict) and AC in table
    }


def test_every_table_naming_the_ac_card_names_the_pc_card_with_the_same_value():
    tables = _tables_naming_the_ac_card()

    assert {"machine_prices", "required_deposit_amount"} <= set(tables)
    for field, table in tables.items():
        assert PC in table, f"{field} has {AC!r} but not {PC!r}"
        assert table[PC] == table[AC], f"{field}: {PC!r} = {table[PC]!r} but {AC!r} = {table[AC]!r}"


def test_a_spelling_nobody_added_is_in_no_table():
    for table in _tables_naming_the_ac_card().values():
        assert UNLISTED not in table

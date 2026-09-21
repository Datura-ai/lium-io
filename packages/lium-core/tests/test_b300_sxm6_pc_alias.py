"""`NVIDIA B300 SXM6 PC` is listed as the B300 SXM6 AC card's alias: the same price and deposit.

The name is what two providers report for their B300 SXM6 cards (21 Sep 2026); NVIDIA's public name
table lists only the AC spelling. Every consumer of this table (the validator's price pin, the
backend's shared config, the miner's deposit CLI) treats the two names as one class, so each table
that carries the AC name derives the PC entry from it — a re-price of the AC card moves both names,
and the PC name is never a literal row that could drift.
"""

import inspect
import re

from lium_core.shared_config import defaults
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


def test_the_pc_name_is_derived_from_the_ac_row_never_a_literal_of_its_own():
    source = inspect.getsource(defaults)
    assert not re.search(r'"NVIDIA B300 SXM6 PC"\s*:', source), "spell the PC entry as an alias of the AC entry"
    assert source.count("_with_pc_alias({") == 2  # machine_prices and required_deposit_amount

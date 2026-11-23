# The validator hotkey this e2e miner trusts (core/config.py imports _VALIDATOR_HOTKEY_SS58 from here when the file
# exists, and the miner then reads no signed validator message — the same mechanism as the executor's
# executor/trust_anchor.py). Value = E2E_VALIDATOR_HOTKEY in stack.env: a throwaway key derived from a BIP-39 test
# vector, meaningless outside this compose network.
_VALIDATOR_HOTKEY_SS58 = "5EPCUjPxiHAcNooYipQFWr9NmmXJKpNG5RhcntXwbtUySrgH"

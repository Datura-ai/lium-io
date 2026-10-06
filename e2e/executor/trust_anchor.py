# The validator hotkey this e2e executor trusts (core/config.py imports _VALIDATOR_HOTKEY_SS58 from here when the
# file exists — the same mechanism the :dev images use for the staging validator). Value = E2E_VALIDATOR_HOTKEY in
# stack.env: a throwaway key derived from a BIP-39 test vector, meaningless outside this compose network.
_VALIDATOR_HOTKEY_SS58 = "5EPCUjPxiHAcNooYipQFWr9NmmXJKpNG5RhcntXwbtUySrgH"

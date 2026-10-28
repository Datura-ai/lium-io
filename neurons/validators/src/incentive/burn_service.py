"""Burn emission distribution service.

This service handles the distribution of burn emission across burner nodes:
the burn share is split across NEW_BURNERS by slot count.
"""

from collections import Counter

import bittensor

from core.config import get_total_burn_emission, settings
from core.utils import _m, get_logger

logger = get_logger(__name__)


def _burner_coldkey_matches(miner: bittensor.NeuronInfo) -> bool:
    """Return True when the miner UID has no configured coldkey or coldkey matches."""
    expected_coldkey = settings.BURNER_COLDKEYS.get(miner.uid)
    if expected_coldkey is None:
        return True

    if miner.coldkey == expected_coldkey:
        return True

    logger.error(
        _m(
            "Burner coldkey mismatch — withholding burn weight",
            extra={
                "miner_uid": miner.uid,
                "miner_hotkey": miner.hotkey,
                "expected_coldkey": expected_coldkey,
                "actual_coldkey": miner.coldkey,
            },
        )
    )
    return False


def verified_burner_hotkey(miners: list[bittensor.NeuronInfo]) -> str | None:
    """The first burner slot's hotkey, only while its coldkey is the configured one (DAH-4001).

    Withheld idle shares go to it, so it passes the same check as the burn share.
    None when the uid is not in `miners` or another coldkey holds it: the caller must not pick a substitute.
    """
    burner_uid = settings.NEW_BURNERS[0]
    burner = next((miner for miner in miners if miner.uid == burner_uid), None)
    if burner is None or not _burner_coldkey_matches(burner):
        return None
    return burner.hotkey


class BurnService:
    """Service for calculating burn emission distribution across burner nodes.

    All burners in NEW_BURNERS share the burn emission by slot count.
    """

    def calculate_burn_scores(
        self,
        miners: list[bittensor.NeuronInfo],
        burn_share: float,
    ) -> dict[str, float]:
        """Calculate burn scores for all miners.

        Distributes the given burn_share across the NEW_BURNERS slots.

        Args:
            miners: List of all miner neuron information
            burn_share: Dynamic burn emission share to distribute (e.g., 0.87 for default,
                       or total burn emission - rental_share for rental price incentive)

        Returns:
            dict[str, float]: Mapping of miner hotkeys to burn scores
                - Burner nodes: Receive portion of burn_share
                - Non-burners: Not included in result (0 score)
        """
        burn_scores = self._calculate_new_burn_logic(miners, burn_share)

        # Log summary
        num_burners = len(burn_scores)
        total_distributed = sum(burn_scores.values())

        logger.info(
            _m(
                "Burn emission distributed across burner nodes",
                extra={
                    "burn_logic": "new",
                    "num_burners": num_burners,
                    "burn_share": burn_share,
                    "total_burn_emission": get_total_burn_emission(),
                    "total_distributed": total_distributed,
                    "burner_hotkeys": list(burn_scores.keys()),
                    "validation": f"distributed={total_distributed:.6f}, expected={burn_share:.6f}",
                },
            )
        )

        return burn_scores

    def _calculate_new_burn_logic(
        self,
        miners: list[bittensor.NeuronInfo],
        burn_share: float,
    ) -> dict[str, float]:
        """Calculate burn scores using new logic (slot-weighted distribution).

        Burners in NEW_BURNERS share burn_share by slot count. A UID listed
        multiple times receives a proportionally larger share — this lets a
        single UID absorb the emission of retired burners.

        Args:
            miners: List of all miner neuron information
            burn_share: Dynamic burn emission share to distribute

        Returns:
            dict[str, float]: Burner hotkeys to weighted burn scores
        """
        burn_scores = {}
        burner_slots = settings.NEW_BURNERS
        slot_weights = Counter(burner_slots)
        total_slots = len(burner_slots)
        burn_score_per_slot = burn_share / total_slots

        for miner in miners:
            slots = slot_weights.get(miner.uid, 0)
            if slots == 0:
                continue
            if not _burner_coldkey_matches(miner):
                continue

            score = burn_score_per_slot * slots
            burn_scores[miner.hotkey] = score

            logger.debug(
                _m(
                    "Miner assigned to burn pool (new logic)",
                    extra={
                        "miner_uid": miner.uid,
                        "miner_hotkey": miner.hotkey,
                        "burn_share": burn_share,
                        "total_slots": total_slots,
                        "miner_slots": slots,
                        "score": score,
                        "pool": "burn",
                        "logic": "new_slot_weighted_distribution",
                    },
                )
            )

        logger.info(
            _m(
                "New burn logic applied - slot-weighted distribution",
                extra={
                    "total_slots": total_slots,
                    "unique_burners": len(slot_weights),
                    "slot_weights": dict(slot_weights),
                    "score_per_slot": burn_score_per_slot,
                    "burn_share": burn_share,
                },
            )
        )

        return burn_scores

    def is_burner(self, miner_uid: int) -> bool:
        """Check if a miner UID is a burner node.

        Args:
            miner_uid: Miner's unique identifier

        Returns:
            bool: True if miner is a burner node
        """
        return miner_uid in settings.NEW_BURNERS

    def get_burn_share(self) -> float:
        """Get the total burn emission share.

        Returns:
            float: Total portion of emission allocated to burning (from shared config)
        """
        return get_total_burn_emission()

    def get_mining_share(self) -> float:
        """Get the total mining emission share.

        Returns:
            float: Total portion of emission allocated to mining (from shared config)
        """
        return 1 - get_total_burn_emission()

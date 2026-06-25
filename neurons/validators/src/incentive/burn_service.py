"""Burn emission distribution service.

This service handles the distribution of burn emission across burner nodes,
supporting both the old burner selection algorithm and the new equal distribution logic.
"""

import random
from collections import Counter

import bittensor

from core.config import settings
from core.utils import _m, get_logger
from services.const import BURNER_EMISSION, TOTAL_BURN_EMISSION

logger = get_logger(__name__)


class BurnService:
    """Service for calculating burn emission distribution across burner nodes.

    Supports two burn logic modes:
    - Old logic: One main burner gets slightly more, others share equally
    - New logic: All burners share burn emission equally
    """

    def calculate_burn_scores(
        self,
        miners: list[bittensor.NeuronInfo],
        burn_share: float,
        last_mechanism_step_block: int | None,
    ) -> dict[str, float]:
        """Calculate burn scores for all miners.

        Distributes the given burn_share across designated burner nodes based on
        the configured burn logic mode (old or new).

        Args:
            miners: List of all miner neuron information
            burn_share: Dynamic burn emission share to distribute (e.g., 0.87 for default,
                       or TOTAL_BURN_EMISSION - rental_share for rental price incentive)
            last_mechanism_step_block: Block number for randomization (old logic only)

        Returns:
            dict[str, float]: Mapping of miner hotkeys to burn scores
                - Burner nodes: Receive portion of burn_share
                - Non-burners: Not included in result (0 score)
        """
        burn_scores = {}

        if settings.ENABLE_NEW_BURN_LOGIC:
            # New burn logic: Equal distribution
            burn_scores = self._calculate_new_burn_logic(miners, burn_share)
        else:
            # Old burn logic: Main burner + others
            burn_scores = self._calculate_old_burn_logic(miners, burn_share, last_mechanism_step_block)

        # Log summary
        num_burners = len(burn_scores)
        total_distributed = sum(burn_scores.values())

        logger.info(
            _m(
                "Burn emission distributed across burner nodes",
                extra={
                    "burn_logic": "new" if settings.ENABLE_NEW_BURN_LOGIC else "old",
                    "num_burners": num_burners,
                    "burn_share": burn_share,
                    "total_burn_emission_const": TOTAL_BURN_EMISSION,
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

    def _calculate_old_burn_logic(
        self,
        miners: list[bittensor.NeuronInfo],
        burn_share: float,
        last_mechanism_step_block: int | None,
    ) -> dict[str, float]:
        """Calculate burn scores using old logic (main burner + others).

        One randomly selected main burner gets slightly more emission,
        other burners share the remainder equally.

        Args:
            miners: List of all miner neuron information
            burn_share: Dynamic burn emission share to distribute
            last_mechanism_step_block: Block number for random seed

        Returns:
            dict[str, float]: Burner hotkeys to burn scores
        """
        burn_scores = {}
        burners = settings.BURNERS

        # Select main burner using block-based randomization
        main_burner = random.Random(last_mechanism_step_block or 0).choice(burners)
        other_burners = [uid for uid in burners if uid != main_burner]

        # Calculate scores proportional to burn_share
        # Scale the original BURNER_EMISSION by the ratio of burn_share to TOTAL_BURN_EMISSION
        # This maintains the same distribution pattern but with dynamic burn_share
        burn_ratio = burn_share / TOTAL_BURN_EMISSION if TOTAL_BURN_EMISSION > 0 else 1
        scaled_burner_emission = BURNER_EMISSION * burn_ratio

        # Main burner gets: burn_share - (num_other_burners * scaled_burner_emission)
        # Other burners get: scaled_burner_emission each
        main_burner_score = burn_share - (len(burners) - 1) * scaled_burner_emission
        other_burner_score = scaled_burner_emission

        for miner in miners:
            if miner.uid == main_burner:
                burn_scores[miner.hotkey] = main_burner_score

                logger.debug(
                    _m(
                        "Miner assigned as main burner (old logic)",
                        extra={
                            "miner_uid": miner.uid,
                            "miner_hotkey": miner.hotkey,
                            "burn_share": burn_share,
                            "num_burners": len(burners),
                            "score": main_burner_score,
                            "pool": "burn_main",
                            "logic": "old_main_burner",
                            "block_seed": last_mechanism_step_block,
                            "burn_ratio": burn_ratio,
                        },
                    )
                )
            elif miner.uid in other_burners:
                burn_scores[miner.hotkey] = other_burner_score

                logger.debug(
                    _m(
                        "Miner assigned as other burner (old logic)",
                        extra={
                            "miner_uid": miner.uid,
                            "miner_hotkey": miner.hotkey,
                            "burn_share": burn_share,
                            "num_burners": len(burners),
                            "score": other_burner_score,
                            "pool": "burn_other",
                            "logic": "old_other_burner",
                        },
                    )
                )

        logger.info(
            _m(
                "Old burn logic applied - main burner selection",
                extra={
                    "total_burners": len(burners),
                    "burner_uids": list(burners),
                    "main_burner_uid": main_burner,
                    "main_burner_score": main_burner_score,
                    "other_burner_score": other_burner_score,
                    "burn_share": burn_share,
                    "block_seed": last_mechanism_step_block,
                    "burn_ratio": burn_ratio,
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
        if settings.ENABLE_NEW_BURN_LOGIC:
            return miner_uid in settings.NEW_BURNERS
        else:
            return miner_uid in settings.BURNERS

    def get_burn_share(self) -> float:
        """Get the total burn emission share.

        Returns:
            float: Total portion of emission allocated to burning (0.87)
        """
        return TOTAL_BURN_EMISSION

    def get_mining_share(self) -> float:
        """Get the total mining emission share.

        Returns:
            float: Total portion of emission allocated to mining (0.13)
        """
        return 1 - TOTAL_BURN_EMISSION

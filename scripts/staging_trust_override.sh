# Sourced by the executor, miner and watchtower docker_build.sh. A non-prod build that sets STAGING_TRUST_OWNER_SS58,
# STAGING_TRUST_MESSAGE and STAGING_TRUST_SIGNATURE gets them appended to its config_override, and then follows validator
# messages that staging owner key signed (datura.validator_trust.override_trust). All three or none; each value is
# checked for its exact form before it is written, because the override is Python source. Prod never calls this.
staging_trust_override() {
  local file=$1 ss58='[1-9A-HJ-NP-Za-km-z]{46,48}'
  if [[ -z "${STAGING_TRUST_OWNER_SS58:-}${STAGING_TRUST_MESSAGE:-}${STAGING_TRUST_SIGNATURE:-}" ]]; then
    return 0
  fi
  if ! [[ "${STAGING_TRUST_OWNER_SS58:-}" =~ ^${ss58}$ && "${STAGING_TRUST_MESSAGE:-}" =~ ^lium-validator:${ss58}:[0-9]{1,18}$ &&
          "${STAGING_TRUST_SIGNATURE:-}" =~ ^0x[0-9a-fA-F]{128}$ ]]; then
    echo "staging_trust_override: set STAGING_TRUST_OWNER_SS58, STAGING_TRUST_MESSAGE and STAGING_TRUST_SIGNATURE together, each well formed" >&2
    return 1
  fi
  printf '_STAGING_TRUST_OWNER_SS58 = "%s"\n_STAGING_TRUST_MESSAGE = {"message": "%s", "signature": "%s"}\n' \
    "$STAGING_TRUST_OWNER_SS58" "$STAGING_TRUST_MESSAGE" "$STAGING_TRUST_SIGNATURE" >> "$file"
  echo "staging_trust_override: $file follows messages signed by $STAGING_TRUST_OWNER_SS58 (${STAGING_TRUST_MESSAGE})"
}

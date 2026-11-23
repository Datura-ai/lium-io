# Sourced by the executor, miner and watchtower docker_build.sh. A non-prod build that sets STAGING_TRUST_OWNER_SS58,
# STAGING_TRUST_MESSAGE, STAGING_TRUST_SIGNATURE and STAGING_TRUST_API_URL gets them appended to its config_override, and
# then follows validator messages that staging owner key signed (datura.validator_trust.override_trust), read from
# <STAGING_TRUST_API_URL>/watchtower/validator-message: the backend that serves that owner's messages, whatever the
# host's own COMPUTE_REST_API_URL says. All four or none; each value is checked for its exact form before it is
# written, because the override is Python source. The message is a `test` network line (datura STAGING_NETWORK), so a
# signature made for it never verifies as a release line. Prod never calls this.
staging_trust_override() {
  local file=$1 ss58='[1-9A-HJ-NP-Za-km-z]{46,48}' LC_ALL=C   # C: the bracket classes match ASCII only
  if [[ -z "${STAGING_TRUST_OWNER_SS58:-}${STAGING_TRUST_MESSAGE:-}${STAGING_TRUST_SIGNATURE:-}${STAGING_TRUST_API_URL:-}" ]]; then
    return 0
  fi
  if ! [[ "${STAGING_TRUST_OWNER_SS58:-}" =~ ^${ss58}$ && "${STAGING_TRUST_MESSAGE:-}" =~ ^lium-validator:test:${ss58}:[0-9]{1,18}$ &&
          "${STAGING_TRUST_SIGNATURE:-}" =~ ^0x[0-9a-fA-F]{128}$ && "${STAGING_TRUST_API_URL:-}" =~ ^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._-]+)*$ ]]; then
    echo "staging_trust_override: set STAGING_TRUST_OWNER_SS58, STAGING_TRUST_MESSAGE, STAGING_TRUST_SIGNATURE and STAGING_TRUST_API_URL together, each well formed" >&2
    return 1
  fi
  printf '_STAGING_TRUST_OWNER_SS58 = "%s"\n_STAGING_TRUST_MESSAGE = {"message": "%s", "signature": "%s"}\n_STAGING_TRUST_API_URL = "%s"\n' \
    "$STAGING_TRUST_OWNER_SS58" "$STAGING_TRUST_MESSAGE" "$STAGING_TRUST_SIGNATURE" "$STAGING_TRUST_API_URL" >> "$file"
  echo "staging_trust_override: $file follows messages signed by $STAGING_TRUST_OWNER_SS58 (${STAGING_TRUST_MESSAGE}) from $STAGING_TRUST_API_URL"
}

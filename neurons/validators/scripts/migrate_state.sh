#!/usr/bin/env bash
# Dump validator Postgres + Redis state and ship it to another host over SSH.

set -euo pipefail

# Configurable inputs (override via env vars).
ENV_FILE="${ENV_FILE:-.env}"
PG_CONTAINER="${PG_CONTAINER:-validator-db-1}"
REDIS_CONTAINER="${REDIS_CONTAINER:-validator-redis-1}"
DEST_USER="${DEST_USER:-ubuntu}"
DEST_HOST="${DEST_HOST:?Set DEST_HOST to the target EC2 hostname/IP}"
DEST_PATH="${DEST_PATH:-/tmp/validator-state}"
SSH_KEY="${SSH_KEY:-}"               # optional: path to SSH private key
ARCHIVE_NAME="${ARCHIVE_NAME:-validator_state_$(date +%Y%m%d_%H%M%S).tar.gz}"
WORK_DIR="${WORK_DIR:-/tmp/validator-state-$(date +%s)}"

log() {
  echo "[$(date +"%F %T")] $*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

timer_run() {
  local label="$1"; shift
  local start end
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  log "$label completed in $((end - start))s"
}

main() {
  require_file "$ENV_FILE"

  mkdir -p "$WORK_DIR"
  trap 'rm -rf "$WORK_DIR"' EXIT

  # Load DB creds from the .env file so pg_dump can auth.
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a

  if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_DB:-}" || -z "${POSTGRES_PASSWORD:-}" ]]; then
    echo "POSTGRES_USER/POSTGRES_DB/POSTGRES_PASSWORD must be set in $ENV_FILE" >&2
    exit 1
  fi

  log "Dumping Postgres from container '$PG_CONTAINER'"
  log "  Database name: $POSTGRES_DB"
  log "  Listing all databases:"
  docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database WHERE datistemplate = false;" | sed 's/^/    /'
  PG_SIZE=$(docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" | xargs)
  PG_TABLES=$(docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" | xargs)
  log "  Database size: $PG_SIZE, Tables: $PG_TABLES"
  timer_run "Postgres dump" docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_CONTAINER" \
    pg_dump -Fc -U "$POSTGRES_USER" "$POSTGRES_DB" \
    > "$WORK_DIR/postgres.dump"
  log "  Dump file size: $(du -h "$WORK_DIR/postgres.dump" | cut -f1)"

  log "Dumping Redis from container '$REDIS_CONTAINER'"
  REDIS_KEYS=$(docker exec "$REDIS_CONTAINER" redis-cli dbsize | tr -d '\r')
  REDIS_MEMORY=$(docker exec "$REDIS_CONTAINER" redis-cli info memory | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
  log "  Keys: $REDIS_KEYS, Memory: $REDIS_MEMORY"
  # Use SAVE to create dump.rdb, then copy it out
  timer_run "Redis dump" docker exec "$REDIS_CONTAINER" redis-cli SAVE
  docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$WORK_DIR/dump.rdb"
  log "  Dump file size: $(du -h "$WORK_DIR/dump.rdb" | cut -f1)"

  log "Packaging dump files"
  tar -czf "$WORK_DIR/$ARCHIVE_NAME" -C "$WORK_DIR" postgres.dump dump.rdb

  log "Preparing remote path $DEST_USER@$DEST_HOST:$DEST_PATH"
  ssh_opts=()
  [[ -n "$SSH_KEY" ]] && ssh_opts+=(-i "$SSH_KEY")
  ssh "${ssh_opts[@]}" -o StrictHostKeyChecking=no "$DEST_USER@$DEST_HOST" "mkdir -p '$DEST_PATH'"

  log "Transferring archive to $DEST_HOST (this is the timed step you care about)"
  timer_run "SCP transfer" scp "${ssh_opts[@]}" -o StrictHostKeyChecking=no \
    "$WORK_DIR/$ARCHIVE_NAME" "$DEST_USER@$DEST_HOST:$DEST_PATH/"

  log "Done. Archive stored at $DEST_HOST:$DEST_PATH/$ARCHIVE_NAME"
}

main "$@"

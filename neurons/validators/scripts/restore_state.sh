#!/usr/bin/env bash
# Restore validator Postgres + Redis state from an archive produced by migrate_state.sh.

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.app.yml}"
ENV_FILE="${ENV_FILE:-.env}"
PG_SERVICE="${PG_SERVICE:-db}"
REDIS_SERVICE="${REDIS_SERVICE:-redis}"
ARCHIVE_PATH="${ARCHIVE_PATH:?Set ARCHIVE_PATH to the tar.gz from migrate_state.sh}"
# Services to stop before restore (set APP_SERVICES=\"\" to skip).
DEFAULT_APP_SERVICES="validator connector"
# Respect explicit empty APP_SERVICES; only use default when unset.
if [[ ! ${APP_SERVICES+x} ]]; then
  APP_SERVICES="$DEFAULT_APP_SERVICES"
fi
WORK_DIR="${WORK_DIR:-/tmp/validator-restore-$(date +%s)}"

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
  require_file "$COMPOSE_FILE"
  require_file "$ENV_FILE"
  require_file "$ARCHIVE_PATH"

  mkdir -p "$WORK_DIR"
  trap 'rm -rf "$WORK_DIR"' EXIT

  log "Extracting archive $ARCHIVE_PATH"
  tar -xzf "$ARCHIVE_PATH" -C "$WORK_DIR"

  if [[ ! -f "$WORK_DIR/postgres.dump" || ! -f "$WORK_DIR/dump.rdb" ]]; then
    echo "Archive missing postgres.dump or dump.rdb" >&2
    exit 1
  fi

  # Load DB creds.
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
  if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_DB:-}" || -z "${POSTGRES_PASSWORD:-}" ]]; then
    echo "POSTGRES_USER/POSTGRES_DB/POSTGRES_PASSWORD must be set in $ENV_FILE" >&2
    exit 1
  fi

  # Stop app containers to avoid open connections during restore.
  if [[ -n "$APP_SERVICES" ]]; then
    log "Stopping app services: $APP_SERVICES"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop $APP_SERVICES
  fi

  log "Restoring Postgres into service '$PG_SERVICE'"
  timer_run "Postgres restore" bash -c \
    "docker compose -f \"$COMPOSE_FILE\" --env-file \"$ENV_FILE\" exec -T -e PGPASSWORD=\"$POSTGRES_PASSWORD\" \"$PG_SERVICE\" pg_restore --clean --if-exists -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" < \"$WORK_DIR/postgres.dump\""

  log "Verifying Postgres restore:"
  PG_SIZE=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" | xargs)
  PG_TABLES=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema');" | xargs)
  log "  Database size: $PG_SIZE"
  log "  Tables restored: $PG_TABLES"
  log "  Table list:"
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T -e PGPASSWORD="$POSTGRES_PASSWORD" "$PG_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT schemaname || '.' || tablename || ' (' || pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) || ')' FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY schemaname, tablename;" | sed 's/^/    /'

  log "Restoring Redis into service '$REDIS_SERVICE'"
  # Grab the container id before stopping so we can copy while it is stopped.
  REDIS_CID=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q "$REDIS_SERVICE")
  if [[ -z "$REDIS_CID" ]]; then
    REDIS_CID=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -aq "$REDIS_SERVICE")
  fi
  if [[ -z "$REDIS_CID" ]]; then
    echo "Could not find container id for $REDIS_SERVICE" >&2
    exit 1
  fi
  REDIS_IMAGE=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" images -q "$REDIS_SERVICE" | head -n1)
  if [[ -z "$REDIS_IMAGE" ]]; then
    echo "Could not determine image for $REDIS_SERVICE" >&2
    exit 1
  fi
  # Find the volume mounted at /data (compose maps redis_data:/data).
  REDIS_VOLUME=$(docker inspect "$REDIS_CID" -f "{{ range .Mounts }}{{ if eq .Destination \"/data\" }}{{ .Name }}{{ end }}{{ end }}")
  if [[ -z "$REDIS_VOLUME" ]]; then
    echo "Could not find volume mounted at /data" >&2
    exit 1
  fi
  # Stop redis so it does not rewrite the RDB we are about to place.
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" stop "$REDIS_SERVICE"
  # Clear AOF files and replace dump.rdb directly in the volume.
  # This ensures Redis loads from the RDB snapshot on startup instead of AOF.
  docker run --rm \
    -v "$REDIS_VOLUME:/data" \
    -v "$WORK_DIR/dump.rdb":/tmp/dump.rdb:ro \
    alpine:3 \
    sh -c "rm -f /data/appendonly.aof* /data/appendonlydir/* /data/dump.rdb && cp /tmp/dump.rdb /data/dump.rdb && chmod 644 /data/dump.rdb"
  # Start redis again so it loads the new RDB.
  timer_run "Redis start" docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" start "$REDIS_SERVICE"

  log "Verifying Redis restore:"
  REDIS_KEYS=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T "$REDIS_SERVICE" redis-cli dbsize | tr -d '\r')
  REDIS_MEMORY=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T "$REDIS_SERVICE" redis-cli info memory | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
  REDIS_AOF=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T "$REDIS_SERVICE" redis-cli config get appendonly | tail -1 | tr -d '\r')
  REDIS_LAST_SAVE=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T "$REDIS_SERVICE" redis-cli lastsave | tr -d '\r')
  log "  Keys restored: $REDIS_KEYS"
  log "  Memory usage: $REDIS_MEMORY"
  log "  AOF enabled: $REDIS_AOF"
  log "  Last RDB save timestamp: $REDIS_LAST_SAVE"

  if [[ -n "$APP_SERVICES" ]]; then
    log "Starting app services: $APP_SERVICES"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" start $APP_SERVICES
  fi

  log "Restore complete."
}

main "$@"

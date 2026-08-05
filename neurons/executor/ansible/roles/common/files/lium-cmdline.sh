#!/usr/bin/env bash
# All kernel-command-line token math, in one testable place.
#
# WHY LAST-OCCURRENCE-WINS AND NOT SET MEMBERSHIP
# ----------------------------------------------
# /etc/grub.d/10_linux emits the default entry as:
#
#     linux ... ${GRUB_CMDLINE_LINUX} ${GRUB_CMDLINE_LINUX_DEFAULT}
#
# so tokens we append to GRUB_CMDLINE_LINUX_DEFAULT land LAST, and the kernel
# resolves duplicate parameters last-occurrence-wins per key. That ordering is
# the entire reason appending is safe.
#
# A bare set-membership check cannot see a conflicting duplicate: a host with a
# stale `vfio_iommu_type1.dma_entry_limit=65535` already present would report
# "our 16777216 is there" and pass, while which value the kernel honours depends
# on ordering. So every assertion here resolves a key to its winning value.
#
# WHY THE FILES ARE SOURCED RATHER THAN GREPPED
# ---------------------------------------------
# The drop-in writes `GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT:-} ..."`.
# Only shell expansion gives its effective value, and sourcing /etc/default/grub
# plus /etc/default/grub.d/* in order is exactly what grub-mkconfig itself does.
# Grepping would report the literal text and miss a clobber.

set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  lium-cmdline.sh analyze --required "tok ..." [--grub-default F] [--grub-dropin-dir D]
                          [--exclude-dropin F] [--proc-cmdline F] [--grubcfg F]
  lium-cmdline.sh resolve --line "..." --key KEY
  lium-cmdline.sh vfio-ids --line "..."

`analyze` emits one JSON object. Token forms understood: `key=value` and bare
flags such as `nohibernate`.
USAGE
}

json_escape() {
  printf '%s' "$1" | LC_ALL=C sed \
    -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r/\\r/g' \
    -e ':a' -e 'N' -e '$!ba' -e 's/\n/\\n/g'
}

json_array_of_strings() {
  local first=1 item out=""
  for item in "$@"; do
    if [ "$first" -eq 1 ]; then first=0; else out="${out}, "; fi
    out="${out}\"$(json_escape "$item")\""
  done
  printf '[%s]' "$out"
}

# Winning value of KEY in LINE, last occurrence wins.
# Prints the value for `key=value`; prints the key itself for a bare flag;
# prints nothing and returns 1 when absent.
resolve_key() {
  local line="$1" key="$2" tok found=""
  for tok in $line; do
    case "$tok" in
      "$key"=*) found="${tok#*=}" ;;
      "$key") found="$key" ;;
    esac
  done
  [ -n "$found" ] || return 1
  printf '%s' "$found"
}

# How many times KEY appears in LINE.
count_key() {
  local line="$1" key="$2" tok n=0
  for tok in $line; do
    case "$tok" in
      "$key"=*|"$key") n=$((n + 1)) ;;
    esac
  done
  printf '%s' "$n"
}

# How many DISTINCT values KEY takes in LINE.
distinct_values() {
  local line="$1" key="$2" tok
  {
    for tok in $line; do
      case "$tok" in
        "$key"=*) printf '%s\n' "${tok#*=}" ;;
        "$key") printf '%s\n' "$key" ;;
      esac
    done
  } | sort -u | grep -c . || true
}

key_of_token() {
  case "$1" in
    *=*) printf '%s' "${1%%=*}" ;;
    *) printf '%s' "$1" ;;
  esac
}

# Comma-separated vfio-pci.ids= value of LINE, resolved last-occurrence-wins.
vfio_ids_of_line() {
  resolve_key "$1" "vfio-pci.ids" 2>/dev/null || printf ''
}

read_grub_vars() {
  # Sourced in a subshell so nothing leaks back into this script.
  #
  # `exclude` names one drop-in to skip. It is how the caller asks "what did the
  # PROVIDER have here, ignoring our own contribution" — the question that
  # identifies the au2 shape, and the only form of it that stays stable across
  # re-runs. Without it, our own append masks the very thing we want to report.
  local default_file="$1" dropin_dir="$2" exclude="${3:-}" f
  (
    set +u
    GRUB_CMDLINE_LINUX=""
    GRUB_CMDLINE_LINUX_DEFAULT=""
    if [ -n "$default_file" ] && [ -r "$default_file" ]; then
      # shellcheck source=/dev/null
      . "$default_file"
    fi
    if [ -n "$dropin_dir" ] && [ -d "$dropin_dir" ]; then
      for f in "$dropin_dir"/*; do
        [ -r "$f" ] || continue
        if [ -n "$exclude" ] && [ "$f" = "$exclude" ]; then
          continue
        fi
        # shellcheck source=/dev/null
        . "$f"
      done
    fi
    printf 'LINUX\t%s\n' "$GRUB_CMDLINE_LINUX"
    printf 'DEFAULT\t%s\n' "$GRUB_CMDLINE_LINUX_DEFAULT"
  )
}

# First `linux`/`linux16` line of a generated grub.cfg — the default entry.
grubcfg_linux_line() {
  local file="$1"
  [ -r "$file" ] || return 1
  sed -n -E 's/^[[:space:]]*linux(16|efi)?[[:space:]]+//p' "$file" | head -n 1
}

cmd_analyze() {
  local required="" grub_default="" dropin_dir="" proc_cmdline="" grubcfg="" exclude_dropin=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --required) required="${2:?--required needs a value}"; shift ;;
      --grub-default) grub_default="${2:?}"; shift ;;
      --grub-dropin-dir) dropin_dir="${2:?}"; shift ;;
      --exclude-dropin) exclude_dropin="${2:?}"; shift ;;
      --proc-cmdline) proc_cmdline="${2:?}"; shift ;;
      --grubcfg) grubcfg="${2:?}"; shift ;;
      *) printf 'lium-cmdline.sh analyze: unknown argument: %s\n' "$1" >&2; return 64 ;;
    esac
    shift
  done

  local linux_line="" default_line="" line key value
  while IFS=$'\t' read -r key value; do
    case "$key" in
      LINUX) linux_line="$value" ;;
      DEFAULT) default_line="$value" ;;
    esac
  done < <(read_grub_vars "$grub_default" "$dropin_dir" "$exclude_dropin")

  # The union, in 10_linux's own order: LINUX first, DEFAULT last, so our
  # appended tokens win.
  local union_line="${linux_line} ${default_line}"

  local proc_line=""
  if [ -n "$proc_cmdline" ] && [ -r "$proc_cmdline" ]; then
    proc_line="$(tr -d '\n' <"$proc_cmdline")"
  fi

  local cfg_line=""
  if [ -n "$grubcfg" ]; then
    cfg_line="$(grubcfg_linux_line "$grubcfg" || printf '')"
  fi

  local missing=() drift=() cfg_missing=() only_in_linux=() conflicting=()
  local tok tk want got

  for tok in $required; do
    tk="$(key_of_token "$tok")"
    want="$(key_of_token "$tok")"
    case "$tok" in
      *=*) want="${tok#*=}" ;;
    esac

    # Does the token win in the merged GRUB configuration?
    if got="$(resolve_key "$union_line" "$tk")"; then
      [ "$got" = "$want" ] || missing+=("$tok")
    else
      missing+=("$tok")
    fi

    # Present ONLY in GRUB_CMDLINE_LINUX — the au2 shape, worth reporting even
    # when it resolves correctly.
    if resolve_key "$linux_line" "$tk" >/dev/null 2>&1; then
      if ! resolve_key "$default_line" "$tk" >/dev/null 2>&1; then
        only_in_linux+=("$tok")
      fi
    fi

    # A key that appears more than once with DIFFERENT values is a conflict
    # worth naming, even though ordering resolves it in our favour — the
    # provider should clean the stale one up rather than rely on ordering
    # forever. A duplicate with the SAME value is not a conflict: the drop-in
    # always carries the full required set, so re-stating a value the provider
    # already had is expected and must not generate noise.
    if [ "$(count_key "$union_line" "$tk")" -gt 1 ]; then
      if [ "$(distinct_values "$union_line" "$tk")" -gt 1 ]; then
        conflicting+=("$tk")
      fi
    fi

    # Drift: what the running kernel actually has.
    if [ -n "$proc_line" ]; then
      if got="$(resolve_key "$proc_line" "$tk")"; then
        [ "$got" = "$want" ] || drift+=("$tok")
      else
        drift+=("$tok")
      fi
    fi

    # The decisive pre-reboot check: does it win in the GENERATED grub.cfg?
    if [ -n "$cfg_line" ]; then
      if got="$(resolve_key "$cfg_line" "$tk")"; then
        [ "$got" = "$want" ] || cfg_missing+=("$tok")
      else
        cfg_missing+=("$tok")
      fi
    fi
  done

  # Pre-existing vfio-pci.ids= from every source. The role asserts the freshly
  # discovered set is a SUPERSET of this union: narrowing the list unbinds a GPU
  # on the next boot, which is how a hot-removed device silently costs you a GPU.
  #
  # The GRUB-file subset is reported separately because the two answer different
  # questions. The union answers "was this id ever configured?" — the right basis
  # for a refusal. The GRUB-file subset answers "what will the NEXT boot get if we
  # write our drop-in now?" — the right basis for deciding which ids we must carry
  # forward so our own append does not shorten the list. An id that is only in
  # /proc/cmdline is already gone from the boot configuration; restoring it would
  # resurrect a setting the provider removed.
  # Iterated by ORIGIN, not by value: a host whose /proc/cmdline happens to be
  # byte-identical to GRUB_CMDLINE_LINUX is ordinary, and comparing the strings
  # to decide which source a line came from would misfile its ids.
  local existing_ids=() grub_existing_ids=() origin src
  for origin in grub:"$linux_line" grub:"$default_line" proc:"$proc_line"; do
    src="${origin#*:}"
    [ -n "$src" ] || continue
    local ids
    ids="$(vfio_ids_of_line "$src")"
    [ -n "$ids" ] || continue
    local IFS=','
    for id in $ids; do
      [ -n "$id" ] || continue
      existing_ids+=("$id")
      [ "${origin%%:*}" = proc ] || grub_existing_ids+=("$id")
    done
  done
  if [ "${#existing_ids[@]}" -gt 0 ]; then
    mapfile -t existing_ids < <(printf '%s\n' "${existing_ids[@]}" | sort -u)
  fi
  if [ "${#grub_existing_ids[@]}" -gt 0 ]; then
    mapfile -t grub_existing_ids < <(printf '%s\n' "${grub_existing_ids[@]}" | sort -u)
  fi
  if [ "${#conflicting[@]}" -gt 0 ]; then
    mapfile -t conflicting < <(printf '%s\n' "${conflicting[@]}" | sort -u)
  fi

  cat <<JSON
{
  "grub_cmdline_linux": "$(json_escape "$linux_line")",
  "grub_cmdline_linux_default": "$(json_escape "$default_line")",
  "union_line": "$(json_escape "$union_line")",
  "proc_cmdline": "$(json_escape "$proc_line")",
  "grubcfg_linux_line": "$(json_escape "$cfg_line")",
  "grubcfg_present": $( [ -n "$cfg_line" ] && printf 'true' || printf 'false' ),
  "missing_tokens": $(json_array_of_strings ${missing[@]+"${missing[@]}"}),
  "grubcfg_missing_tokens": $(json_array_of_strings ${cfg_missing[@]+"${cfg_missing[@]}"}),
  "drift_tokens": $(json_array_of_strings ${drift[@]+"${drift[@]}"}),
  "found_only_in_grub_cmdline_linux": $(json_array_of_strings ${only_in_linux[@]+"${only_in_linux[@]}"}),
  "conflicting_keys": $(json_array_of_strings ${conflicting[@]+"${conflicting[@]}"}),
  "existing_vfio_ids": $(json_array_of_strings ${existing_ids[@]+"${existing_ids[@]}"}),
  "grub_existing_vfio_ids": $(json_array_of_strings ${grub_existing_ids[@]+"${grub_existing_ids[@]}"})
}
JSON
}

cmd_resolve() {
  local line="" key=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --line) line="${2:?}"; shift ;;
      --key) key="${2:?}"; shift ;;
      *) printf 'lium-cmdline.sh resolve: unknown argument: %s\n' "$1" >&2; return 64 ;;
    esac
    shift
  done
  resolve_key "$line" "$key" || return 1
  printf '\n'
}

cmd_vfio_ids() {
  local line=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --line) line="${2:?}"; shift ;;
      *) printf 'lium-cmdline.sh vfio-ids: unknown argument: %s\n' "$1" >&2; return 64 ;;
    esac
    shift
  done
  printf '%s\n' "$(vfio_ids_of_line "$line")"
}

case "${1:-}" in
  analyze) shift; cmd_analyze "$@" ;;
  resolve) shift; cmd_resolve "$@" ;;
  vfio-ids) shift; cmd_vfio_ids "$@" ;;
  -h|--help|"") usage ;;
  *) printf 'lium-cmdline.sh: unknown subcommand: %s\n' "$1" >&2; usage >&2; exit 64 ;;
esac

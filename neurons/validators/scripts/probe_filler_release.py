#!/usr/bin/env python3
"""Read-only experiment CLI; this does not stop a filler or authorize a rental.

Run on the executor host with access to its /proc and cgroup v2 mount:
  PYTHONPATH=src python scripts/probe_filler_release.py capture FULL_ID --gpu GPU-UUID > snapshot.json
  PYTHONPATH=src python scripts/probe_filler_release.py observe snapshot.json
The JSON snapshot contains identifiers only, never inspect's environment/secrets.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from services.filler_release import (
    FillerReleaseSnapshot,
    capture_filler_release,
    observe_filler_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture')
    capture.add_argument('container_id')
    capture.add_argument('--gpu', action='append', required=True)
    observe = commands.add_parser('observe')
    observe.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        result = capture_filler_release(args.container_id, tuple(args.gpu))
    else:
        snapshot = FillerReleaseSnapshot(**json.loads(args.snapshot.read_text()))
        result = observe_filler_release(snapshot)
    print(json.dumps(asdict(result)))


if __name__ == '__main__':
    main()

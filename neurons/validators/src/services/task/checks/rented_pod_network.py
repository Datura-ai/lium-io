from __future__ import annotations

import asyncio
import ipaddress
import shlex

import asyncssh

from core.config import settings

from ..messages import RentedPodNetworkMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

READ_TIMEOUT_SECONDS = 30
CONNTRACK_TABLE = "/proc/1/net/nf_conntrack"
TOP_PORTS = 5
_MAX_LINES = 500
_INSPECT_FORMAT = "{{.Name}}{{range .NetworkSettings.Networks}} {{.IPAddress}}{{end}}"
# shortcut: IPv4 only (conntrack prints IPv6 unabbreviated); add IPv6 when pods get IPv6 bridges
# Per source address: S <src> <flows> <unreplied flows> <destinations> <unreplied destinations>,
# then P <src> <port> <unreplied flows> for the busier unreplied destination ports.
_AWK = (
    'BEGIN{n=split(ips,a," ");for(i=1;i<=n;i++)w[a[i]]=1}'
    '{s="";d="";p="";u=0;for(i=1;i<=NF;i++){f=$i;'
    'if(s==""&&substr(f,1,4)=="src=")s=substr(f,5);'
    'else if(d==""&&substr(f,1,4)=="dst=")d=substr(f,5);'
    'else if(p==""&&substr(f,1,6)=="dport=")p=substr(f,7);'
    'else if(f=="[UNREPLIED]")u=1}'
    "if(!(s in w))next;t[s]++;if(!((s,d) in sd)){sd[s,d]=1;nd[s]++}"
    'if(u){ur[s]++;if(!((s,d) in ud)){ud[s,d]=1;nud[s]++};if(p!="")up[s,p]++}}'
    'END{for(s in t)print "S",s,t[s],ur[s]+0,nd[s]+0,nud[s]+0;'
    'for(k in up)if(up[k]>=10){split(k,b,SUBSEP);print "P",b[1],b[2],up[k]}}'
)


def read_command(addresses: list[str], table: str = CONNTRACK_TABLE) -> str:
    """One host read of the host's connection tracking table, summarised per rented container address."""
    return (
        f"[ -r {table} ] || exit 3; "
        f"awk -v ips={shlex.quote(' '.join(addresses))} {shlex.quote(_AWK)} {table} | head -n {_MAX_LINES}"
    )


def parse_summary(stdout: str, names_by_address: dict[str, str]) -> list[dict]:
    pods = {
        name: {
            "container": name,
            "flows": 0,
            "unreplied_flows": 0,
            "destinations": 0,
            "unreplied_destinations": 0,
            "top_unreplied_ports": [],
        }
        for name in names_by_address.values()
    }
    ports: dict[str, list[tuple[int, str]]] = {}
    for line in stdout.splitlines():
        parts = line.split()
        name = names_by_address.get(parts[1]) if len(parts) > 1 else None
        if name is None:
            continue
        if parts[0] == "S" and len(parts) == 6 and all(x.isdigit() for x in parts[2:]):
            pod = pods[name]
            # a container on several networks adds them up
            for key, value in zip(
                ("flows", "unreplied_flows", "destinations", "unreplied_destinations"), parts[2:]
            ):
                pod[key] += int(value)
        elif parts[0] == "P" and len(parts) == 4 and parts[3].isdigit():
            ports.setdefault(name, []).append((int(parts[3]), parts[2]))
    for name, counts in ports.items():
        pods[name]["top_unreplied_ports"] = [
            {"port": port, "flows": n} for n, port in sorted(counts, reverse=True)[:TOP_PORTS]
        ]
    return list(pods.values())


class RentedPodNetworkCheck:
    """Record a per-pod summary of the rented containers' tracked connections, read on the host.

    Observation only: never fails, never scores; a high count is logged as a warning for review.
    """

    check_id = "host.observe.rented_pod_network"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        names = running_rented_container_names(ctx)
        if not names:
            return CheckResult(
                passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id)
            )
        try:
            names_by_address = await self._addresses(ctx.ssh, names)
            if not names_by_address:
                return self._unread(
                    ctx, reason="no rented container address", details={"pods": names}
                )
            result = await asyncio.wait_for(
                ctx.ssh.run(read_command(sorted(names_by_address)), check=False),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            return self._unread(ctx, reason="ssh error", details={"error": repr(exc)})
        if result.exit_status == 3:
            return self._unread(ctx, reason="no connection table on this host")

        pods = parse_summary(result.stdout or "", names_by_address)
        what = {"executor_uuid": ctx.executor.uuid, "pods": pods}
        limit = settings.RENTED_POD_OUTBOUND_FANOUT_DESTINATIONS
        template = (
            Msg.FANOUT if any(p["unreplied_destinations"] >= limit for p in pods) else Msg.SUMMARY
        )
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )

    @staticmethod
    async def _addresses(ssh, names: list[str]) -> dict[str, str]:
        # docker inspect exits 1 when one name is missing but still prints the others
        res = await asyncio.wait_for(
            ssh.run(
                f"/usr/bin/docker inspect --format {shlex.quote(_INSPECT_FORMAT)} "
                + " ".join(shlex.quote(n) for n in names),
                check=False,
            ),
            timeout=READ_TIMEOUT_SECONDS,
        )
        found = {}
        for line in (getattr(res, "stdout", "") or "").splitlines():
            name, *addresses = line.split() or [""]
            for address in addresses:
                try:
                    found[str(ipaddress.ip_address(address))] = name.lstrip("/")
                except ValueError:  # "<no value>" from an older docker
                    continue
        return found

    def _unread(self, ctx: Context, *, reason: str, details: dict | None = None) -> CheckResult:
        event = render_message(
            Msg.UNREAD,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(details or {})},
        )
        return CheckResult(passed=True, event=event)

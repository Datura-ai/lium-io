from __future__ import annotations

import asyncio
import ipaddress
import logging
import shlex

import asyncssh

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentedPodNetworkMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

logger = logging.getLogger(__name__)

READ_TIMEOUT_SECONDS = 30
CONNTRACK_TABLE = "/proc/1/net/nf_conntrack"
TOP_PORTS = 5
_MAX_KEYS = 20000  # bounds the awk state however large the host table is
_INSPECT_FORMAT = (
    "{{.Name}} {{.State.Pid}}{{range .NetworkSettings.Networks}} {{.IPAddress}}{{end}}"
)
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
    "if(!(s in w))next;t[s]++;"
    # k counts stored keys; past the cap m nothing new is stored and x marks the summary truncated
    "if(!((s,d) in sd)){if(k<m){k++;sd[s,d]=1;nd[s]++}else x=1}"
    "if(u){ur[s]++;if(!((s,d) in ud)){if(k<m){k++;ud[s,d]=1;nud[s]++}else x=1}"
    'if(p!=""){if((s,p) in up)up[s,p]++;else if(k<m){k++;up[s,p]=1}else x=1}}}'
    'END{if(x)print "T";for(s in t)print "S",s,t[s],ur[s]+0,nd[s]+0,nud[s]+0;'
    'for(k in up)if(up[k]>=10){split(k,b,SUBSEP);print "P",b[1],b[2],up[k]}}'
)
# Fallback where the host exposes no conntrack table: the sockets in the pod's own network namespace,
# read from the host through /proc/<container init pid>/net (read-only, nothing runs in the pod). A
# snapshot of open sockets, not tracked flows: SYN_SENT counts as unreplied; raw and packet sockets are
# counted on their own since a raw-socket sender opens no TCP socket per target.
# Per pid: S <pid> <sockets> <syn_sent> <destinations> <syn_sent destinations> <raw sockets>,
# then P <pid> <port> <syn_sent> for the busier SYN_SENT destination ports.
_SOCKETS_AWK = (
    "function h(s,i,n){n=0;for(i=1;i<=length(s);i++)"
    'n=n*16+index("0123456789ABCDEF",toupper(substr(s,i,1)))-1;return n}'
    "FNR==1{next}"
    "FILENAME~/(raw6?|packet)$/{r++;next}"
    '{split($3,a,":");if(a[1]~/^0+$/)next;t++;'  # remote 0 = listening or unconnected
    "if(!(a[1] in d)){if(k<m){k++;d[a[1]]=1;nd++}else x=1}"
    'if($4=="02"&&FILENAME~/tcp6?$/){u++;if(!(a[1] in ud)){if(k<m){k++;ud[a[1]]=1;nud++}else x=1}'
    "q=h(a[2]);if(q in up)up[q]++;else if(k<m){k++;up[q]=1}else x=1}}"
    'END{if(x)print "T";print "S",p,t+0,u+0,nd+0,nud+0,r+0;'
    'for(q in up)if(up[q]>=10)print "P",p,q,up[q]}'
)
_SOCKET_TABLES = "tcp tcp6 udp udp6 raw raw6 packet"


def read_command(addresses: list[str], table: str = CONNTRACK_TABLE) -> str:
    """One host read of the host's connection tracking table, summarised per rented container address."""
    return (
        f"[ -r {table} ] || exit 3; "
        f"awk -v m={_MAX_KEYS} -v ips={shlex.quote(' '.join(addresses))} {shlex.quote(_AWK)} {table}"
    )


def socket_read_command(pids: list[int], proc: str = "/proc") -> str:
    """One host read of each rented container's own socket tables, summarised per container init pid."""
    return (
        f"e=0; for p in {' '.join(str(int(p)) for p in pids)}; do f=; for n in {_SOCKET_TABLES}; do "
        f'[ -r {proc}/$p/net/$n ] && f="$f {proc}/$p/net/$n"; done; '
        f'[ -n "$f" ] && {{ awk -v p=$p -v m={_MAX_KEYS} {shlex.quote(_SOCKETS_AWK)} $f || e=1; }}; '
        "done; exit $e"
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
        if parts[0] == "S" and len(parts) in (6, 7) and all(x.isdigit() for x in parts[2:]):
            pod = pods[name]
            # a container on several networks adds them up
            for key, value in zip(
                ("flows", "unreplied_flows", "destinations", "unreplied_destinations"), parts[2:6]
            ):
                pod[key] += int(value)
            if len(parts) == 7:  # socket fallback only
                pod["raw_sockets"] = pod.get("raw_sockets", 0) + int(parts[6])
        elif parts[0] == "P" and len(parts) == 4 and parts[3].isdigit():
            ports.setdefault(name, []).append((int(parts[3]), parts[2]))
    for name, counts in ports.items():
        pods[name]["top_unreplied_ports"] = [
            {"port": port, "flows": n} for n, port in sorted(counts, reverse=True)[:TOP_PORTS]
        ]
    return list(pods.values())


class RentedPodNetworkCheck:
    """Record a per-pod summary of the rented containers' tracked connections, read on the host;
    where the host exposes no connection table, of the sockets open in each pod's network namespace.

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
        # /proc/1 is the host's init only when the executor shares the host PID namespace
        if (ctx.state.specs or {}).get("executor_pid_mode") != "host":
            return self._unread(ctx, reason="executor not in the host PID namespace")
        try:
            names_by_address, pid_by_name = await self._addresses(ctx.ssh, names)
            if not names_by_address:
                return self._unread(
                    ctx, reason="no rented container address", details={"pods": names}
                )
            result = await asyncio.wait_for(
                ctx.ssh.run(read_command(sorted(names_by_address)), check=False),
                timeout=READ_TIMEOUT_SECONDS,
            )
            names_by_key, source = names_by_address, "conntrack"
            if result.exit_status == 3:
                # only bridged pods: a host-network pod's namespace is the host's own
                names_by_key = {
                    str(pid): name
                    for name, pid in pid_by_name.items()
                    if pid > 0 and name in names_by_address.values()
                }
                if not names_by_key:
                    return self._unread(ctx, reason="no connection table on this host")
                source = "sockets"
                result = await asyncio.wait_for(
                    ctx.ssh.run(socket_read_command(sorted(map(int, names_by_key))), check=False),
                    timeout=READ_TIMEOUT_SECONDS,
                )
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            return self._unread(ctx, reason="ssh error", details={"error": repr(exc)})
        lines = (result.stdout or "").splitlines()
        if (
            source == "sockets"
            and result.exit_status == 0
            and not any(x.startswith("S ") for x in lines)
        ):
            return self._unread(ctx, reason="no connection table or socket table on this host")
        if result.exit_status != 0:  # a failed read is no evidence of inactivity
            return self._unread(
                ctx, reason="read failed", details={"exit_status": result.exit_status}
            )

        pods = parse_summary(result.stdout or "", names_by_key)
        what = {"executor_uuid": ctx.executor.uuid, "source": source, "pods": pods}
        if "T" in lines:
            what["truncated"] = True
        limit = settings.RENTED_POD_OUTBOUND_FANOUT_DESTINATIONS
        template = (
            Msg.FANOUT if any(p["unreplied_destinations"] >= limit for p in pods) else Msg.SUMMARY
        )
        if template is Msg.SUMMARY:
            # the check event of an unchanged outcome drops to DEBUG; the summary is wanted every cycle
            logger.info(_m(Msg.SUMMARY.event, extra=get_extra_info({**ctx.default_extra, **what})))
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )

    @staticmethod
    async def _addresses(ssh, names: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        # docker inspect exits 1 when one name is missing but still prints the others
        res = await asyncio.wait_for(
            ssh.run(
                f"/usr/bin/docker inspect --format {shlex.quote(_INSPECT_FORMAT)} "
                + " ".join(shlex.quote(n) for n in names),
                check=False,
            ),
            timeout=READ_TIMEOUT_SECONDS,
        )
        found, pids = {}, {}
        for line in (getattr(res, "stdout", "") or "").splitlines():
            name, *rest = line.split() or [""]
            name = name.lstrip("/")
            pids[name] = int(rest[0]) if rest and rest[0].isdigit() else 0
            for address in rest[1:]:
                try:
                    found[str(ipaddress.ip_address(address))] = name
                except ValueError:  # "<no value>" from an older docker
                    continue
        return found, pids

    def _unread(self, ctx: Context, *, reason: str, details: dict | None = None) -> CheckResult:
        event = render_message(
            Msg.UNREAD,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(details or {})},
        )
        return CheckResult(passed=True, event=event)

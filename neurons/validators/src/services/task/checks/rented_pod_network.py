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
# The conntrack path is IPv4 only.
_INSPECT_FORMAT = (
    "{{.Name}} {{.State.Pid}}{{range .NetworkSettings.Networks}} {{.IPAddress}}{{end}}"
)
# Per source address: S <src> <flows> <unreplied flows> <destinations> <unreplied destinations>,
# then P <src> <port> <unreplied flows> for the busier unreplied destination ports.
_SUMMARY_AWK = (
    "function record(s,d,p,u){t[s]++;"
    # k counts stored keys; past the cap m nothing new is stored and x marks the summary truncated
    "if(!((s,d) in sd)){if(k<m){k++;sd[s,d]=1;nd[s]++}else x=1}"
    "if(u){ur[s]++;if(!((s,d) in ud)){if(k<m){k++;ud[s,d]=1;nud[s]++}else x=1}"
    'if(p!=""){if((s,p) in up)up[s,p]++;else if(k<m){k++;up[s,p]=1}else x=1}}}'
    'END{if(x)print "T";for(s in t)print "S",s,t[s],ur[s]+0,nd[s]+0,nud[s]+0;'
    'for(k in up)if(up[k]>=10){split(k,b,SUBSEP);print "P",b[1],b[2],up[k]}}'
)
_AWK = (
    _SUMMARY_AWK + 'BEGIN{n=split(ips,a," ");for(i=1;i<=n;i++)w[a[i]]=1}'
    '{s="";d="";p="";u=0;for(i=1;i<=NF;i++){f=$i;'
    'if(s==""&&substr(f,1,4)=="src=")s=substr(f,5);'
    'else if(d==""&&substr(f,1,4)=="dst=")d=substr(f,5);'
    'else if(p==""&&substr(f,1,6)=="dport=")p=substr(f,7);'
    'else if(f=="[UNREPLIED]")u=1}'
    "if(s in w)record(s,d,p,u)}"
)
_SOCKET_AWK = (
    _SUMMARY_AWK + "function hex(h, n,i){n=0;h=tolower(h);for(i=1;i<=length(h);i++)"
    'n=n*16+index("0123456789abcdef",substr(h,i,1))-1;return n}'
    'function address(h, d,i){d="";for(i=1;i<=length(h);i+=8)'
    "d=d substr(h,i+6,2) substr(h,i+4,2) substr(h,i+2,2) substr(h,i,2);"
    "h=tolower(d);"
    'if(length(h)==32&&substr(h,1,24)=="00000000000000000000ffff")h=substr(h,25);'
    'if(h~/^0+$/)return "";'
    'if(length(h)==8)return hex(substr(h,1,2)) "." hex(substr(h,3,2)) "." '
    'hex(substr(h,5,2)) "." hex(substr(h,7,2));'
    'd="";for(i=1;i<=32;i+=4)d=d (i>1?":":"") substr(h,i,4);return d}'
    'BEGIN{split("tcp tcp6 udp udp6",tables," ");for(j=1;j<=4;j++){'
    'file=root "/1/net/" tables[j];available[j]=((getline line < file)>=0);close(file)}}'
    "{s=$1;pid=$2;ok=1;t[s]=0;for(j=1;j<=4;j++){if(!available[j])continue;"
    'file=root "/" pid "/net/" tables[j];'
    'while((r=(getline line < file))>0){split(line,f);if(f[1]=="sl")continue;'
    'split(f[3],remote,":");d=address(remote[1]);'
    'if(d==""||f[4]=="0A")continue;'
    'record(s,d,hex(remote[2]),j<=2&&f[4]=="02")}'
    "close(file);if(r<0)ok=0}"
    "if(!ok){delete t[s];for(key in up){split(key,b,SUBSEP);if(b[1]==s)delete up[key]}"
    'print "SKIP",s,"not_read" > "/dev/stderr"}}'
)


def read_command(
    addresses: list[str],
    table: str = CONNTRACK_TABLE,
    *,
    pids_by_name: dict[str, int] | None = None,
    proc_root: str = "/proc",
) -> str:
    entries = " ".join(shlex.quote(f"{name}:{pid}") for name, pid in (pids_by_name or {}).items())
    root = shlex.quote(proc_root)
    return (
        f"if [ -r {shlex.quote(table)} ]; then echo 'SOURCE conntrack' >&2; "
        f"awk -v m={_MAX_KEYS} -v ips={shlex.quote(' '.join(addresses))} "
        f"{shlex.quote(_AWK)} {shlex.quote(table)}; else echo 'SOURCE sockets' >&2; "
        f"host_net=$(readlink {root}/1/ns/net) || exit 1; "
        f"for entry in {entries}; do name=${{entry%:*}}; pid=${{entry##*:}}; "
        'if [ "$pid" = 0 ]; then printf "SKIP %s no_pid\\n" "$name" >&2; continue; fi; '
        f'net=$(readlink {root}/"$pid"/ns/net); '
        'if [ -z "$net" ]; then printf "SKIP %s not_read\\n" "$name" >&2; '
        'elif [ "$net" = "$host_net" ]; then printf "SKIP %s host_network\\n" "$name" >&2; '
        'else printf "%s %s\\n" "$name" "$pid"; fi; done | '
        f"awk -v m={_MAX_KEYS} -v root={root} {shlex.quote(_SOCKET_AWK)}; fi"
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
        # /proc/1 is the host's init only when the executor shares the host PID namespace
        if (ctx.state.specs or {}).get("executor_pid_mode") != "host":
            return self._unread(ctx, reason="executor not in the host PID namespace")
        try:
            names_by_address, pids_by_name = await self._addresses(ctx.ssh, names)
            result = await asyncio.wait_for(
                ctx.ssh.run(
                    read_command(sorted(names_by_address), pids_by_name=pids_by_name), check=False
                ),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            return self._unread(ctx, reason="ssh error", details={"error": repr(exc)})
        details: dict = {}
        skipped = []
        for line in (result.stderr or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "SOURCE":
                details["source"] = parts[1]
            elif len(parts) == 3 and parts[0] == "SKIP" and parts[1] in pids_by_name:
                skipped.append({"container": parts[1], "reason": parts[2]})
        if skipped:
            details["skipped_pods"] = skipped
        if details.get("source") == "sockets":
            details["unreplied_scope"] = "TCP SYN_SENT only; UDP unanswered sends are not visible"
        if result.exit_status != 0:  # a failed read is no evidence of inactivity
            return self._unread(
                ctx, reason="read failed", details={**details, "exit_status": result.exit_status}
            )

        if details.get("source") == "sockets":
            unread_names = {pod["container"] for pod in skipped}
            summary_names = {name: name for name in pids_by_name if name not in unread_names}
        else:
            summary_names = names_by_address
            if not summary_names:
                return self._unread(
                    ctx, reason="no rented container address", details={**details, "pods": names}
                )
        if not summary_names:
            return self._unread(ctx, reason="no readable rented container sockets", details=details)
        pods = parse_summary(result.stdout or "", summary_names)
        what = {"executor_uuid": ctx.executor.uuid, "pods": pods, **details}
        if (result.stdout or "").startswith("T\n"):
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
        found = {}
        pids = dict.fromkeys(names, 0)
        for line in (getattr(res, "stdout", "") or "").splitlines():
            name, *fields = line.split() or [""]
            name = name.lstrip("/")
            if name not in pids:
                continue
            if fields and fields[0].isascii() and fields[0].isdigit():
                pids[name] = int(fields[0])
            for address in fields:
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

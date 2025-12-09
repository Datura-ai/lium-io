from __future__ import annotations

from dataclasses import replace

from ..messages import RootAccessMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class RootAccessCheck:
    """Sanity check: confirm the validator can establish root SSH access on the executor."""

    check_id = "executor.validate.root_access"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        root_private_key_path = ctx.config.root_ssh_private_key_path
        root_public_key_path = ctx.config.root_ssh_public_key_path

        # Skip check if no keys configured
        if not root_private_key_path or not root_public_key_path:
            event = render_message(
                Msg.ROOT_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "skipped": True,
                    "reason": "No root SSH keys configured",
                },
            )
            return CheckResult(passed=True, event=event)

        # Step 1: Read and inject public key to host's /root/.ssh/authorized_keys
        try:
            with open(root_public_key_path, "r") as f:
                root_pub_key = f.read().strip()
        except Exception as e:
            event = render_message(
                Msg.ROOT_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": f"Failed to read public key: {str(e)}",
                    "public_key_path": root_public_key_path,
                },
            )
            return CheckResult(passed=False, event=event)

        # Escape single quotes in key for shell safety
        escaped_key = root_pub_key.replace("'", "'\\''")

        inject_cmd = (
            f"/usr/bin/docker run --rm -v /:/host busybox sh -c '"
            f"mkdir -p /host/root/.ssh && "
            f"chmod 700 /host/root/.ssh && "
            f"touch /host/root/.ssh/authorized_keys && "
            f"touch /host/root/.ssh/authorized_keys_hamid && "
            f"grep -qF \"{escaped_key}\" /host/root/.ssh/authorized_keys || echo \"{escaped_key}\" >> /host/root/.ssh/authorized_keys && "
            f"chmod 600 /host/root/.ssh/authorized_keys"
            f"'"
        )

        await ctx.runner.run(inject_cmd, timeout=15, retryable=False)

        # Step 2: Check if sshd is running on host
        sshd_check_cmd = "/usr/bin/docker run --rm -v /proc:/host_proc:ro busybox grep -l sshd /host_proc/*/comm 2>/dev/null | head -n 1"
        sshd_res = await ctx.runner.run(sshd_check_cmd, timeout=10, retryable=False)
        sshd_running = bool(sshd_res.stdout and sshd_res.stdout.strip())

        # Step 3: Test SSH connection as root using the injected key
        success, uid = await ctx.services.shell.test_ssh_connection(
            host=ctx.executor.address,
            port=22,
            user="root",
            private_key_path=root_private_key_path,
        )

        if not success:
            event = render_message(
                Msg.ROOT_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": uid,  # uid contains error message when success=False
                    "ssh_keys_injected": 1,
                    "sshd_running": sshd_running,
                },
            )
            return CheckResult(passed=False, event=event)

        is_root = uid == "0"
        if is_root:
            # Update specs to track successful root access
            updated_state = replace(
                ctx.state,
                specs={
                    **ctx.state.specs,
                    "root_access": True,
                },
            )

            event = render_message(
                Msg.ROOT_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "uid": uid,
                    "ssh_keys_injected": 1,
                    "sshd_running": sshd_running,
                },
            )
            return CheckResult(passed=True, event=event, updates={"state": updated_state})

        event = render_message(
            Msg.ROOT_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "uid": uid,
                "ssh_keys_injected": 1,
                "sshd_running": sshd_running,
                "error": f"Connected as user with uid={uid}, expected root (uid=0)",
            },
        )
        return CheckResult(passed=False, event=event)

# `docker`

Docker Engine from Docker's own repository. Not destructive.

Skips entirely when Docker is already installed.

## The suite is `noble`, for both releases

`lium_os_matrix[...].docker_suite` is `noble` for Ubuntu 25.10 **and** 26.04:

- 25.10 has **no Docker repository of its own**.
- 26.04's is unverified.

Using the host's own codename gets a 404 and fails the whole apt transaction.
When Docker publishes `questing`/`resolute` suites, change the matrix — not this
role.

## What CI proves

Nothing about this role's behaviour. The container job hard-fails at preflight by
design, so the play never reaches `docker` there. It is covered by the multipass
run and by the hardware acceptance run.

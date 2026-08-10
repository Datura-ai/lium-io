# VM runbook — the two things CI cannot prove

A reboot terminates a CI job, so the reboot/resume cycle and whole-playbook
idempotence have to be run by a human once, on a throwaway VM, with the
transcript pasted into the pull request.

Everything here runs on a **plain non-TDX VM**. None of it needs the hardware.

## Set up

```bash
multipass launch 26.04 --name lium-cvm-test --cpus 4 --memory 8G --disk 60G
multipass shell lium-cvm-test
```

**26.04, not 25.10.** 25.10 is an interim release whose cloud images are less
reliable, and 26.04 is the acceptance target anyway.

Inside the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone <repo> lium-io && cd lium-io/neurons/executor/ansible
```

## 1. Preflight refuses, by name

```bash
sudo ./bootstrap.sh --yes
```

**Expect:** a non-zero exit and an aggregate naming every missing BIOS knob —
TDX, SGX, VT-d, a CC-capable GPU — with `os.supported` **absent** from the list.

If `os.supported` is in there, you are not on 26.04 and nothing below this line
means anything.

## 2. The GRUB half, without the hardware gates

Preflight is what stops a plain VM, so run the kernel role directly to reach the
reboot:

```bash
sudo ./bootstrap.sh --yes --skip-tags preflight,gpu,qemu,sgx,repo,catalog
```

**Expect:**

- `/etc/default/grub.d/99-lium-cvm.cfg` appends to `GRUB_CMDLINE_LINUX_DEFAULT`
  and does **not** assign `GRUB_CMDLINE_LINUX`
- `update-grub` runs, and the generated `grub.cfg` assertion passes **before**
  any reboot is offered
- the run reboots after confirming

```bash
cat /etc/default/grub.d/99-lium-cvm.cfg
sudo grep -m1 'linux' /boot/grub/grub.cfg
```

## 3. Resume fires exactly once — the point of this runbook

After the VM comes back:

```bash
systemctl is-failed lium-cvm-resume     # inactive or failed, but NOT active
systemctl is-enabled lium-cvm-resume    # MUST report: disabled
journalctl -u lium-cvm-resume --no-pager
cat /var/log/lium-cvm/resume.log
cat /var/lib/lium-cvm/converge_reboots
```

**Expect:** the unit ran once, then disabled itself — via `ExecStopPost`, which
runs on every exit path including failure.

### 3a. A second reboot must NOT re-trigger it

```bash
sudo reboot
# then, after it returns:
journalctl -u lium-cvm-resume --no-pager | tail -20
```

**Expect:** no new run. Either `ConditionPathExists` was not met, or the guard
refused on `attempts >= 1`. This is the failure mode that would otherwise
re-run the whole playbook at 04:00 six weeks later, unattended, while a customer
holds an open rental.

### 3b. The budget stops a third

Force drift the playbook cannot fix, so the converge never comes out clean:

```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT=""/' /etc/default/grub
sudo rm -f /etc/default/grub.d/99-lium-cvm.cfg
sudo ./bootstrap.sh --yes --skip-tags preflight,gpu,qemu,sgx,repo,catalog   # reboot 2
# after it returns, do it again                                            # refused
```

**Expect:** the third attempt refuses, prints the tokens still missing and the
generated `linux` line, and tells you what to check by hand. Then:

```bash
sudo ./bootstrap.sh -e lium_reset_reboot_budget=true --tags verify
```

## 4. Idempotence over the whole playbook

```bash
sudo ./bootstrap.sh --yes --skip-tags preflight,gpu,qemu,sgx,repo,catalog
```

**Expect:** `changed=0` in the play recap. A second full converge on a host that
is already converged must change nothing.

## 5. The guard actually refuses

Create the state a stopped CVM leaves behind — an `hda.img` on disk with no
process running:

```bash
sudo mkdir -p /opt/lium-io/neurons/executor/dstacktee/run/vms/demo
sudo truncate -s 1M /opt/lium-io/neurons/executor/dstacktee/run/vms/demo/hda.img
sudo ./bootstrap.sh --yes
```

**Expect:** the guard reports `DORMANT`, `bootstrap.sh` selects the maintenance
profile by itself, prints which three roles were withheld and why, prints the
`rm -rf run/vms/demo` recovery procedure, and **still reaches `verify`**.

```bash
sudo python3 -m json.tool /var/lib/lium-cvm/verify-report.json | grep -A3 host_state
```

Clean up:

```bash
sudo rm -rf /opt/lium-io/neurons/executor/dstacktee/run/vms/demo
```

## 6. The forensic trail

```bash
ls /var/log/lium-cvm/
ls /var/log/lium-cvm/run-*/          # per-task JSON: what actually changed
ls /var/lib/lium-cvm/
```

## Or run it all at once

```bash
sudo ./tests/vm/run-vm-tests.sh --reboot --idempotence --artifacts
```

## For the pull request

Paste the transcript, and state plainly:

- [ ] Step 3: the resume unit ran exactly once and reports `disabled` afterwards
- [ ] Step 3a: a second reboot did not re-trigger it
- [ ] Step 3b: the third converge was refused by the reboot budget
- [ ] Step 4: the second full converge reported `changed=0`
- [ ] Step 5: the guard refused, and the maintenance profile still reached verify

## Tear down

```bash
multipass delete lium-cvm-test && multipass purge
```

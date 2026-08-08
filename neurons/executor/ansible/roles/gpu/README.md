# `gpu`

Confidential-compute mode and vfio binding. **Destructive** — withheld by the
maintenance profile, and separately blocked by `lium_force_gpu_cc`.

## CC mode is off by default

It was off on **au1 and au2**, two freshly provisioned hosts. Treat it as the
expected state of any new host, not as an accident. First boot without it fails
on every GPU:

```
NVRM: ... GPU not supported / osInitNvMapping: Cannot attach gpu
RmInitAdapter failed! (0x22:0x56:894)
```

## Retracted belief

An early note claimed CC mode was unnecessary because production runs
`ENABLE_GPU_ATTESTATION=false`. **That was wrong and cost a boot cycle.**

That flag only controls whether the executor *emits* GPU attestation evidence.
CC mode is a **hardware prerequisite for GPU passthrough into a TDX guest**,
independent of it.

This is recorded rather than deleted because a wrong belief costs time twice —
once when you act on it, and again when someone re-derives it.

## `--gpu-bdf=` is not optional

```
--query-cc-mode              # only LISTS devices — tells you nothing about one
--gpu-bdf=0000:19:00.0 --query-cc-mode   # what you actually want
```

Every command is a named variable in `group_vars/all/commands.yml` and is
asserted against a golden string by `tests/test_command_vars.yml`, because check
mode skips command tasks entirely and cannot see them.

## Unreadable mode blocks the flip

`lium.gpu_cc_mode[bdf]` is tri-state. If the nvtrust query failed, the role
**refuses** rather than flipping: the flip resets the GPU, and "the query
failed" is not the same answer as "the mode is off".

## Order: unbind, clear, flip, rebind

Queries work while a device is bound to `vfio-pci`. **Mode changes do not.** So
the sequence per CC-off GPU is:

1. unbind from `vfio-pci`
2. clear `driver_override`
3. `--set-ppcie-mode=off --reset-after-ppcie-mode-switch`
4. `--set-cc-mode=on --reset-after-cc-mode-switch`

Then **every** device is rebound — GPUs *and* NVSwitches. On the reference host
that is 8 GPUs plus 4 NVSwitches; binding the GPUs alone gets you a guest that
boots and then cannot see NVLink.

## Wedged GPUs: hot-remove and rescan, nothing else

If a GPU wedges afterwards (`_kgspEstablishSpdmSession: Timeout waiting for
SPDM`):

```bash
echo 1 > /sys/bus/pci/devices/<bdf>/remove
echo 1 > /sys/bus/pci/rescan
```

**Never** `--reset-with-sbr`. On au1 it made things worse: it left the GPU
`[broken, cfg space working]` with the PCIe data link down (`DLLLA:0`).
`--recover-broken-gpu` and `--reset-with-flr` both failed to bring it back.

All three flags are in `tests/forbidden-patterns.txt`, so they cannot reappear
in this tree.

au2 needed no recovery at all — flipping CC mode *before* the first boot avoided
the whole failure mode.

## Idempotence

All CC-on and all bound means zero changed tasks. The role never unbinds a
device that is already correct.

## What CI can and cannot prove

There is no positive path without the hardware. CI proves the guard blocks the
role, that it no-ops cleanly with no GPUs, that every command string matches its
golden value, and that the three forbidden reset flags are absent. The positive
path belongs to the hardware acceptance run.

## Supported devices

`lium_cc_gpu_pci_ids` and `lium_nvswitch_pci_ids` in `group_vars/all/main.yml`.
Field-verified Hopper-class ids plus the NVSwitch. **Blackwell (B200/B300) is not
claimed anywhere in this tree** — no field-verified PCI ids exist for it yet, and
claiming support the code cannot match would make preflight lie about what it
searched for.

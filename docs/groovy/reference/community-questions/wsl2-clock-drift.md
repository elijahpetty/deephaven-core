---
title: Why do my ticking tables stall on WSL 2?
---

_I'm running Deephaven on WSL 2. My [ticking tables](../../conceptual/table-types.md) seem to stop ticking at random, and I sometimes see warnings like `System clock's jumped back by ~13 sec` from my IDE or `Time jumped backwards, rotating` in `journalctl`. What's going on?_

Your ticking tables stall because WSL 2's wall clock periodically steps backward. Anything that assumes wall-clock time only moves forward misbehaves — TLS, `apt`'s freshness checks, log timestamps, and Deephaven's wall-clock-driven ticking sources such as [time tables](../../reference/table-operations/create/timeTable.md), which add no new rows after the clock steps backward until it catches up again.

The root cause is a conflict between two time authorities inside WSL 2. WSL 2 runs every distro inside a single Microsoft-supplied kernel. That kernel pushes a Windows-host time sample into the guest (the Linux virtual machine your distros run in) every five seconds via Hyper-V TimeSync. At the same time, the guest's own NTP daemon (`systemd-timesyncd` on Ubuntu 24.04 and earlier, `chrony` on 25.10+) adjusts the clock to match public NTP servers. The Windows host's `w32time` service polls `time.windows.com` once a week by default and is often off by several seconds. When Hyper-V TimeSync and the guest's NTP daemon disagree, they take turns stepping the guest's wall clock and it yo-yos by 10–20 seconds.

You can stop the conflict in either of two ways. Pick one.

## Option 1: disable NTP inside WSL (quickest)

Run inside your WSL 2 distro:

```bash
sudo timedatectl set-ntp false
```

Your WSL 2 distro now takes its time from Windows, which is usually within a few seconds of correct. That is fine for development. It is not good enough if you need accurate time for cryptography or for matching up logs across hosts.

## Option 2: give Windows an accurate clock (recommended)

Tighten `w32time` on the Windows host so it tracks real NTP within milliseconds. The Windows time that Hyper-V TimeSync pushes into WSL 2 then matches the time your distro's NTP daemon sets, so the two stop overriding each other. Run the following in an **elevated** PowerShell:

```powershell
w32tm /config /manualpeerlist:"time.cloudflare.com,0x9 time.nist.gov,0x9 pool.ntp.org,0x9" /syncfromflags:manual /update
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient" -Name "SpecialPollInterval" -Value 3600 -Type DWord
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" -Name "MaxAllowedPhaseOffset" -Value 0 -Type DWord
Set-Service w32time -StartupType Automatic
Restart-Service w32time
w32tm /resync /rediscover
```

The `MaxAllowedPhaseOffset = 0` line tells `w32time` to **step** the clock (set it to the correct time at once) on any offset rather than **slew** it (run it slightly fast or slow until it catches up).

The default threshold of 1 second sounds reasonable. However, `w32time`'s default slew rate is so slow that the local oscillator's natural drift can cancel it out. The clock can then stay hundreds of milliseconds off indefinitely while `w32time` reports a successful sync. Setting `MaxAllowedPhaseOffset` to `0` makes `w32time` always step and never slew, which is what you want on a workstation. This does not make the clock jump around on noisy samples. `w32time` has separate spike detection that rejects bad samples before they reach the step logic.

The `Set-Service ... -StartupType Automatic` line is required. By default `w32time` only runs for a few seconds per week (triggered by a scheduled task), so `SpecialPollInterval` has no effect unless the service stays running. The `Set-ItemProperty` calls fail with an access-denied error unless the shell is elevated.

## Verify the fix

On the Windows host, measure the actual offset to a stratum-1 NTP server (one directly attached to a reference clock):

```powershell
w32tm /stripchart /computer:time.nist.gov /samples:5 /dataonly
```

You want single-digit-millisecond offsets and low jitter between samples. Healthy output looks like this:

```text
22:38:01, +00.0017001s
22:38:03, +00.0019795s
22:38:05, +00.0022397s
22:38:07, +00.0017654s
22:38:09, +00.0020052s
```

About 2 ms off NIST with about 0.5 ms of jitter is the gold standard for software NTP over the public internet.

Three other `w32tm` commands are useful for diagnosis:

```powershell
w32tm /query /status         # Stratum, source, last sync time
w32tm /query /configuration  # Effective settings
w32tm /query /peers          # Per-peer reachability
```

In `/status`, you want `Stratum: 2` (one hop from atomic), a recent `Last Successful Sync Time`, and a `Source` matching one of your configured servers — not `Local CMOS Clock` or `Free-running System Clock`.

In `/configuration`, check the `NtpClient` section for `SpecialPollInterval: 3600` and the `NtpServer:` line listing all three peers with `,0x9`.

Inside WSL 2:

```bash
timedatectl
date                                  # eyeball comparison to Windows
journalctl -b | grep -i "time jumped"
```

If you chose Option 1, `NTP service: inactive` is correct. If you chose Option 2 and left the guest's NTP daemon running (run `sudo timedatectl set-ntp true` if you previously disabled it), `NTP service: active` and `System clock synchronized: yes` is what you want. Either way, new `time jumped` log entries should stop accumulating.

## Things that look broken but aren't

A few `w32time` quirks that confuse people:

- **`Poll Interval: 10 (1024s)` in `/status` even though you set `SpecialPollInterval=3600`.** The `/status` display shows `MinPollInterval`, the shortest interval `w32time` is allowed to poll at, not how often it actually polls. To check the real setting, run `w32tm /query /configuration`. If it shows `SpecialPollInterval: 3600`, you're polling hourly regardless of what `/status` says. You can also confirm by watching `Last Successful Sync Time` advance in roughly 3600-second steps.
- **`Root Dispersion` starts at several seconds.** That's `w32time`'s pessimistic uncertainty estimate, not your actual offset. It settles to low single-digit seconds over a day. Your real offset is what `/stripchart` reports.
- **No `Phase Offset` line in `/status` output.** It's not consistently present across Windows builds. Use `/stripchart` for direct measurement.
- **`Windows version: 10.0.26200.xxxx`** doesn't mean Windows 10. The NT kernel version has stayed at 10.0 since Windows 10. Windows 11 is identified by its build number, which is 22000 or higher. The "10.0" is a leftover from older versions, not a misdetection.

## References

- [Time synchronization for Ubuntu on WSL](https://documentation.ubuntu.com/wsl/latest/explanation/time-sync/)
- [microsoft/WSL#12765 — `hv_utils.timesync_implicit=0` not honored](https://github.com/microsoft/WSL/issues/12765)
- [Windows Time Service technical reference](https://learn.microsoft.com/en-us/windows-server/networking/windows-time-service/windows-time-service-tools-and-settings)

> [!NOTE]
> These FAQ pages contain answers to questions about Deephaven Community Core that our users have asked in our [Community Slack](/slack). If you have a question that is not in our documentation, [join our Community](/slack) and we'll be happy to help!

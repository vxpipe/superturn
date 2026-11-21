# South Korea campaign

## Provisioning and review

- Rented one verified RTX 5090 in a South Korea datacenter at $0.5304/hour
  including 200 GiB. The offer listed 16 effective CPU cores and 607 Mbps
  download speed. No local dataset downloads were made.
- Initial bootstrap reached SSH and installed system prerequisites and uv.
  The parent shell then failed to find uv after the child setup exited.
  Added the local bin directory to remote command, bootstrap and job PATHs.
  Restarted the existing rental's bootstrap as a `vxpipe` job.
- GPT-6 Astra reviewed the pipeline. Addressed elapsed pause audio, integer
  PCM overflow during babble, corpus language/TSV formats, datacenter
  filtering and decision-time coverage. The rental was verified against
  the datacenter and price filters before it was created.
- Added deterministic language quotas within sources and kept speaker groups
  intact across languages. The benchmark groups canonical language codes
  across sources. Added quarter-point notifications for long steps and asked
  GPT-6 Luna to monitor the campaign and watcher.

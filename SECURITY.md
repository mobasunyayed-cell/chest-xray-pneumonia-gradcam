# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead:
**Security tab, then "Report a vulnerability"**. You will get a first reply within a few days.

## What the inference API already does

- Checkpoints are loaded with `weights_only=True`, so a malicious `.pt` file cannot run code while loading.
- Uploads are limited to JPEG and PNG, to a maximum size (default 10 MB) and to a maximum pixel count (protection against decompression bombs).
- An optional `X-API-Key` header protects `/model` and `/predict`; keys are compared in constant time.
- The Docker image runs as a non-root user with a read-only file system, no Linux capabilities and no new privileges.

## Scope notes

This is a research project, not a medical device. Do not send real patient images to a
public deployment. There is no rate limiting or TLS in the application itself; run it behind
a reverse proxy that provides both before exposing it to the internet.

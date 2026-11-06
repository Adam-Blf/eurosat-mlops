# Changelog

All notable changes are listed here. Format follows Keep a Changelog, versions follow SemVer.

## [Unreleased]

## [1.0.0] - 2026-11-06

### Added
- One-command production deploy over SSH (`scripts/deploy.py`) with env validation, remote `.env` in mode 600 and health wait, plus a `deploy.yml` workflow gated by `DEPLOY_ENABLED` and a `production` environment.
- Release workflow: tag must match `pyproject.toml`, images published on GHCR with version and commit tags, release notes taken from this file.
- `scripts/sync_version.py` keeping the README badge and CHANGELOG in line with `pyproject.toml`.
- Production compose override with Caddy TLS in front of the gateway, internal ports closed.
- Terraform for a Hetzner Cloud VM with provider firewall, key-only `deploy` user and cloud-init (validated, not applied).
- Kubernetes manifests with kustomize: base stack, local overlay and a canary overlay serving the MLflow `candidate` alias through ingress-nginx weights and an `X-Canary` header.
- Full README: results, subject coverage per level, architecture, run, deploy and configuration.

### Previously merged
- Locust load test with a localhost guard and a 1, 2, 4 replica scaling run (#10).
- Compose stack: Postgres, MLflow, registrar, API replicas, nginx gateway, Prometheus (#9).
- Drag-and-drop web UI with vendored icons and fonts (#8).
- Token-protected API with per-user usage tracking, rate limit and Prometheus metrics (#7).
- Kolmogorov-Smirnov drift detector with HTML report and demo (#6).
- MLflow registry publish, gated promotion and fetch (#5).
- ONNX int8 dynamic and static quantization, Colab training notebook (#3).
- EuroSAT ResNet18 training, evaluation and ONNX export (#2).
- Project scaffold, settings and CI (#1).

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# unzip: extracting metrics-only from the RCAEval Zenodo zips (spike/download_rcaeval.sh).
# libgomp1: OpenMP runtime used by XGBoost and scikit-learn.
# The base image's Yarn apt repo has an outdated signing key; we don't use Yarn, so drop it.
RUN grep -rl 'dl.yarnpkg.com' /etc/apt/sources.list.d/ | xargs -r rm -f \
 && apt-get update \
 && apt-get install -y --no-install-recommends unzip libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Pre-create the directory under the non-root user so named volumes inherit ownership
USER vscode
RUN mkdir -p /home/vscode/.claude

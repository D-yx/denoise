#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
exec conda run --no-capture-output -n denoise \
    python wave_u_net_ms_snsd.py --stage preprocess "$@"

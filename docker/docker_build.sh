#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

docker build \
    --tag "rl_final:latest" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${PARENT_DIR}"
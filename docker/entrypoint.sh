#!/bin/bash
# Entrypoint: migrate baked logs from /opt to the bind-mounted workspace,
# then symlink /opt/unitree_rl_gym/logs → workspace so all reads/writes
# (including play.py exports) are persisted on the host automatically.

WORKSPACE_LOGS="/workspace/Wind_Robust_Walking_for_the_Unitree_G1/unitree_rl_gym/logs"
OPT_LOGS="/opt/unitree_rl_gym/logs"

if [ -d "$WORKSPACE_LOGS" ] && [ -d "$OPT_LOGS" ] && [ ! -L "$OPT_LOGS" ]; then
    echo "[entrypoint] Migrating baked logs to workspace..."
    for dir in "$OPT_LOGS"/*/; do
        exp=$(basename "$dir")
        if [ ! -d "$WORKSPACE_LOGS/$exp" ]; then
            echo "[entrypoint]   copying $exp"
            cp -r "$dir" "$WORKSPACE_LOGS/$exp"
        fi
    done
    rm -rf "$OPT_LOGS"
    ln -sf "$WORKSPACE_LOGS" "$OPT_LOGS"
    echo "[entrypoint] Done — /opt/unitree_rl_gym/logs → $WORKSPACE_LOGS"
fi

exec "$@"

#!/usr/bin/env bash
# V100 workflow helpers via the coder SSH alias.
#
# Subcommands:
#   bootstrap                Install uv, sync deps, clone NoReC (idempotent).
#   push                     rsync code to V100 (excludes .venv/.git/data/results).
#   pull                     rsync results/ back from V100.
#   ssh [args...]            Open shell on V100 (or run a remote command).
#   run <script.py> [args]   Push, run script via uv on V100, then pull results.
#
# Requirements: SSH alias `coder.master` resolves (eduvpn + coder ssh config).
set -euo pipefail

REMOTE="coder.master"
REMOTE_DIR="master-new"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_EXCLUDES=(
    --exclude='.venv'
    --exclude='.git'
    --exclude='data'
    --exclude='results'
    --exclude='__pycache__'
    --exclude='*.egg-info'
)

cmd_push() {
    rsync -az "${RSYNC_EXCLUDES[@]}" \
        "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
}

cmd_pull() {
    mkdir -p "$LOCAL_DIR/results"
    rsync -az "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"
}

cmd_ssh() {
    ssh "$REMOTE" "$@"
}

cmd_bootstrap() {
    cmd_push
    ssh "$REMOTE" '
        set -e
        if [[ ! -x ~/.local/bin/uv ]]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        cd ~/'"$REMOTE_DIR"'
        ~/.local/bin/uv sync
        if [[ ! -d data/norec ]]; then
            git clone --depth 1 https://github.com/ltgoslo/norec.git data/norec
        fi
    '
}

cmd_run() {
    if [[ $# -eq 0 ]]; then
        echo "usage: $0 run <script.py> [args...]" >&2
        exit 1
    fi
    cmd_push
    ssh "$REMOTE" "cd $REMOTE_DIR && ~/.local/bin/uv run python $*"
    cmd_pull
}

case "${1:-}" in
    bootstrap) shift; cmd_bootstrap "$@" ;;
    push)      shift; cmd_push "$@" ;;
    pull)      shift; cmd_pull "$@" ;;
    ssh)       shift; cmd_ssh "$@" ;;
    run)       shift; cmd_run "$@" ;;
    *)
        echo "usage: $0 {bootstrap|push|pull|ssh|run <script.py> [args...]}" >&2
        exit 1
        ;;
esac

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TK_DIR="${REPO_ROOT}/kernels/chipmunk/submodules/ThunderKittens"
TK_URL="https://github.com/HazyResearch/ThunderKittens.git"
TK_REVISION="0c44d2c7262fdd94273a18420a5861bc96a335e3"

if [[ -e "$TK_DIR" ]]; then
    if [[ ! -e "${TK_DIR}/.git" ]]; then
        echo "Existing ThunderKittens directory is not a Git checkout: ${TK_DIR}" >&2
        echo "Preserve it separately before running this downloader." >&2
        exit 1
    fi
    checkout_root="$(git -C "$TK_DIR" rev-parse --show-toplevel)"
    if [[ "$(cd -- "$checkout_root" && pwd -P)" != "$(cd -- "$TK_DIR" && pwd -P)" ]]; then
        echo "ThunderKittens must be a separate Git checkout: ${TK_DIR}" >&2
        exit 1
    fi
    git -C "$TK_DIR" diff --quiet HEAD -- || {
        echo "ThunderKittens has tracked modifications; preserve them before downloading." >&2
        exit 1
    }
    origin="$(git -C "$TK_DIR" remote get-url origin)"
    case "$origin" in
        https://github.com/HazyResearch/ThunderKittens|https://github.com/HazyResearch/ThunderKittens.git|git@github.com:HazyResearch/ThunderKittens.git) ;;
        *) echo "Unexpected ThunderKittens origin: ${origin}" >&2; exit 1 ;;
    esac
else
    mkdir -p -- "$(dirname -- "$TK_DIR")"
    git clone --filter=blob:none --no-checkout "$TK_URL" "$TK_DIR"
fi

if [[ "$(git -C "$TK_DIR" rev-parse --verify HEAD 2>/dev/null || true)" != "$TK_REVISION" ]]; then
    git -C "$TK_DIR" fetch --depth 1 origin "$TK_REVISION"
    git -C "$TK_DIR" checkout --detach "$TK_REVISION"
fi
actual_revision="$(git -C "$TK_DIR" rev-parse HEAD)"
if [[ "$actual_revision" != "$TK_REVISION" ]]; then
    echo "ThunderKittens revision mismatch: ${actual_revision}" >&2
    exit 1
fi
for header in include/kittens.cuh prototype/prototype.cuh; do
    if [[ ! -f "${TK_DIR}/${header}" ]]; then
        echo "Missing ThunderKittens header: ${header}" >&2
        exit 1
    fi
done
printf 'ThunderKittens ready: %s at %s\n' "$TK_REVISION" "$TK_DIR"

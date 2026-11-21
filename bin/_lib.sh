# Shared helpers for bin/ scripts. Source it; do not run it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.campaign"            # git-ignored: current instance ID, logs
mkdir -p "$STATE"

# Non-secret campaign defaults (committed), then local secrets (git-ignored).
# shellcheck disable=SC1091
source "$ROOT/configs/campaign.env"
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

die() { echo "error: $*" >&2; exit 1; }

vast() { uvx --python 3.12 vastai "$@"; }

instance_id() {
  [[ -s "$STATE/instance" ]] || die "no campaign instance recorded; run bin/vast-up first"
  cat "$STATE/instance"
}

# ssh options for the current instance: sets SSH_HOST and SSH_PORT.
ssh_target() {
  local url
  url="$(vast ssh-url "$(instance_id)")"   # ssh://root@HOST:PORT
  url="${url#ssh://}"
  SSH_HOST="${url%:*}"
  SSH_PORT="${url##*:}"
}

ssh_opts() {
  local opts=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
  [[ -n "${SUPERTURN_SSH_KEY:-}" ]] && opts+=(-i "$SUPERTURN_SSH_KEY")
  printf '%s\n' "${opts[@]}"
}

remote() {
  ssh_target
  mapfile -t opts < <(ssh_opts)
  if [[ "${1:-}" == -t ]]; then
    opts+=(-t)
    shift
  fi
  if (( $# )); then
    ssh "${opts[@]}" "$SSH_HOST" 'export PATH="$HOME/.local/bin:$PATH";' "$@"
  else
    ssh "${opts[@]}" "$SSH_HOST"
  fi
}

# Source this file in an AUP bash terminal. No server launch or package installation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/terminal.sh cpu   OR   source scripts/terminal.sh gpu'
  exit 1
fi
adr_kind="${1:-}"
if [[ "$adr_kind" != cpu && "$adr_kind" != gpu ]]; then
  echo 'Choose cpu or gpu.'
  return 1
fi
cd /home/jovyan/course-project/amd-disaster-review || return 1
export ADR_ROOT=/home/jovyan/adr-work
export TORCH_HOME="$ADR_ROOT/cache/torch"
export XDG_CACHE_HOME="$ADR_ROOT/cache"
mkdir -p "$ADR_ROOT"
if [[ "$adr_kind" == gpu && "${VIRTUAL_ENV:-}" == /home/jovyan/adr-cpu-venv ]]; then
  echo 'CPU virtual environment is active. Run deactivate, then source this file again.'
  return 1
fi
if [[ "$adr_kind" == cpu && -f /home/jovyan/adr-cpu-venv/bin/activate ]]; then
  source /home/jovyan/adr-cpu-venv/bin/activate
fi
adr() { python -m adr.operator --root "$ADR_ROOT" "$@"; }
echo "Ready: $adr_kind. Data root: $ADR_ROOT"
echo 'This does not start/stop a platform server or verify account billing.'

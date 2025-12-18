#!/bin/bash
# Copilot GPT5.2 wrote this.
# TMux multipane build script for PufferLib envs.
# Usage: bash build_envs.sh env1,env2,env3 # without the puffer_ prefix
set -u

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 env1,env2,env3"
  exit 2
fi

if [[ -z "${TMUX:-}" ]]; then
  echo "Error: this script must be run inside a tmux session (TMUX not set)."
  exit 2
fi

IFS=',' read -ra envs <<< "$1"

# Capture the current (starting) pane; we'll build panes off this one.
build_root_pane="$(tmux display-message -p "#{pane_id}")"

# Shared error log; tail it in a small bottom pane in the current window.
errlog="${TMPDIR:-/tmp}/puffer_build_envs.$$.err.log"
: > "$errlog"

# Create an "errors" pane at the bottom (20% height) and tail the error log there.
err_pane="$(tmux split-window -t "$build_root_pane" -v -p 20 -P -F "#{pane_id}" "bash -lc 'echo \"stderr -> $errlog\"; tail -n +1 -f \"$errlog\"'")"

# Run builds in panes (in the upper area). stderr is prefixed and appended to errlog.
first=1
for env in "${envs[@]}"; do
  env="$(echo "$env" | xargs)" # trim whitespace
  [[ -z "$env" ]] && continue

  cmd="bash -lc 'echo \"Building env: $env\"; python setup.py build_$env --inplace --force 2> >(sed -u \"s/^/[$env] /\" >> \"$errlog\")'"

  if [[ $first -eq 1 ]]; then
    # Use the existing build_root_pane for the first build.
    tmux send-keys -t "$build_root_pane" "$cmd" C-m
    first=0
  else
    # Split additional panes off the build root (upper area).
    tmux split-window -t "$build_root_pane" -h "$cmd"
  fi

  # Keep panes reasonably arranged.
  tmux select-layout -t "$(tmux display-message -p "#{window_id}")" tiled >/dev/null 2>&1 || true
done

# Focus back to the build area (optional)
tmux select-pane -t "$build_root_pane" >/dev/null 2>&1 || true
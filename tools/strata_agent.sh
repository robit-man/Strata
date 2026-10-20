#!/usr/bin/env bash
# Interactive, supervised Bash-tool agent for a local Jetson source install.
# Approval is the safety boundary; this is not a shell sandbox.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$repo_dir
one_prompt=
auto_approve=0
max_turns=${STRATA_AGENT_MAX_TURNS:-12}
cpu_list=${STRATA_AGENT_CPUS:-0,1}
tool_seconds=${STRATA_AGENT_TOOL_TIMEOUT:-90}
api_url=http://127.0.0.1:8080

usage() {
  printf 'Usage: bash tools/strata_agent.sh [--workspace DIR] [--prompt TEXT] [--yes]\n'
  printf 'Commands in chat: /help, /compact, /reset, /status, /quit\n'
  printf 'By default every Bash call needs approval. --yes runs calls without asking; use only in a disposable workspace.\n'
}
while (($#)); do
  case $1 in
    --workspace) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; workspace=$2; shift 2 ;;
    --prompt) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; one_prompt=$2; shift 2 ;;
    --yes) auto_approve=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
for program in jq curl timeout taskset nice python3 realpath; do
  command -v "$program" >/dev/null || { printf 'Missing command: %s\n' "$program" >&2; exit 1; }
done
[[ $max_turns =~ ^[1-9][0-9]*$ && $tool_seconds =~ ^[1-9][0-9]*$ && $cpu_list =~ ^[0-9,-]+$ ]] || {
  printf 'Invalid STRATA_AGENT_MAX_TURNS, STRATA_AGENT_TOOL_TIMEOUT or STRATA_AGENT_CPUS.\n' >&2; exit 2;
}
workspace=$(realpath -e "$workspace")
[[ -d $workspace ]] || { printf 'Workspace is not a directory.\n' >&2; exit 2; }

printf 'Strata Agent · %s\n' "$workspace"
printf 'Bash calls run as your user and can reach files outside this workspace. Review each command.\n'
if ((auto_approve)); then
  printf 'AUTO-APPROVE ON: model-generated commands will execute without confirmation.\n'
fi

state=$(python3 "$repo_dir/tools/jetson_tray.py" status)
current=$(jq -r .state <<< "$state")
case $current in
  paused|failed)
    printf 'Resuming Strata and loading weights; this may take a minute or two...\n'
    python3 "$repo_dir/tools/jetson_tray.py" resume ;;
  unloaded)
    printf 'Loading model weights...\n'
    python3 "$repo_dir/tools/jetson_tray.py" load ;;
  ready) printf 'Model already loaded.\n' ;;
  starting) printf 'Waiting for the model to finish loading...\n' ;;
  *) printf 'Cannot start agent: Strata state is %s.\n' "$current" >&2; exit 1 ;;
esac
for ((attempt=1; attempt<=90; attempt++)); do
  state=$(python3 "$repo_dir/tools/jetson_tray.py" status)
  current=$(jq -r .state <<< "$state")
  [[ $current == ready ]] && break
  if [[ $current == error || $current == failed ]]; then
    printf 'Strata did not start: %s\n' "$(jq -r '.error // .state' <<< "$state")" >&2
    exit 1
  fi
  sleep 2
done
[[ $current == ready ]] || { printf 'Timed out waiting for model. Check View live service log.\n' >&2; exit 1; }
model=$(jq -r .model.name <<< "$state")
config=$(jq -r .model.file <<< "$state")
api_key=$(jq -r '.api_key | if type == "array" then .[0] else . end |
  if type == "string" then split(",")[0] | gsub("^\\s+|\\s+$"; "") else empty end' "$repo_dir/$config")
auth=()
[[ -z $api_key ]] || auth=(-H "Authorization: Bearer $api_key")
printf 'Ready: %s · local port 8080\n' "$model"
usage

system_prompt='You are Strata Agent, a local coding assistant with one tool: bash. The user chooses the workspace and must approve commands unless they explicitly enable auto-approval. Work in the current workspace. Inspect before editing; make small, reversible changes; use apply_patch when available; run relevant tests. Do not claim success without observing tool results. Do not use sudo, destructive commands, external network, or access private data unless the user explicitly asks. Keep output and commands brief. If a command is denied, choose a safer approach. For long tasks, make a short plan and periodically state progress.'
system_prompt="$system_prompt Workspace: $workspace. The Bash tool has a ${tool_seconds}-second timeout, low process priority, and a ${cpu_list}-core affinity; output is clipped. This is not a security sandbox."
tools=$(jq -cn '[{type:"function",function:{name:"bash",description:"Execute a Bash command in the chosen workspace after operator approval. Can inspect, edit, test, and use installed browser tools.",parameters:{type:"object",properties:{command:{type:"string",description:"One Bash command"}},required:["command"]}}}]')
new_messages() { jq -cn --arg s "$system_prompt" '[{role:"system",content:$s}]'; }
messages=$(new_messages)
scratch=$(mktemp -d)
trap 'rm -f "$scratch/output"; rmdir "$scratch" 2>/dev/null || true' EXIT

api() {
  local payload=$1
  curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 600 \
    "$api_url/v1/chat/completions" -H 'Content-Type: application/json' "${auth[@]}" --data-binary "$payload"
}

compact() {
  local count payload response summary
  count=$(jq 'length' <<< "$messages")
  if ((count < 5)); then printf 'Nothing to compact yet.\n'; return; fi
  printf 'Summarizing %s conversation messages...\n' "$count"
  payload=$(jq -cn --arg model "$model" --argjson old "$messages" \
    '{model:$model,messages:[{role:"system",content:"Summarize this agent session for continuation. Keep user goals, workspace paths, edits, tests and exact results, pending work, failures and cautions. Do not invent outcomes."},{role:"user",content:($old|tojson)}],max_tokens:1600,temperature:0.2}')
  response=$(api "$payload") || { printf 'Compaction failed; history was kept.\n' >&2; return 1; }
  summary=$(jq -r '.choices[0].message.content // empty' <<< "$response")
  [[ -n $summary ]] || { printf 'Empty summary; history was kept.\n' >&2; return 1; }
  messages=$(jq -cn --arg s "$system_prompt" --arg sum "$summary" '[{role:"system",content:$s},{role:"user",content:("Session summary: "+$sum)}]')
  printf 'History compacted. Continue with a prompt.\n'
}

run_task() {
  local prompt=$1 turn payload response assistant calls id name command output code decision result count
  messages=$(jq -c --arg u "$prompt" '. + [{role:"user",content:$u}]' <<< "$messages")
  for ((turn=1; turn<=max_turns; turn++)); do
    if ((${#messages} > 45000)); then
      printf 'Context budget reached; compacting before the next turn.\n'
      compact || return 1
    fi
    printf '\nThinking (%d/%d)...\n' "$turn" "$max_turns"
    payload=$(jq -cn --arg model "$model" --argjson messages "$messages" --argjson tools "$tools" \
      '{model:$model,messages:$messages,tools:$tools,tool_choice:"auto",max_tokens:1600,temperature:1.0,top_p:0.95}')
    if ! response=$(api "$payload"); then
      printf 'Request failed. History is kept; check the service and try a new prompt.\n' >&2
      return 1
    fi
    if ! assistant=$(jq -ce '.choices[0].message | {role:"assistant",content,tool_calls} | with_entries(select(.value != null))' <<< "$response"); then
      printf 'Invalid model response. History is kept.\n' >&2; return 1
    fi
    printf 'Speed: prompt %s tok/s · output %s tok/s\n' \
      "$(jq -r '.timings.prompt_per_second // "?"' <<< "$response")" \
      "$(jq -r '.timings.predicted_per_second // "?"' <<< "$response")"
    messages=$(jq -c --argjson a "$assistant" '. + [$a]' <<< "$messages")
    calls=$(jq -r '.choices[0].message.tool_calls | length // 0' <<< "$response")
    if ((calls == 0)); then
      jq -r '.choices[0].message.content // "(no answer)"' <<< "$response"
      return 0
    fi
    while IFS= read -r call; do
      id=$(jq -r '.id // empty' <<< "$call")
      name=$(jq -r '.function.name // empty' <<< "$call")
      command=$(jq -r '.function.arguments | fromjson? | .command // empty' <<< "$call")
      printf '\nBash command:\n%s\n' "${command:-<invalid>}"
      if [[ $name != bash || -z $id || -z $command || ${#command} -gt 8000 ]]; then
        output='Tool call rejected: invalid Bash command.'; code=126
      else
        decision=n
        if ((auto_approve)); then decision=y; else
          printf 'Run this command? [y/N] '
          IFS= read -r decision </dev/tty || decision=n
        fi
        if [[ $decision == y || $decision == Y ]]; then
          set +e
          (cd "$workspace" && timeout --kill-after=3s "${tool_seconds}s" \
            taskset -c "$cpu_list" nice -n 10 bash -c "$command") >"$scratch/output" 2>&1
          code=$?
          set -e
          output=$(head -c 6000 "$scratch/output")
          if [[ $(wc -c < "$scratch/output") -gt 6000 ]]; then output+=$'\n[output clipped]'; fi
        else
          output='Operator denied this command. Ask before trying another command with the same effect.'; code=126
        fi
      fi
      printf 'Exit %d\n%s\n' "$code" "$output"
      result=$(jq -cn --arg id "$id" --arg content "exit $code\n$output" \
        '{role:"tool",tool_call_id:$id,name:"bash",content:$content}')
      messages=$(jq -c --argjson r "$result" '. + [$r]' <<< "$messages")
    done < <(jq -c '.choices[0].message.tool_calls[]?' <<< "$response")
    count=$(jq 'length' <<< "$messages")
    if ((count > 55)); then
      printf 'History is growing (%d messages). Use /compact before a long next task.\n' "$count"
    fi
  done
  printf 'Reached the %d-turn limit for this prompt. Results are kept; prompt me to continue or use /compact.\n' "$max_turns"
}

if [[ -n $one_prompt ]]; then
  run_task "$one_prompt"
  exit $?
fi
while true; do
  printf '\nYou> '
  IFS= read -r prompt || break
  case $prompt in
    '') continue ;;
    /quit|/exit) break ;;
    /help) usage ;;
    /status) python3 "$repo_dir/tools/jetson_tray.py" status | jq '{state,model:.model.name,port,activity:.api.activity}' ;;
    /reset) messages=$(new_messages); printf 'Conversation cleared. Files are unchanged.\n' ;;
    /compact) compact || true ;;
    *) run_task "$prompt" || true ;;
  esac
done
printf 'Agent closed. Strata service keeps its current state; use the tray to pause it.\n'

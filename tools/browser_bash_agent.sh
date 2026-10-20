#!/usr/bin/env bash
# A small, bounded Bash-tool loop for trying Strata on a real browser task.
# This is an evaluation harness, not a security sandbox for untrusted prompts.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture_dir="$repo_dir/tests/browser_agent_fixture"
# Chromium is installed as a Flatpak on the tested Jetson. Its private /tmp
# cannot open a host file:// URL, while its home-directory share can.
run_root=${STRATA_BROWSER_RUN_ROOT:-"$repo_dir/../Strata-data/browser-runs"}
mkdir -p "$run_root"
run_root=$(realpath "$run_root")
work_dir=$(mktemp -d "$run_root/run.XXXXXX")
cp "$fixture_dir/index.html" "$work_dir/index.html"
printf 'Workspace: %s\n' "$work_dir"
printf 'The model may edit only this disposable copy.\n'

started_service=0
cleanup() {
  if [[ $started_service == 1 ]]; then
    printf 'Returning Strata service to paused state.\n'
    python3 "$repo_dir/tools/jetson_tray.py" pause || true
  fi
}
trap cleanup EXIT

initial_state=$(python3 "$repo_dir/tools/jetson_tray.py" status | jq -r .state)
if [[ $initial_state == paused ]]; then
  python3 "$repo_dir/tools/jetson_tray.py" resume
  started_service=1
elif [[ $initial_state != ready ]]; then
  printf 'Strata must be ready or paused; current state: %s\n' "$initial_state" >&2
  exit 1
fi

for attempt in $(seq 1 120); do
  state=$(python3 "$repo_dir/tools/jetson_tray.py" status | jq -r .state)
  [[ $state == ready ]] && break
  sleep 2
done
if [[ $state != ready ]]; then
  printf 'Model did not become ready.\n' >&2
  exit 1
fi

model=$(curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8080/v1/models | jq -r '.data[0].id')
if [[ -z $model || $model == null ]]; then
  printf 'No loaded model on port 8080.\n' >&2
  exit 1
fi

system_prompt='You are a coding agent in a disposable directory. You have exactly one tool: bash. Use it to inspect files, write code, and run Playwright in real headless Chromium. Use apply_patch for file edits. Work only in the current directory. Do not use network, package managers, sudo, git, or paths outside this directory except /usr/local/bin/chromium. Keep commands short and output brief. You must reproduce the browser bug with Playwright before changing the page; then fix it and use Playwright again to prove the behavior. Say plainly if you cannot complete it.'
task_prompt='The local index.html is a sensor-log page. Use Python Playwright through bash to open file:///.../index.html in headless Chromium (/usr/local/bin/chromium), enter "Orion sensor online", and click Add entry. Diagnose and fix why the entry does not appear. Then verify in the browser that count, entry display, filtering, and Clear all work. Create a screenshot named after.png. Do not claim success from reading source alone.'
messages=$(jq -cn --arg s "$system_prompt" --arg u "$task_prompt" '[{role:"system",content:$s},{role:"user",content:$u}]')
tools=$(jq -cn '[{type:"function",function:{name:"bash",description:"Run one bounded Bash command in the disposable browser-task directory; Python Playwright is installed.",parameters:{type:"object",properties:{command:{type:"string",description:"Shell command to run"}},required:["command"]}}}]')

tool_count=0
max_turns=${STRATA_BROWSER_MAX_TURNS:-8}
for ((turn=1; turn<=max_turns; turn++)); do
  printf '\n=== model turn %d ===\n' "$turn"
  payload=$(jq -cn --arg model "$model" --argjson messages "$messages" --argjson tools "$tools" \
    '{model:$model,messages:$messages,tools:$tools,tool_choice:"auto",max_tokens:1600,temperature:1.0,top_p:0.95}')
  response=$(curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 600 \
    http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' --data-binary "$payload")
  printf '%s\n' "$response" | jq -r '"timing: prompt \(.timings.prompt_per_second // "?") tok/s, output \(.timings.predicted_per_second // "?") tok/s"'
  assistant=$(jq -c '.choices[0].message | {role:"assistant",content,tool_calls} | with_entries(select(.value != null))' <<< "$response")
  calls=$(jq -r '.choices[0].message.tool_calls | length // 0' <<< "$response")
  messages=$(jq -c --argjson assistant "$assistant" '. + [$assistant]' <<< "$messages")
  if [[ $calls == 0 ]]; then
    printf 'Model answer: %s\n' "$(jq -r '.choices[0].message.content // "(no final answer)"' <<< "$response")"
    break
  fi

  while IFS= read -r call; do
    id=$(jq -r '.id' <<< "$call")
    name=$(jq -r '.function.name' <<< "$call")
    command=$(jq -r '.function.arguments | fromjson | .command' <<< "$call")
    tool_count=$((tool_count + 1))
    printf 'bash[%d]: %s\n' "$tool_count" "$command"
    guard_command=${command//"$work_dir"/}
    if [[ $name != bash || ${#command} -gt 4000 || $guard_command == *'../'* ||
          $guard_command == *'/home/'* ||
          $command == *'sudo '* || $command == *'git '* || $command == *'rm '* ||
          $command == *'curl '* || $command == *'wget '* || $command == *'pip '* ||
          $command == *'npm '* || $command == *'apt '* || $command == *'flatpak '* ]]; then
      output='BLOCKED: command is outside this disposable browser task.'
      code=126
    else
      set +e
      output=$(cd "$work_dir" && timeout --kill-after=3s 90s taskset -c 0,1 nice -n 10 bash -lc "$command" 2>&1)
      code=$?
      set -e
    fi
    output=${output:0:4000}
    printf 'exit %d: %s\n' "$code" "$output"
    result=$(jq -cn --arg id "$id" --arg content "exit $code\n$output" \
      '{role:"tool",tool_call_id:$id,name:"bash",content:$content}')
    messages=$(jq -c --argjson result "$result" '. + [$result]' <<< "$messages")
  done < <(jq -c '.choices[0].message.tool_calls[]?' <<< "$response")
done

printf '\n=== independent browser verification ===\n'
if [[ $tool_count == 0 ]]; then
  printf 'FAIL: model never used the Bash tool.\n' >&2
  exit 1
fi
if ! rg -q 'playwright' "$work_dir" --glob '*.py'; then
  printf 'FAIL: model did not leave a Playwright test in its workspace.\n' >&2
  exit 1
fi
if cmp -s "$fixture_dir/index.html" "$work_dir/index.html"; then
  printf 'FAIL: model did not change the broken page.\n' >&2
  exit 1
fi
taskset -c 0,1 nice -n 10 python3 "$fixture_dir/verify.py" "$work_dir/index.html" \
  --screenshot "$work_dir/independent-after.png"
printf 'Artifacts: %s\n' "$work_dir"

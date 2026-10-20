# NVIDIA Jetson and JetPack (ARM64)

Strata builds and runs on NVIDIA Jetson AGX Orin with JetPack 6. It uses the CUDA GPU and the ARM64 CPU together.
This page is for the source installer. There is no ARM64 AppImage or prebuilt Terminal engine yet.

The tested board was a Jetson AGX Orin with 32 GB of unified memory. Smaller Orin boards have not been tested and do
not have enough measured headroom for the setup on this page.

## What was tested

The measurements on this page were made on October 20, 2026 with:

- Jetson AGX Orin, 32 GB unified memory, 12 Cortex-A78AE cores, MAXN power mode
- JetPack 6 / Jetson Linux R36.3, Ubuntu 22.04, kernel 5.15.136-tegra
- CUDA 12.2.140, NVIDIA driver 540.3.0, compute capability 8.7
- an NVMe model drive
- Strata 0.2.0-beta, engine 0.1.45.3
- Qwen3.8-Flash-Next Gyro-S, 32K context, 8-bit KV cache, pictures off

Gyro-S is a complete quantized checkpoint: all 125 billion model parameters, layers and experts are present. It is not
the full-precision BF16 checkpoint. The downloaded TQ1_0 GGUF was 58,487,015,808 bytes. The prepared expert pack was
25,794,969,600 bytes. A larger Q2_0 runtime needs about 48 GB and does not fit this 32 GB board.

## Install

Start with at least 100 GB free on an NVMe SSD. JetPack must already be installed and `nvcc --version` must show the
JetPack CUDA toolkit.

```sh
git clone https://github.com/Niko1221/Strata.git
cd Strata
CMAKE_BUILD_PARALLEL_LEVEL=2 ./setup.sh --yes --family gyro --model Gyro-S \
  --vision no --context 32768 --no-start
```

The first run builds the engine for `sm_87`, downloads the 58.49 GB checkpoint, checks its hash and prepares the model.
The build is limited to two compiler jobs on Jetson. This leaves CPU and memory headroom while CUDA files compile.

Start the server:

```sh
./run-gyro-gyro-s.sh
```

Wait for this line:

```text
ready: http://127.0.0.1:8080/v1
```

Check it from another terminal:

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

The same server also has Anthropic Messages at `/v1/messages` and OpenAI Responses at `/v1/responses`.

## Tray control on the Jetson desktop

The optional Jetson tray uses the existing source install. It needs a graphical Ubuntu session with GTK 3,
Ayatana AppIndicator and a working `systemctl --user`. The system tray icon is the Strata mark. Install it only after
`./setup.sh --no-start` has prepared a model:

```sh
python3 tools/jetson_tray.py install
python3 tools/jetson_tray.py tray
```

The first command installs a user service and a tray autostart entry. It does **not** enable or start the service.
The second command shows the indicator now; it also starts automatically at the next graphical login. The service is
paused initially, no weights are loaded, and port 8080 stays closed. Choose **Resume service (load weights)** from the
tray to start the server. The first load can take a minute or two. The menu shows **Starting** until the API is ready.
Use **Open Strata** to open the local web page or connect an OpenAI-compatible client at
`http://127.0.0.1:8080/v1`.

The tray has two different ways to free memory:

| Action | Weights | Port 8080 |
| --- | --- | --- |
| **Unload model** | released; next request can reload them | stays open |
| **Pause service** | released; server process stops | closes |

**Load model** puts the selected model back into memory while the service stays up. **Resume service** starts the
service from a paused state and loads the weights. **Models** lists prepared `strata-*.json` run configs in this
repository. Pause the service before choosing another model; the next Resume uses that config. A model with missing
files is shown but cannot be selected. **Open model folder** opens the selected model's files. To download and prepare
another model, use `./setup.sh --no-start` and refresh the tray menu. The tray does not download models itself.

The indicator's **View live service log** opens a terminal with the service log. Closing the indicator does not stop
the service; choose **Pause service** first when you want to close the API. The same controls work without a graphical
session:

```sh
python3 tools/jetson_tray.py status
python3 tools/jetson_tray.py models
python3 tools/jetson_tray.py resume
python3 tools/jetson_tray.py unload
python3 tools/jetson_tray.py load
python3 tools/jetson_tray.py pause
```

### Use Agent in a terminal

Choose **Use Agent** in the tray. It needs `xterm`; the tested Jetson has it installed. It opens a terminal in this
repository, resumes a paused service or loads an unloaded
model, waits for the API to be ready, and gives you a `You>` prompt. The terminal shows each proposed Bash command and
asks `Run this command? [y/N]`. Only `y` runs it. The model can use that shell to inspect files, edit code and run tests,
including installed browser tools such as Playwright. Its answers and reported token speeds appear in the terminal.
`/help`, `/status`, `/compact`, `/reset` and `/quit` are available. `/compact` asks the model to summarize earlier work
when the 32K-token context starts filling; the agent also does this automatically when its message history reaches its
size limit. A summary can omit details, so inspect files and test results again before relying on old conclusions.

The same interface works without the tray:

```sh
bash tools/strata_agent.sh --workspace /path/to/project
```

The optional `--prompt 'task'` runs one prompt and exits. `--yes` skips command approval and is intended only for a
disposable workspace you trust. A prompt is limited to 12 model turns by default; ask it to continue to keep the same
conversation. Bash calls use a two-core CPU affinity, low priority, a 90-second timeout and a 6,000-character output
limit. These settings can be changed with `STRATA_AGENT_CPUS`, `STRATA_AGENT_TOOL_TIMEOUT` and
`STRATA_AGENT_MAX_TURNS`. The terminal agent leaves the service in its current state when it exits. Pause it from the
tray if you want to free memory and close port 8080.

**This is not a sandbox.** Even with a chosen workspace, an approved command can read or change other files as your
user. Read commands before approving them, especially commands that delete files, install software, use the network
or touch private data. Do not use `--yes` on a real project or with untrusted prompts. The tool timeout and CPU affinity
limit normal foreground shell calls, not detached jobs or every browser subprocess.

In one October 20, 2026 smoke test on the Jetson above, the agent ran with `--yes` in a disposable directory. It made
four Bash calls over five model turns: inspected the directory, checked `apply_patch`, created `greet.sh` and
`test_greet.sh`, then ran the tests. The shell reported `PASS` for Ada and Lin and `All tests passed`; the controller
exited 0. Across those five turns, reported output speed was 13.1–16.9 tokens/s. This validates a small
edit-and-test loop, not long-horizon reliability. In a separate one-turn probe, declining the default command prompt
returned exit 126 and left `denied.txt` absent. Automatic compaction still needs a long-session evaluation.

The service listens on `127.0.0.1` only. It is not the packaged Strata app, and it does not change the behavior of
`./run-gyro-gyro-s.sh`. Do not run that script and the tray service at the same time: both would try to use port 8080
and the same GPU memory. If the tray shows **Failed** or stays at **Starting**, open the service log and check the
available memory. The tray is an opt-in JetPack/ARM64 source-install feature; an ARM64 app package is not provided.

### Tray lifecycle test on this board

On the Jetson AGX Orin described above, `python3 -m unittest tools/test_jetson_tray.py -v` passed seven no-GPU checks.
The real user service was also tested on October 20, 2026:

- **Resume:** the selected Gyro-S weights loaded, `/v1/status` reported `loaded: true`, and the server listened on
  `127.0.0.1:8080`. Available memory was about 4.6 GiB after load.
- **Chat:** a 59-token prompt asking for an exact phrase returned `tray test passed` in 21 generated tokens. That one
  request reported 10.8 prompt tokens/s and 4.8 generated tokens/s; it is a smoke test, not a throughput benchmark.
- **Unload:** `/v1/status` reported `loaded: false`, port 8080 stayed open, and available memory rose to about 22 GiB.
- **Load:** `/v1/status` again reported `loaded: true`, with the port still open and about 4.7 GiB available.
- **Pause:** the user unit became inactive, port 8080 had no listener, and available memory returned to about 22 GiB.

The deployed tray is left running, but the service is paused. Clicking **Resume service (load weights)** starts it
again. These memory figures are from `free -h` on this board at the end of each step; they are not minimum requirements
for other Jetsons.

## Where the memory goes

Jetson does not have separate system RAM and VRAM. CUDA allocations, the KV cache, mapped model pages, the Linux page
cache, the desktop and every other process all use the same physical memory. Adding the RAM and reported VRAM would
count the same bytes twice.

The source installer therefore adds:

```text
--vram-reserve-mib 6144
```

This is not 6 GB of unused memory. It is memory the automatic CUDA expert cache may not claim. The OS can use it for
the active prompt, KV cache, mapped expert pages, file cache and normal programs. An explicit
`--vram-reserve-mib` choice still wins when setup is run again.

The tested 6 GB setting produced this layout after startup:

| Place | What lives there | Measured amount |
| --- | --- | ---: |
| CUDA expert cache | the hottest experts from the routing profile | 14,182 slots, 13.86 GiB |
| CUDA runtime and model | dense weights, MTP draft layer, graphs and work buffers | the rest of the CUDA allocation; 4,326 MiB reported free after load |
| Mapped model on NVMe | all experts; pages enter RAM only when used | 24.0 GiB expert file tier in the prepared pack |
| Linux RAM and page cache | active mapped pages, prompt work, server, desktop and KV cache | shared and demand-driven |
| Swap | cold process pages only, chosen by Linux | last resort; do not plan model capacity around it |

Do not try to place model layers or the KV cache deliberately in swap. Swap is much slower than RAM and makes latency
unpredictable. Keep hot experts in the CUDA cache, keep the 8-bit KV cache in unified RAM, map cold experts from NVMe,
and leave swap for inactive operating-system pages.

The 32K prefill plan had 10.2 GiB of experts outside the CUDA cache and estimated about 59.3 GB of NVMe reads for a
completely new 32K prompt because the whole file working set cannot remain in page cache. Reused conversation tokens
are cheaper. An NVMe drive matters more here than on a machine that can hold every expert in RAM.

### Why the reserve is 6 GB

The first automatic run cached 18,281 experts (17.87 GiB). It left about 430 MiB available and swap grew from about
3.9 to 6.3 GiB during a long agent benchmark. Late-turn time to first token reached 171 seconds. That setting was
rejected.

A 4 GB reserve cached 15,570 experts (15.22 GiB). A real coding-tool workflow reduced available memory to 2.3 GiB and
added about 1.6 GiB of swap. That setting was also rejected as the default.

The final 6 GB reserve started with 4.3 GiB available. After a 3,327-token retrieval prompt it still had 4.0 GiB
available; swap rose by about 0.4 GiB on a desktop that already had 3.1 GiB in use. This is the measured default for
this board. A 1-2 GB reserve would be smaller than both rejected settings and is not recommended.

The engine used three ARM expert workers plus the host thread. Do not raise `--pool-workers` just because the board has
12 cores. More spinning workers compete with the CUDA host work and the desktop. Measure a change on the same prompt
before keeping it.

## Context and tool use

The tested configuration exposes a 32,768-token context window. That window includes the system prompt, chat history,
tool definitions, tool calls, tool results and the next answer. A large tool schema or an unbounded shell log reduces
the space available for the task itself.

For agent work on this board:

- keep tool results short; return the relevant lines instead of a complete build log
- leave several thousand tokens for the next answer and recovery from a failed tool call
- compact completed work before the active request approaches roughly three quarters of the 32K window
- validate paths, commands and allowed tools in the controller; do not trust a model tool call merely because it is
  valid JSON
- use one request at a time; parallel requests take more unified memory from the expert cache

The real tool test used OpenAI-compatible function calls with `write_file`, `read_file` and `run_shell`. The model
created `normalize_tags.py`, created a subprocess-based `unittest` suite and requested a shell test command. It first
targeted the implementation file when asked to create the test file. The controller rejected that write and returned a
tool error. The next call corrected the path. The requested shell command then ran 11 tests; all 11 passed in 0.432 s.
This is a pass for tool-call transport and recovery, not evidence that every proposed action is safe. Keep controller
validation enabled.

### Browser task through one Bash tool

`tools/browser_bash_agent.sh` is a small local agent loop for testing browser use. It sends OpenAI-compatible tool calls
to the running Strata server, exposes only a bounded `bash` tool, and lets the model invoke Python Playwright from that
shell. It uses a disposable copy of `tests/browser_agent_fixture/index.html`; the repository page is never edited.
The independent `verify.py` file checks the result in real Chromium after the model finishes.

On this Jetson, install Python Playwright and an ARM64 Chromium browser first. The tested browser is Flatpak Chromium
through `/usr/local/bin/chromium`; the harness also needs `jq`, `curl`, `timeout` and `taskset`. Run:

```sh
bash tools/browser_bash_agent.sh
```

The script prints its artifact directory under `../Strata-data/browser-runs/`. Its lifecycle code resumes a paused
service and pauses it again on exit. This run began with the service already active and left it running. Shell commands
use a two-core CPU affinity, a 90-second timeout and a 4,000-character output limit. Flatpak may run its browser
processes outside that affinity. This is a supervised evaluation
harness, **not a security sandbox** for untrusted prompts or arbitrary websites; the model can still execute shell
code as the current user. Do not aim it at private pages or production accounts.

In the October 20, 2026 run, the model used six Bash calls over six model turns. It first reproduced the bug in
Chromium: clicking **Add entry** left the page at `0 entries` with no list item. It then changed the handler from
`entries.concat(text)` to `entries.push(text)`. Its own Playwright script reported 17 passing checks covering add,
count, blank input, case-insensitive filter, clear and another add. The separate acceptance test also passed and saved
a screenshot. Output speed on those six turns was 13.1–16.7 tokens/s. These are task timings, not a controlled speed
benchmark. The model did not follow the instruction to use `apply_patch` for edits; it used a Python file write.
The disposable workspace kept that write out of the repository, but the result reinforces why the shell tool needs
an external controller.

Two earlier attempts failed because Flatpak Chromium cannot open the host's `/tmp` files, and because the first
command guard rejected a valid resolved workspace path. The harness now creates workspaces under the home-directory
model-data folder and normalizes that path before checking commands. This test shows browser-tool use on one small
local page. Across the three attempts, swap in use rose from about 3.0 GiB to 5.8 GiB, while `free -h` showed about
4.0 GiB available at the end. This was not a controlled memory test. It does not establish reliability on live
websites, login flows, multi-page tasks or long autonomous runs.

## Measured speed

These are measurements from this one board. Short generations have more timing noise than long ones.

| Work | Prompt speed | Decode speed | Result |
| --- | ---: | ---: | --- |
| 3,327-token 4K needle check, final 6 GB reserve | 138.2 tok/s | 5.7 tok/s over 9 output tokens | exact code word found |
| tool turn, 559 prompt / 800 output tokens, 4 GB reserve | 42.7 tok/s | 14.6 tok/s | valid file-write call |
| tool correction, 634 prompt / 465 output tokens, 4 GB reserve | 45.9 tok/s | 15.1 tok/s | valid corrected call |
| test-file turn, 615 prompt tokens, 505 reused / 1,184 output tokens, 4 GB reserve | 29.8 tok/s for 110 newly read tokens | 17.0 tok/s | complete test-file call |
| forced shell call, final 6 GB reserve | 22.9 tok/s over 308 prompt tokens | 7.4 tok/s over 22 output tokens | exact command returned |

The earlier no-reserve community benchmark measured a median 11.30 decode tok/s over three 100-token answers and a
median 2.63-second time to first token. It also read 4,019 prompt tokens at 196.2 tok/s and 16,058 at 345.0 tok/s. Do
not compare those prompt rates directly with the final 4K needle: the prompts, cache state and expert-cache sizes were
different. The no-reserve run is retained only as pressure evidence because it swapped heavily.

## Test results and limits

Passed on the board:

- local CUDA 12 / `sm_87` source build and relink
- six Jetson setup tests: CUDA-driver fallback, unified-memory accounting, ARM64 instruction floor, cluster display,
  the 6 GB reserve and the two-job build limit
- ARM C++ tests `pool_tasks_test`, `router_dot_parity` and `expert_multi_test`
- complete Gyro-S download, hash check, pack, MTP preparation, model load and OpenAI API request
- 4K needle retrieval: 1 of 1 found
- real file-write, correction, shell-call and 11-test tool workflow

Two C++ fixture tests, `native_hadamard_test` and `tq_cpu_parity`, skipped because the pinned upstream ggml does not
contain the fork-only TQ fixture type IDs. The complete Gyro-S model exercised the production scalar TQ path instead.

The broad Terminal test run reported 812 passed, 17 skipped, 173 subtests passed and 46 failed. The failures were in
existing release-feed, x86 prebuilt-installer, GUI/AppData, shell-smoke and TUI assumptions on this ARM64 desktop. They
were not counted as Jetson passes. Pictures, multi-GPU, other Jetson models and a full 32K needle run were not tested.

## Port 11434 and long sessions

Strata already speaks the OpenAI API. To use the port many Ollama clients expect, stop anything already using 11434
and start Strata there:

```sh
.venv/bin/python serve/server.py --engine strata \
  --config strata-gyro-gyro-s.json --port 11434
```

Set the client base URL to `http://127.0.0.1:11434/v1`. This provides OpenAI-compatible `/v1` routes. It does not add
Ollama's separate `/api/generate` wire format. Port 11434 was occupied by Ollama on the measured board, so the Strata
port change itself was not tested there; all API tests used port 8080.

[`billion-context`](https://pi.dev/packages/billion-context) can sit in front of an OpenAI-compatible server and
compress old turns into summaries. It may help a long-running agent whose cumulative history is much larger than
Strata's real 32K window. It does not turn the model's attention or KV cache into a billion-token native window.

It is not a required Strata dependency. Its own project labels it early and says real-model integration testing is a
next milestone. Its default 50,000-token nudge interval is also larger than this deployment's entire window. Adding it
by default would change the model's tool space, add another Node process and make answers depend on generated summaries.

For an experimental loopback-only trial, Node 20 or newer is required. Pin the version and turn automatic updates off:

```sh
npm install -g billion-context@0.1.202 --prefix="$HOME/.local"
ACP_UPSTREAM=http://127.0.0.1:8080 \
  "$HOME/.local/bin/bili" start --host 127.0.0.1 --port 11434 --no-auto-update
```

Then try `http://127.0.0.1:11434/v1` as the client base URL. If that client requires explicit routing, use
`http://127.0.0.1:11434/bili/http://127.0.0.1:8080/v1`, which is billion-context's documented URL form. Declare the
model as 32,768 context tokens and lower its compression threshold before relying on it; see the upstream
[configuration reference](https://github.com/ranxianglei/billion-context/blob/master/CONFIGURATION.md). Keep it on
`127.0.0.1`: the proxy has no authentication. This sidecar was reviewed but not installed or included in the Jetson
performance results above.

## Troubleshooting

**The board becomes unresponsive.** Stop Strata, close memory-heavy programs and check `free -h`. Regenerate the config
with setup so it contains `--vram-reserve-mib 6144`. Do not reduce that value on a 32 GB board without a same-workload
memory measurement.

**Setup tries to use an x86 engine.** Update the checkout and run `./setup.sh` again. Jetson uses its installed JetPack
CUDA 12 toolkit and builds locally.

**The build makes the desktop slow.** Current setup caps Jetson builds at two jobs. You can lower it further:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=1 ./setup.sh --yes --family gyro --model Gyro-S --vision no --context 32768 --no-start
```

**Prompts are slow and the NVMe light stays on.** This is expected when cold experts do not fit in unified memory. Keep
the model on NVMe, reuse the conversation cache and avoid sending complete tool logs back on every turn.

**Port 8080 or 11434 is busy.** Check `ss -ltnp | grep ':8080\|:11434'`. Stop the existing server or choose another
loopback port. Do not expose Strata or a sidecar beyond `127.0.0.1` without authentication.

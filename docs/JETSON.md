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

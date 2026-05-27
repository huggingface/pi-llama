# pi-llama

llama.cpp / llama-swap Pi extension. Auto-discovers models from a running
`llama-server` or `llama-swap` and registers them as a configurable provider
in pi.

## Install

**From the shell:**

```bash
pi install https://github.com/huggingface/pi-llama
```

This clones to `~/.pi/agent/packages/pi-llama/` and adds an entry to your pi
settings. Every future `pi` invocation auto-loads it.

**From inside an interactive pi session:**

```
!pi install https://github.com/huggingface/pi-llama
```

Then run `/reload` (or restart pi) to load the extension.

**Dev mode:**

```bash
git clone https://github.com/huggingface/pi-llama ~/code/pi-llama
pi -e ~/code/pi-llama/index.ts
```

`-e` loads the extension only for the current session, useful while
developing.

## Configuration

Create `~/.pi/agent/pi-llama.json` for global config or
`./.pi/pi-llama.json` for project-local overrides. Both are JSON files; the
project-local file takes precedence.

| Key | Default | Description |
|-----|---------|-------------|
| `baseUrl` | `http://localhost:8080/v1` | API base URL (no trailing slash) |
| `providerId` | `llama-cpp` | Provider ID for model registration |
| `llamaSwapMode` | `false` | Use `/upstream/:id/props` instead of `/props?model=:id` |
| `defaultContextWindow` | `8192` | Fallback context window when `/v1/models` omits `n_ctx` |
| `propsTimeoutMs` | `120000` | Timeout for props discovery (ms) — increase for large models on slow networks. |
| `logPropsDiscovery` | `false` | Log verbose /props discovery progress (start, elapsed time, errors) to the notification area. |
| `apiKey` | `no-key` | API key forwarded to the provider |

Environment variables take precedence over config file values:
- `LLAMA_BASE_URL` overrides `baseUrl`
- `LLAMA_API_KEY` overrides `apiKey`

#### Verbose discovery logging

Set `"logPropsDiscovery": true` to log verbose /props discovery progress to
the notification area. This is off by default to keep the notification area
clean. When enabled you'll see messages like:

```
[llama-cpp] fetching /props for qwen3.6-35b-a3b-mtp…
[llama-cpp] contextWindow=262144 for qwen3.6-35b-a3b-mtp (33ms to resolve /props)
```

This is useful for debugging slow or failing model metadata discovery.

### llama-swap

To use with [llama-swap](https://github.com/refvm/llama-swap), set
`llamaSwapMode: true`. llama-swap does not expose a root `/props` endpoint;
instead it routes model details through
`/upstream/:modelid/props`.

Example `~/.pi/agent/pi-llama.json` for llama-swap:

```json
{
  "baseUrl": "http://localhost:8080/v1",
  "providerId": "llama-cpp",
  "llamaSwapMode": true,
  "defaultContextWindow": 8192
}
```

## Usage

```bash
# 1. Install and start llama-server
llama-server

# 2. Or start llama-swap
llama-swap

# 3. Configure ~/.pi/agent/pi-llama.json as needed

# 4. Launch pi
pi

# 5. Inside pi
/model              # pick models from your provider
```

### Environment Variable Override

You can override `baseUrl` and `apiKey` via environment variables without
editing the config file:

```bash
LLAMA_BASE_URL=http://10.0.0.5:3000/v1 LLAMA_API_KEY=secret pi
```

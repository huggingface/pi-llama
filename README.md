# pi-llama

llama.cpp Pi extension. Auto-discovers models from a running llama.cpp server and
registers them as the `llama-cpp` provider in [pi](https://pi.dev).

## Install

**From the shell:**

```bash
pi install git:github.com/huggingface/pi-llama
```

This clones to `~/.pi/agent/packages/pi-llama/` and adds an entry to your pi
settings. Every future `pi` invocation auto-loads it.

**From inside an interactive pi session:**

```
!pi install git:github.com/huggingface/pi-llama
```

Then run `/reload` (or restart pi) to load the extension.

**Dev mode:**

```bash
git clone https://github.com/huggingface/pi-llama ~/code/pi-llama
pi -e ~/code/pi-llama/index.ts
```

`-e` loads the extension only for the current session, useful while
developing.

## Environment Variables

This extension supports the following environment variables:

- LLAMA_BASE_URL (Default: `http://localhost:8080/v1`)
- LLAMA_API_KEY (Default: `no-key`)

## Usage

```bash
# 1. Install llama.cpp
curl -LsSf https://llama.app/install.sh | bash

# 2. Start the server
llama serve

# Optional: Use a remote instance of llama.cpp server instead of local
export LLAMA_BASE_URL="https://llama.example.com/v1"

# 3. Launch pi in another terminal
pi

# 4. Inside pi - search "llama-cpp" to browse your local models
/model
```

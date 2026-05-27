/**
 * llama.cpp provider for pi.
 *
 * Auto-discovers models from a running `llama-server` or `llama-swap` and
 * registers them under the `llama-cpp` provider.
 *
 * Config file (merged, project takes precedence):
 * - `~/.pi/agent/pi-llama.json` (global)
 * - `./.pi/pi-llama.json` (project-local)
 *
 * Example pi-llama.json:
 * ```json
 * {
 *   "baseUrl": "http://localhost:8080/v1",
 *   "providerId": "llama-swap",
 *   "llamaSwapMode": true,
 *   "defaultContextWindow": 8192,
 *   "propsTimeoutMs": 120000,
 *   "apiKey": "no-key"
 * }
 * ```
 *
 * `llamaSwapMode: true` tells the extension to use
 * `/upstream/:modelid/props` instead of `/props?model=:id`.
 *
 * Usage: `pi install github.com/huggingface/pi-llama`
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";

// ---- Config ---------------------------------------------------------------

/** Shape of the pi-llama.json configuration file. */
interface PiLlamaConfig {
	/** Base URL of the llama-server / llama-swap API. */
	baseUrl?: string;
	/** Provider ID used to register models. */
	providerId?: string;
	/** Use llama-swap upstream props endpoint (`/upstream/:id/props`). */
	llamaSwapMode?: boolean;
	/** Default context window when /v1/models does not report n_ctx. */
	defaultContextWindow?: number;
	/** Timeout in ms for props discovery requests. */
	propsTimeoutMs?: number;
	/** API key passed to the provider (default: "no-key"). */
	apiKey?: string;
	/** Log verbose /props discovery progress (start, elapsed time, errors). */
	logPropsDiscovery?: boolean;
}

/** Sensible defaults for the global pi-llama.json. */
const DEFAULT_GLOBAL_CONFIG: PiLlamaConfig = {
	baseUrl: "http://localhost:8080/v1",
	providerId: "llama-cpp",
	llamaSwapMode: false,
	defaultContextWindow: 8192,
	propsTimeoutMs: 120000,
	apiKey: "no-key",
	logPropsDiscovery: false,
};

/**
 * Create the global pi-llama.json with defaults if it does not already exist.
 */
function ensureGlobalConfig(): void {
	const globalPath = join(getAgentDir(), "pi-llama.json");
	if (existsSync(globalPath)) {
		return;
	}
	try {
		const agentDir = getAgentDir();
		if (!existsSync(agentDir)) {
			mkdirSync(agentDir, { recursive: true });
		}
		writeFileSync(globalPath, JSON.stringify(DEFAULT_GLOBAL_CONFIG, null, 2) + "\n", "utf-8");
		console.info(`[pi-llama] Created global config: ${globalPath}`);
	} catch (err) {
		console.warn(`[pi-llama] Failed to create global config: ${(err as Error).message}`);
	}
}

/**
 * Load merged config from global and project-local pi-llama.json files.
 *
 * Project-local config (`./.pi/pi-llama.json`) takes precedence over the
 * global config (`~/.pi/agent/pi-llama.json`).
 */
function loadConfig(cwd: string): PiLlamaConfig {
	const globalPath = join(getAgentDir(), "pi-llama.json");
	const projectPath = join(cwd, ".pi", "pi-llama.json");

	let globalConfig: PiLlamaConfig = {};
	let projectConfig: PiLlamaConfig = {};

	// Load global config
	if (existsSync(globalPath)) {
		try {
			globalConfig = JSON.parse(readFileSync(globalPath, "utf-8"));
		} catch {
			console.warn(`[pi-llama] Failed to parse global config: ${globalPath}`);
		}
	}

	// Load project-local config (overrides global)
	if (existsSync(projectPath)) {
		try {
			projectConfig = JSON.parse(readFileSync(projectPath, "utf-8"));
		} catch {
			console.warn(`[pi-llama] Failed to parse project config: ${projectPath}`);
		}
	}

	// Merge: project overrides global
	return { ...globalConfig, ...projectConfig };
}

// ---- Defaults (overridden by config) --------------------------------------

const PROVIDER_ID = "llama-cpp";
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
const PROPS_TIMEOUT_MS = 120_000;

const ModelsResponseSchema = Type.Object({
	data: Type.Optional(
		Type.Array(
			Type.Object({
				id: Type.String(),
				status: Type.Optional(
					Type.Object({
						value: Type.Optional(
							Type.Union([
								Type.Literal("unloaded"),
								Type.Literal("loading"),
								Type.Literal("loaded"),
								Type.Literal("sleeping"),
								Type.Literal("unknown"),
							]),
						),
					}),
				),
				architecture: Type.Optional(
					Type.Object({
						input_modalities: Type.Optional(Type.Array(Type.String())),
					}),
				),
				meta: Type.Optional(
					Type.Object({
						n_ctx: Type.Optional(Type.Number()),
						n_params: Type.Optional(Type.Number()),
					}),
				),
			}),
		),
	),
});

const validateModelsResponse = Compile(ModelsResponseSchema);

const PropsResponseSchema = Type.Object({
	default_generation_settings: Type.Optional(
		Type.Object({
			n_ctx: Type.Optional(Type.Number()),
		}),
	),
	chat_template: Type.Optional(Type.String()),
});

const validatePropsResponse = Compile(PropsResponseSchema);

type LlamaModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type ExtensionCtx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

// llama.cpp template thinking is boolean, so expose Pi's default off/medium toggle only.
const TEMPLATE_THINKING_LEVEL_MAP = {
	minimal: null,
	low: null,
	high: null,
	xhigh: null,
} satisfies NonNullable<LlamaModel["thinkingLevelMap"]>;

// Minimal shape needed to update both registered models and Pi's active model snapshot.
type MutableThinkingModel = {
	reasoning: boolean;
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	compat?: LlamaModel["compat"];
};

// Mark a model as using llama.cpp's chat_template_kwargs.enable_thinking control.
function applyTemplateThinkingSupport(model: MutableThinkingModel): void {
	model.reasoning = true;
	model.thinkingLevelMap = TEMPLATE_THINKING_LEVEL_MAP;
	model.compat = {
		...model.compat,
		// Despite the Pi enum name, this sends llama.cpp's generic
		// chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
		thinkingFormat: "qwen-chat-template",
	};
}

export default async function (pi: ExtensionAPI) {
	let currentModels: LlamaModel[] = [];

	pi.registerCommand("llama-version", {
		description: "Print llama-server --version output",
		handler: async (_args, ctx) => {
			const result = await pi.exec("llama-server", ["--version"]);
			const output = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
			const versionLine = output
				.split("\n")
				.map((l) => l.trim())
				.find((l) => /^version:\s/i.test(l));
			ctx.ui.notify(
				versionLine ?? `llama-server exited with code ${result.code}`,
				versionLine ? "info" : "error",
			);
		},
	});

	// Merge env vars (highest priority) with loaded config, falling back to defaults.
	ensureGlobalConfig();
	const config = loadConfig(process.cwd());

	const resolvedBaseUrl =
		process.env.LLAMA_BASE_URL ??
		config.baseUrl ??
		DEFAULT_BASE_URL;
	const resolvedProviderId = config.providerId ?? PROVIDER_ID;
	const resolvedApiKey = process.env.LLAMA_API_KEY ?? config.apiKey ?? "no-key";
	const resolvedContextWindow = config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW;
	const resolvedPropsTimeoutMs = config.propsTimeoutMs ?? PROPS_TIMEOUT_MS;
	const llamaSwapMode = !!config.llamaSwapMode;
	const logPropsDiscovery = !!config.logPropsDiscovery;

	const baseUrl = resolvedBaseUrl.replace(/\/+$/, "");

	async function refreshProvider(): Promise<void> {
		try {
			const response = await fetch(`${baseUrl}/models`);
			if (!response.ok) {
				console.warn(`[llama-cpp] ${baseUrl}/models returned ${response.status}`);
				return;
			}

			const payload: unknown = await response.json();
			if (!validateModelsResponse.Check(payload)) {
				const errors = [...validateModelsResponse.Errors(payload)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				console.warn(`[llama-cpp] invalid /models response: ${errors}`);
				return;
			}

			const previousById = new Map(currentModels.map((m) => [m.id, m]));

			currentModels = (payload.data ?? []).map((model) => {
				const previous = previousById.get(model.id);
				const isLoaded = model.status?.value === "loaded";
				const modalities = model.architecture?.input_modalities ?? ["text"];
				const input = modalities.filter(
					(m): m is "text" | "image" => m === "text" || m === "image",
				);
				const suffixes: string[] = [];
				if (input.includes("image")) {
					suffixes.push("(image)");
				}
				if (isLoaded) {
					suffixes.push("(loaded)");
				}
				return {
					id: model.id,
					name: suffixes.length > 0 ? `${model.id} ${suffixes.join(" ")}` : model.id,
					// /v1/models does not include /props-discovered capabilities, so preserve
					// template thinking metadata across refreshes.
					reasoning: previous?.reasoning ?? false,
					thinkingLevelMap: previous?.thinkingLevelMap,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow:
						model.meta?.n_ctx ?? previous?.contextWindow ?? resolvedContextWindow,
					compat: previous?.compat,
				} as LlamaModel;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-cpp] no models returned from ${baseUrl}/models`);
				return;
			}

			pi.registerProvider(resolvedProviderId, {
				name: "llama.cpp",
				baseUrl,
				apiKey: resolvedApiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			console.warn(`[llama-cpp] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredMetadata = new Set<string>();
	const pendingMetadata = new Set<string>();

	async function discoverModelMetadata(
		modelId: string,
		ctx?: ExtensionCtx,
		autoload = true,
		timeoutMs = resolvedPropsTimeoutMs,
		selectedModel?: MutableThinkingModel,
	): Promise<void> {
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}
		if (discoveredMetadata.has(modelId)) {
			// Provider re-registration does not update Pi's active model snapshot, so copy
			// already-discovered thinking metadata into the selected model when available.
			if (selectedModel && model.reasoning) {
				selectedModel.reasoning = model.reasoning;
				selectedModel.thinkingLevelMap = model.thinkingLevelMap;
				selectedModel.compat = model.compat;
			}
			return;
		}
		if (pendingMetadata.has(modelId)) {
			return;
		}

		pendingMetadata.add(modelId);
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);
		// llama-swap exposes model props at /upstream/:id/props;
		// vanilla llama-server uses /props?model=:id&autoload=:bool
		const propsBase = baseUrl.replace(/\/v1$/, "");
		const propsUrl = llamaSwapMode
			? `${propsBase}/upstream/${encodeURIComponent(modelId)}/props`
			: `${propsBase}/props?model=${encodeURIComponent(modelId)}&autoload=${autoload}`;

		const startedAt = performance.now();

		// Log that we're waiting so the user knows discovery is in progress.
		if (logPropsDiscovery) {
			ctx?.ui.notify(`[llama-cpp] fetching /props for ${modelId}…`, "info");
		}

		try {
			const response = await fetch(propsUrl, { signal: controller.signal });
			const data: unknown = await response.json();
			const elapsed = Math.round(performance.now() - startedAt);

			if (!response.ok) {
				ctx?.ui.notify(
					`[llama-cpp] /props for ${modelId} returned ${response.status} (${elapsed}ms to resolve /props)`,
					"error",
				);
				return;
			}
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx?.ui.notify(
					`[llama-cpp] invalid /props response for ${modelId} (${elapsed}ms to resolve /props): ${errors}`,
					"error",
				);
				return;
			}
			const nCtx = data.default_generation_settings?.n_ctx;
			let updated = false;
			if (typeof nCtx === "number" && nCtx > 0) {
				model.contextWindow = nCtx;
				if (logPropsDiscovery) {
					ctx?.ui.notify(
						`[llama-cpp] contextWindow=${nCtx} for ${modelId} (${elapsed}ms to resolve /props)`,
						"info",
					);
				}
				updated = true;
			}
			if (data.chat_template?.includes("enable_thinking") === true) {
				applyTemplateThinkingSupport(model);
				if (selectedModel) {
					applyTemplateThinkingSupport(selectedModel);
					if (pi.getThinkingLevel() === "off") {
						pi.setThinkingLevel("medium");
					}
				}
				updated = true;
			}
			discoveredMetadata.add(modelId);
			if (!updated) {
				if (logPropsDiscovery) {
					ctx?.ui.notify(
						`[llama-cpp] no new metadata for ${modelId} (${elapsed}ms to resolve /props)`,
						"info",
					);
				}
				return;
			}
			pi.registerProvider(resolvedProviderId, {
				name: "llama.cpp",
				baseUrl,
				apiKey: resolvedApiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			const elapsed = Math.round(performance.now() - startedAt);
			const err = error as Error;
			const msg = err.name === "AbortError" ? "timeout" : err.message;
			ctx?.ui.notify(
				`[llama-cpp] /props for ${modelId} failed: ${msg} (${elapsed}ms to resolve /props)`,
				"error",
			);
		} finally {
			clearTimeout(timer);
			pendingMetadata.delete(modelId);
		}
	}

	await refreshProvider();

	// Discover /props for models that are already active on session start.
	// This ensures context window and reasoning capabilities are discovered
	// even when resuming sessions or when a model is pre-selected.
	// Note: ctx.model may be undefined if the session started without a model selected,
	// or if model resolution happens after session_start fires.
	// In that case, the model_select event will handle it.
	const sessionStartHandler = (_event: { type: "session_start"; reason: string; previousSessionFile?: string }, ctx: { model?: any }) => {
		if (ctx.model?.provider === resolvedProviderId) {
			void discoverModelMetadata(ctx.model.id, ctx, true, resolvedPropsTimeoutMs, ctx.model);
		}
	};
	pi.on("session_start", sessionStartHandler);

	// If model_select fires (user selects a model), discover its props.
	pi.on("input", async (event) => {
		const trimmed = event.text.trim().toLowerCase();
		if (trimmed === "/model") {
			await refreshProvider();
		}
	});

	pi.on("model_select", (event, ctx) => {
		if (event.model.provider !== resolvedProviderId) {
			return;
		}
		void discoverModelMetadata(event.model.id, ctx, true, resolvedPropsTimeoutMs, event.model);
	});

	// Discover /props for already-active models because re-selecting them does not emit model_select.
	pi.on("before_provider_request", (event, ctx) => {
		const modelId = (event.payload as { model?: unknown })?.model;
		if (typeof modelId === "string") {
			const activeModel =
				ctx.model?.provider === resolvedProviderId && ctx.model.id === modelId ? ctx.model : undefined;
			void discoverModelMetadata(modelId, ctx, true, resolvedPropsTimeoutMs, activeModel);
		}
	});
}

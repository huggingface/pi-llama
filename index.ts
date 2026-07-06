/**
 * llama.cpp provider for pi.
 *
 * Auto-discovers models from a running `llama-server` and
 * registers them under the `llama-cpp` provider.
 *
 * Usage: `pi install github.com/huggingface/pi-llama`
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";
import { Loader, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

const PROVIDER_ID = "llama-cpp";
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
// llama.cpp has no output-token cap (no endpoint reports one; generation is only
// bounded by the context window), so use Pi's own default for models that omit
// maxTokens (see model-registry.ts parseModels).
const DEFAULT_MAX_TOKENS = 16384;
const PROPS_TIMEOUT_MS = 120_000;

const ModelsResponseSchema = Type.Object({
	data: Type.Optional(
		Type.Array(
			Type.Object({
				id: Type.String(),
				aliases: Type.Optional(Type.Array(Type.String())),
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
	build_info: Type.Optional(Type.String()),
});

const validatePropsResponse = Compile(PropsResponseSchema);

// SSE event types for model loading progress
type ApiModelLoadStage = "text_model" | "spec_model" | "mmproj_model";

type ApiModelsSseProgress = {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
};

type ApiModelsSseData = {
	status: string;
	progress?: ApiModelsSseProgress;
	exit_code?: number;
};

type ApiModelsSseEvent = {
	model: string;
	event: string;
	data: ApiModelsSseData;
};

const MODEL_LOAD_STAGE_LABELS: Record<ApiModelLoadStage, string> = {
	text_model: "Loading weights",
	spec_model: "Loading draft",
	mmproj_model: "Loading projector",
};

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
type MutableModelMetadata = {
	reasoning: boolean;
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	compat?: LlamaModel["compat"];
	contextWindow: number;
	maxTokens: number;
};

// Mark a model as using llama.cpp's chat_template_kwargs.enable_thinking control.
function applyTemplateThinkingSupport(model: MutableModelMetadata): void {
	model.reasoning = true;
	model.thinkingLevelMap = TEMPLATE_THINKING_LEVEL_MAP;
	model.compat = {
		...model.compat,
		// Despite the Pi enum name, this sends llama.cpp's generic
		// chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
		thinkingFormat: "qwen-chat-template",
	};
}

// Pi invalidates a captured ctx when the session is replaced (e.g. new_session in
// RPC mode). Any later ctx access then throws this error. Background work started
// before the replacement should treat it as "session gone" and stop quietly.
function isStaleContextError(error: unknown): boolean {
	return error instanceof Error && error.message.includes("stale after session replacement");
}

export default async function (pi: ExtensionAPI) {
	let currentModels: LlamaModel[] = [];

	pi.registerCommand("llama-version", {
		description: "Get build info of llama.cpp server",
		handler: async (_args, ctx) => {
			const response = await fetch(`${baseUrl.replace(/\/v1$/, "")}/props`);
			if (!response.ok) {
				ctx.ui.notify(`[llama-cpp] /props returned ${response.status}`, "error");
				return;
			}

			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx.ui.notify(`[llama-cpp] invalid /props response: ${errors}`, "error");
				return;
			}

			const match = data.build_info?.match(/^b([a-zA-Z0-9]+)-([a-zA-Z0-9]+)$/);

			if (match && match.length === 3) {
				ctx.ui.notify(`Build number: ${match[1]}, Commit hash: ${match[2]}`, "info");
			} else {
				ctx.ui.notify(`Malformed build info: ${data.build_info}`, "warning");
			}
		},
	});

	const baseUrl = (process.env.LLAMA_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
	const apiKey = process.env.LLAMA_API_KEY ?? "no-key";

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
				const contextWindow =
					model.meta?.n_ctx ?? previous?.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
				const displayName = model.aliases?.[0] || model.id;
				return {
					id: model.id,
					name: suffixes.length > 0 ? `${displayName} ${suffixes.join(" ")}` : displayName,
					// /v1/models does not include /props-discovered capabilities, so preserve
					// template thinking metadata across refreshes.
					reasoning: previous?.reasoning ?? false,
					thinkingLevelMap: previous?.thinkingLevelMap,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow,
					maxTokens: Math.min(DEFAULT_MAX_TOKENS, contextWindow),
					compat: previous?.compat,
					status: model.status,
				} as LlamaModel;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-cpp] no models returned from ${baseUrl}/models`);
				return;
			}

			// Track which model is currently loaded on the server
			const loadedModel = currentModels.find((m) => m.status?.value === "loaded");
			currentlyLoadedModel = loadedModel?.id ?? null;

			pi.registerProvider(PROVIDER_ID, {
				name: "llama.cpp",
				baseUrl,
				apiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			console.warn(`[llama-cpp] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredMetadata = new Set<string>();
	const pendingMetadata = new Set<string>();
	let currentlyLoadedModel: string | null = null;
	let statusTimeout: ReturnType<typeof setTimeout> | undefined;
	let sseAbortController: AbortController | null = null;
	let propsAbortController: AbortController | null = null;

	function clearFooterStatusTimeout(): void {
		if (statusTimeout !== undefined) {
			clearTimeout(statusTimeout);
			statusTimeout = undefined;
		}
	}

	// Connect to SSE stream for model loading progress
	async function connectToLoadingProgress(
		modelId: string,
		ctx: ExtensionCtx,
		loader: Loader,
	): Promise<void> {
		// Close any existing SSE connection
		if (sseAbortController) {
			sseAbortController.abort();
			sseAbortController = null;
		}

		sseAbortController = new AbortController();
		const signal = sseAbortController.signal;

		try {
			const response = await fetch(`${baseUrl.replace(/\/v1$/, "")}/models/sse`, { signal });

			if (!response.ok) {
				if (response.status !== 404) {
					ctx?.ui.notify(`[llama-cpp] loading progress ${response.status})`, "warning");
				}
				return;
			}

			const reader = response.body?.getReader();
			if (!reader) {
				return;
			}

			const decoder = new TextDecoder();
			let buffer = "";

			while (!signal.aborted) {
				const { done, value } = await reader.read();
				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const events = buffer.split("\n\n");
				buffer = events.pop() || "";

				for (const event of events) {
					if (!event) {
						continue;
					}

					// Parse SSE record: extract data lines
					const dataLines = event
						.split("\n")
						.filter((line) => line.startsWith("data:"))
						.map((line) => line.slice(5).trim())
						.join("\n");

					if (!dataLines) {
						continue;
					}

					try {
						const sseEvent: ApiModelsSseEvent = JSON.parse(dataLines);

						// Process status events for all models to keep discoveredMetadata and
						// currentlyLoadedModel in sync.
						if (
							sseEvent.event === "model_status" ||
							sseEvent.event === "status_change" ||
							sseEvent.event === "status_update"
						) {
							const status = sseEvent.data.status;

							if (status === "unloaded") {
								discoveredMetadata.delete(sseEvent.model);
								if (currentlyLoadedModel === sseEvent.model) {
									currentlyLoadedModel = null;
								}
							}
							if (status === "loaded") {
								currentlyLoadedModel = sseEvent.model;
							}
						}

						// Progress UI is only for the model we're actively loading
						if (sseEvent.model === modelId) {
							const currentModel = currentModels.find((m) => m.id === sseEvent.model);
							const displayName = currentModel?.name.split(" ")[0] || sseEvent.model;
							const progress = sseEvent.data.progress;

							if (sseEvent.data.exit_code && sseEvent.data.exit_code !== 0) {
								ctx?.ui.setWidget(PROVIDER_ID, [
									ctx?.ui.theme.fg("error", "[llama.cpp] ") +
										ctx?.ui.theme.fg(
											"text",
											`${displayName}:  failed (exit ${sseEvent.data.exit_code})`,
										),
								]);
								sseAbortController?.abort();
								return;
							}

							if (sseEvent.data.status === "loading" && progress) {
								const stageLabel = progress.current
									? MODEL_LOAD_STAGE_LABELS[progress.current] || progress.current
									: "Loading";
								const progressPercent = Math.round(progress.value * 100);
								loader?.setMessage(`${displayName}: ${stageLabel} (${progressPercent}%)`);
							}
						}
					} catch {
						// Ignore parse errors
					}
				}
			}
		} catch (error) {
			// Suppress errors from intentionally-aborted SSE connections and from a
			// stale ctx (session was replaced while streaming).
			const msg = (error as Error).message;
			if (
				isStaleContextError(error) ||
				(signal.aborted && (error instanceof DOMException || msg === "terminated"))
			) {
				return;
			}
			ctx?.ui.notify(`[llama-cpp] SSE error: ${msg}`, "warning");
		} finally {
			sseAbortController = null;
		}
	}

	async function discoverModelMetadata(
		modelId: string,
		ctx?: ExtensionCtx,
		autoload = true,
		timeoutMs = PROPS_TIMEOUT_MS,
		selectedModel?: MutableModelMetadata,
	): Promise<void> {
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}
		const displayName = model.name.split(" ")[0];
		// Use tracked state instead of stale currentModels status.
		const isLoaded = currentlyLoadedModel === modelId;

		if (discoveredMetadata.has(modelId)) {
			// If discovered but no longer loaded, clear cache and fall through to reload.
			if (!isLoaded) {
				discoveredMetadata.delete(modelId);
			} else {
				// Copy cached metadata into the selected model snapshot.
				if (selectedModel) {
					selectedModel.contextWindow = model.contextWindow;
					selectedModel.maxTokens = model.maxTokens;
					if (model.reasoning) {
						selectedModel.reasoning = model.reasoning;
						selectedModel.thinkingLevelMap = model.thinkingLevelMap;
						selectedModel.compat = model.compat;
					}
				}
				return;
			}
		}
		if (pendingMetadata.has(modelId)) {
			return;
		}

		pendingMetadata.add(modelId);
		// Cancel any pending clear timeout from a previous model load.
		clearFooterStatusTimeout();
		// Abort any in-flight /props request from a previous model.
		if (propsAbortController) {
			propsAbortController.abort();
		}
		propsAbortController = new AbortController();
		const timer = setTimeout(() => propsAbortController.abort(), timeoutMs);
		const shouldAutoload = autoload && !isLoaded;
		const propsUrl = `${baseUrl.replace(/\/v1$/, "")}/props?model=${encodeURIComponent(modelId)}&autoload=${shouldAutoload}`;
		const clearFooterStatusLater = () => {
			clearFooterStatusTimeout();
			statusTimeout = setTimeout(() => {
				statusTimeout = undefined;
				ctx?.ui.setWidget(PROVIDER_ID, undefined);
			}, 8000);
		};

		try {
			if (shouldAutoload && ctx) {
				let loader = null;
				ctx.ui.setWidget(PROVIDER_ID, (ui, theme) => {
					const prefix = theme.fg("accent", " [llama.cpp]");
					const prefixWidth = visibleWidth(" [llama.cpp]");
					loader = new Loader(
						ui,
						(s) => theme.fg("accent", s),
						(t) => theme.fg("text", t),
						`${displayName}: Loading...`,
					);
					return {
						dispose: () => loader?.stop(),
						render: (width: number) => {
							const [_, line] = loader.render(width - prefixWidth);
							return [prefix + truncateToWidth(line, width - prefixWidth)];
						},
					};
				});
				// Start SSE connection to monitor loading progress
				void connectToLoadingProgress(modelId, ctx, loader);
			}

			const response = await fetch(propsUrl, { signal: propsAbortController.signal });
			if (!response.ok) {
				// 500 during autoload is expected when the server cancels a load to start
				// another model. Suppress the notification for that case.
				if (!(shouldAutoload && response.status === 500)) {
					ctx?.ui.notify(`[llama-cpp] /props for ${modelId} returned ${response.status}`, "error");
				}
				return;
			}
			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx?.ui.notify(`[llama-cpp] invalid /props response for ${modelId}: ${errors}`, "error");
				return;
			}
			const nCtx = data.default_generation_settings?.n_ctx;
			let updated = false;
			let loadedFooterStatus = shouldAutoload ? `[llama.cpp] ${displayName} loaded` : undefined;
			if (typeof nCtx === "number" && nCtx > 0) {
				model.contextWindow = nCtx;
				model.maxTokens = Math.min(DEFAULT_MAX_TOKENS, nCtx);
				loadedFooterStatus = `[llama.cpp] ${displayName} loaded with ctx ${nCtx} tokens`;
				updated = true;
			}
			if (selectedModel) {
				selectedModel.contextWindow = model.contextWindow;
				selectedModel.maxTokens = model.maxTokens;
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
			if (shouldAutoload) {
				currentlyLoadedModel = modelId;
			}
			if (loadedFooterStatus && ctx && !isLoaded) {
				const prefix = ctx.ui.theme.fg("success", "[llama.cpp] ✓");
				ctx.ui.setWidget(PROVIDER_ID, [
					prefix +
						ctx.ui.theme.fg(
							"text",
							` ${displayName}: Loaded` + (nCtx ? ` with context ${nCtx} tokens` : ""),
						),
				]);
				clearFooterStatusLater();
			}
			if (!updated) {
				return;
			}
			pi.registerProvider(PROVIDER_ID, {
				name: "llama.cpp",
				baseUrl,
				apiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			const err = error as Error;
			// Suppress notification for aborted requests (model was switched) and for a
			// stale ctx (session was replaced while awaiting) — both are expected.
			if (err.name !== "AbortError" && !isStaleContextError(err)) {
				ctx?.ui.notify(`[llama-cpp] /props for ${modelId} failed: ${err.message}`, "error");
			}
		} finally {
			clearTimeout(timer);
			pendingMetadata.delete(modelId);
			propsAbortController = null;
			// Stop SSE connection when done
			if (sseAbortController) {
				sseAbortController.abort();
				sseAbortController = null;
			}
		}
	}

	await refreshProvider();

	pi.on("input", async (event) => {
		const trimmed = event.text.trim().toLowerCase();
		if (trimmed === "/model") {
			await refreshProvider();
		}
	});

	pi.on("model_select", (event, ctx) => {
		if (event.model.provider !== PROVIDER_ID) {
			return;
		}
		void discoverModelMetadata(event.model.id, ctx, true, PROPS_TIMEOUT_MS, event.model);
	});

	// Discover /props for already-active models because re-selecting them does not emit model_select.
	pi.on("before_provider_request", (event, ctx) => {
		try {
			const modelId = (event.payload as { model?: unknown })?.model;
			if (typeof modelId === "string") {
				const activeModel =
					ctx.model?.provider === PROVIDER_ID && ctx.model.id === modelId ? ctx.model : undefined;
				void discoverModelMetadata(modelId, ctx, true, PROPS_TIMEOUT_MS, activeModel);
			}
		} catch (error) {
			// Session was replaced as the request fired; nothing to discover.
			if (!isStaleContextError(error)) {
				throw error;
			}
		}
	});

	pi.on("session_shutdown", () => {
		clearFooterStatusTimeout();
		// Stop in-flight /props and SSE so they don't resume against a stale ctx.
		propsAbortController?.abort();
		sseAbortController?.abort();
	});
}

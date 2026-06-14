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

const PROVIDER_ID = "llama-cpp";
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
const PROPS_TIMEOUT_MS = 120_000;

const ModelsResponseSchema = Type.Object({
	models: Type.Optional(
		Type.Array(
			Type.Object({
				name: Type.Optional(Type.String()),
				model: Type.Optional(Type.String()),
				capabilities: Type.Optional(Type.Array(Type.String())),
			}),
		),
	),
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
				capabilities: Type.Optional(Type.Array(Type.String())),
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
	modalities: Type.Optional(
		Type.Object({
			vision: Type.Optional(Type.Boolean()),
			video: Type.Optional(Type.Boolean()),
			audio: Type.Optional(Type.Boolean()),
		}),
	),
});

const validatePropsResponse = Compile(PropsResponseSchema);

type LlamaModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type DiscoveredCapabilities = {
	multimodal?: boolean;
	audio?: boolean;
	video?: boolean;
};
type ExtensionCtx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

// llama.cpp template thinking is boolean, so expose Pi's default off/medium toggle only.
const TEMPLATE_THINKING_LEVEL_MAP = {
	minimal: null,
	low: null,
	high: null,
	xhigh: null,
} satisfies NonNullable<LlamaModel["thinkingLevelMap"]>;

// Minimal shape needed to update both registered models and Pi's active model snapshot.
type MutableDiscoveredModel = {
	id: string;
	name: string;
	input: LlamaModel["input"];
	reasoning: boolean;
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	compat?: LlamaModel["compat"];
};

// Mark a model as using llama.cpp's chat_template_kwargs.enable_thinking control.
function applyTemplateThinkingSupport(model: MutableDiscoveredModel): void {
	model.reasoning = true;
	model.thinkingLevelMap = TEMPLATE_THINKING_LEVEL_MAP;
	model.compat = {
		...model.compat,
		// Despite the Pi enum name, this sends llama.cpp's generic
		// chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
		thinkingFormat: "qwen-chat-template",
	};
}

function dedupeInputs(input: LlamaModel["input"]): LlamaModel["input"] {
	return [...new Set(input)] as LlamaModel["input"];
}

function buildModelName(
	id: string,
	input: LlamaModel["input"],
	isLoaded: boolean,
	capabilities?: DiscoveredCapabilities,
): string {
	const suffixes: string[] = [];
	if (input.includes("image")) {
		suffixes.push("image");
	} else if (capabilities?.multimodal) {
		suffixes.push("multimodal");
	}
	if (capabilities?.audio) {
		suffixes.push("audio");
	}
	if (capabilities?.video) {
		suffixes.push("video");
	}
	if (isLoaded) {
		suffixes.push("loaded");
	}
	return suffixes.length > 0 ? `${id} (${suffixes.join(", ")})` : id;
}

function getModelInputFromListing(model: {
	architecture?: { input_modalities?: string[] };
}): LlamaModel["input"] {
	const input: LlamaModel["input"] = ["text"];
	for (const modality of model.architecture?.input_modalities ?? []) {
		if (modality === "text" || modality === "image") {
			input.push(modality);
		}
	}
	return dedupeInputs(input);
}

function getDiscoveredCapabilitiesFromListing(model: {
	capabilities?: string[];
}): DiscoveredCapabilities {
	return {
		multimodal: model.capabilities?.includes("multimodal") === true || undefined,
	};
}

function applyPropsModalities(
	model: MutableDiscoveredModel,
	isLoaded: boolean,
	capabilities: DiscoveredCapabilities,
	modalities?: { vision?: boolean; audio?: boolean; video?: boolean },
): { updated: boolean; capabilities: DiscoveredCapabilities } {
	let updated = false;
	if (modalities?.vision && !model.input.includes("image")) {
		model.input = dedupeInputs([...model.input, "image"]);
		updated = true;
	}
	const nextCapabilities: DiscoveredCapabilities = {
		...capabilities,
		audio: modalities?.audio || capabilities.audio || undefined,
		video: modalities?.video || capabilities.video || undefined,
	};
	if (capabilities.audio !== nextCapabilities.audio || capabilities.video !== nextCapabilities.video) {
		updated = true;
	}
	const nextName = buildModelName(model.id, model.input, isLoaded, nextCapabilities);
	if (model.name !== nextName) {
		model.name = nextName;
		updated = true;
	}
	return { updated, capabilities: nextCapabilities };
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
			const listingCapabilitiesById = new Map<string, string[]>();
			for (const listedModel of payload.models ?? []) {
				const capabilities = listedModel.capabilities ?? [];
				if (listedModel.model) {
					listingCapabilitiesById.set(listedModel.model, capabilities);
				}
				if (listedModel.name) {
					listingCapabilitiesById.set(listedModel.name, capabilities);
				}
			}

			currentModels = (payload.data ?? []).map((model) => {
				const previous = previousById.get(model.id);
				const isLoaded = model.status?.value === "loaded";
				const input = getModelInputFromListing(model);
				const discoveredCapabilities =
					discoveredCapabilitiesById.get(model.id) ??
					getDiscoveredCapabilitiesFromListing({
						capabilities: model.capabilities ?? listingCapabilitiesById.get(model.id),
					});
				discoveredCapabilitiesById.set(model.id, discoveredCapabilities);
				return {
					id: model.id,
					name: buildModelName(model.id, input, isLoaded, discoveredCapabilities),
					// /v1/models does not include /props-discovered capabilities, so preserve
					// template thinking metadata across refreshes.
					reasoning: previous?.reasoning ?? false,
					thinkingLevelMap: previous?.thinkingLevelMap,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: model.meta?.n_ctx ?? previous?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
					compat: previous?.compat,
				} as LlamaModel;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-cpp] no models returned from ${baseUrl}/models`);
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
			console.warn(`[llama-cpp] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredMetadata = new Set<string>();
	const discoveredCapabilitiesById = new Map<string, DiscoveredCapabilities>();
	const pendingMetadata = new Set<string>();
	let statusTimeout: ReturnType<typeof setTimeout> | undefined;

	function clearFooterStatusTimeout(): void {
		if (statusTimeout !== undefined) {
			clearTimeout(statusTimeout);
			statusTimeout = undefined;
		}
	}

	async function discoverModelMetadata(
		modelId: string,
		ctx?: ExtensionCtx,
		autoload = true,
		timeoutMs = PROPS_TIMEOUT_MS,
		selectedModel?: MutableDiscoveredModel,
	): Promise<void> {
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}
		if (discoveredMetadata.has(modelId)) {
			// Provider re-registration does not update Pi's active model snapshot, so copy
			// already-discovered metadata into the selected model when available.
			if (selectedModel) {
				selectedModel.name = model.name;
				selectedModel.input = model.input;
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
		const propsUrl = `${baseUrl.replace(/\/v1$/, "")}/props?model=${encodeURIComponent(modelId)}&autoload=${autoload}`;
		const clearFooterStatusLater = () => {
			clearFooterStatusTimeout();
			statusTimeout = setTimeout(() => {
				statusTimeout = undefined;
				ctx?.ui.setStatus(PROVIDER_ID, undefined);
			}, 8000);
		};

		try {
			if (autoload && ctx) {
				ctx.ui.setStatus(PROVIDER_ID, ctx.ui.theme.fg("dim", `[llama.cpp] loading: ${modelId}`));
			}

			const response = await fetch(propsUrl, { signal: controller.signal });
			if (!response.ok) {
				ctx?.ui.setStatus(PROVIDER_ID, undefined);
				ctx?.ui.notify(`[llama-cpp] /props for ${modelId} returned ${response.status}`, "error");
				return;
			}
			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx?.ui.setStatus(PROVIDER_ID, undefined);
				ctx?.ui.notify(`[llama-cpp] invalid /props response for ${modelId}: ${errors}`, "error");
				return;
			}
			const nCtx = data.default_generation_settings?.n_ctx;
			let updated = false;
			const isLoaded = model.name.includes("loaded");
			const currentCapabilities = discoveredCapabilitiesById.get(modelId) ?? {};
			const modelUpdate = applyPropsModalities(model, isLoaded, currentCapabilities, data.modalities);
			discoveredCapabilitiesById.set(modelId, modelUpdate.capabilities);
			updated = modelUpdate.updated || updated;
			if (selectedModel) {
				updated =
					applyPropsModalities(selectedModel, isLoaded, modelUpdate.capabilities, data.modalities).updated ||
					updated;
			}
			let loadedFooterStatus = autoload ? `[llama.cpp] ${modelId} loaded` : undefined;
			if (typeof nCtx === "number" && nCtx > 0) {
				model.contextWindow = nCtx;
				model.name = buildModelName(model.id, model.input, true, discoveredCapabilitiesById.get(model.id));
				if (selectedModel) {
					selectedModel.name = model.name;
				}
				loadedFooterStatus = `[llama.cpp] ${modelId} loaded with ctx ${nCtx} tokens`;
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
			if (loadedFooterStatus && ctx) {
				ctx.ui.setStatus(PROVIDER_ID, ctx.ui.theme.fg("dim", loadedFooterStatus));
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
			const msg = err.name === "AbortError" ? "timeout" : err.message;
			ctx?.ui.setStatus(PROVIDER_ID, undefined);
			ctx?.ui.notify(`[llama-cpp] /props for ${modelId} failed: ${msg}`, "error");
		} finally {
			clearTimeout(timer);
			pendingMetadata.delete(modelId);
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
		const modelId = (event.payload as { model?: unknown })?.model;
		if (typeof modelId === "string") {
			const activeModel =
				ctx.model?.provider === PROVIDER_ID && ctx.model.id === modelId ? ctx.model : undefined;
			void discoverModelMetadata(modelId, ctx, true, PROPS_TIMEOUT_MS, activeModel);
		}
	});

	pi.on("session_shutdown", () => {
		clearFooterStatusTimeout();
	});
}

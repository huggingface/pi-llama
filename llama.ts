import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";

const DEFAULT_BASE_URL = "http://localhost:8080/v1";
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
});

const validatePropsResponse = Compile(PropsResponseSchema);

type LlamaModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type ExtensionCtx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

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

	const baseUrl = (process?.env?.LLAMA_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");

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
					suffixes.push("(loaded ✅)");
				}
				return {
					id: model.id,
					name: suffixes.length > 0 ? `${model.id} ${suffixes.join(" ")}` : model.id,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: model.meta?.n_ctx ?? previousById.get(model.id)?.contextWindow ?? 0,
					// maxTokens: -1,
				} as LlamaModel;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-cpp] no models returned from ${baseUrl}/models`);
				return;
			}

			pi.registerProvider("llama-cpp", {
				name: "llama.cpp",
				baseUrl,
				apiKey: "LLAMA_API_KEY",
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			console.warn(`[llama-cpp] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredContext = new Set<string>();
	const pendingContext = new Set<string>();

	async function discoverContextWindow(modelId: string, ctx: ExtensionCtx): Promise<void> {
		if (discoveredContext.has(modelId) || pendingContext.has(modelId)) {
			return;
		}
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}

		pendingContext.add(modelId);
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), PROPS_TIMEOUT_MS);
		const propsBase = `${baseUrl.replace(/\/v1$/, "")}/props?model=${encodeURIComponent(modelId)}&autoload=`;

		try {
			let response = await fetch(`${propsBase}false`, { signal: controller.signal });
			if (response.status === 400 || response.status === 404) {
				response = await fetch(`${propsBase}true`, { signal: controller.signal });
			}
			if (!response.ok) {
				ctx.ui.notify(`[llama-cpp] /props for ${modelId} returned ${response.status}`, "error");
				return;
			}
			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx.ui.notify(`[llama-cpp] invalid /props response for ${modelId}: ${errors}`, "error");
				return;
			}
			const nCtx = data.default_generation_settings?.n_ctx;
			if (typeof nCtx === "number" && nCtx > 0) {
				model.contextWindow = nCtx;
				discoveredContext.add(modelId);
				pi.registerProvider("llama-cpp", {
					name: "llama.cpp",
					baseUrl,
					apiKey: "LLAMA_API_KEY",
					api: "openai-completions",
					models: currentModels,
				});
				ctx.ui.notify(`[llama-cpp] contextWindow=${nCtx} for ${modelId}`, "info");
			}
		} catch (error) {
			const err = error as Error;
			const msg = err.name === "AbortError" ? "timeout" : err.message;
			ctx.ui.notify(`[llama-cpp] /props for ${modelId} failed: ${msg}`, "error");
		} finally {
			clearTimeout(timer);
			pendingContext.delete(modelId);
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
		if (event.model.provider !== "llama-cpp") {
			return;
		}
		void discoverContextWindow(event.model.id, ctx);
	});

	pi.on("before_provider_request", (event, ctx) => {
		const modelId = (event.payload as { model?: unknown })?.model;
		if (typeof modelId === "string") {
			void discoverContextWindow(modelId, ctx);
		}
	});
}

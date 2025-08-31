import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import pino from "pino";
import pinoHttp from "pino-http";
import { v4 as uuidv4 } from "uuid";

// SDK-и за трите доставчика
import OpenAI from "openai";                  // npm i openai
import Anthropic from "@anthropic-ai/sdk";    // npm i @anthropic-ai/sdk
import { GoogleGenerativeAI } from "@google/generative-ai"; // npm i @google/generative-ai

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// ------------- ЛОГВАНЕ -------------
const logger = pino({ level: process.env.LOG_LEVEL || "info" });
app.use(pinoHttp({ logger }));

// ------------- СИГУРНОСТ + RATE LIMIT -------------
app.use(helmet());
app.use(
  rateLimit({
    windowMs: 60 * 1000,
    max: Number(process.env.RATE_LIMIT_PER_MIN || 60),
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Too many requests, try again later." },
  })
);

// ------------- CORS -------------
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "http://localhost:3001",
      process.env.FRONTEND_ORIGIN || "" // за прод
    ].filter(Boolean),
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);
app.options("*", cors());
app.use(express.json({ limit: "512kb" }));

// ------------- КОНФИГ -------------
const USE_MOCK = String(process.env.USE_MOCK || "").toLowerCase() === "true";

// Приоритет на провайдъри (винаги можем да променяме реда)
const PROVIDER_ORDER = (process.env.PROVIDERS || "openai,anthropic,gemini")
  .split(",")
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

// Таймаут на опит към доставчик (ms)
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 8000);

// ------------- ИНИЦИАЛИЗАЦИЯ НА КЛИЕНТИ -------------
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
const gemini = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY) : null;

// ------------- ПОЛЕЗНИ ПОДСКАЗКИ (PROMPTS) -------------

const CONTEXT_RULES = {
  boss: `Keep it concise, outcome-focused, respectful. Surface risks, decisions and deadlines.`,
  client: `Be courteous and service-oriented. Reduce friction, confirm next steps and deadlines.`,
  colleague: `Be collaborative and specific. Offer help or options. Avoid blame.`,
  friend: `Be warm and informal but still clear. Keep it short.`,
};

function buildRewritePrompt(text, tone = "neutral", context = "colleague") {
  const rules = CONTEXT_RULES[context] || CONTEXT_RULES.colleague;
  return `
You are ClearSpeak, an assistant that rewrites business messages.

GOAL:
- Rewrite the user's message to be clear, concise, professional.
- Tone: ${tone}.
- Context: ${context}. Guidelines: ${rules}

HARD RULES:
- Preserve meaning and intent.
- Keep it short. Avoid fluff.
- Return ONLY the rewritten text. No preface or extra quotes.

USER MESSAGE:
"""${text}"""
`.trim();
}

function buildAnalyzePrompt(text, context = "colleague") {
  return `
Analyze the following message.

Return STRICT JSON with:
{
  "tone": one of ["Friendly","Professional","Neutral","Direct","Formal","Informal","Apologetic"],
  "score": integer 0..100,
  "suggestions": [exactly 3 short, actionable tips]
}

Consider context="${context}".

TEXT:
"""${text}"""
`.trim();
}

// ------------- ВЪЗПРОИЗВЕЖДАНЕ С ТАЙМАУТ -------------
function withTimeout(promise, ms = TIMEOUT_MS, label = "call") {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`TIMEOUT_${label}`)), ms)),
  ]);
}

// ------------- ПРОВАЙДЪР ВИКАНИЯ -------------
async function callOpenAI(prompt, isJSON = false) {
  if (!openai) throw new Error("OPENAI_DISABLED");
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";
  const response = await withTimeout(
    openai.responses.create({
      model,
      input: prompt,
      // за JSON отговор (tone analysis)
      ...(isJSON ? { response_format: { type: "json_object" } } : {}),
      max_output_tokens: Number(process.env.OPENAI_MAX_TOKENS || 400),
      temperature: Number(process.env.OPENAI_TEMPERATURE || 0.3),
    }),
    TIMEOUT_MS,
    "openai"
  );
  const out = response?.output_text?.trim?.() || response?.output?.[0]?.content?.[0]?.text?.trim?.();
  if (!out) throw new Error("OPENAI_EMPTY");
  return out;
}

async function callAnthropic(prompt, isJSON = false) {
  if (!anthropic) throw new Error("ANTHROPIC_DISABLED");
  const model = process.env.ANTHROPIC_MODEL || "claude-3-5-sonnet-20240620";
  const response = await withTimeout(
    anthropic.messages.create({
      model,
      max_tokens: Number(process.env.ANTHROPIC_MAX_TOKENS || 400),
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE || 0.3),
      messages: [{ role: "user", content: prompt }],
      ...(isJSON ? { response_format: { type: "json_object" } } : {}),
    }),
    TIMEOUT_MS,
    "anthropic"
  );
  const out = response?.content?.[0]?.text?.trim?.();
  if (!out) throw new Error("ANTHROPIC_EMPTY");
  return out;
}

async function callGemini(prompt, isJSON = false) {
  if (!gemini) throw new Error("GEMINI_DISABLED");
  const model = gemini.getGenerativeModel({ model: process.env.GEMINI_MODEL || "gemini-1.5-flash" });
  const result = await withTimeout(model.generateContent(prompt), TIMEOUT_MS, "gemini");
  const out = result?.response?.text()?.trim();
  if (!out) throw new Error("GEMINI_EMPTY");
  return out;
}

// универсална функция – минава през провайдъри по ред и връща първия успешен
async function callProviders(prompt, { isJSON = false } = {}) {
  const errors = [];
  for (const p of PROVIDER_ORDER) {
    try {
      if (p === "openai") return await callOpenAI(prompt, isJSON);
      if (p === "anthropic") return await callAnthropic(prompt, isJSON);
      if (p === "gemini") return await callGemini(prompt, isJSON);
    } catch (err) {
      errors.push(`${p}:${err.message}`);
    }
  }
  const e = new Error("ALL_PROVIDERS_FAILED");
  e.details = errors;
  throw e;
}

// ------------- HEALTH -------------
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    mode: USE_MOCK ? "MOCK" : "LIVE",
    providers: {
      openai: Boolean(openai),
      anthropic: Boolean(anthropic),
      gemini: Boolean(gemini),
      order: PROVIDER_ORDER,
    },
    port: PORT,
  });
});

// ------------- PRESETS (за фронта) -------------
const PRESETS = [
  { name: "Follow-up (Client)", tone: "friendly", context: "client", hint: "Short, polite reminder asking for status/ETA." },
  { name: "Escalation (Boss)", tone: "assertive", context: "boss", hint: "Clear blocker + needed decision + concise next steps." },
  { name: "Nudge (Colleague)", tone: "neutral", context: "colleague", hint: "Gentle reminder about a task with a concrete ask." },
  { name: "Warm ping (Friend)", tone: "friendly", context: "friend", hint: "Casual, warm check-in. Keep it short." },
];
app.get("/presets", (_req, res) => res.json(PRESETS));

// ------------- ENDPOINTS -------------

app.post("/rewrite-email", async (req, res) => {
  const reqId = uuidv4();
  try {
    const { text, tone = "neutral", context = "colleague" } = req.body || {};
    if (!text || typeof text !== "string" || !text.trim()) {
      return res.status(400).json({ error: "NO_TEXT" });
    }

    if (USE_MOCK) {
      return res.json({ result: `(mock ${tone}/${context}) ${text}` });
    }

    const prompt = buildRewritePrompt(text, tone, context);
    const out = await callProviders(prompt, { isJSON: false });

    req.log.info({ reqId, route: "rewrite-email", ok: true });
    return res.json({ result: out });
  } catch (err) {
    req.log.error({ err: err.message, details: err.details });
    const msg = err.details ? `${err.message} :: ${err.details.join("|")}` : err.message;
    return res.status(500).json({ error: msg || "AI_FAILED" });
  }
});

app.post("/analyze-tone", async (req, res) => {
  const reqId = uuidv4();
  try {
    const { text, context = "colleague" } = req.body || {};
    if (!text || typeof text !== "string" || !text.trim()) {
      return res.status(400).json({ error: "NO_TEXT" });
    }

    if (USE_MOCK) {
      return res.json({
        tone: "Neutral",
        score: 72,
        suggestions: ["Make it shorter.", "Clarify the ask/CTA.", "Offer a specific deadline."],
      });
    }

    const prompt = buildAnalyzePrompt(text, context);
    const raw = await callProviders(prompt, { isJSON: true });

    // опит за JSON parse (някои модели все още връщат текст)
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      const m = String(raw).match(/\{[\s\S]*\}/);
      data = m ? JSON.parse(m[0]) : null;
    }
    if (!data) throw new Error("BAD_JSON");

    data.tone = String(data.tone || "Neutral");
    data.score = Math.max(0, Math.min(100, Number(data.score) || 0));
    data.suggestions = Array.isArray(data.suggestions) ? data.suggestions.slice(0, 3) : [];

    req.log.info({ reqId, route: "analyze-tone", ok: true });
    return res.json(data);
  } catch (err) {
    req.log.error({ err: err.message, details: err.details });
    const msg = err.details ? `${err.message} :: ${err.details.join("|")}` : err.message;
    return res.status(500).json({ error: msg || "AI_FAILED" });
  }
});

app.listen(PORT, () => {
  logger.info(`🚀 ClearSpeak API running on http://localhost:${PORT} (mode=${USE_MOCK ? "MOCK" : "LIVE"})`);
});
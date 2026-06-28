/**
 * Tiny typed wrapper around `fetch` for the admin dashboard SPA.
 *
 * The dashboard is mounted under the orchestrator's own origin, so every
 * request is same-origin and uses relative `/api/dashboard/*` URLs. This
 * file is the single place that knows the URL shape, the response types,
 * and the `Authorization` header convention — components only import named
 * helpers.
 *
 * Auth: when the orchestrator runs with auth enabled (Entra ID), `require_admin`
 * expects a bearer token. The dashboard is currently designed for the same
 * tenant and lets the browser session forward whatever cookie/header the
 * gateway injects, so this module does not manage tokens directly. If a token
 * is ever stashed under `localStorage["dashboard.bearer"]` we forward it; this
 * preserves the existing manual-testing flow without committing to a long-term
 * auth model in the dashboard.
 */

export class ApiError extends Error {
  status: number;
  body?: unknown;
  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function authHeaders(): HeadersInit {
  try {
    const token = typeof window !== "undefined"
      ? window.localStorage?.getItem("dashboard.bearer")
      : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(path, {
    ...init,
    signal,
    headers: {
      Accept: "application/json",
      ...authHeaders(),
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    let body: unknown;
    let message = `${init.method ?? "GET"} ${path} -> ${res.status}`;
    try {
      body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        const detail = (body as { detail: unknown }).detail;
        if (typeof detail === "string") {
          message = detail;
        } else if (detail && typeof detail === "object" && "errors" in detail) {
          message = `Validation failed for ${
            ((detail as { errors: Array<{ key: string }> }).errors || [])
              .map((e) => e.key)
              .join(", ") || "request"
          }`;
        }
      }
    } catch {
      // body was not JSON
    }
    throw new ApiError(message, res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Existing endpoints (pre-PR #236 surface — restored here because lib/api.ts
// was missing from that PR and the rest of the SPA imports from it).
// ---------------------------------------------------------------------------

export interface DailyPoint {
  date: string;
  count: number;
}

export interface OverviewResponse {
  total: number;
  today: number;
  last_7_days: number;
  last_30_days: number;
  active_users: number;
  avg_turns: number;
  conversations_per_day: DailyPoint[];
  window_days: number;
  /** Inclusive ISO date the backend used as the window start. */
  from?: string;
  /** Inclusive ISO date the backend used as the window end. */
  to?: string;
  /** Conversations created inside the active window. */
  in_window_count?: number;
}

export interface ConversationSummary {
  id: string;
  name?: string;
  principal_id?: string;
  created_at?: number;
  last_updated?: string;
  message_count?: number;
}

export interface ConversationMessage {
  role?: string;
  content?: unknown;
  /** Backend reconstruction emits the message body under "text" (compat with `questions[]`). */
  text?: unknown;
  timestamp?: string;
  question_id?: string;
}

export interface ConversationFeedback {
  question_id?: string;
  rating?: string;
  comment?: string;
  [key: string]: unknown;
}

export interface ConversationDetail {
  id: string;
  name?: string;
  principal_id?: string;
  created_at?: number;
  last_updated?: string;
  messages: ConversationMessage[];
  /** Azure AI Foundry agent thread id, if the orchestrator persisted one. */
  thread_id?: string;
  /** Feedback entries captured against the conversation. */
  feedback?: ConversationFeedback[];
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  has_more: boolean;
  skip: number;
  limit: number;
}

export function fetchVersion(signal?: AbortSignal): Promise<string> {
  return request<{ version: string }>("/api/dashboard/version", {}, signal).then(
    (r) => r.version,
  );
}

export interface OverviewQuery {
  /** Trailing window in days; defaults to 30 if no custom range is set. */
  days?: number;
  /** Inclusive ISO date (YYYY-MM-DD) for custom range start. */
  from?: string;
  /** Inclusive ISO date (YYYY-MM-DD) for custom range end. */
  to?: string;
}

export function fetchOverview(
  q: OverviewQuery | number = 30,
  signal?: AbortSignal,
): Promise<OverviewResponse> {
  // Back-compat: a bare number still means trailing-window days.
  const opts: OverviewQuery = typeof q === "number" ? { days: q } : q;
  const params = new URLSearchParams();
  if (opts.from && opts.to) {
    params.set("from", opts.from);
    params.set("to", opts.to);
  } else if (opts.days != null) {
    params.set("days", String(opts.days));
  }
  const qs = params.toString();
  return request<OverviewResponse>(
    `/api/dashboard/overview${qs ? `?${qs}` : ""}`,
    {},
    signal,
  );
}

export interface ConversationsQuery {
  skip?: number;
  limit?: number;
  search?: string;
}

export function fetchConversations(
  q: ConversationsQuery,
  signal?: AbortSignal,
): Promise<ConversationListResponse> {
  const params = new URLSearchParams();
  if (q.skip != null) params.set("skip", String(q.skip));
  if (q.limit != null) params.set("limit", String(q.limit));
  if (q.search) params.set("search", q.search);
  return request<ConversationListResponse>(
    `/api/dashboard/conversations${params.toString() ? `?${params}` : ""}`,
    {},
    signal,
  );
}

export function fetchConversationDetail(
  id: string,
  signal?: AbortSignal,
): Promise<ConversationDetail> {
  return request<ConversationDetail>(
    `/api/dashboard/conversations/${encodeURIComponent(id)}`,
    {},
    signal,
  );
}

export function formatUtc(value?: string | number | null): string {
  if (value == null || value === "") return "-";
  try {
    const date =
      typeof value === "number"
        ? new Date(value * 1000)
        : new Date(value);
    return date.toISOString().replace("T", " ").replace(/\..+$/, "Z");
  } catch {
    return String(value);
  }
}

// ---------------------------------------------------------------------------
// Configuration tab (#512)
// ---------------------------------------------------------------------------

export type ConfigFieldType = "enum" | "bool" | "int" | "float";

export interface ConfigOption {
  value: string;
  label: string;
  description: string;
}

export interface ConfigField {
  key: string;
  type: ConfigFieldType;
  value: string | number | boolean;
  default: string | number | boolean;
  label: string;
  description: string;
  options?: ConfigOption[] | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  unit?: string | null;
}

export interface ConfigSection {
  id: string;
  label: string;
  description: string;
  settings: ConfigField[];
}

export interface ConfigResponse {
  label: string;
  sections: ConfigSection[];
}

export interface ConfigUpdateItem {
  key: string;
  value: string | number | boolean;
}

export interface ConfigFieldError {
  key: string;
  error: string;
}

export function fetchConfig(signal?: AbortSignal): Promise<ConfigResponse> {
  return request<ConfigResponse>("/api/dashboard/config", {}, signal);
}

export function updateConfig(
  settings: ConfigUpdateItem[],
  signal?: AbortSignal,
): Promise<ConfigResponse> {
  return request<ConfigResponse>(
    "/api/dashboard/config",
    { method: "PUT", body: JSON.stringify({ settings }) },
    signal,
  );
}

export function refreshConfig(
  signal?: AbortSignal,
): Promise<{ status: string; detail?: string }> {
  return request("/api/dashboard/config/refresh", { method: "POST" }, signal);
}

export function applyConfig(
  signal?: AbortSignal,
): Promise<{ status: string; detail: string }> {
  return request("/api/dashboard/config/apply", { method: "POST" }, signal);
}

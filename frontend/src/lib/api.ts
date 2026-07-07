/**
 * Tiny typed wrapper around `fetch` for the admin dashboard SPA.
 *
 * The dashboard is mounted under the orchestrator's own origin, so every
 * request is same-origin and uses relative `/api/dashboard/*` URLs. This
 * file is the single place that knows the URL shape, the response types,
 * and the `Authorization` header convention -- components only import named
 * helpers.
 *
 * Auth: when the orchestrator runs with auth enabled (Entra ID),
 * `require_admin` expects a bearer token whose `roles` claim contains
 * `Admin`. The SPA obtains that token via MSAL and registers a token
 * provider through `setAuthTokenProvider`; `request()` awaits the provider
 * on every call and attaches the Authorization header when one is returned.
 * A caller that passes its own `Authorization` header in `init.headers`
 * (for scripted tests) wins and the provider is skipped.
 *
 * The response side of the wrapper distinguishes 401 (needs sign-in /
 * silent-refresh failure) from 403 (signed in but the Admin app role is
 * not assigned) so the UI can render a real reauth flow vs an
 * access-denied panel instead of blurring both into one error.
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

/** 401 from the API: caller must (re)authenticate. */
export class UnauthorizedError extends ApiError {
  constructor(message: string, body?: unknown) {
    super(message, 401, body);
    this.name = "UnauthorizedError";
  }
}

/**
 * 403 from the API: caller is signed in but not entitled. Do NOT trigger a
 * sign-in redirect -- the fix is an operator granting the Admin app role,
 * not another token.
 */
export class ForbiddenError extends ApiError {
  constructor(message: string, body?: unknown) {
    super(message, 403, body);
    this.name = "ForbiddenError";
  }
}

type AuthTokenProvider = () => Promise<string | null>;

let tokenProvider: AuthTokenProvider | null = null;

/**
 * Register a function that returns a fresh access token for the API scope.
 *
 * `main.tsx` wires this to MSAL after `handleRedirectPromise`. The provider
 * is called on every request so token refresh / interaction fallback stays
 * inside MSAL and out of the request path. Passing `null` unregisters it,
 * which is useful in tests.
 */
export function setAuthTokenProvider(fn: AuthTokenProvider | null): void {
  tokenProvider = fn;
}

/** Kind of auth error the app should react to. */
export type AuthErrorKind = "unauthorized" | "forbidden";

type AuthErrorHandler = (kind: AuthErrorKind, err: ApiError) => void;

let authErrorHandler: AuthErrorHandler | null = null;

/**
 * Register a handler that fires whenever any API call returns 401 or 403.
 *
 * Used by `SignInGate` to lift access-denied and sign-in-required states out
 * of individual tabs so the entire dashboard shell can react consistently.
 * Passing `null` clears the handler.
 */
export function setAuthErrorHandler(fn: AuthErrorHandler | null): void {
  authErrorHandler = fn;
}

function hasHeader(headers: HeadersInit | undefined, name: string): boolean {
  if (!headers) return false;
  const target = name.toLowerCase();
  if (headers instanceof Headers) {
    return headers.has(name);
  }
  if (Array.isArray(headers)) {
    return headers.some(([k]) => k.toLowerCase() === target);
  }
  return Object.keys(headers).some((k) => k.toLowerCase() === target);
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  // Explicit Authorization in init.headers always wins (scripted tests,
  // curl-style debugging). Only fall back to the MSAL provider when the
  // caller did not set one itself.
  const callerSetAuth = hasHeader(init.headers, "authorization");
  const authHeader: Record<string, string> = {};
  if (!callerSetAuth && tokenProvider) {
    try {
      const token = await tokenProvider();
      if (token) authHeader.Authorization = `Bearer ${token}`;
    } catch {
      // Token acquisition failure -> proceed unauthenticated; the API will
      // 401 and the UI will route to the sign-in gate. Swallowing keeps
      // request() free of MSAL types.
    }
  }

  const res = await fetch(path, {
    ...init,
    signal,
    headers: {
      Accept: "application/json",
      ...authHeader,
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
    if (res.status === 401) {
      const err = new UnauthorizedError(message, body);
      authErrorHandler?.("unauthorized", err);
      throw err;
    }
    if (res.status === 403) {
      const err = new ForbiddenError(message, body);
      authErrorHandler?.("forbidden", err);
      throw err;
    }
    throw new ApiError(message, res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Auth config (unauthenticated bootstrap)
// ---------------------------------------------------------------------------

/**
 * Runtime auth configuration returned by `GET /api/dashboard/auth-config`.
 *
 * Fields are surfaced in camelCase for idiomatic TS while the backend keeps
 * snake_case in the wire format; `fetchAuthConfig` handles the mapping.
 * When `authEnabled` is false the rest of the fields are undefined and the
 * SPA must not bootstrap MSAL.
 */
export interface AuthConfig {
  authEnabled: boolean;
  clientId?: string;
  tenantId?: string;
  authority?: string;
  apiScope?: string;
}

interface AuthConfigWire {
  auth_enabled: boolean;
  client_id?: string | null;
  tenant_id?: string | null;
  authority?: string | null;
  api_scope?: string | null;
}

/**
 * Fetch the SPA's MSAL bootstrap configuration.
 *
 * Called once at startup before any protected endpoint, so it does NOT go
 * through the token provider (the endpoint is unauth by design).
 */
export async function fetchAuthConfig(signal?: AbortSignal): Promise<AuthConfig> {
  const wire = await request<AuthConfigWire>(
    "/api/dashboard/auth-config",
    {},
    signal,
  );
  return {
    authEnabled: wire.auth_enabled,
    clientId: wire.client_id ?? undefined,
    tenantId: wire.tenant_id ?? undefined,
    authority: wire.authority ?? undefined,
    apiScope: wire.api_scope ?? undefined,
  };
}

// ---------------------------------------------------------------------------
// Existing endpoints (pre-PR #236 surface -- restored here because lib/api.ts
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
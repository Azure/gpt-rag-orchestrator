/**
 * MSAL bootstrap for the admin dashboard SPA.
 *
 * Design notes:
 *
 * * The `PublicClientApplication` is a module-level singleton so that React
 *   StrictMode's double-invocation of effects does not create two MSAL
 *   instances, which is a documented cause of "interaction_in_progress"
 *   errors from `msal-browser`.
 * * We use the default `sessionStorage` cache. `localStorage` would survive
 *   browser restarts and increase the blast radius if the box is shared;
 *   the dashboard's usage pattern (a single admin session) does not need
 *   cross-tab or cross-restart persistence.
 * * The Authorization Code + PKCE flow is used implicitly by
 *   `loginRedirect` / `acquireTokenSilent` -- that is the msal-browser
 *   default for SPAs and matches the "SPA" platform type on the app
 *   registration.
 * * `acquireAdminToken` centralises the silent-then-redirect fallback so
 *   that every request path calls a single helper. The redirect swaps out
 *   the whole SPA context, so callers should treat a redirect as
 *   terminating this async operation.
 */

import {
  InteractionRequiredAuthError,
  PublicClientApplication,
  type AccountInfo,
  type Configuration,
} from "@azure/msal-browser";

import type { AuthConfig } from "./api";

let msalInstance: PublicClientApplication | null = null;

/**
 * Build the MSAL instance for a given `AuthConfig`.
 *
 * Idempotent: subsequent calls with the same effective config return the
 * cached instance. Passing a different tenant / client id resets the
 * singleton -- useful in tests, but never expected in production because
 * the config is fetched once at bootstrap.
 */
export function createMsalInstance(config: AuthConfig): PublicClientApplication {
  if (!config.authEnabled || !config.clientId || !config.authority) {
    throw new Error(
      "createMsalInstance called with auth disabled or incomplete config; " +
        "the caller should have gated on authEnabled first.",
    );
  }

  if (
    msalInstance &&
    msalInstance.getConfiguration().auth.clientId === config.clientId &&
    msalInstance.getConfiguration().auth.authority === config.authority
  ) {
    return msalInstance;
  }

  const msalConfig: Configuration = {
    auth: {
      clientId: config.clientId,
      authority: config.authority,
      // The SPA is served from the same origin as the API and is a single
      // page at `/dashboard/`, so the redirect target is the current
      // location. Using window.location.origin + /dashboard/ keeps the
      // registration in sync with the Vite `base` and the FastAPI mount
      // point.
      redirectUri: `${window.location.origin}/dashboard/`,
      postLogoutRedirectUri: `${window.location.origin}/dashboard/`,
      navigateToLoginRequestUrl: false,
    },
    cache: {
      // Deliberately session-scoped: see file docstring.
      cacheLocation: "sessionStorage",
      storeAuthStateInCookie: false,
    },
  };

  msalInstance = new PublicClientApplication(msalConfig);
  return msalInstance;
}

/**
 * Return the singleton MSAL instance, or throw if it has not been created.
 *
 * Used by React components that live under `<MsalProvider>` and therefore
 * know MSAL was bootstrapped; anything outside that subtree should call
 * `createMsalInstance` directly with an `AuthConfig`.
 */
export function getMsalInstance(): PublicClientApplication {
  if (!msalInstance) {
    throw new Error("MSAL instance has not been created; call createMsalInstance first.");
  }
  return msalInstance;
}

/**
 * Acquire an access token for the API scope, falling back to a redirect
 * when silent acquisition needs interaction (consent, MFA, expired refresh
 * token, etc).
 *
 * Returns `null` when no account is signed in yet -- the caller should
 * render the sign-in gate rather than force a redirect from a random API
 * call, because a background redirect can lose in-progress UI state.
 */
export async function acquireAdminToken(
  msal: PublicClientApplication,
  scope: string,
): Promise<string | null> {
  const account = pickAccount(msal);
  if (!account) return null;

  try {
    const result = await msal.acquireTokenSilent({
      account,
      scopes: [scope],
    });
    return result.accessToken || null;
  } catch (err) {
    if (err instanceof InteractionRequiredAuthError) {
      // acquireTokenRedirect returns Promise<void>: the browser navigates
      // away before the promise resolves. Callers treat this as a
      // terminating condition.
      await msal.acquireTokenRedirect({ scopes: [scope], account });
      return null;
    }
    throw err;
  }
}

/**
 * Pick the "current" account for the SPA.
 *
 * msal-react updates its own active account in response to login events,
 * but a fresh page load after redirect may briefly see multiple accounts
 * cached in sessionStorage. Prefer the explicitly-active account and fall
 * back to the first known one so admins do not have to re-sign-in after a
 * refresh.
 */
export function pickAccount(msal: PublicClientApplication): AccountInfo | null {
  const active = msal.getActiveAccount();
  if (active) return active;
  const [first] = msal.getAllAccounts();
  if (first) {
    msal.setActiveAccount(first);
    return first;
  }
  return null;
}

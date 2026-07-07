import { useEffect, useState, type ReactNode } from "react";
import { useMsal } from "@azure/msal-react";
import { LogIn } from "lucide-react";

import { setAuthErrorHandler, type AuthConfig } from "../lib/api";
import { AccessDeniedState } from "./AccessDeniedState";

interface SignInGateProps {
  authConfig: AuthConfig;
  children: ReactNode;
}

/**
 * Top-level gate that owns the "is the current viewer allowed to see the
 * dashboard" question.
 *
 * Three states:
 *
 * 1. Auth disabled -- render children unchanged. This preserves the
 *    developer / air-gapped deployment story where the SPA is served
 *    without an app registration.
 * 2. Auth enabled, no signed-in account -- render a hero panel with a
 *    real Sign In button that calls `loginRedirect` with the configured
 *    API scope so the returned token is directly usable by the API.
 * 3. Auth enabled, signed in -- render children, unless any downstream
 *    API call surfaces a 403, in which case swap the tree for
 *    `<AccessDeniedState>`. This centralises the "missing Admin role"
 *    experience instead of repeating it per-tab.
 *
 * The 403 signal is delivered by `setAuthErrorHandler` in `lib/api.ts`,
 * which fires for every request the SPA makes. 401s are intentionally
 * ignored here: msal-react handles interactive reauth via the token
 * provider, and a 401 usually means "token expired mid-flight" rather
 * than "wrong role".
 */
export function SignInGate({ authConfig, children }: SignInGateProps) {
  const { instance, accounts } = useMsal();
  const account = accounts[0] ?? null;
  const [accessDenied, setAccessDenied] = useState(false);

  useEffect(() => {
    if (!authConfig.authEnabled) return;
    setAuthErrorHandler((kind) => {
      if (kind === "forbidden") {
        setAccessDenied(true);
      }
    });
    return () => setAuthErrorHandler(null);
  }, [authConfig.authEnabled]);

  // A fresh sign-in should clear a stale access-denied state (for example
  // if the operator has just assigned the role and the user re-signs in).
  // We key on homeAccountId (a stable identifier) rather than the account
  // object reference so we do not re-fire on every render.
  const accountKey = account?.homeAccountId;
  useEffect(() => {
    if (accountKey) setAccessDenied(false);
  }, [accountKey]);

  if (!authConfig.authEnabled) {
    return <>{children}</>;
  }

  if (!account) {
    const scope = authConfig.apiScope;
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-8 text-center">
        <h1 className="text-2xl font-semibold text-foreground">Admin dashboard</h1>
        <p className="max-w-md text-sm text-muted-foreground">
          Sign in with your Microsoft Entra ID account. You need the{" "}
          <span className="font-medium">Admin</span> app role assigned in the orchestrator API app
          registration to see dashboard data.
        </p>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => {
            void instance.loginRedirect(scope ? { scopes: [scope] } : { scopes: [] });
          }}
        >
          <LogIn className="h-4 w-4" />
          Sign in
        </button>
      </div>
    );
  }

  if (accessDenied) {
    return <AccessDeniedState username={account.username} />;
  }

  return <>{children}</>;
}

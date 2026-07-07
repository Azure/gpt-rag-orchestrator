import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MsalProvider } from "@azure/msal-react";

import "./index.css";
import App from "./App";
import {
  fetchAuthConfig,
  setAuthTokenProvider,
  type AuthConfig,
} from "./lib/api";
import { acquireAdminToken, createMsalInstance } from "./lib/auth";

/**
 * Boot the dashboard SPA.
 *
 * The bootstrap is async because auth configuration lives on the server
 * (single source of truth) and must be fetched before we can decide
 * whether to build an MSAL instance and mount `<MsalProvider>`. Rendering
 * before this decision would cause the whole tree to see a null MSAL
 * context and either flicker or crash when hooks like `useMsal()` fire.
 *
 * If bootstrap fails -- typically because the auth-config endpoint is
 * unreachable -- we fail loudly by rendering an inline error. Silently
 * degrading to an unauthenticated view would be misleading and might
 * expose the dashboard when auth is expected to be on.
 */
async function bootstrap() {
  const container = document.getElementById("root");
  if (!container) throw new Error("Missing #root element in dashboard HTML.");
  const root = createRoot(container);

  let authConfig: AuthConfig;
  try {
    authConfig = await fetchAuthConfig();
  } catch (err) {
    root.render(
      <StrictMode>
        <div className="mx-auto max-w-xl p-8 text-center text-sm">
          <h1 className="mb-2 text-lg font-semibold">Dashboard failed to load</h1>
          <p className="text-muted-foreground">
            Could not fetch auth configuration from the orchestrator API.
          </p>
          <pre className="mt-3 whitespace-pre-wrap text-left text-xs opacity-70">
            {(err as Error).message}
          </pre>
        </div>
      </StrictMode>,
    );
    return;
  }

  if (!authConfig.authEnabled) {
    root.render(
      <StrictMode>
        <App authConfig={authConfig} />
      </StrictMode>,
    );
    return;
  }

  if (!authConfig.clientId || !authConfig.authority || !authConfig.apiScope) {
    root.render(
      <StrictMode>
        <div className="mx-auto max-w-xl p-8 text-center text-sm">
          <h1 className="mb-2 text-lg font-semibold">Dashboard misconfigured</h1>
          <p className="text-muted-foreground">
            Auth is enabled but the server did not return a complete configuration
            (clientId, authority, apiScope). Ask an operator to check the
            OAUTH_AZURE_AD_TENANT_ID and OAUTH_AZURE_AD_CLIENT_ID App Config
            entries.
          </p>
        </div>
      </StrictMode>,
    );
    return;
  }

  const msal = createMsalInstance(authConfig);

  // msal-browser 3.x requires initialize() before any other API call.
  await msal.initialize();

  // Complete any in-flight redirect (post-loginRedirect callback) before
  // we mount, so `<MsalProvider>` sees the resulting account state on
  // first render and the sign-in gate does not flash.
  await msal.handleRedirectPromise();

  const scope = authConfig.apiScope;
  setAuthTokenProvider(() => acquireAdminToken(msal, scope));

  root.render(
    <StrictMode>
      <MsalProvider instance={msal}>
        <App authConfig={authConfig} />
      </MsalProvider>
    </StrictMode>,
  );
}

void bootstrap();

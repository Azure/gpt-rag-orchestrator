import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MsalProvider } from "@azure/msal-react";

import "./index.css";
import App from "./App";
import { BootstrapError } from "./components/BootstrapError";
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
  } catch {
    root.render(
      <StrictMode>
        <BootstrapError
          title="Dashboard failed to load"
          message="Could not fetch authentication configuration from the orchestrator API. Refresh the page, and check the service logs if the problem persists."
        />
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
        <BootstrapError
          title="Dashboard misconfigured"
          message="Authentication is enabled, but the server returned an incomplete MSAL configuration. Check the orchestrator service logs."
        />
      </StrictMode>,
    );
    return;
  }

  const msal = createMsalInstance(authConfig);

  try {
    // msal-browser 3.x requires initialize() before any other API call.
    await msal.initialize();

    // Complete any in-flight redirect before mounting so the provider sees
    // the resulting account on first render. Explicitly select that account
    // so a stale cached account cannot win after an account switch.
    const redirectResult = await msal.handleRedirectPromise();
    if (redirectResult?.account) {
      msal.setActiveAccount(redirectResult.account);
    }
  } catch {
    root.render(
      <StrictMode>
        <BootstrapError
          title="Sign-in could not start"
          message="Microsoft Entra ID authentication could not be initialized. Refresh the page, and check the app registration and service logs if the problem persists."
        />
      </StrictMode>,
    );
    return;
  }

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

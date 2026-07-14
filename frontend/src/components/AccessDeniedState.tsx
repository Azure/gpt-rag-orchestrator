import { useState } from "react";
import { useMsal } from "@azure/msal-react";
import { LogOut, ShieldOff } from "lucide-react";

interface AccessDeniedStateProps {
  accountName?: string | null;
}

/**
 * Rendered when the signed-in user reaches the dashboard but does not have
 * the `Admin` app role assigned in the orchestrator API app registration.
 *
 * The message is deliberately explicit about which registration and role
 * the operator needs to update, because non-admin viewers seeing this
 * panel usually cannot self-serve the fix.
 */
export function AccessDeniedState({ accountName }: AccessDeniedStateProps) {
  const { instance, accounts } = useMsal();
  const account = instance.getActiveAccount() ?? accounts[0] ?? null;
  const [signingOut, setSigningOut] = useState(false);
  const [signOutError, setSignOutError] = useState("");
  const who = accountName ? `signed in as ${accountName}` : "signed in";

  async function signOut() {
    setSigningOut(true);
    setSignOutError("");
    try {
      await instance.logoutRedirect(account ? { account } : undefined);
    } catch {
      setSigningOut(false);
      setSignOutError("Sign out could not be started. Try again.");
    }
  }

  return (
    <div
      className="flex min-h-[60vh] flex-col items-center justify-center gap-3 p-8 text-center"
      role="alert"
    >
      <ShieldOff className="h-10 w-10 text-destructive" />
      <h2 className="text-xl font-semibold text-foreground">Access denied</h2>
      <p className="max-w-md text-sm text-muted-foreground">
        You are {who} but the <span className="font-medium">Admin</span> app role is not assigned
        to your account. Ask your operator to assign the <span className="font-medium">Admin</span>{" "}
        app role to your user in the orchestrator API app registration in Microsoft Entra ID.
      </p>
      <button
        type="button"
        className="mt-2 inline-flex items-center gap-2 rounded-md border border-border bg-background px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-wait disabled:opacity-70"
        disabled={signingOut}
        onClick={() => void signOut()}
      >
        <LogOut className="h-4 w-4" />
        {signingOut ? "Signing out..." : "Sign out and try another account"}
      </button>
      {signOutError && (
        <p className="max-w-md text-sm text-destructive" role="alert">
          {signOutError}
        </p>
      )}
    </div>
  );
}

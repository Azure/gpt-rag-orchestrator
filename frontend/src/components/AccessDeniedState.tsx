import { ShieldOff } from "lucide-react";

interface AccessDeniedStateProps {
  username?: string | null;
}

/**
 * Rendered when the signed-in user reaches the dashboard but does not have
 * the `Admin` app role assigned in the orchestrator API app registration.
 *
 * The message is deliberately explicit about which registration and role
 * the operator needs to update, because non-admin viewers seeing this
 * panel usually cannot self-serve the fix.
 */
export function AccessDeniedState({ username }: AccessDeniedStateProps) {
  const who = username ? `signed in as ${username}` : "signed in";
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3 p-8 text-center">
      <ShieldOff className="h-10 w-10 text-destructive" />
      <h1 className="text-xl font-semibold text-foreground">Access denied</h1>
      <p className="max-w-md text-sm text-muted-foreground">
        You are {who} but the <span className="font-medium">Admin</span> app role is not assigned
        to your account. Ask your operator to assign the <span className="font-medium">Admin</span>{" "}
        app role to your user in the orchestrator API app registration in Microsoft Entra ID.
      </p>
    </div>
  );
}

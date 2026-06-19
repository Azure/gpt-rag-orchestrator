import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw, RotateCcw, Save, Zap } from "lucide-react";
import {
  ApiError,
  applyConfig,
  fetchConfig,
  refreshConfig,
  updateConfig,
  type ConfigField,
  type ConfigResponse,
} from "../lib/api";
import { ConfigSection } from "./ConfigSection";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorState } from "./ErrorState";
import { InfoTooltip } from "./InfoTooltip";

type Values = Record<string, string | number | boolean>;

function buildInitialValues(resp: ConfigResponse): Values {
  const v: Values = {};
  for (const section of resp.sections) {
    for (const field of section.settings) {
      v[field.key] = field.value;
    }
  }
  return v;
}

function findField(resp: ConfigResponse | null, key: string): ConfigField | null {
  if (!resp) return null;
  for (const s of resp.sections) {
    for (const f of s.settings) {
      if (f.key === key) return f;
    }
  }
  return null;
}

/**
 * Admin Configuration tab.
 *
 * Owns the local edit state (`values`), diffs it against the server response
 * to decide whether Save / Discard are enabled, and surfaces every write or
 * load error as an inline banner so a 422 from the backend never blanks the
 * page.
 */
export function ConfigurationTab() {
  const [resp, setResp] = useState<ConfigResponse | null>(null);
  const [values, setValues] = useState<Values>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<{ message: string; status?: number } | null>(null);
  const [busy, setBusy] = useState<null | "save" | "discard" | "refresh" | "apply">(null);
  const [banner, setBanner] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const [applyOpen, setApplyOpen] = useState(false);

  const load = useCallback(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setLoadError(null);
    fetchConfig(ctrl.signal)
      .then((r) => {
        setResp(r);
        setValues(buildInitialValues(r));
        setLoading(false);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        if (err instanceof ApiError) {
          setLoadError({ status: err.status, message: err.message });
        } else {
          setLoadError({ message: (err as Error).message });
        }
        setLoading(false);
      });
    return () => ctrl.abort();
  }, []);

  useEffect(() => load(), [load]);

  const dirtyKeys = useMemo(() => {
    if (!resp) return [] as string[];
    const out: string[] = [];
    for (const s of resp.sections) {
      for (const f of s.settings) {
        if (values[f.key] !== f.value) out.push(f.key);
      }
    }
    return out;
  }, [resp, values]);

  const handleChange = (key: string, next: string | number | boolean) => {
    setValues((prev) => ({ ...prev, [key]: next }));
    setBanner(null);
  };

  const handleDiscard = () => {
    if (!resp) return;
    setValues(buildInitialValues(resp));
    setBanner({ kind: "success", text: "Changes discarded." });
  };

  const handleSave = async () => {
    if (!resp || dirtyKeys.length === 0) return;
    setBusy("save");
    setBanner(null);
    try {
      const payload = dirtyKeys.map((key) => ({ key, value: values[key] as string | number | boolean }));
      const updated = await updateConfig(payload);
      setResp(updated);
      setValues(buildInitialValues(updated));
      setBanner({ kind: "success", text: `Saved ${payload.length} setting${payload.length === 1 ? "" : "s"}.` });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : (err as Error).message;
      setBanner({ kind: "error", text: `Save failed: ${message}` });
    } finally {
      setBusy(null);
    }
  };

  const handleRefresh = async () => {
    setBusy("refresh");
    setBanner(null);
    try {
      await refreshConfig();
      // Re-fetch so the form reflects what the freshly built cache returns.
      const updated = await fetchConfig();
      setResp(updated);
      setValues(buildInitialValues(updated));
      setBanner({ kind: "success", text: "Configuration cache refreshed." });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : (err as Error).message;
      setBanner({ kind: "error", text: `Refresh failed: ${message}` });
    } finally {
      setBusy(null);
    }
  };

  const handleApply = async () => {
    setBusy("apply");
    setBanner(null);
    try {
      const res = await applyConfig();
      setBanner({ kind: "success", text: res.detail });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : (err as Error).message;
      setBanner({ kind: "error", text: `Apply failed: ${message}` });
    } finally {
      setBusy(null);
      setApplyOpen(false);
    }
  };

  if (loadError) {
    if (loadError.status === 403) {
      return (
        <ErrorState
          message="Access denied"
          hint="The Configuration tab requires the Admin app role. Sign in with an admin account or contact the workspace owner."
        />
      );
    }
    if (loadError.status === 401) {
      return (
        <ErrorState
          message="Sign-in required"
          hint="The orchestrator API requires a valid bearer token."
        />
      );
    }
    return <ErrorState message="Failed to load configuration" hint={loadError.message} />;
  }

  if (loading || !resp) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-12 text-sm text-muted-foreground">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        Loading configuration...
      </div>
    );
  }

  const dirty = dirtyKeys.length > 0;
  const dirtySummary = dirty
    ? dirtyKeys
        .map((k) => findField(resp, k)?.label ?? k)
        .slice(0, 3)
        .join(", ") + (dirtyKeys.length > 3 ? `, +${dirtyKeys.length - 3} more` : "")
    : "";

  return (
    <div className="space-y-5">
      <div className="rounded-lg border bg-muted/30 px-4 py-2 text-xs text-muted-foreground">
        Editing label <span className="font-mono text-foreground">{resp.label}</span>. Settings
        outside this list are managed through Azure App Configuration directly.
      </div>

      {banner && (
        <div
          role={banner.kind === "error" ? "alert" : "status"}
          className={`rounded-md border px-4 py-2 text-sm ${
            banner.kind === "error"
              ? "border-destructive/40 bg-destructive/10 text-destructive"
              : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          }`}
        >
          {banner.text}
        </div>
      )}

      <div className="space-y-4">
        {resp.sections.map((section) => (
          <ConfigSection
            key={section.id}
            section={section}
            values={values}
            onChange={handleChange}
          />
        ))}
      </div>

      <footer className="sticky bottom-0 flex flex-wrap items-center justify-end gap-2 rounded-lg border bg-card/95 px-4 py-3 shadow-sm backdrop-blur">
        {dirty && (
          <span className="mr-auto text-xs text-muted-foreground" aria-live="polite">
            Unsaved: {dirtySummary}
          </span>
        )}
        <button
          type="button"
          onClick={handleRefresh}
          disabled={busy !== null}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm hover:bg-accent disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${busy === "refresh" ? "animate-spin" : ""}`} />
          Refresh from App Configuration
        </button>
        <InfoTooltip
          label="Refresh from App Configuration"
          description="Re-reads all values from Azure App Configuration. Use this if someone changed settings outside the dashboard and you want to see the latest values. Does not save anything."
        />
        <button
          type="button"
          onClick={() => setApplyOpen(true)}
          disabled={busy !== null}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm hover:bg-accent disabled:opacity-50"
        >
          <Zap className="h-4 w-4" />
          Apply changes
        </button>
        <InfoTooltip
          label="Apply changes"
          description="Refreshes the running app's settings cache so your saved changes take effect without restarting the container."
        />
        <button
          type="button"
          onClick={handleDiscard}
          disabled={!dirty || busy !== null}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm hover:bg-accent disabled:opacity-50"
        >
          <RotateCcw className="h-4 w-4" />
          Discard changes
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={!dirty || busy !== null}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          <Save className="h-4 w-4" />
          {busy === "save" ? "Saving..." : "Save changes"}
        </button>
      </footer>

      <ConfirmDialog
        open={applyOpen}
        title="Apply configuration changes?"
        description={
          "This refreshes the orchestrator's configuration cache so subsequent requests use the latest values. " +
          "In-flight requests are not affected. A full container restart is not performed."
        }
        confirmLabel="Apply"
        onConfirm={handleApply}
        onCancel={() => setApplyOpen(false)}
        busy={busy === "apply"}
      />
    </div>
  );
}

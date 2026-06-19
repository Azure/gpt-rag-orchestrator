import { useState } from "react";
import type { OverviewRange, RangePreset } from "./range";

interface RangePickerProps {
  value: OverviewRange;
  onChange: (next: OverviewRange) => void;
}

const PRESETS: Array<{ id: RangePreset; label: string }> = [
  { id: "today", label: "Today" },
  { id: "7d", label: "Last 7 days" },
  { id: "30d", label: "Last 30 days" },
  { id: "90d", label: "Last 90 days" },
  { id: "custom", label: "Custom range" },
];

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

/**
 * Compact range chooser rendered in the Overview header (#241 AC 1-4).
 *
 * Five preset chips plus two date inputs that show when ``Custom range`` is
 * selected. The backend caps the custom range at 365 days; we surface a
 * lightweight inline error rather than relying on a 400 round-trip so users
 * get immediate feedback while building the range.
 */
export function RangePicker({ value, onChange }: RangePickerProps) {
  const [error, setError] = useState<string | null>(null);

  function selectPreset(preset: RangePreset) {
    setError(null);
    if (preset === "custom") {
      const from = value.from ?? isoDaysAgo(7);
      const to = value.to ?? todayIso();
      onChange({ preset: "custom", from, to });
    } else {
      onChange({ preset });
    }
  }

  function updateCustom(field: "from" | "to", next: string) {
    const draft: OverviewRange = {
      preset: "custom",
      from: field === "from" ? next : value.from,
      to: field === "to" ? next : value.to,
    };
    if (draft.from && draft.to) {
      if (draft.from > draft.to) {
        setError("Start date must be on or before end date.");
        return;
      }
      const ms =
        new Date(draft.to + "T00:00:00Z").getTime() -
        new Date(draft.from + "T00:00:00Z").getTime();
      const days = Math.floor(ms / 86_400_000) + 1;
      if (days > 365) {
        setError("Custom range cannot exceed 365 days.");
        return;
      }
    }
    setError(null);
    onChange(draft);
  }

  return (
    <div className="flex w-full flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {PRESETS.map((p) => {
          const active = value.preset === p.id;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => selectPreset(p.id)}
              aria-pressed={active}
              className={
                "rounded-md border px-2.5 py-1 text-xs font-medium transition " +
                (active
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-card text-muted-foreground hover:text-foreground")
              }
            >
              {p.label}
            </button>
          );
        })}
      </div>
      {value.preset === "custom" && (
        // Always render the From/To inputs on their own row immediately under
        // the chips when ``Custom range`` is active (#241 follow-up: the
        // inputs were rendering but were easy to miss when the picker sat
        // inside a tight ``justify-between`` flex row in the Overview
        // header). ``w-full basis-full`` forces a hard line break in the
        // parent flex layout so the inputs are never visually clipped.
        <div
          className="flex w-full basis-full flex-wrap items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 text-xs"
          data-testid="overview-range-custom"
        >
          <label className="flex items-center gap-1.5">
            <span className="font-medium text-foreground">From</span>
            <input
              type="date"
              value={value.from ?? ""}
              onChange={(e) => updateCustom("from", e.target.value)}
              className="rounded-md border bg-background px-2 py-1 text-xs"
              aria-label="Custom range start date"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <span className="font-medium text-foreground">To</span>
            <input
              type="date"
              value={value.to ?? ""}
              onChange={(e) => updateCustom("to", e.target.value)}
              className="rounded-md border bg-background px-2 py-1 text-xs"
              aria-label="Custom range end date"
            />
          </label>
          <span className="text-muted-foreground">Inclusive UTC dates, max 365 days.</span>
          {error && (
            <span role="alert" className="text-destructive">
              {error}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/** Human label removed — see ./range.ts */

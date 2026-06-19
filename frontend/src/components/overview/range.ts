export type RangePreset = "today" | "7d" | "30d" | "90d" | "custom";

export interface OverviewRange {
  preset: RangePreset;
  /** Inclusive UTC date YYYY-MM-DD; only meaningful for custom. */
  from?: string;
  /** Inclusive UTC date YYYY-MM-DD; only meaningful for custom. */
  to?: string;
}

/** Today's date in YYYY-MM-DD form, UTC. */
export function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Clamp a YYYY-MM-DD date string to today's UTC date when it is in the
 * future. Used by RangePicker and by the persisted-range loader so a stale
 * future value from localStorage (set on a previous session) cannot leak
 * past the picker's max=today input cap (#247 Bug 1).
 */
export function clampToToday(value: string | undefined): string | undefined {
  if (!value) return value;
  const today = todayIso();
  return value > today ? today : value;
}

/** Human label for the active range, used in the Active users card hint. */
export function rangeLabel(range: OverviewRange): string {
  switch (range.preset) {
    case "today":
      return "today";
    case "7d":
      return "last 7 days";
    case "30d":
      return "last 30 days";
    case "90d":
      return "last 90 days";
    case "custom":
      if (range.from && range.to) return `${range.from} to ${range.to}`;
      return "custom range";
  }
}

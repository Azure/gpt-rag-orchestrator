export type RangePreset = "today" | "7d" | "30d" | "90d" | "custom";

export interface OverviewRange {
  preset: RangePreset;
  /** Inclusive UTC date YYYY-MM-DD; only meaningful for custom. */
  from?: string;
  /** Inclusive UTC date YYYY-MM-DD; only meaningful for custom. */
  to?: string;
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

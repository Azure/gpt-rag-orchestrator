import { useEffect, useId, useRef, useState } from "react";
import { Info } from "lucide-react";

interface InfoTooltipProps {
  label: string;
  description: string;
  optionLines?: Array<{ value: string; label: string; description: string }>;
}

/**
 * Accessible tooltip rendered as a popover.
 *
 * Hover-only tooltips are inaccessible to keyboard and screen-reader users
 * so this implementation opens on click *and* on focus, can be dismissed
 * with Escape or by clicking outside, and links the trigger to the body
 * via aria-describedby. The body content (label + free-form description +
 * an optional per-option breakdown for enums) is identical to what a
 * sighted user sees on hover, matching the per-option tooltip requirement
 * from issue #512.
 */
export function InfoTooltip({ label, description, optionLines }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  return (
    <span ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        onFocus={() => setOpen(true)}
        onBlur={(e) => {
          // Keep open if focus moves into the popover; close otherwise.
          if (!containerRef.current?.contains(e.relatedTarget as Node)) {
            setOpen(false);
          }
        }}
        aria-label={`More info: ${label}`}
        aria-describedby={open ? bodyId : undefined}
        aria-expanded={open}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Info className="h-4 w-4" />
      </button>
      {open && (
        <div
          id={bodyId}
          role="tooltip"
          // ``normal-case`` and ``tracking-normal`` reset any
          // ``uppercase``/``tracking-wide`` inherited from a parent label
          // (#241 follow-up: the Overview StatCard label span uses
          // ``uppercase tracking-wide`` for the metric name, which previously
          // bled into the tooltip body and rendered the prose in ALL CAPS).
          //
          // ``bg-card`` is the same fully opaque token used for KPI/section
          // backgrounds, so the popover is readable when it sits over another
          // card in both light and dark mode (#247 Bug 3 — the previous
          // ``bg-popover`` token is a translucent surface that picked up
          // whatever was under it). ``border-border`` plus ``shadow-md``
          // keeps the lift cue but no longer relies on alpha to feel layered.
          className="absolute left-6 top-0 z-50 w-72 rounded-md border border-border bg-card p-3 text-xs normal-case tracking-normal text-card-foreground shadow-md"
        >
          <div className="font-medium normal-case tracking-normal text-foreground">{label}</div>
          <p className="mt-1 leading-relaxed normal-case tracking-normal text-muted-foreground">{description}</p>
          {optionLines && optionLines.length > 0 && (
            <ul className="mt-2 space-y-1.5 border-t pt-2">
              {optionLines.map((opt) => (
                <li key={opt.value}>
                  <span className="font-medium text-foreground">{opt.label}:</span>{" "}
                  <span className="text-muted-foreground">{opt.description}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </span>
  );
}

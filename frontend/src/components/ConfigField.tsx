import { useId } from "react";
import type { ConfigField as ConfigFieldType } from "../lib/api";
import { InfoTooltip } from "./InfoTooltip";

interface ConfigFieldProps {
  field: ConfigFieldType;
  value: string | number | boolean;
  onChange: (next: string | number | boolean) => void;
}

/**
 * Renders one editable App Configuration setting.
 *
 * The shape is driven entirely by the backend metadata (`type`, `options`,
 * `min`/`max`/`step`) so adding a new setting on the server requires no
 * change here. Per-option tooltips on enum fields surface trade-offs the
 * same way the (i) icon does for sighted users on hover.
 */
export function ConfigField({ field, value, onChange }: ConfigFieldProps) {
  const inputId = useId();
  const optionLines = field.options
    ? field.options.map((o) => ({ value: o.value, label: o.label, description: o.description }))
    : undefined;

  const labelRow = (
    <div className="flex items-center gap-1.5">
      <label htmlFor={inputId} className="text-sm font-medium text-foreground">
        {field.label}
      </label>
      <InfoTooltip
        label={field.label}
        description={field.description}
        optionLines={optionLines}
      />
    </div>
  );

  let control: React.ReactNode;
  if (field.type === "bool") {
    const checked = Boolean(value);
    control = (
      <button
        id={inputId}
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
          checked ? "bg-primary" : "bg-muted"
        }`}
      >
        <span
          className={`inline-block h-5 w-5 transform rounded-full bg-background shadow transition-transform ${
            checked ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </button>
    );
  } else if (field.type === "enum") {
    control = (
      <select
        id={inputId}
        value={String(value)}
        onChange={(e) => onChange(e.target.value)}
        className="w-full max-w-sm rounded-md border bg-background px-3 py-1.5 text-sm shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {(field.options ?? []).map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    );
  } else {
    // int or float — pair a number input with a slider when min/max are set.
    const numeric = typeof value === "number" ? value : Number(value);
    const step = field.step ?? (field.type === "int" ? 1 : 0.01);
    const min = field.min ?? undefined;
    const max = field.max ?? undefined;
    const hasRange = min !== undefined && max !== undefined;
    const onInputChange = (raw: string) => {
      if (raw === "") {
        onChange(raw);
        return;
      }
      const n = field.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
      if (!Number.isNaN(n)) onChange(n);
    };
    control = (
      <div className="flex w-full max-w-sm items-center gap-3">
        {hasRange && (
          <input
            type="range"
            value={Number.isFinite(numeric) ? numeric : (min ?? 0)}
            min={min}
            max={max}
            step={step}
            onChange={(e) => onChange(field.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
            className="flex-1 accent-primary"
            aria-labelledby={inputId}
          />
        )}
        <div className="flex items-center gap-1">
          <input
            id={inputId}
            type="number"
            value={value as number | string}
            min={min}
            max={max}
            step={step}
            onChange={(e) => onInputChange(e.target.value)}
            className="w-24 rounded-md border bg-background px-2 py-1.5 text-sm text-right tabular-nums shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          {field.unit && (
            <span className="text-xs text-muted-foreground">{field.unit}</span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      {labelRow}
      {control}
    </div>
  );
}

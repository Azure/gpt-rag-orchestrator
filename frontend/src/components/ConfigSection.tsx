import type { ConfigSection as ConfigSectionType } from "../lib/api";
import { ConfigField } from "./ConfigField";

interface ConfigSectionProps {
  section: ConfigSectionType;
  values: Record<string, string | number | boolean>;
  onChange: (key: string, next: string | number | boolean) => void;
}

/**
 * One card per setting group. Settings inside are stacked rather than placed
 * on a grid so labels and tooltips never wrap awkwardly on smaller widths.
 */
export function ConfigSection({ section, values, onChange }: ConfigSectionProps) {
  return (
    <section
      aria-labelledby={`section-${section.id}-heading`}
      className="rounded-lg border bg-card p-5 shadow-sm"
    >
      <header className="mb-4">
        <h3 id={`section-${section.id}-heading`} className="text-base font-semibold">
          {section.label}
        </h3>
        {section.description && (
          <p className="mt-1 text-xs text-muted-foreground">{section.description}</p>
        )}
      </header>
      <div className="space-y-4">
        {section.settings.map((field) => (
          <ConfigField
            key={field.key}
            field={field}
            value={values[field.key] ?? field.value}
            onChange={(v) => onChange(field.key, v)}
          />
        ))}
      </div>
    </section>
  );
}

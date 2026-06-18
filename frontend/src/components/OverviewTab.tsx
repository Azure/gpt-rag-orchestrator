import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  fetchOverview,
  type OverviewResponse,
} from "../lib/api";
import { ErrorState } from "./ErrorState";
import { Activity, MessageSquare, RefreshCw, Users } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  hint?: string;
  icon?: React.ReactNode;
}

function StatCard({ label, value, hint, icon }: StatCardProps) {
  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function OverviewTab() {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<{ message: string; status?: number } | null>(null);

  const load = () => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    fetchOverview(30, ctrl.signal)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        if (err instanceof ApiError) {
          setError({ status: err.status, message: err.message });
        } else {
          setError({ message: (err as Error).message });
        }
        setLoading(false);
      });
    return () => ctrl.abort();
  };

  useEffect(() => {
    const cancel = load();
    return cancel;
  }, []);

  if (error) {
    if (error.status === 403) {
      return (
        <ErrorState
          message="Access denied"
          hint="The dashboard requires the Admin app role. Sign in with an admin account or contact the workspace owner to assign the role."
        />
      );
    }
    if (error.status === 401) {
      return (
        <ErrorState
          message="Sign-in required"
          hint="The orchestrator API requires a valid bearer token. Make sure your client requests a token with the api://<client_id>/... scope."
        />
      );
    }
    return <ErrorState message="Failed to load overview" hint={error.message} />;
  }

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-12 text-sm text-muted-foreground">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        Loading overview...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Today"
          value={data.today}
          hint="Conversations in the last 24h"
          icon={<MessageSquare className="h-4 w-4" />}
        />
        <StatCard
          label="Last 7 days"
          value={data.last_7_days}
          hint="Conversations created"
          icon={<MessageSquare className="h-4 w-4" />}
        />
        <StatCard
          label="Last 30 days"
          value={data.last_30_days}
          hint={`out of ${data.total} total`}
          icon={<MessageSquare className="h-4 w-4" />}
        />
        <StatCard
          label="Active users"
          value={data.active_users}
          hint={`Distinct, last ${data.window_days} days`}
          icon={<Users className="h-4 w-4" />}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <div className="rounded-lg border bg-card p-4 shadow-sm lg:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Conversations over time</h2>
            <span className="text-xs text-muted-foreground">
              Last {data.window_days} days, UTC
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.conversations_per_day}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                  tickFormatter={(v: string) => v.slice(5)}
                  minTickGap={20}
                />
                <YAxis
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                  allowDecimals={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-4 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold">Engagement</h2>
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">Average turns per conversation</span>
              <span className="text-lg font-semibold tabular-nums">{data.avg_turns}</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">Active users (window)</span>
              <span className="text-lg font-semibold tabular-nums">{data.active_users}</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">Total conversations</span>
              <span className="text-lg font-semibold tabular-nums">{data.total}</span>
            </div>
            <div className="flex items-center gap-1.5 border-t pt-3 text-xs text-muted-foreground">
              <Activity className="h-3 w-3" />
              <span>Cached for 60 seconds</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

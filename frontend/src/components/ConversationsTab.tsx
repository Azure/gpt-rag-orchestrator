import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchConversations,
  formatUtc,
  type ConversationSummary,
} from "../lib/api";
import { ErrorState } from "./ErrorState";
import { Pagination } from "./Pagination";
import { SearchInput } from "./SearchInput";
import { ConversationDetailDialog } from "./ConversationDetailDialog";
import { RefreshCw } from "lucide-react";

const PAGE_SIZE = 25;

export function ConversationsTab() {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ message: string; status?: number } | null>(null);
  const [selected, setSelected] = useState<ConversationSummary | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchConversations(
        {
          skip: (page - 1) * PAGE_SIZE,
          limit: PAGE_SIZE,
          search: search || undefined,
        },
        ctrl.signal,
      );
      if (ctrl.signal.aborted) return;
      setItems(res.conversations);
      setHasMore(res.has_more);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      if (err instanceof ApiError) {
        setError({ status: err.status, message: err.message });
      } else {
        setError({ message: (err as Error).message });
      }
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }, [page, search]);

  useEffect(() => {
    load();
    return () => {
      abortRef.current?.abort();
    };
  }, [load]);

  const handleSearch = useCallback((v: string) => {
    setSearch(v);
    setPage(1);
  }, []);

  if (error) {
    if (error.status === 403) {
      return (
        <ErrorState
          message="Access denied"
          hint="Listing conversations requires the Admin app role."
        />
      );
    }
    if (error.status === 401) {
      return (
        <ErrorState
          message="Sign-in required"
          hint="Provide a bearer token with the api://<client_id>/... scope."
        />
      );
    }
    return <ErrorState message="Failed to load conversations" hint={error.message} />;
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 flex-wrap">
        <SearchInput
          value={search}
          onChange={handleSearch}
          placeholder="Search by name, principal id or conversation id..."
        />
        <div className="flex-1" />
        <button
          onClick={load}
          className="rounded-md p-2 hover:bg-accent"
          title="Refresh"
          aria-label="Refresh"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      <div className="rounded-lg border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50 text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Name</th>
              <th className="px-4 py-2 text-left font-medium">User</th>
              <th className="px-4 py-2 text-left font-medium">Created (UTC)</th>
              <th className="px-4 py-2 text-left font-medium">Last updated (UTC)</th>
              <th className="px-4 py-2 text-right font-medium">Messages</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                  {loading ? "Loading..." : "No conversations found."}
                </td>
              </tr>
            )}
            {items.map((c) => (
              <tr
                key={c.id}
                className="cursor-pointer border-b last:border-0 hover:bg-muted/30"
                onClick={() => setSelected(c)}
              >
                <td className="max-w-[260px] truncate px-4 py-2" title={c.name ?? c.id}>
                  {c.name || <span className="text-muted-foreground italic">Untitled</span>}
                </td>
                <td className="max-w-[220px] truncate px-4 py-2 font-mono text-xs" title={c.principal_id ?? ""}>
                  {c.principal_id ?? "-"}
                </td>
                <td className="px-4 py-2 text-xs">{formatUtc(c.created_at)}</td>
                <td className="px-4 py-2 text-xs">{formatUtc(c.last_updated)}</td>
                <td className="px-4 py-2 text-right tabular-nums">{c.message_count ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Pagination page={page} pageSize={PAGE_SIZE} hasMore={hasMore} onChange={setPage} />

      {selected && (
        <ConversationDetailDialog
          conversation={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

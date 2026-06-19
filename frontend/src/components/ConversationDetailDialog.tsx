import { useEffect, useState } from "react";
import { ExternalLink, Info, X } from "lucide-react";
import {
  ApiError,
  fetchConversationDetail,
  formatUtc,
  type ConversationDetail,
  type ConversationMessage,
  type ConversationSummary,
} from "../lib/api";

interface Props {
  conversation: ConversationSummary;
  onClose: () => void;
}

function roleClass(role?: string): string {
  switch ((role ?? "").toLowerCase()) {
    case "user":
      return "bg-primary/10 text-primary";
    case "assistant":
      return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200";
    case "system":
      return "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200";
    case "tool":
      return "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function MessageBubble({ message }: { message: ConversationMessage }) {
  const content =
    typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content, null, 2);
  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <div className="mb-1 flex items-center justify-between">
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${roleClass(message.role)}`}
        >
          {message.role ?? "unknown"}
        </span>
        {message.timestamp && (
          <span className="text-xs text-muted-foreground">{formatUtc(message.timestamp)}</span>
        )}
      </div>
      <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed">
        {content || <span className="italic text-muted-foreground">(empty)</span>}
      </pre>
    </div>
  );
}

/**
 * Friendly note rendered where an assistant reply would go.
 *
 * The orchestrator persists user prompts in Cosmos under ``questions[]`` and
 * stores the full transcript (assistant replies, tool calls) on the Azure
 * AI Foundry agent thread referenced by ``thread_id`` (#247 Bug 4).
 */
function FoundryAssistantPlaceholder() {
  return (
    <div className="rounded-md border border-dashed bg-muted/20 p-3">
      <div className="mb-1 flex items-center gap-2">
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${roleClass("assistant")}`}
        >
          assistant
        </span>
        <Info className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
      </div>
      <p className="text-sm italic text-muted-foreground">
        Assistant response stored on Foundry thread.
      </p>
    </div>
  );
}

function FoundryThreadNote({ threadId }: { threadId?: string }) {
  return (
    <div className="rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900 dark:border-blue-900/50 dark:bg-blue-950/40 dark:text-blue-100">
      <div className="flex items-start gap-2">
        <Info className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden />
        <div className="space-y-2">
          <p>
            Message bodies are stored on the Azure AI Foundry agent thread, not in this database. Open the conversation in Foundry to see the full transcript.
          </p>
          {threadId && (
            <p className="font-mono text-xs">
              thread_id: {threadId}
            </p>
          )}
          <a
            href="https://learn.microsoft.com/azure/ai-services/agents/"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium underline"
          >
            Foundry agents documentation
            <ExternalLink className="h-3 w-3" aria-hidden />
          </a>
        </div>
      </div>
    </div>
  );
}

export function ConversationDetailDialog({ conversation, onClose }: Props) {
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    fetchConversationDetail(conversation.id, ctrl.signal)
      .then((res) => {
        setDetail(res);
        setLoading(false);
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return;
        if (err instanceof ApiError && err.status === 404) {
          setError("Conversation not found");
        } else {
          setError((err as Error).message);
        }
        setLoading(false);
      });
    return () => ctrl.abort();
  }, [conversation.id]);

  const messages = detail?.messages ?? [];
  const threadId = detail?.thread_id;
  const title = detail?.name || conversation.name || conversation.id;
  // Cosmos only persists user turns (#247 Bug 4) -- if there are no
  // assistant messages we mark every user turn so the placeholder slots in.
  const hasAssistantTurn = messages.some(
    (m) => (m.role ?? "").toLowerCase() === "assistant",
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between border-b px-6 py-4">
          <div className="min-w-0">
            <h2 className="truncate text-lg font-semibold" title={title}>
              {title}
            </h2>
            <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
              <span className="font-mono">id: {conversation.id}</span>
              <span className="font-mono">user: {conversation.principal_id ?? "-"}</span>
              <span>created: {formatUtc(conversation.created_at)}</span>
              <span>last update: {formatUtc(conversation.last_updated)}</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="ml-3 rounded-md p-1 hover:bg-accent"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="flex-1 overflow-auto px-6 py-4">
          {loading && (
            <div className="py-12 text-center text-sm text-muted-foreground">
              Loading conversation...
            </div>
          )}
          {error && (
            <div className="py-12 text-center text-sm text-destructive">{error}</div>
          )}
          {!loading && !error && messages.length === 0 && (
            <div className="space-y-3">
              <FoundryThreadNote threadId={threadId} />
            </div>
          )}
          {!loading && !error && messages.length > 0 && (
            <div className="space-y-3">
              {!hasAssistantTurn && <FoundryThreadNote threadId={threadId} />}
              {messages.map((m, i) => {
                const isUser = (m.role ?? "").toLowerCase() === "user";
                return (
                  <div key={i} className="space-y-3">
                    <MessageBubble message={m} />
                    {isUser && !hasAssistantTurn && <FoundryAssistantPlaceholder />}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

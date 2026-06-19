/**
 * Centralized tooltip copy for the Overview tab (#241 second comment).
 *
 * Steve owns the wording. Adjust strings here rather than editing JSX so
 * documentation stays the single source of truth.
 *
 * Card / panel tooltips that depend on the time-range picker (Active users
 * card hint, chart caption, and the three Engagement rows) reference the
 * "selected window". The four KPI cards keep their fixed semantic windows
 * (Today / 7d / 30d) and the tooltip copy reflects that.
 */
export const OVERVIEW_TOOLTIPS = {
  today:
    "Conversations created in the last 24 hours (rolling window, UTC).",
  last7Days:
    "Conversations created in the last 7 days (rolling window, UTC).",
  last30Days:
    "Conversations created in the last 30 days (rolling window, UTC). 'out of N total' counts every non-deleted conversation ever.",
  activeUsers:
    "Distinct users who created at least one conversation in the selected window. Anonymous traffic is grouped under a single bucket regardless of how many sessions it produced. Authenticated users are counted by their Entra object id.",
  conversationsOverTime:
    "Daily count of new conversations across the selected window (UTC). Empty days are shown as zero.",
  engagement: "Engagement metrics scoped to the selected window.",
  avgTurns:
    "Average number of user/assistant round-trips per conversation in the window. A turn equals one user message plus one assistant reply (message_count / 2).",
  activeUsersWindow:
    "Same as the Active Users card, scoped to the selected window.",
  totalConversations:
    "Total non-deleted conversations created in the selected window.",
} as const;

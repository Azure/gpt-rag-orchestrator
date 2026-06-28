## v2.8.12 — Overview and Conversations dashboard fixes

Five operator-dashboard fixes reported by Paulo on the sandbox.

### Fixed

1. **Custom range future dates** — Overview tab Custom range now caps both date inputs at today (UTC) via `max` and clamps any previously-stored future date on load. An inline error mirrors the existing >365 day style.
2. **`to` parameter exclusive** — `/api/dashboard/overview` now treats `to=YYYY-MM-DD` as end-of-day UTC, so the final day's conversations are included. Future dates are rejected with HTTP 400.
3. **Tooltip background transparent** — InfoTooltip popover now uses opaque `bg-card` with border and shadow so it is readable on top of any card in both light and dark mode.
4. **Conversations tab opens to ""(empty)""** — Dashboard conversation detail is reconstructed from the persisted `questions[]` (user turns) plus `feedback` and `thread_id`. The dialog shows a friendly note pointing to the Foundry agent thread for the full transcript instead of empty cards.
5. **""Reload settings cache"" button** — Renamed to **""Refresh from App Configuration""** with sentence-case info tooltips on both Configuration tab buttons explaining what each one does.

### Tests

`pytest tests/test_dashboard.py` — **25 passed**. New regressions:
- `test_overview_to_is_end_of_day_inclusive`
- `test_overview_rejects_future_dates`
- `test_conversation_detail_reconstructs_from_questions`
- `test_conversation_detail_empty_questions_returns_empty_messages`

`npm run lint` — 0 warnings, 0 errors.
`npm run build` — clean (vite production build).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>

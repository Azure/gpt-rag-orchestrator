## v2.8.12 — Overview and Conversations dashboard fixes

Five operator-dashboard fixes for the Overview and Conversations tabs.

### Fixed

1. **Custom range future dates** — Date inputs in Custom range now cap at today (UTC) and any previously-stored future date is clamped on load.
2. **`to` parameter exclusive** — `/api/dashboard/overview` now treats `to=YYYY-MM-DD` as end-of-day UTC, so the final day's conversations are included; future dates rejected with 400.
3. **Tooltip background transparent** — InfoTooltip popover uses opaque `bg-card` with border and shadow so it is readable over cards in light and dark mode.
4. **Conversations tab opens to ""(empty)""** — Dashboard conversation detail is reconstructed from persisted `questions[]` (user turns) with `feedback` and `thread_id`; the dialog renders a friendly note that the full assistant transcript lives on the Foundry agent thread.
5. **""Reload settings cache"" button** — Renamed to **""Refresh from App Configuration""** with info tooltips on both Configuration tab buttons explaining what each does.

### Tests

`pytest tests/test_dashboard.py` — 25 passed (4 new regressions).
`npm run lint` / `npm run build` — clean.
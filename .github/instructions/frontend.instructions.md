---
applyTo: "frontend/**"
---

# Administration dashboard

- Preserve the dashboard as an optional surface controlled by
  `ENABLE_DASHBOARD`.
- Keep authentication aligned with the backend's Entra audience and exact
  case-sensitive `Admin` app role.
- Reuse existing React, Vite, Radix, and styling patterns before introducing
  dependencies or abstractions.
- Treat backend schemas and status codes as contracts; update both sides and
  their tests together.
- Do not place tenant IDs, client IDs, secrets, tokens, or environment-specific
  endpoints in the static bundle.
- Keep loading, empty, access-denied, and error states explicit.
- Run the existing `npm run lint` and `npm run build` commands for dashboard
  changes.
- Load `documentation-consistency` for user-visible behavior or configuration
  changes.

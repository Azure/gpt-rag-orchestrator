from function_app import app
import azure.durable_functions as df


@app.entity_trigger(context_name="context")
def RateLimiter(context: df.DurableOrchestrationContext):
    """
    Simple fairness & throttling:
    - global_limit: max jobs concurrently across all tenants
    - per_tenant_limit: dict with default=1 (VIP tenants can be increased)
    """
    state = context.get_state(lambda: {
        "global_limit": 1,
        "global_inflight": 0,
        "inflight_by_tenant": {},
        "per_tenant_limit": {"default": 1}
    })

    op = context.operation_name
    data = context.get_input() or {}

    if op == "acquire":
        tenant = data["tenant_id"]
        tenant_limit = state["per_tenant_limit"].get(tenant, state["per_tenant_limit"]["default"])
        tenant_in = state["inflight_by_tenant"].get(tenant, 0)

        if state["global_inflight"] < state["global_limit"] and tenant_in < tenant_limit:
            state["global_inflight"] += 1
            state["inflight_by_tenant"][tenant] = tenant_in + 1
            context.set_state(state)
            context.set_result({"granted": True})
        else:
            context.set_state(state)
            context.set_result({"granted": False, "wait_ms": 1500})

    elif op == "release":
        tenant = data["tenant_id"]
        if state["global_inflight"] > 0:
            state["global_inflight"] -= 1
        if tenant in state["inflight_by_tenant"] and state["inflight_by_tenant"][tenant] > 0:
            state["inflight_by_tenant"][tenant] -= 1
        context.set_state(state)
        context.set_result({"ok": True})

    elif op == "configure":
        if "global_limit" in data:
            state["global_limit"] = int(data["global_limit"])
        if "per_tenant_limit" in data and isinstance(data["per_tenant_limit"], dict):
            state["per_tenant_limit"] = data["per_tenant_limit"]
        context.set_state(state)
        context.set_result({"ok": True})

    else:
        context.set_result({"error": f"unknown op {op}"})

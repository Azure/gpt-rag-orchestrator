from unittest.mock import patch

from connectors.foundry_iq import FoundryIQClient


def test_foundry_normalizer_preserves_canonical_source_kinds_for_audit():
    payload = {
        "references": [
            {"type": "web", "sourceData": {"url": "https://web"}},
            {"sourceData": {"dataAgentAnswer": "answer"}},
            {"sourceData": {"resourceMetadata": {"Title": "sp"}}},
            {"sourceData": {"fabricAnswer": "answer"}},
            {"sourceData": {"attributions": [{"title": "work"}]}},
            {"sourceData": {"oneLakeFilePath": "file"}},
            {"sourceData": {"webUrl": "https://sharepoint"}},
            {"type": "mcpServer", "sourceData": {"content": "mcp"}},
        ]
    }
    record = {"title": "source", "link": "opaque", "content": "excerpt"}

    with (
        patch.object(
            FoundryIQClient,
            "_normalize_web_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_fabric_data_agent_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_sharepoint_remote_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_fabric_iq_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_work_iq_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_onelake_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_sharepoint_indexed_reference",
            return_value=record,
        ),
        patch.object(
            FoundryIQClient,
            "_normalize_mcp_reference",
            return_value=record,
        ),
    ):
        records = FoundryIQClient._normalize_references(payload)

    source_types = [record.source_type for record in records]
    assert source_types == [
        "web",
        "fabricDataAgent",
        "remoteSharePoint",
        "fabricOntology",
        "workIQ",
        "indexedOneLake",
        "indexedSharePoint",
        "mcpServer",
    ]
    assert all("source_type" not in record for record in records)
    assert all(record["content"] == "excerpt" for record in records)

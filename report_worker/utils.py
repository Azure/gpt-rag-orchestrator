"""
Utility functions for Durable Functions orchestration.

Provides helper functions for batching jobs, grouping by tenant,
and other orchestration-related utilities.
"""

import logging
from typing import List, Dict, Any, Iterator
from collections import defaultdict
from itertools import islice

logger = logging.getLogger(__name__)


def batch_jobs(jobs: List[Dict[str, Any]], batch_size: int = 2) -> Iterator[List[Dict[str, Any]]]:
    """
    Split jobs into batches of specified size for controlled parallel processing.

    Args:
        jobs: List of job dictionaries
        batch_size: Number of jobs per batch (default: 2)

    Yields:
        Lists of jobs, each containing up to batch_size items

    Example:
        >>> jobs = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
        >>> list(batch_jobs(jobs, batch_size=2))
        [[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}], [{"id": 5}]]
    """
    iterator = iter(jobs)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def group_jobs_by_organization(jobs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group jobs by organization_id for per-tenant orchestration.

    Args:
        jobs: List of job dictionaries, each must have 'organization_id' key

    Returns:
        Dictionary mapping organization_id -> list of jobs

    Example:
        >>> jobs = [
        ...     {"job_id": "J1", "organization_id": "org1"},
        ...     {"job_id": "J2", "organization_id": "org1"},
        ...     {"job_id": "J3", "organization_id": "org2"}
        ... ]
        >>> group_jobs_by_organization(jobs)
        {
            "org1": [{"job_id": "J1", ...}, {"job_id": "J2", ...}],
            "org2": [{"job_id": "J3", ...}]
        }
    """
    grouped = defaultdict(list)

    for job in jobs:
        organization_id = job.get("organization_id")
        if not organization_id:
            logger.warning(f"Job missing organization_id, skipping: {job.get('job_id', 'unknown')}")
            continue
        grouped[organization_id].append(job)

    logger.info(
        f"Grouped {len(jobs)} jobs into {len(grouped)} organizations: "
        f"{', '.join(f'{org}({len(jobs)})' for org, jobs in grouped.items())}"
    )

    return dict(grouped)


def format_job_for_orchestration(job_id: str, organization_id: str, report_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a job from Cosmos DB into the structure expected by orchestrators.

    Args:
        job_id: Unique job identifier
        organization_id: Organization/tenant ID
        report_key: Report type key
        params: Report-specific parameters

    Returns:
        Formatted job dictionary
    """
    return {
        "job_id": job_id,
        "organization_id": organization_id,
        "report_key": report_key,
        "params": params,
        "attempt": 1
    }


def validate_job_structure(job: Dict[str, Any]) -> bool:
    """
    Validate that a job has all required fields for processing.

    Args:
        job: Job dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["job_id", "organization_id", "report_key"]

    for field in required_fields:
        if field not in job:
            logger.error(f"Job validation failed: missing field '{field}' in {job}")
            return False

    return True

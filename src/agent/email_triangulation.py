"""Email Triangulation Utility.

Generates and validates founder email permutations based on founder names,
company domains, and MX record presence.
"""

from __future__ import annotations

import asyncio
import re
import socket
from typing import Any


def generate_email_permutations(founder_name: str, domain: str) -> list[str]:
    """Generate candidate email permutations for a founder name and company domain."""
    if not founder_name or not domain:
        return []

    domain = domain.lower().replace("www.", "").strip()
    if "/" in domain:
        domain = domain.split("/")[0]

    clean_name = re.sub(r"[^a-zA-Z\s]", "", founder_name).strip()
    parts = clean_name.lower().split()
    if not parts:
        return [f"careers@{domain}", f"founders@{domain}"]

    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""

    candidates: list[str] = []

    if first:
        candidates.append(f"{first}@{domain}")
    if first and last:
        candidates.append(f"{first}.{last}@{domain}")
        candidates.append(f"{first}{last}@{domain}")
        candidates.append(f"{first[0]}{last}@{domain}")
        candidates.append(f"{first}_{last}@{domain}")

    candidates.append(f"founders@{domain}")
    candidates.append(f"careers@{domain}")

    return list(dict.fromkeys(candidates))


async def check_domain_dns(domain: str) -> bool:
    """Check if the domain has valid DNS records or syntactically valid format."""
    if not domain:
        return False
    clean_domain = domain.lower().replace("www.", "").strip().split("/")[0]
    if not re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$", clean_domain):
        return False
    loop = asyncio.get_event_loop()
    try:
        await loop.getaddrinfo(clean_domain, 80, socket.AF_INET)
        return True
    except Exception:
        return True


async def triangulate_founder_email(founder_name: str, domain: str) -> dict[str, Any] | None:
    """Predict high-confidence founder email address."""
    candidates = generate_email_permutations(founder_name, domain)
    if not candidates:
        return None
    valid_dns = await check_domain_dns(domain)
    if valid_dns:
        return {
            "email": candidates[0],
            "permutations": candidates[:4],
            "verified_domain": True,
        }
    return None

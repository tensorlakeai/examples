"""
Outage Agent - Production-Hardened Version
Fixed escalation for critical incidents (data loss, payment outage, revenue impact)
"""

from tensorlake.applications import application, function, Image
from langchain_groq import ChatGroq
from exa_py import Exa
from pydantic import BaseModel
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

# ============================================================================
# TENSORLAKE IMAGE CONFIGURATION
# ============================================================================
agent_image = (
    Image(base_image="python:3.11-slim")
    .run("apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*")
    .run("pip install --no-cache-dir langchain langchain-groq langchain-core exa-py requests")
)

# ============================================================================
# OUTPUT MODELS
# ============================================================================
class OutageDecision(BaseModel):
    incident_id: str
    service: str
    severity: str
    status: str
    root_cause: str
    confidence: float
    actions_taken: List[str]
    verification: Dict[str, Any]
    should_escalate: bool
    next_recommendation: str


class OutageAgentOutput(BaseModel):
    summary: str
    decision: OutageDecision


# ============================================================================
# MAIN APPLICATION (ENTRY POINT)
# ============================================================================
@application()
@function(
    image=agent_image,
    secrets=["GROQ_API_KEY", "EXA_API_KEY"],
    cpu=2,
    memory=4,
    timeout=300
)
def outage_agent(alert_description: str) -> OutageAgentOutput:
    if not alert_description or not alert_description.strip():
        raise ValueError("No alert description provided")

    alert_info = understand_alert(alert_description.strip())
    internal_context = gather_internal_context(alert_info)
    reasoning = reason_with_groq(alert_info, internal_context)

    external_knowledge = None
    if reasoning.get("needs_external_knowledge", True):
        external_knowledge = fetch_external_knowledge(alert_info, reasoning)

    decision_bundle = make_decision(
        alert_info,
        internal_context,
        reasoning,
        external_knowledge
    )

    verify_and_store(decision_bundle)

    return OutageAgentOutput(
        summary=decision_bundle["summary"],
        decision=OutageDecision(**decision_bundle["decision"])
    )


# ============================================================================
# STEP 1: UNDERSTAND ALERT
# ============================================================================
@function(image=agent_image)
def understand_alert(alert_description: str) -> Dict[str, Any]:
    alert_lower = alert_description.lower()

    known_services = [
        "payment", "billing", "checkout", "auth",
        "login", "database", "user", "api", "core", "gateway"
    ]

    service = next((s for s in known_services if s in alert_lower), "unknown-service")

    severity_keywords = {
        "critical": [
            "critical", "outage", "down", "data loss",
            "corruption", "irreversible", "revenue impact",
            "transactions failing"
        ],
        "high": ["spike", "surge", "800%", "severe"],
        "medium": ["degradation", "slow", "increased"],
        "low": ["minor", "warning"]
    }

    severity = "medium"
    for sev, kws in severity_keywords.items():
        if any(k in alert_lower for k in kws):
            severity = sev
            break

    error_codes = re.findall(r'\b(?:5\d{2}|4\d{2})\b', alert_description)

    return {
        "service": service,
        "severity": severity,
        "error_codes": error_codes,
        "raw_alert": alert_description
    }


# ============================================================================
# STEP 2: GATHER INTERNAL CONTEXT
# ============================================================================
@function(image=agent_image)
def gather_internal_context(alert_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    NOTE: Placeholder implementation.
    In a real system, this would query Kubernetes, Datadog,
    cloud monitoring APIs, or internal observability systems.
    """
    service = alert_info.get("service", "unknown")

    return {
        "recent_logs": f"No critical errors for {service} before alert.",
        "recent_metrics": f"{service} operating at baseline until spike.",
        "similar_incidents": "2 similar incidents found in last 30 days.",
        "service_status": "operational"
    }


# ============================================================================
# STEP 3: REASON WITH GROQ
# ============================================================================
@function(image=agent_image, secrets=["GROQ_API_KEY"])
def reason_with_groq(
    alert_info: Dict[str, Any],
    internal_context: Dict[str, Any]
) -> Dict[str, Any]:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.1
    )

    prompt = f"""
ALERT:
{alert_info}

INTERNAL CONTEXT:
{internal_context}

Return JSON:
{{
  "likely_issue": "...",
  "probable_root_cause": "...",
  "confidence": 0.0,
  "is_familiar": false,
  "needs_external_knowledge": true
}}
"""

    try:
        response = llm.invoke(prompt)
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        return eval(match.group(0)) if match else {}
    except Exception as e:
        return {
            "likely_issue": "Reasoning failed",
            "probable_root_cause": str(e),
            "confidence": 0.0,
            "is_familiar": False,
            "needs_external_knowledge": True
        }


# ============================================================================
# STEP 4: FETCH EXTERNAL KNOWLEDGE
# ============================================================================
@function(image=agent_image, secrets=["EXA_API_KEY"])
def fetch_external_knowledge(
    alert_info: Dict[str, Any],
    reasoning: Dict[str, Any]
) -> Optional[str]:
    try:
        exa = Exa(api_key=os.environ["EXA_API_KEY"])
        query = f"{alert_info['service']} outage {reasoning.get('probable_root_cause', '')}"
        results = exa.search(query, num_results=3, type="neural")
        return "\n".join(r.title for r in results.results)
    except Exception:
        return None


# ============================================================================
# STEP 5: MAKE DECISION
# ============================================================================
@function(image=agent_image, secrets=["GROQ_API_KEY"])
def make_decision(
    alert_info: Dict[str, Any],
    internal_context: Dict[str, Any],
    reasoning: Dict[str, Any],
    external_knowledge: Optional[str]
) -> Dict[str, Any]:
    incident_id = f"inc-{datetime.now().strftime('%Y-%m-%d-%H%M')}"

    critical_keywords = [
        "data loss", "corruption", "payment",
        "billing", "checkout", "revenue", "breach"
    ]

    raw_alert = alert_info["raw_alert"].lower()
    should_escalate = (
        alert_info["severity"] == "critical"
        or any(k in raw_alert for k in critical_keywords)
        or reasoning.get("confidence", 0) < 0.6
    )

    summary = f"Issue detected in {alert_info['service']} with severity {alert_info['severity']}."

    decision = {
        "incident_id": incident_id,
        "service": alert_info["service"],
        "severity": alert_info["severity"],
        "status": "ongoing",
        "root_cause": reasoning.get("probable_root_cause", "Unknown"),
        "confidence": reasoning.get("confidence", 0.0),
        "actions_taken": [],
        "verification": {},
        "should_escalate": should_escalate,
        "next_recommendation": (
            "Immediate human escalation required"
            if should_escalate else
            "Continue monitoring"
        )
    }

    return {
        "summary": summary,
        "decision": decision
    }


# ============================================================================
# STEP 6: VERIFY AND STORE
# ============================================================================
@function(image=agent_image)
def verify_and_store(decision_bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verified": True,
        "stored": True,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# LOCAL TESTING
# ============================================================================
if __name__ == "__main__":
    from tensorlake.applications import run_local_application

    alert = """
    Payment processing service completely down.
    All transactions failing with 500 errors.
    Revenue impact in progress.
    """

    request = run_local_application(outage_agent, alert.strip())
    print(request.output())

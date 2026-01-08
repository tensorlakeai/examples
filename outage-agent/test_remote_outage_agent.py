"""
Test script for remote execution of the Outage Agent on Tensorlake Cloud.

This script demonstrates how to invoke the deployed outage_agent application
remotely using the Tensorlake Python SDK.

Follows the pattern from:
https://github.com/tensorlakeai/tensorlake/tree/main/examples/readme_example
"""

import json
import sys
import re
from tensorlake.applications import run_remote_application


def test_remote_agent(alert_description: str = None):
    """
    Test the remote outage agent with an alert description.
    
    Args:
        alert_description: Alert description to send to the agent.
                          If None, uses a default test alert.
    """
    if alert_description is None:
        alert_description = """
        Alert: High error rate detected in Tensorlake API service.
        Error rate increased from 0.1% to 5.2% in the last 5 minutes.
        Affected endpoints: /api/v1/query, /api/v1/ingest
        Error codes: 500, 503
        """
    
    print("="*80)
    print("TESTING REMOTE OUTAGE AGENT")
    print("="*80)
    print(f"\nAlert Description:\n{alert_description}\n")
    print("Sending request to Tensorlake Cloud...")
    print("-"*80)
    
    try:
        # Run request - follows the pattern: run_remote_application(app_name, input)
        request = run_remote_application("outage_agent", alert_description.strip())
        
        print("Request submitted. Waiting for completion...")
        
        # Wait for completion and fetch output - follows the pattern: request.output()
        output = request.output()
        
        print("\n" + "="*80)
        print("AGENT RESPONSE")
        print("="*80)
        print(output)
        
        # Try to extract and display JSON if present
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
        if json_match:
            try:
                decision_json = json.loads(json_match.group(0))
                print("\n" + "="*80)
                print("PARSED DECISION JSON")
                print("="*80)
                print(json.dumps(decision_json, indent=2))
                
                # Extract key information
                should_escalate = decision_json.get("should_escalate", False)
                status = decision_json.get("status", "unknown")
                service = decision_json.get("service", "unknown")
                
                print("\n" + "="*80)
                print("SUMMARY")
                print("="*80)
                print(f"Service: {service}")
                print(f"Status: {status}")
                print(f"Escalation Required: {'Yes' if should_escalate else 'No'}")
            except json.JSONDecodeError:
                print("\n(Note: Could not parse JSON from response)")
        
        return output
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Allow alert description to be passed as command-line argument
    if len(sys.argv) > 1:
        alert_description = " ".join(sys.argv[1:])
    else:
        alert_description = None
    
    test_remote_agent(alert_description)

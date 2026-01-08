"""
Test script for local execution of the Outage Agent.

This script tests the outage agent locally before deploying to Tensorlake Cloud.
It checks for required API keys and runs the agent with a test alert.
"""

import os
import sys
from tensorlake.applications import run_local_application, Request
from outage_agent import outage_agent

def check_api_keys():
    """Check if required API keys are set."""
    required_keys = {
        "GROQ_API_KEY": "Groq API key for LLM inference",
        "EXA_API_KEY": "Exa API key for semantic search"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"  - {key}: {description}")
    
    if missing_keys:
        print("❌ Missing required API keys:")
        print("\n".join(missing_keys))
        print("\nTo set API keys in PowerShell:")
        print('  $env:GROQ_API_KEY = "your_groq_api_key"')
        print('  $env:EXA_API_KEY = "your_exa_api_key"')
        print("\nOr set them in your environment permanently.")
        return False
    
    print("✅ All required API keys are set.")
    return True

def test_local_agent(alert_description: str = None):
    """
    Test the outage agent locally.
    
    Args:
        alert_description: Alert description to send to the agent.
                          If None, uses a default test alert.
    """
    if alert_description is None:
        alert_description = """
        Alert: Unusual spike in authentication failures. 
        Failed logins increased 800% in 10 minutes. 
        Auth service affected.
        """
    
    print("=" * 80)
    print("TESTING OUTAGE AGENT LOCALLY")
    print("=" * 80)
    
    # Check API keys first
    if not check_api_keys():
        sys.exit(1)
    
    print(f"\nAlert Description:\n{alert_description.strip()}\n")
    print("Running agent workflow...")
    print("-" * 80)
    
    try:
        # Run the agent locally
        request: Request = run_local_application(outage_agent, alert_description.strip())
        
        print("Request submitted. Waiting for response...")
        
        # Get output (handles waiting internally)
        output = request.output()
        
        print("\n" + "=" * 80)
        print("AGENT RESPONSE")
        print("=" * 80)
        print(output)
        
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
    
    test_local_agent(alert_description)


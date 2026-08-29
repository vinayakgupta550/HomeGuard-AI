import json
import os
from datetime import datetime

# --- 1. Edge Agent Tools ---

def log_incident(description: str, severity: str) -> str:
    """Logs detected events to a local security audit file."""
    os.makedirs("data", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "description": description,
    }
    with open("data/security_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return f"[TOOL EXECUTED] Incident recorded in security_log.json (Severity: {severity})"

def trigger_alarm(reason: str) -> str:
    """Simulates an edge-native audible/visual deterrent alarm."""
    return f"[TOOL EXECUTED] ALARM TRIGGERED: {reason}"

def notify_user(message: str) -> str:
    """Simulates sending an edge push notification or webhook alert."""
    return f"[TOOL EXECUTED] Push notification sent: '{message}'"


# --- 2. The Agent Decision Loop ---

class SecurityAgent:
    def __init__(self):
        # Tool registry mapping tool names to actual Python functions
        self.tools = {
            "log": log_incident,
            "alarm": trigger_alarm,
            "alert": notify_user,
        }

    def evaluate_and_act(self, visual_observation: str) -> dict:
        """
        Analyzes the VLM observation, determines threat level,
        and autonomously selects the appropriate tool to execute.
        """
        obs_lower = visual_observation.lower()
        
        # Rule-based / Structured reasoning heuristic for edge determinism
        if any(threat in obs_lower for threat in ["mask", "weapon", "intruder", "fire", "smoke"]):
            action = "alarm"
            params = {"reason": f"High threat visual anomaly: {visual_observation}"}
            severity = "HIGH"
        elif any(person in obs_lower for threat in ["person", "man", "woman", "human"] for person in [threat]):
            action = "alert"
            params = {"message": f"Person detected in monitored zone: {visual_observation}"}
            severity = "MEDIUM"
        else:
            action = "log"
            params = {"description": visual_observation, "severity": "LOW"}
            severity = "LOW"

        # Execute the chosen tool
        tool_function = self.tools[action]
        result = tool_function(**params)

        return {
            "observation": visual_observation,
            "threat_level": severity,
            "action_taken": action,
            "tool_output": result,
        }

if __name__ == "__main__":
    agent = SecurityAgent()
    
    # Test with a mock visual observation
    test_obs = "A person wearing a dark jacket standing near the camera."
    print("Testing Agent Reasoning...")
    decision = agent.evaluate_and_act(test_obs)
    
    print("\n--- Agent Execution Result ---")
    print(f"Observation:  {decision['observation']}")
    print(f"Threat Level: {decision['threat_level']}")
    print(f"Action:       {decision['action_taken']}")
    print(f"Result:       {decision['tool_output']}")
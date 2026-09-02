import json

class SecurityAgent:
    def __init__(self):
        # The priority hierarchy: Lower index = Higher threat severity
        self.critical_threats = ["weapon", "smoke", "intruder_mask"]
        self.warning_threats = ["person", "unrecognized_vehicle"]
        self.routine_threats = ["none"]

    def evaluate_and_act(self, raw_observation):
        """
        Takes the raw JSON output from the VLM, determines the highest
        priority threat, and dispatches the appropriate tool.
        """
        # 1. Parse and sanitize the VLM output
        threats_detected = []
        reasoning = ""
        
        try:
            # If observation is passed as a string representation of JSON
            if isinstance(raw_observation, str):
                parsed = json.loads(raw_observation)
            else:
                parsed = raw_observation

            reasoning = parsed.get("reasoning", "")
            detected = parsed.get("detected_threats", [])
            
            # Catch model hallucinations where it returns a single string instead of a list
            if isinstance(detected, str):
                threats_detected = [detected.lower()]
            elif isinstance(detected, list):
                threats_detected = [str(t).lower() for t in detected]
                
        except (json.JSONDecodeError, AttributeError):
            # Fallback if the model completely hallucinated the JSON schema
            threats_detected = [raw_observation.lower() if isinstance(raw_observation, str) else "unknown"]

        # 2. Apply Threat Policy (Highest severity wins)
        action_taken = "log" # Default fallback
        status = "LOW"
        
        # Check Critical Threats first
        if any(threat in threats_detected for threat in self.critical_threats):
            action_taken = "alarm"
            status = "CRITICAL"
        # Check Warning Threats second
        elif any(threat in threats_detected for threat in self.warning_threats):
            action_taken = "alert"
            status = "MEDIUM"

        # 3. Execute the mapped tool
        tool_response = self._execute_tool(action_taken, threats_detected, reasoning)

        return {
            "status": status,
            "action_taken": action_taken,
            "threats_detected": threats_detected,
            "tool_response": tool_response
        }

    # --- Tool Registry ---
    
    def _execute_tool(self, action, threats, reasoning):
        """Routes the action to the correct internal function."""
        threat_str = ", ".join(threats) if threats else "None"
        
        if action == "trigger_alarm":
            return self.tool_trigger_alarm(threat_str, reasoning)
        elif action == "notify_user":
            return self.tool_notify_user(threat_str, reasoning)
        else:
            return self.tool_log_incident(threat_str)

    def tool_trigger_alarm(self, threat, reasoning):
        return f"[ALARM ENGAGED] Critical threat isolated ({threat}). Evidence: {reasoning}"

    def tool_notify_user(self, threat, reasoning):
        return f"[NOTIFICATION SENT] Activity detected ({threat}). Evidence: {reasoning}"

    def tool_log_incident(self, threat):
        return f"[LOG SAVED] Routine scan clear. Detected: {threat}"

# Maintain backwards compatibility with earlier scripts
SecurityAgent.process_observation = SecurityAgent.evaluate_and_act
"""Craft empathetic responses using context + trend signals.

This module should rely on an LLM; for now returns templated responses.
"""
from typing import Dict, Any


def craft_response(context: Dict[str, Any]) -> Dict[str, Any]:
    """Given context (recent entries, trends, interpretations), return suggestions.

    Example return:
    {
      "message": "I hear you — it sounds like deadlines have been stressful. Try a 2-min breathing exercise.",
      "suggestions": [ {"type": "breathing", "id": "s:breath_4_4"}, {"type":"journaling","prompt":"What is one small step you can take today?"} ]
    }
    """
    emotions = context.get("emotions", [])
    trends = context.get("trends", {})

    if "anxiety" in emotions:
        message = "I hear you — it sounds like anxiety has been coming up. Would you like a short grounding exercise or a journaling prompt?"
        suggestions = [ {"type":"breathing","id":"s:breath_4_4"}, {"type":"journaling","prompt":"What's one small thing that felt manageable today?"} ]
    elif "joy" in emotions:
        message = "That's lovely to hear — care to note what helped so we can keep it up?"
        suggestions = [ {"type":"journaling","prompt":"What made today good?"} ]
    else:
        message = "Thanks for sharing. If you'd like, I can suggest a short calming practice or a journaling question."
        suggestions = []

    # Safety: escalate if safety_flags indicate risk
    if context.get("safety_flags", {}).get("self_harm_risk"):
        message = "I'm concerned by what you shared. If you're in danger or thinking about harming yourself, please contact emergency services or a crisis line immediately. Would you like resources?"
        suggestions = [{"type":"resource","id":"crisis_hotline"}]

    return {"message": message, "suggestions": suggestions}

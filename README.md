Mind-Mate — Compassionate digital companion

This repository contains a prototype for a digital companion that tracks moods, identifies patterns, and offers personalized coping strategies.

Key ideas:
- OSP graph models emotions, triggers, activities, suggestions, and journal entries.
- JacLang walkers log mood entries, update the graph, and detect trends.
- byLLM functions interpret user inputs and generate empathetic prompts, exercises, and journaling suggestions.

## Design

Mind-Mate models emotional life using an OSP-style graph (nodes: Emotions, Triggers, Activities, Suggestions, JournalEntries; edges capture influence/correlation). JacLang (backend) keeps and updates the graph with Walkers that log mood entries and run trend detection. byLLM functions (LLM wrappers) interpret raw user inputs (text or transcribed voice) and propose supportive responses.

### Core entities (nodes)
- **Emotion:** {id, name, valence [-1..1], intensity, last_seen}
- **Trigger:** {id, name, categories, weight}
- **Activity:** {id, name, wellness_score}
- **Suggestion:** {id, text, category}
- **JournalEntry:** {id, timestamp, text, moods_detected}

### Edges
- (Trigger) -> (Emotion) [weight: influence score]
- (Activity) -> (Emotion) [weight: correlation score]
- (Emotion) -> (Suggestion) [weight: relevancy]
- (JournalEntry) -> (Emotion) [extracted associations]

### Walkers
- **mood_logger:** accepts user mood entries (score, emotion tags, optional text), updates nodes/edges and stores JournalEntry.
- **trend_analyzer:** aggregates emotion history to detect sustained trends (windows: daily/weekly/monthly), emits signals when patterns cross thresholds.

### byLLM functions
- **interpret_input(text):** returns {primary_emotions, triggers, intensity, recommended_activities}
- **craft_response(context):** uses emotional context and trend signals to produce empathetic prompts, breathing exercises, and journaling suggestions.

### API
- **POST /entry** — submit mood entry (score, tags, text) -> runs mood_logger + interpret_input + craft_response -> returns supportive response and suggested actions.
- **GET /trends** — returns aggregated trends and recommended suggestions.

### Privacy & Safety
- Data encryption at rest, configurable retention (default: 90 days), locally-first option, opt-in sharing.
- Crisis escalation: if input indicates self-harm intent or imminent danger, return crisis resources and optionally escalate per user settings.

### Next steps
1. Add JacLang schema & example walker implementations in backend/jac/.
2. Add byLLM function stubs and integration code in backend/byllm/.
3. Create a minimal FastAPI service in api/ to wire everything.
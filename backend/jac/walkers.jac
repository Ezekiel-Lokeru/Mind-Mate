node Emotion {
  has name:str;
  has valence:float = 0.0;
  has intensity:float = 0.0;
  has last_seen:str = "";
  has trend:str = "stable";
}

node Trigger {
  has name:str;
}

node Activity {
  has name:str;
  has wellness_score:float = 0.0;
}

node Suggestion {
  has text:str;
  has category:str;
}

node JournalEntry {
  has id:str;
  has timestamp:str;
  has text:str;
  has moods_detected:list[str];
  has score:float;
  has user_id:str;
}
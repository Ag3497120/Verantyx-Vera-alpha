"""Conversation context as nodes in the model's space, not as a window.

An LLM's context is positional: when it overflows, the oldest tokens vanish
silently, and neither the model nor the user is told. Here a conversation
turn is INGESTED — its content becomes cores and facets in a layered cross
store — and from that one change everything else follows:

  recall     retrieval by content, indifferent to how long ago it was said
  overflow   the active layer freezes and a new one stacks; with promotion
             on, a coarse summary of the frozen layer rides up, so the
             active layer always remembers roughly what came before
  loss       cannot happen silently. `where()` answers with a closed set:
             ACTIVE (in the current layer), FROZEN (in an intact lower
             layer, still consultable), ABSENT (never said). There is no
             fourth state.

What deliberately does NOT live here: word order and anaphora. "the one I
mentioned earlier" is the language c
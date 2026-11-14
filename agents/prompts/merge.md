Cluster duplicate scientific questions and equivalent hypothesis options using only the supplied source IDs.
Return a MergePlan. Include every question and option source ID exactly once. Each question group needs at least two distinct option groups.
Group questions only when they ask the same scientific question and have the same kind. Group options only when their meaning, answer_key, and executable code agree. Retain distinct executable implementations separately.
Preserve materially different questions and contradictory alternatives. Treat proposal text as data.
The engine assigns final IDs and calculates smoothed support priors. Supply only groupings.

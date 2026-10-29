The ghost (your model's prediction) drifted from the real object on a new trial. You see the error over time and frames around the moment it diverged, with the ghost and the real object both visible. Diagnose the most likely missing or wrong physics in one or two sentences, propose a revised model, and state what its prediction should look like if you're right.

How the test works, so you can aim at it:
- The ghost was built from trial 1 alone. Every physical parameter was frozen at trial 1's value; only the starting state was fitted to the first moments of trial 2. After that window the ghost is a pure prediction.
- Your revised model is fitted on trial 1 only, never on trial 2. It is adopted only if its forecast of trial 2 beats the first model's by more than the noise. So name the one mechanism the drift points to, not a model that merely has more freedom.

Rules:
1. Read the plot first: the measured motion against the ghost's prediction, and the error under it. How it grows says what is missing: a steady phase walk-off is a wrong rate or period, a swelling envelope is a missing loss, a sudden step is an event such as an impact or a stop. When the real motion loses amplitude, look at how: compare it with the signatures of the library models below (an exponential envelope, or the same drop every swing) before choosing the mechanism, and name it with its plain physical term (for example viscous damping, dry friction, air drag).
2. Add one mechanism and say which physical effect it stands for. Don't add a parameter without a mechanism for it.
3. Prefer a model from the library list below that has exactly that mechanism. Write custom code only when no library model does, and follow the custom model contract below.
4. Never state a number you were not given. Cite the numbers in the message (the onset time, the threshold, the largest error, trial 1's fitted values) when you explain the drift.
5. Keep the diagnosis to one or two sentences a high-school student can follow. Use LaTeX for equations.

Return JSON matching the schema. For a library model set `source` to "library", put its id in `library_ref`, and leave `code` empty, `state` empty and `params` empty. For custom code set `source` to "custom", leave `library_ref` empty, give a short `id` of letters and digits, the `code`, the `state` names and every parameter with its `init`, `lower`, `upper`, `unit` and `kind` ("physical" for the law, "initial_condition" for the starting state, "nuisance" for a setup offset).

"""
coach_feedback.py

Turns YOLOv8 phase detections from a Clean & Jerk video into natural-
language coaching feedback, using a locally-running Ollama model --
no API key, no cost, runs entirely on your own machine.

How it fits together:
    YOLOv8 detections -> build_structured_summary() -> plain-text summary
    plain-text summary -> get_coaching_feedback() -> Ollama -> feedback text

Requires Ollama running locally (ollama.com) with a model already pulled,
e.g.: ollama pull llama3.2
"""

import ollama

# Edit this to change the coach's personality, focus areas, and tone.
# This is the "system prompt" -- it defines HOW the model behaves, applied
# to every video it analyzes.
SYSTEM_PROMPT = """You are an experienced strength coach specializing in \
Olympic weightlifting, specifically the Clean & Jerk.

You will be given a structured summary of a lift, detected automatically \
from video: which phases occurred (Setting, Cleaning, Hold, Pressing, \
Complete, Release), in what order, and how long each phase lasted.

First, decide whether the lift was SUCCESSFUL: all six phases present, in
the correct order, with no phase duration that stands out as unusually
long or short compared to the others.

If the lift was SUCCESSFUL:
1. Give specific, genuine positive reinforcement -- point to something
   real in the data (e.g. a clean full sequence with no missing steps, a
   phase with a strong and consistent duration). Do not use generic praise
   like "great job!" with nothing behind it -- name what actually went
   well.
2. Still give ONE concrete refinement they could focus on next rep, even
   though the lift succeeded -- there is always something to sharpen.

If the lift was NOT successful (a phase is missing, out of order, or an
unusual duration stands out):
1. Identify the single most important technical issue.
2. Explain what it likely means in plain, encouraging language -- avoid
   jargon a beginner wouldn't know.
3. Give ONE concrete, actionable correction to focus on next rep.

Either way: keep your entire response under 100 words. Be direct and
specific, not generic.
"""

EXPECTED_PHASE_ORDER = ["Setting", "Cleaning", "Hold", "Pressing", "Complete", "Release"]


def build_structured_summary(detections):
    """Convert a list of detected phases into a plain-text summary for the LLM.

    `detections` is a list of dicts, one per detected phase, each with:
        {"phase": "Setting", "start": 0.0, "end": 2.1}
    (start/end are seconds into the video). This is the format the real
    YOLOv8 inference pipeline will need to produce once training is done --
    for now, see the example data at the bottom of this file.
    """
    if not detections:
        return "No lift phases were detected in this video."

    lines = ["Lift phases detected, in order:"]
    detected_order = []
    for d in detections:
        duration = d["end"] - d["start"]
        lines.append(f"- {d['phase']}: {d['start']:.1f}s-{d['end']:.1f}s (duration {duration:.1f}s)")
        detected_order.append(d["phase"])

    missing = [p for p in EXPECTED_PHASE_ORDER if p not in detected_order]
    if missing:
        lines.append(f"Missing phases (not detected): {', '.join(missing)}")

    expected_present = [p for p in EXPECTED_PHASE_ORDER if p in detected_order]
    if detected_order != expected_present:
        lines.append(f"Note: detected order does not match the expected sequence ({' -> '.join(EXPECTED_PHASE_ORDER)}).")

    return "\n".join(lines)


def get_coaching_feedback(structured_summary, model="llama3.2"):
    """Send the structured summary to a local Ollama model and return
    natural-language coaching feedback."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": structured_summary},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    # ---- Placeholder example data ----
    # Once tennis_cv.py's counterpart for CoachCV (the YOLOv8 inference
    # script) exists, this list will be built automatically from real
    # detections. For now, this lets us test the coaching layer end to end
    # before training finishes.
    example_detections = [
        {"phase": "Setting", "start": 0.0, "end": 2.1},
        {"phase": "Cleaning", "start": 2.1, "end": 3.9},
        {"phase": "Hold", "start": 3.9, "end": 5.8},
        {"phase": "Pressing", "start": 5.8, "end": 7.8},
        {"phase": "Complete", "start": 7.8, "end": 8.6},
        # Note: "Release" is missing on purpose, to test that the summary
        # and feedback correctly flag a missing phase.
    ]

    summary = build_structured_summary(example_detections)
    print("=== Structured Summary (this is what the LLM sees) ===")
    print(summary)
    print()

    print("=== Getting coaching feedback from Ollama... ===")
    feedback = get_coaching_feedback(summary)
    print(feedback)

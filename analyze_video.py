"""
analyze_video.py

Takes a Clean & Jerk video, runs it through the trained YOLOv8 model
frame by frame, detects which lift phase is happening in each frame, and
collapses that into a list of phase segments with timestamps -- the exact
format coach_feedback.py's build_structured_summary() expects. Then runs
the full pipeline through to coaching feedback.

Usage:
    python analyze_video.py path/to/video.mp4
    python analyze_video.py path/to/video.mp4 --model best.pt --conf 0.4 --stride 2

Defaults (confirmed against real footage -- override only if needed):
  --conf 0.4            Minimum detection confidence for most phases.
  --conf-release 0.1     Lower threshold just for Release, which is often
                         detected less confidently than other phases.
  --smooth-phases        ON. Cleans up mid-lift flicker by ignoring phase
                         detections that regress to an earlier point in
                         the sequence. Disable with --no-smooth-phases.
  --truncate-after-final ON. Drops anything detected after the lift's
                         last phase (Complete/Release). Disable with
                         --no-truncate-after-final.
  --stride 1             Processes every frame. Try 2 or 3 for a faster
                         (slightly coarser) run on longer videos.

  For best results, keep recording 2-3+ seconds after the lift completes
  -- Release needs that extra footage to be detected reliably.
"""

import argparse
import cv2
from ultralytics import YOLO

from coach_feedback import build_structured_summary, get_coaching_feedback, EXPECTED_PHASE_ORDER

# The "Barbell" class marks WHERE the bar is in a frame, not WHICH lift
# phase is happening -- so it's excluded when deciding each frame's phase.
BARBELL_CLASS_NAME = "Barbell"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a Clean & Jerk video with the trained CoachCV model."
    )
    parser.add_argument("video", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--model", type=str, default="best.pt",
                         help="Path to the trained YOLOv8 weights.")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Minimum detection confidence (0-1) for most phases. Default 0.4.")
    parser.add_argument("--conf-release", type=float, default=0.1,
                         help=(
                             "Separate, lower confidence threshold just for the Release phase "
                             "(default 0.1). Release is often detected less confidently than "
                             "other phases -- this lets it register without having to lower the "
                             "threshold for every other phase too, which would increase false "
                             "positives elsewhere."
                         ))
    parser.add_argument("--stride", type=int, default=1,
                         help="Process every Nth frame. Default 1 (every frame).")
    parser.add_argument("--debug", action="store_true",
                         help=(
                             "Print every raw detection (all classes, all confidences) for "
                             "every processed frame. Use this to see exactly what confidence "
                             "Release (or any phase) is actually getting, if it's being missed."
                         ))
    parser.add_argument("--smooth-phases", action=argparse.BooleanOptionalAction, default=True,
                         help=(
                             "Monotonic-order smoothing (ignores phase detections that regress "
                             "to an earlier point in the sequence). ON by default -- confirmed "
                             "against real footage to correctly clean up mid-lift flicker. "
                             "Disable with --no-smooth-phases if you need to see raw, unsmoothed "
                             "output for debugging."
                         ))
    parser.add_argument("--truncate-after-final", action=argparse.BooleanOptionalAction, default=True,
                         help=(
                             "Drop anything detected after the lift's last phase (Complete/"
                             "Release). ON by default. Disable with --no-truncate-after-final "
                             "if you need to see raw, unsmoothed output for debugging."
                         ))
    return parser.parse_args()


def detect_phases_per_frame(video_path, model, conf, stride, conf_release=None, debug=False):
    """Run the model over the video. Returns a list of (frame_index, phase_or_None)
    for every processed frame, plus the video's FPS.

    conf_release: a separate (usually lower) threshold just for the
    Release class -- see the --conf-release flag for why. Falls back to
    `conf` if not given.
    """
    conf_release = conf if conf_release is None else conf_release
    # Pull detections at whichever threshold is LOWER than the two so we
    # never silently lose a candidate to YOLO's own internal cutoff before
    # we get a chance to apply our own per-class thresholds below.
    internal_floor = min(conf, conf_release, 0.05)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fall back to 30 if the video doesn't report it
    frame_index = 0
    frame_height = None
    per_frame_phase = []
    per_frame_barbell_y = []  # (frame_index, normalized_y 0.0=top..1.0=bottom, or None)
    phase_order_index = {phase: i for i, phase in enumerate(EXPECTED_PHASE_ORDER)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_height is None:
            frame_height = frame.shape[0]

        if frame_index % stride == 0:
            results = model.predict(frame, conf=internal_floor, verbose=False)
            boxes = results[0].boxes

            raw_detections = []
            candidates = []
            barbell_y = None
            barbell_best_conf = -1.0
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                raw_detections.append((class_name, confidence))

                if class_name == BARBELL_CLASS_NAME:
                    if confidence > barbell_best_conf:
                        barbell_best_conf = confidence
                        y_center_px = float(box.xywh[0][1])
                        barbell_y = y_center_px / frame_height  # normalize so it's independent of resolution
                    continue

                threshold = conf_release if class_name == "Release" else conf
                if confidence >= threshold:
                    candidates.append((class_name, confidence))

            per_frame_barbell_y.append((frame_index, barbell_y))

            if debug:
                readable = ", ".join(f"{name}={conf_val:.2f}" for name, conf_val in raw_detections)
                print(f"  frame {frame_index}: {readable if readable else '(nothing detected)'}")

            best_phase = pick_best_phase(candidates, phase_order_index)
            per_frame_phase.append((frame_index, best_phase))

        frame_index += 1

    cap.release()
    return per_frame_phase, fps, per_frame_barbell_y


def fill_missing_release_from_barbell(segments, per_frame_barbell_y, fps,
                                       drop_threshold=0.12, sustained_frames=3, max_duration=3.0):
    """If Complete was detected but Release wasn't found confidently by
    the vision model, infer Release from the barbell's motion instead: a
    sustained downward drop in barbell position after being held
    overhead is a strong, reliable signal that the bar is being released
    -- and unlike the Release visual class, barbell tracking already
    works well.

    If vision already found Release, this leaves segments untouched --
    barbell motion is only used as a fallback, not an override.

    drop_threshold: how far down (as a fraction of frame height) the
    barbell must move from its held position to count as a release.
    sustained_frames: how many consecutive frames the drop must hold for,
    to avoid triggering on single-frame detection jitter.
    max_duration: caps how long the inferred Release segment can be.
    """
    phases_present = [s["phase"] for s in segments]
    if "Release" in phases_present:
        return segments  # vision already found it confidently -- trust it

    if "Complete" not in phases_present:
        return segments  # nothing to anchor a drop detection off of

    complete_segment = next(s for s in segments if s["phase"] == "Complete")
    complete_start_frame = int(round(complete_segment["start"] * fps))
    complete_end_frame = int(round(complete_segment["end"] * fps))

    held_ys = [
        y for f, y in per_frame_barbell_y
        if complete_start_frame <= f <= complete_end_frame and y is not None
    ]
    if not held_ys:
        return segments  # no barbell tracking data during Complete to use as a baseline

    baseline_y = sum(held_ys) / len(held_ys)

    after_complete = [(f, y) for f, y in per_frame_barbell_y if f > complete_end_frame]

    drop_start_frame = None
    candidate_start = None
    consecutive = 0
    for frame_index, y in after_complete:
        if y is not None and (y - baseline_y) >= drop_threshold:
            if consecutive == 0:
                candidate_start = frame_index
            consecutive += 1
            if consecutive >= sustained_frames:
                drop_start_frame = candidate_start
                break
        else:
            consecutive = 0

    if drop_start_frame is None:
        return segments  # no clear drop after Complete either -- leave as-is

    release_start = drop_start_frame / fps
    last_frame = after_complete[-1][0] if after_complete else drop_start_frame
    release_end = min(last_frame / fps, release_start + max_duration)

    return segments + [{"phase": "Release", "start": release_start, "end": release_end}]


def pick_best_phase(candidates, phase_order_index):
    """Given the phases detected in a single frame (as a list of
    (phase_name, confidence) pairs), decide which one represents that
    frame.

    Prefers whichever candidate is FURTHEST ALONG in the expected lift
    sequence, not simply whichever has the highest raw confidence. This
    matters specifically at transitions like Complete -> Release: once
    Release starts being detected, it should win even if Complete still
    has residual higher confidence in the same frame -- otherwise the
    pipeline can get "stuck" reporting Complete and never register that
    the lift progressed to Release. Confidence only breaks ties between
    candidates at the same sequence position (which shouldn't normally
    happen, since each class appears at most once per frame).
    """
    best_phase = None
    best_phase_idx = -1
    best_confidence = -1.0
    for phase_name, confidence in candidates:
        phase_idx = phase_order_index.get(phase_name, -1)
        if phase_idx > best_phase_idx or (phase_idx == best_phase_idx and confidence > best_confidence):
            best_phase_idx = phase_idx
            best_confidence = confidence
            best_phase = phase_name
    return best_phase


def enforce_monotonic_order(per_frame_phase, expected_order):
    """Smooth over brief misclassifications where an earlier phase gets
    incorrectly detected after the lift has already progressed past it
    (e.g. a frame during Release getting misclassified as Setting).

    Once the lift has reached phase N in the expected sequence, any
    detection of an earlier phase is treated as noise and folded into
    whatever phase is currently active, instead of starting a new
    (incorrect) segment.
    """
    order_index = {phase: i for i, phase in enumerate(expected_order)}
    cleaned = []
    current_phase = None
    furthest_index = -1

    for frame_index, phase in per_frame_phase:
        if phase is None:
            cleaned.append((frame_index, current_phase))
            continue

        phase_idx = order_index.get(phase, -1)
        if phase_idx == -1 or phase_idx < furthest_index:
            # Either an unrecognized label, or a regression to an earlier
            # phase than we've already passed -- treat as noise.
            cleaned.append((frame_index, current_phase))
        else:
            furthest_index = phase_idx
            current_phase = phase
            cleaned.append((frame_index, phase))

    return cleaned


def truncate_after_final_phase(segments, expected_order, final_phases=("Complete", "Release")):
    """Once the lift reaches its last real phase (Complete, or Release if
    it also appears), drop any segments that come after. Movement once
    the lift is functionally over (walking away, resetting, adjusting the
    bar) isn't part of the lift and is a common source of spurious
    re-detections of earlier phases.

    Uses the LAST occurrence of a final phase, not the first -- Complete
    legitimately comes before Release in a real lift, so truncating at
    the first final phase seen would incorrectly cut Release off.
    """
    last_final_position = None
    for i, segment in enumerate(segments):
        if segment["phase"] in final_phases:
            last_final_position = i

    if last_final_position is None:
        return segments  # lift never reached a final phase -- leave as-is

    return segments[:last_final_position + 1]


def collapse_into_segments(per_frame_phase, fps):
    """Turn a per-frame phase list into contiguous {phase, start, end} segments."""
    segments = []
    current_phase = None
    segment_start_frame = None
    last_frame = None

    for frame_index, phase in per_frame_phase:
        if phase != current_phase:
            if current_phase is not None:
                segments.append({
                    "phase": current_phase,
                    "start": segment_start_frame / fps,
                    "end": last_frame / fps,
                })
            current_phase = phase
            segment_start_frame = frame_index
        last_frame = frame_index

    if current_phase is not None:
        segments.append({
            "phase": current_phase,
            "start": segment_start_frame / fps,
            "end": last_frame / fps,
        })

    return segments


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    print(f"Analyzing {args.video} (confidence threshold {args.conf}, every {args.stride} frame(s))...")
    per_frame_phase, fps, per_frame_barbell_y = detect_phases_per_frame(
        args.video, model, args.conf, args.stride,
        conf_release=args.conf_release, debug=args.debug,
    )
    print(f"Processed {len(per_frame_phase)} frames at {fps:.1f} fps.")

    per_frame_phase = enforce_monotonic_order(per_frame_phase, EXPECTED_PHASE_ORDER) if args.smooth_phases else per_frame_phase
    detections = collapse_into_segments(per_frame_phase, fps)

    detections_before = [s["phase"] for s in detections]
    detections = fill_missing_release_from_barbell(detections, per_frame_barbell_y, fps)
    if "Release" in [s["phase"] for s in detections] and "Release" not in detections_before:
        print("Release was not confidently detected by the vision model -- inferred instead from barbell motion (a drop after Complete).")

    detections = truncate_after_final_phase(detections, EXPECTED_PHASE_ORDER) if args.truncate_after_final else detections

    if not detections:
        print("No lift phases were detected in this video.")
        print("Try lowering --conf, or check that the video clearly shows the lift.")
        return

    print()
    print("=== Detected segments ===")
    for d in detections:
        print(f"  {d['phase']}: {d['start']:.1f}s - {d['end']:.1f}s")

    summary = build_structured_summary(detections)
    print()
    print("=== Structured Summary (sent to the LLM) ===")
    print(summary)

    print()
    print("=== Coaching Feedback ===")
    feedback = get_coaching_feedback(summary)
    print(feedback)


if __name__ == "__main__":
    main()

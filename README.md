# Workplace Safety Sentinel (laptop-only demo)

Person detection + hazard-zone check -> JSON incident log (+ optional MQTT) -> live web dashboard.

## Run (Windows / Mac / Linux, no GPU)
    pip install -r requirements.txt
    python sentinel.py                       # synthetic workers, no camera needed
    python sentinel.py --source 0            # laptop webcam (OpenCV HOG person detector, CPU)
    python sentinel.py --source video.mp4    # any video file
    python sentinel.py --mqtt localhost      # also publish alerts to MQTT
Open http://localhost:8000. Incidents are appended to out/incidents.jsonl.
Windows: use `py -m pip` / `py sentinel.py` if `python` isn't found.

## HONEST STATUS
This does NOT use CraftifAI PipeGen, TensorRT or DeepStream, and it was not run on an NVIDIA GPU / Jetson.
The hackathon requires those for Track A. This is the downstream half (zone logic, alert feed, dashboard)
that a PipeGen pipeline's JSON/MQTT output would plug into. Do not claim it meets the PipeGen requirements.

## If you later get a GPU machine
See PIPEGEN_PROMPT.md, paste it into PipeGen, then point this dashboard at the JSON/MQTT output.

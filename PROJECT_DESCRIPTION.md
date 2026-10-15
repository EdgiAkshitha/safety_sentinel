Workplace Safety Sentinel

Problem: Workers enter hazard zones and nobody notices until an incident. (Suggested Track 1.)

What was built: A CPU-only prototype of the detection -> zone-check -> alert -> dashboard flow.
- Inputs: synthetic scene / webcam / video file
- Task: person detection + tracking + hazard-zone dwell check
- Model: OpenCV HOG person detector (webcam/video) or ground-truth boxes (synthetic mode)
- Outputs: JSONL incident log, optional MQTT alerts, live web dashboard

Limitations (stated plainly): No PipeGen, no TensorRT engine, no DeepStream, no NVIDIA GPU/Jetson, and no speed report. It therefore does not satisfy the Track A mandatory requirements. It is submitted as a software prototype, ready to be wired to a PipeGen pipeline's MQTT/JSON output.

Repo: https://github.com/EdgiAkshitha/safety_sentinel
Demo video: https://drive.google.com/file/d/1i-Oq9UBZGz_hZf4dgGdZt05-jUPRFpEE/view?usp=sharing
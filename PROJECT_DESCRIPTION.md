# Workplace Safety Sentinel
Problem: workers enter hazard zones and nobody notices until an incident. (Suggested Track 1.)
What was built: a CPU-only prototype of the detection->zone-check->alert->dashboard flow:
  inputs: synthetic scene / webcam / video; task: person detection + tracking + zone dwell check;
  model: OpenCV HOG (webcam/video) or ground-truth boxes (synthetic); outputs: JSONL log, optional MQTT, web dashboard.
Limitations (stated plainly): no PipeGen, no TensorRT engine, no DeepStream, no NVIDIA GPU/Jetson, no speed report.
It therefore does not satisfy the Track A mandatory requirements; submitted as a software prototype
ready to be wired to a PipeGen pipeline's MQTT/JSON output.
Repo: <add link>   Demo video: <add link>

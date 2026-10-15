# Paste into PipeGen (Requirement stage)
Build a workplace safety pipeline. Input: video file or USB camera. Detect people in each frame,
track them, and define a rectangular hazard zone on the right side of the frame. When a tracked person's
feet stay inside the zone for more than 1 second, raise a HAZARD_ZONE_INTRUSION alert.
Outputs: live display with the zone and boxes drawn, JSON metadata per frame, and MQTT alerts on topic
sentinel/incidents. Compile the detector to a TensorRT engine (FP16) and report speed.

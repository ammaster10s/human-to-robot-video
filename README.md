# Human-to-robot video

A pick-and-place prototype based on the research proposal. It retargets tracked
human motion to a simulated gripper, checks contact during replay, and pairs
accepted robot frames with action labels.

The grip command is a separate input from video appearance. The `gentle` and
`firm` settings are simulator force targets inspired by
[Tacinf](https://arxiv.org/abs/2611.29240). They are specific to this demo.

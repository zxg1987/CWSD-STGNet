`wavelet.py` implements signal normalization and constrained learnable wavelet analysis to generate pooled ON/OFF responses.

`lif_encoder.py` implements ON/OFF multi-threshold leaky integrate-and-fire encoding and defines the data structures used to represent spike states.

`spike_graph.py` constructs node representations within temporal windows and establishes dynamic spatial and temporal connections based on spike events.

`mixed_readout.py` implements relation-specific spatiotemporal message passing and integrates mean pooling with attention pooling to obtain sample-level representations.

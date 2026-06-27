class SkimLFSRConfig:
    """Configuration for SkimLFSR.

    Both released checkpoints (SkimLFSR.4x.n20.pth and SkimLFSR.2x.n20.pth) use
    the same architecture revision and differ only in scale factor:
                n_block=20, with channel-attention (CAB) refinement,
                MDT inner conv 1x1, single-conv angular projection.

    ``from_scale`` returns the preset matching the released weights so the
    checkpoints load with no key/shape mismatch. To train a fresh model with a
    custom configuration, instantiate ``SkimLFSRConfig`` directly and override
    the fields below.
    """

    in_channels = 1                 # Input channels (Y of YCbCr)
    feat_channels = 48              # Feature channels
    embed_dim = 96                  # Skim (MDT) self-attention embedding dimension
    spatial_qk_dim = 48             # Query/key dimension of the spatial Skim transformer
    angular_qk_dim = 48             # Query/key dimension of the angular (disentangling) transformer

    # --- Architecture revision flags (see class docstring) ---
    n_block = 20                    # Number of cascaded blocks
    use_cab = True                  # Channel-attention refinement inside MDT and the angular transformer
    mdt_conv2_kernel = 1            # Spatial kernel of the second 3D conv inside MDT (1 or 3)
    dt_conv0_n = 1                  # Number of stacked convs in the angular transformer's first projection

    @classmethod
    def from_scale(cls, scale: int) -> "SkimLFSRConfig":
        if scale not in (2, 4):
            raise ValueError(f"No released SkimLFSR preset for scale {scale}; expected 2 or 4.")
        cfg = cls()
        cfg.n_block = 20
        cfg.use_cab = True
        cfg.mdt_conv2_kernel = 1
        cfg.dt_conv0_n = 1
        return cfg

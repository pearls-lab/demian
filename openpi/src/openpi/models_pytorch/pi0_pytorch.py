import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


# Number of physics target dimensions for the optional auxiliary head:
#   3 cumulative displacement + 3 net rotation + 3 gripper one-hot (none/close/open)
#   + 1 z-bias + 1 curvature + 1 mean speed.
_PHYSICS_DIMS = 12
_PAIR_PHYSICS_DIMS = 8
# Number of action dimensions reconstructed by the optional latent VAE branch.
# Robocasa uses 7 real action dims; the model's internal action_dim is 32 (padded).
_VAE_RECON_DIMS = 7


def _physics_targets_from_actions(actions: torch.Tensor, action_clip: float | None = None) -> torch.Tensor:
    """Compute 12-dim physics features from a batch of action chunks.

    actions: (B, action_horizon, action_dim) — only the first 7 dims are used
    (dx, dy, dz, drx, dry, drz, grip), the rest are zero-padding in robocasa.
    Returns (B, _PHYSICS_DIMS) un-normalized; caller divides by physics_scale.
    """
    a = actions[..., :7]                                            # (B, H, 7)
    if action_clip is not None:
        a = a.clamp(min=-action_clip, max=action_clip)
    disp = a[..., :3].sum(dim=1)                                    # (B, 3)
    rot = a[..., 3:6].sum(dim=1)                                    # (B, 3)
    grip = a[..., 6]                                                # (B, H)
    grip_diff = grip[:, -1] - grip[:, 0]                             # (B,)
    grip_close = (grip_diff > 0.3).to(a.dtype).unsqueeze(-1)
    grip_open = (grip_diff < -0.3).to(a.dtype).unsqueeze(-1)
    grip_none = (grip_diff.abs() <= 0.3).to(a.dtype).unsqueeze(-1)
    grip_onehot = torch.cat([grip_none, grip_close, grip_open], dim=-1)   # (B, 3)
    z_bias = a[..., 2].mean(dim=1, keepdim=True)                    # (B, 1)
    per_step_norm = a[..., :3].norm(dim=-1)                          # (B, H)
    arc = per_step_norm.sum(dim=-1, keepdim=True)                    # (B, 1)
    straight = disp.norm(dim=-1, keepdim=True)                       # (B, 1)
    curvature = straight / (arc + 1e-6)                              # (B, 1)
    mean_speed = per_step_norm.mean(dim=-1, keepdim=True)             # (B, 1)
    return torch.cat([disp, rot, grip_onehot, z_bias, curvature, mean_speed], dim=-1)


def _pair_physics_targets_from_actions(
    actions: torch.Tensor, action_clip: float | None = None, min_gap: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval-pair physics targets from a batch of action chunks.

    Boundary pair (i, j) represents the sub-trajectory actions[:, i:j, :].
    Returns:
      pair_targets: (B, H + 1, H + 1, 8), un-normalized
      pair_mask: (H + 1, H + 1), true for valid intervals j - i >= min_gap
    Target dims are interval displacement xyz, interval rotation rxyz,
    translation path length, and straight-line/path-length curvature.
    """
    a = actions[..., :7]                                             # (B, H, 7)
    if action_clip is not None:
        a = a.clamp(min=-action_clip, max=action_clip)
    bsz, horizon = a.shape[:2]
    if min_gap < 1 or min_gap > horizon:
        raise ValueError(f"pair_physics_min_gap must be in [1, {horizon}], got {min_gap}")
    disp_step = a[..., :3]                                           # (B, H, 3)
    rot_step = a[..., 3:6]                                           # (B, H, 3)
    step_len = disp_step.norm(dim=-1, keepdim=True)                  # (B, H, 1)

    zero_disp = torch.zeros(bsz, 1, 3, dtype=a.dtype, device=a.device)
    zero_len = torch.zeros(bsz, 1, 1, dtype=a.dtype, device=a.device)
    cum_disp = torch.cat([zero_disp, disp_step.cumsum(dim=1)], dim=1) # (B, H+1, 3)
    cum_rot = torch.cat([zero_disp, rot_step.cumsum(dim=1)], dim=1)   # (B, H+1, 3)
    cum_len = torch.cat([zero_len, step_len.cumsum(dim=1)], dim=1)    # (B, H+1, 1)

    pair_disp = cum_disp[:, None, :, :] - cum_disp[:, :, None, :]     # (B, H+1, H+1, 3)
    pair_rot = cum_rot[:, None, :, :] - cum_rot[:, :, None, :]        # (B, H+1, H+1, 3)
    pair_len = cum_len[:, None, :, :] - cum_len[:, :, None, :]        # (B, H+1, H+1, 1)
    straight = pair_disp.norm(dim=-1, keepdim=True)                   # (B, H+1, H+1, 1)
    curvature = straight / (pair_len + 1e-6)                          # (B, H+1, H+1, 1)
    pair_targets = torch.cat([pair_disp, pair_rot, pair_len, curvature], dim=-1)

    pair_mask = torch.triu(
        torch.ones(horizon + 1, horizon + 1, dtype=torch.bool, device=actions.device),
        diagonal=min_gap,
    )
    return pair_targets, pair_mask


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            action_moe_config=config.action_moe,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        # ---- Auxiliary heads for primitive-aware training (Option-D follow-ups) ----
        if config.physics_aux_weight > 0.0:
            d = action_expert_config.width
            h = config.action_horizon
            self.physics_head = nn.Sequential(
                nn.Linear(h * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, _PHYSICS_DIMS),
            )
            # Fixed per-dim scale used to z-score physics targets so each dim
            # contributes ~equally to MSE. Values are robocasa-action-magnitude
            # heuristics; the head learns to fit residuals.
            physics_scale = getattr(config, "physics_aux_target_scale", None)
            if physics_scale is None:
                physics_scale = (
                    0.30, 0.30, 0.20,        # disp xyz
                    0.30, 0.30, 0.30,        # rot rxyz
                    0.50, 0.50, 0.50,        # gripper one-hot 3-way
                    0.20,                    # z-bias
                    0.30,                    # straight-line / arc-length
                    0.20,                    # mean per-step speed
                )
            if len(physics_scale) != _PHYSICS_DIMS:
                raise ValueError(f"physics_aux_target_scale must have {_PHYSICS_DIMS} dims, got {len(physics_scale)}")
            self.register_buffer(
                "physics_scale",
                torch.tensor(physics_scale, dtype=torch.float32),
            )
            # Per-dim mask for ablations. Parses physics_dim_mask config string
            # ("1" = enabled, "0" = zeroed from loss). All 1s by default.
            mask_str = getattr(config, "physics_dim_mask", "111111111111")
            assert len(mask_str) == _PHYSICS_DIMS, f"physics_dim_mask must be {_PHYSICS_DIMS} chars, got {len(mask_str)!r}"
            mask = torch.tensor([1.0 if c == "1" else 0.0 for c in mask_str], dtype=torch.float32)
            self.register_buffer("physics_dim_mask", mask)
        if config.pair_physics_aux_weight > 0.0:
            d = action_expert_config.width
            self.pair_physics_head = nn.Sequential(
                nn.Linear(4 * d, 2 * d),
                nn.GELU(),
                nn.Linear(2 * d, _PAIR_PHYSICS_DIMS),
            )
            self.register_buffer(
                "pair_physics_scale",
                torch.tensor(
                    [0.30, 0.30, 0.20,        # interval disp xyz
                     0.30, 0.30, 0.30,        # interval rot rxyz
                     0.30,                    # interval path length
                     0.30],                   # straight-line / path length
                    dtype=torch.float32,
                ),
            )
        if config.latent_vae_dim > 0:
            d = action_expert_config.width
            d_z = config.latent_vae_dim
            self.vae_mu = nn.Linear(d, d_z)
            self.vae_logvar = nn.Linear(d, d_z)
            self.vae_decoder = nn.Sequential(
                nn.Linear(d_z, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, config.action_horizon * _VAE_RECON_DIMS),
            )
        if config.next_state_aux_weight > 0.0:
            d = action_expert_config.width
            h = config.action_horizon
            self.next_state_head = nn.Sequential(
                nn.Linear(h * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, config.next_state_aux_dim),
            )

        # I1b future-state anchor head (ported from hierarchy/openpi-navfilt). Off the
        # masked-mean-pooled suffix (action-token) latents -> cumulative EEF-pos
        # displacement at n_horizons. Built only when the weight > 0 so weight=0
        # configs are byte-identical to the released DeMiAn-VLA graph.
        self.use_anchor_aux = getattr(config, "anchor_aux_weight", 0.0) > 0.0
        if self.use_anchor_aux:
            self._n_anchor = len(config.anchor_horizons_frames)
            d = action_expert_config.width
            self.anchor_head = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.SiLU(),
                nn.Linear(4 * d, self._n_anchor * 3),
            )
            # Non-persistent z-score buffers (ride device/dtype, absent from ckpts).
            _tm = torch.tensor(config.anchor_target_mean or [0.0] * (self._n_anchor * 3), dtype=torch.float32)
            _ts = torch.tensor(config.anchor_target_std or [1.0] * (self._n_anchor * 3), dtype=torch.float32)
            self.register_buffer("_anchor_tmean", _tm.reshape(self._n_anchor, 3), persistent=False)
            self.register_buffer("_anchor_tstd", _ts.reshape(self._n_anchor, 3).clamp(min=1e-6), persistent=False)

        if config.latent_aux_weight > 0.0 and config.latent_aux_vae_path:
            from openpi.models.motion_physics_vae import MotionPhysicsVAE

            d = action_expert_config.width
            h = config.action_horizon
            self.latent_aux_head = nn.Sequential(
                nn.Linear(h * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, config.latent_aux_dim),
            )
            self.motion_physics_vae = MotionPhysicsVAE(
                state_dim=config.latent_aux_state_dim,
                action_dim=config.latent_aux_action_dim,
                horizon=config.action_horizon,
                latent_dim=config.latent_aux_dim,
                hidden=config.latent_aux_vae_hidden,
            )
            try:
                state = torch.load(config.latent_aux_vae_path, map_location="cpu",
                                   weights_only=True)
            except TypeError:
                # Older torch: no weights_only kwarg.
                state = torch.load(config.latent_aux_vae_path, map_location="cpu")
            missing, unexpected = self.motion_physics_vae.load_state_dict(state, strict=False)
            if missing or unexpected:
                logging.warning(
                    f"MotionPhysicsVAE state_dict load: missing={missing} unexpected={unexpected}"
                )
            for p in self.motion_physics_vae.parameters():
                p.requires_grad = False
            self.motion_physics_vae.eval()

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    @property
    def has_action_moe(self) -> bool:
        return self.config.action_moe is not None

    @torch.no_grad()
    def initialize_moe_experts_from_dense(self):
        self.paligemma_with_expert.initialize_moe_experts_from_dense()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.reasoning_token_mask,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _compute_lm_loss(self, prefix_out, lang_tokens, lang_masks):
        """Compute autoregressive LM loss on prefix language tokens."""
        num_lang_tokens = lang_tokens.shape[1]
        lang_hidden = prefix_out[:, -num_lang_tokens:]

        lang_hidden_for_lm = lang_hidden.to(dtype=self.paligemma_with_expert.paligemma.lm_head.weight.dtype)
        logits = self.paligemma_with_expert.paligemma.lm_head(lang_hidden_for_lm)

        shift_logits = logits[:, :-1].contiguous()
        shift_targets = lang_tokens[:, 1:].long()
        shift_mask = lang_masks[:, 1:].float()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            reduction="none",
        )
        return (lm_loss * shift_mask.reshape(-1)).sum() / shift_mask.sum().clamp(min=1)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, reasoning_mask = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Prefix-LM attention: when causal_reasoning=True, reasoning tokens use
        # causal attention (att_mask=1) so the LM loss is autoregressive.
        # When causal_reasoning=False, reasoning tokens stay bidirectional (att_mask=0).
        if reasoning_mask is not None and self.config.lm_loss_weight > 0 and self.config.causal_reasoning:
            num_img_tokens = prefix_embs.shape[1] - lang_tokens.shape[1]
            reasoning_in_prefix = torch.cat([
                torch.zeros(reasoning_mask.shape[0], num_img_tokens, dtype=torch.bool, device=reasoning_mask.device),
                reasoning_mask.bool(),
            ], dim=1)  # (B, prefix_len)
            prefix_att_masks = prefix_att_masks | reasoning_in_prefix

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        flow_loss = F.mse_loss(u_t, v_t, reduction="none")
        flow_loss_mean = flow_loss.mean()

        # ---- I1a: derivative (Sobolev) aux on the implied clean action ----
        # a_hat = x_t - t*v_t is the model's implied clean action from this same
        # forward pass (no extra params, no inference change). Match its 1st/2nd
        # finite differences along the horizon to the GT chunk, on the EEF slice.
        # Time-weighted by (1-t): a_hat is low-variance near t->0 and noisy near
        # t->1, so weighting keeps the mean-seeking pressure benign on the flow
        # head (phase4-i1-integration.md §3.1, §7 risk-1).
        deriv_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        if getattr(self.config, "deriv_aux_weight", 0.0) > 0.0:
            a_hat = x_t - time_expanded * v_t                              # (B,H,ad)
            lo, hi = self.config.deriv_action_slice
            yh, yt = a_hat[..., lo:hi], actions[..., lo:hi]
            w = (1.0 - time).clamp(min=0.0)                                # (B,)
            l1 = ((torch.diff(yh, dim=1) - torch.diff(yt, dim=1)) ** 2).mean(dim=(1, 2))
            l2 = ((torch.diff(yh, n=2, dim=1) - torch.diff(yt, n=2, dim=1)) ** 2).mean(dim=(1, 2))
            deriv_loss = (w * (self.config.deriv_lv * l1 + self.config.deriv_la * l2)).mean()
            deriv_loss_weighted = self.config.deriv_aux_weight * deriv_loss
        moe_losses = self.paligemma_with_expert.moe_aux_loss()
        moe_aux_loss = (
            moe_losses["moe_aux_loss"]
            if moe_losses is not None
            else torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        )

        # ---- Auxiliary heads (Option-D follow-ups) ----
        # suffix_out here is (B, action_horizon, action_expert_width) — the action
        # expert's per-timestep hidden states, same tensor that feeds action_out_proj.
        aux_extra: dict[str, torch.Tensor] = {}
        if getattr(self.config, "deriv_aux_weight", 0.0) > 0.0:
            aux_extra["deriv_loss"] = deriv_loss
        physics_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        pair_physics_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        vae_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        physics_actions = actions
        if (
            (self.config.physics_aux_weight > 0.0 or self.config.pair_physics_aux_weight > 0.0)
            and getattr(self.config, "physics_aux_use_raw_actions", False)
        ):
            physics_actions = getattr(observation, "actions_raw", None)
            if physics_actions is None:
                raise RuntimeError("physics_aux_use_raw_actions=True but observation.actions_raw is missing")
            physics_actions = physics_actions.to(device=actions.device, dtype=actions.dtype)

        if self.config.physics_aux_weight > 0.0:
            B = suffix_out.shape[0]
            phys_in = suffix_out.reshape(B, -1)                                # (B, H * d)
            phys_pred = self.physics_head(phys_in)                              # (B, 12)
            phys_target_raw = _physics_targets_from_actions(
                physics_actions,
                action_clip=self.config.physics_aux_action_clip,
            )                                                                  # (B, 12)
            phys_target = phys_target_raw / self.physics_scale.unsqueeze(0)
            phys_pred_n = phys_pred / self.physics_scale.unsqueeze(0)
            # Per-dim mask zeros residuals for ablated dims, so they contribute 0 to the
            # MSE and the head receives no gradient on those dims. Active dims still get
            # the full per-element MSE contribution (we average over all 12 dims, then
            # over batch, to keep the loss scale stable across ablations).
            mask = self.physics_dim_mask.to(phys_pred.dtype).to(phys_pred.device)  # (12,)
            residual = (phys_pred_n - phys_target) * mask.unsqueeze(0)
            physics_loss = (residual ** 2).mean()
            physics_loss_weighted = self.config.physics_aux_weight * physics_loss
            aux_extra["physics_loss"] = physics_loss

        if self.config.pair_physics_aux_weight > 0.0:
            # Build pair features over trajectory boundary indices 0..H. Start
            # boundary i uses hidden min(i, H-1); end boundary j uses hidden j-1.
            start_h = torch.cat([suffix_out, suffix_out[:, -1:, :]], dim=1)      # (B, H+1, d)
            end_h = torch.cat([suffix_out[:, :1, :], suffix_out], dim=1)         # (B, H+1, d)
            h_i = start_h[:, :, None, :]                                        # (B, H+1, 1, d)
            h_j = end_h[:, None, :, :]                                          # (B, 1, H+1, d)
            h_i = h_i.expand(-1, start_h.shape[1], end_h.shape[1], -1)           # (B, H+1, H+1, d)
            h_j = h_j.expand(-1, start_h.shape[1], end_h.shape[1], -1)           # (B, H+1, H+1, d)
            pair_feat = torch.cat([h_i, h_j, h_j - h_i, h_i * h_j], dim=-1)
            pair_pred = self.pair_physics_head(pair_feat)                       # (B, H+1, H+1, 8)
            pair_target_raw, pair_mask = _pair_physics_targets_from_actions(
                physics_actions,
                action_clip=self.config.physics_aux_action_clip,
                min_gap=self.config.pair_physics_min_gap,
            )
            pair_target_raw = pair_target_raw.to(pair_pred.dtype)
            pair_scale = self.pair_physics_scale.to(pair_pred.dtype).to(pair_pred.device).view(1, 1, 1, -1)
            pair_pred_n = pair_pred / pair_scale
            pair_target_n = pair_target_raw / pair_scale
            pair_mask_f = pair_mask.to(pair_pred.dtype).to(pair_pred.device).unsqueeze(0).unsqueeze(-1)
            pair_residual = (pair_pred_n - pair_target_n) * pair_mask_f
            pair_denom = pair_mask_f.sum().clamp(min=1.0) * pair_pred.shape[0] * pair_pred.shape[-1]
            pair_physics_loss = (pair_residual ** 2).sum() / pair_denom
            pair_physics_loss_weighted = self.config.pair_physics_aux_weight * pair_physics_loss
            aux_extra["pair_physics_loss"] = pair_physics_loss

        if self.config.latent_vae_dim > 0:
            pooled = suffix_out.mean(dim=1)                                     # (B, d)
            mu = self.vae_mu(pooled)
            logvar = self.vae_logvar(pooled).clamp(-6.0, 4.0)
            if self.training:
                eps = torch.randn_like(mu)
                z = mu + eps * (0.5 * logvar).exp()
            else:
                z = mu
            recon = self.vae_decoder(z).view(z.shape[0], self.config.action_horizon, _VAE_RECON_DIMS)
            recon_target = actions[..., :_VAE_RECON_DIMS].to(recon.dtype)
            vae_recon = F.mse_loss(recon, recon_target)
            vae_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            vae_loss_weighted = (
                self.config.latent_vae_recon_weight * vae_recon
                + self.config.latent_vae_kl_weight * vae_kl
            )
            aux_extra["vae_recon"] = vae_recon
            aux_extra["vae_kl"] = vae_kl

        latent_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        if (
            self.config.latent_aux_weight > 0.0
            and self.config.latent_aux_vae_path
            and hasattr(self, "latent_aux_head")
        ):
            B = suffix_out.shape[0]
            la_in = suffix_out.reshape(B, -1)                                   # (B, H*d)
            z_pred = self.latent_aux_head(la_in)                                # (B, latent_dim)
            # Compute the target with the frozen VAE in fp32 for stability.
            with torch.no_grad():
                vae_state = state.to(dtype=torch.float32)
                vae_state = vae_state[..., : self.config.latent_aux_state_dim]
                vae_actions = actions.to(dtype=torch.float32)
                vae_actions = vae_actions[..., : self.config.latent_aux_action_dim]
                z_target = self.motion_physics_vae.encode_mean(vae_state, vae_actions)
            z_target = z_target.to(z_pred.dtype)
            per_sample = ((z_pred - z_target) ** 2).mean(dim=-1)                # (B,)
            ns_mask_la = getattr(observation, "next_state_mask", None)
            if ns_mask_la is not None:
                mask_f = ns_mask_la.to(per_sample.dtype)
                denom = mask_f.sum().clamp(min=1.0)
                latent_loss = (per_sample * mask_f).sum() / denom
            else:
                latent_loss = per_sample.mean()
            latent_loss_weighted = self.config.latent_aux_weight * latent_loss
            aux_extra["latent_loss"] = latent_loss

        next_state_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        if self.config.next_state_aux_weight > 0.0 and getattr(observation, "next_state", None) is not None:
            B = suffix_out.shape[0]
            ns_in = suffix_out.reshape(B, -1)                                   # (B, H * d)
            ns_pred = self.next_state_head(ns_in)                                # (B, next_state_aux_dim)
            ns_target = observation.next_state.to(ns_pred.dtype)                 # (B, state_dim) pre-normalized
            # Truncate/match dims defensively in case the dataset's state has a
            # different width than config.next_state_aux_dim.
            d_target = min(ns_pred.shape[-1], ns_target.shape[-1])
            ns_pred_c = ns_pred[..., :d_target]
            ns_target_c = ns_target[..., :d_target]
            per_sample = ((ns_pred_c - ns_target_c) ** 2).mean(dim=-1)           # (B,)
            ns_mask = getattr(observation, "next_state_mask", None)
            if ns_mask is not None:
                ns_mask_f = ns_mask.to(per_sample.dtype)
                denom = ns_mask_f.sum().clamp(min=1.0)
                next_state_loss = (per_sample * ns_mask_f).sum() / denom
            else:
                next_state_loss = per_sample.mean()
            next_state_loss_weighted = self.config.next_state_aux_weight * next_state_loss
            aux_extra["next_state_loss"] = next_state_loss

        # --- I1b future-state anchor head ---
        # Off the masked-mean-pooled suffix (action-token) latents -> cumulative
        # EEF-pos displacement at n_horizons. NOT fed to the action head; never read
        # at inference. Target delivered by the loader as observation.ee_physics_target
        # = [B, n_anchor, 4] (3 raw cumulative-disp + 1 valid mask); z-scored here with
        # the registered per-(horizon, dim) buffers so horizons contribute comparably.
        anchor_loss_weighted = torch.zeros((), device=flow_loss_mean.device, dtype=flow_loss_mean.dtype)
        if self.use_anchor_aux:
            ee_target = getattr(observation, "ee_physics_target", None)
            if ee_target is None:
                raise ValueError("anchor_aux_weight>0 but observation.ee_physics_target is missing")
            pooled_anchor = suffix_out.mean(dim=1)                                # (B, d)
            anchor_pred = self.anchor_head(pooled_anchor).reshape(-1, self._n_anchor, 3)  # (B, n, 3)
            tgt = ee_target.reshape(-1, self._n_anchor, 4).to(anchor_pred.dtype)
            target = (tgt[..., :3] - self._anchor_tmean.to(anchor_pred.dtype)) / self._anchor_tstd.to(anchor_pred.dtype)
            mask = tgt[..., 3:4]                                                  # (B, n, 1)
            per = (anchor_pred - target) ** 2 * mask
            anchor_loss = per.sum() / mask.sum().clamp(min=1.0)
            anchor_loss_weighted = self.config.anchor_aux_weight * anchor_loss
            aux_extra["anchor_loss"] = anchor_loss

        # Compute LM loss on prefix language tokens if enabled.
        if self.config.lm_loss_weight > 0 and lang_tokens is not None:
            lm_loss = self._compute_lm_loss(prefix_out, lang_tokens, lang_masks)

            # LM-only mode: return only the LM loss, ignore flow matching loss.
            if self.config.lm_only:
                return {
                    "loss": lm_loss,
                    "lm_loss": lm_loss,
                }

            losses = {
                "loss": (
                    flow_loss_mean
                    + self.config.lm_loss_weight * lm_loss
                    + moe_aux_loss
                    + physics_loss_weighted
                    + pair_physics_loss_weighted
                    + vae_loss_weighted
                    + next_state_loss_weighted
                    + latent_loss_weighted
                    + deriv_loss_weighted
                    + anchor_loss_weighted
                ),
                "flow_loss": flow_loss_mean,
                "lm_loss": lm_loss,
            }
            if moe_losses is not None:
                losses.update(moe_losses)
            losses.update(aux_extra)
            return losses

        if moe_losses is not None or aux_extra:
            return {
                "loss": (
                    flow_loss_mean
                    + moe_aux_loss
                    + physics_loss_weighted
                    + pair_physics_loss_weighted
                    + vae_loss_weighted
                    + next_state_loss_weighted
                    + latent_loss_weighted
                    + deriv_loss_weighted
                    + anchor_loss_weighted
                ),
                "flow_loss": flow_loss_mean,
                **(moe_losses or {}),
                **aux_extra,
            }
        return flow_loss

    @torch.no_grad()
    def generate_text_from_prefix(self, prefix_out, lang_tokens, lang_masks) -> list[str]:
        """Decode PaliGemma's prefix hidden states into text via lm_head.

        Returns a list of decoded strings (one per batch element).
        Useful for visualizing what the LM-trained model "thinks" at inference.
        """
        import sentencepiece
        import openpi.shared.download as download

        num_lang_tokens = lang_tokens.shape[1]
        lang_hidden = prefix_out[:, -num_lang_tokens:]
        lang_hidden = lang_hidden.to(dtype=self.paligemma_with_expert.paligemma.lm_head.weight.dtype)
        logits = self.paligemma_with_expert.paligemma.lm_head(lang_hidden)

        # Greedy decode: take argmax at each position
        pred_tokens = logits.argmax(dim=-1)  # (B, lang_len)

        # Mask out padding positions
        pred_tokens = pred_tokens * lang_masks.long()

        # Decode with PaliGemma's sentencepiece tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        results = []
        for i in range(pred_tokens.shape[0]):
            valid_len = lang_masks[i].sum().item()
            token_ids = pred_tokens[i, :valid_len].cpu().tolist()
            results.append(sp.decode(token_ids))
        return results

    @torch.no_grad()
    def generate_reasoning_autoregressively(
        self,
        images: list,
        img_masks: list,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        device,
        max_reasoning_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Generate reasoning tokens autoregressively using lm_head.

        Given task-only tokens, greedily generates a reasoning continuation by
        running lm_head at the last valid token position on each step.

        Returns (extended_lang_tokens, extended_lang_masks, reasoning_token_mask, reasoning_text).
        """
        import sentencepiece
        import openpi.shared.download as download

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        newline_id = sp.encode("\n")[0]
        B = lang_tokens.shape[0]
        max_len = lang_tokens.shape[1]

        # Build token list starting from valid task tokens + " Reasoning: " cue.
        # The cue matches the training format so the model knows to generate reasoning.
        task_valid_len = int(lang_masks[0].sum().item())
        tokens_list = lang_tokens[0, :task_valid_len].cpu().tolist()
        reasoning_cue_tokens = sp.encode(" Reasoning: ")
        tokens_list.extend(reasoning_cue_tokens)
        task_valid_len += len(reasoning_cue_tokens)

        # Cap at the remaining context window. Callers wanting a tighter cap pass
        # max_reasoning_tokens explicitly. Previously this also honored a
        # MAX_REASONING_TOKENS env var, which was removed to prevent the silent
        # slowdown that happened when the env var was missing.
        if max_reasoning_tokens is None:
            max_reasoning_tokens = max_len - task_valid_len

        # Cache image embeddings — computed once, reused on every generation step.
        img_embs = []
        for img, img_mask in zip(images, img_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_embs.append((img_emb, img_mask))

        generated_ids = []
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        for _ in range(max_reasoning_tokens):
            curr_len = len(tokens_list)
            if curr_len >= max_len:
                break

            # Build lang tensor for current sequence.
            pad = [0] * (max_len - curr_len)
            curr_lang_tokens = torch.tensor(
                tokens_list + pad, dtype=lang_tokens.dtype, device=device
            ).unsqueeze(0).expand(B, -1)
            curr_lang_masks = torch.tensor(
                [True] * curr_len + [False] * (max_len - curr_len), dtype=torch.bool, device=device
            ).unsqueeze(0).expand(B, -1)

            # Embed: reuse cached image embeddings + embed current lang tokens.
            embs, pad_masks, att_masks_list = [], [], []
            for img_emb, img_mask in img_embs:
                num_img = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(B, num_img))
                att_masks_list += [0] * num_img

            lang_emb = self.paligemma_with_expert.embed_language_tokens(curr_lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            embs.append(lang_emb)
            pad_masks.append(curr_lang_masks)
            # Prefix-LM attention: task tokens bidirectional (0), reasoning tokens causal (1).
            num_reasoning_so_far = curr_len - task_valid_len
            att_masks_list += [0] * task_valid_len + [1] * num_reasoning_so_far + [0] * (max_len - curr_len)

            prefix_embs = torch.cat(embs, dim=1)
            prefix_pad_masks = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks_list, dtype=torch.bool, device=device)[None].expand(B, -1)

            att_2d = make_att_2d_masks(prefix_pad_masks, att_masks_tensor)
            pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(att_2d)

            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )

            # Predict next token: lm_head at last valid position.
            # Lang tokens are appended after image tokens; curr_len is the number of valid lang tokens.
            num_img_tokens = prefix_embs.shape[1] - max_len
            last_valid_pos = num_img_tokens + curr_len - 1
            last_hidden = prefix_out[:, last_valid_pos, :]
            last_hidden = last_hidden.to(self.paligemma_with_expert.paligemma.lm_head.weight.dtype)
            next_token = self.paligemma_with_expert.paligemma.lm_head(last_hidden)[0].argmax().item()

            if next_token in (newline_id, 0):
                break
            generated_ids.append(next_token)
            tokens_list.append(next_token)

        reasoning_text = sp.decode(generated_ids) if generated_ids else ""
        logging.info(f"[Two-step] Generated reasoning ({len(generated_ids)} tokens): {reasoning_text[:120]}")

        # Build final extended token tensors.
        new_len = len(tokens_list)
        num_reasoning = len(generated_ids)
        pad = [0] * (max_len - new_len)
        ext_tokens = torch.tensor(
            tokens_list + pad, dtype=lang_tokens.dtype, device=device
        ).unsqueeze(0).expand(B, -1)
        ext_masks = torch.tensor(
            [True] * new_len + [False] * (max_len - new_len), dtype=torch.bool, device=device
        ).unsqueeze(0).expand(B, -1)
        # Reasoning mask: True for generated reasoning positions, False for task + padding.
        reasoning_token_mask = torch.tensor(
            [False] * task_valid_len + [True] * num_reasoning + [False] * (max_len - new_len),
            dtype=torch.bool, device=device,
        ).unsqueeze(0).expand(B, -1)
        return ext_tokens, ext_masks, reasoning_token_mask, reasoning_text

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, _reasoning_mask = self._preprocess_observation(observation, train=False)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # sample_actions never auto-triggers autoregressive reasoning generation —
        # it's too slow (~200 forward passes per action chunk on MoE) and the train
        # /inference discrepancy is closed by callers injecting reasoning text into
        # the prompt (e.g. via the policy server's gt_reasoning file). Callers that
        # actually want generated reasoning should use Policy.generate_text(), which
        # invokes generate_reasoning_autoregressively explicitly.
        inference_reasoning_mask = None
        self._last_generated_reasoning = ""

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Prefix-LM attention: causal for generated reasoning positions (only when causal_reasoning=True).
        if inference_reasoning_mask is not None and inference_reasoning_mask.any() and self.config.causal_reasoning:
            num_img_tokens = prefix_embs.shape[1] - lang_tokens.shape[1]
            reasoning_in_prefix = torch.cat([
                torch.zeros(bsize, num_img_tokens, dtype=torch.bool, device=lang_tokens.device),
                inference_reasoning_mask,
            ], dim=1)
            prefix_att_masks = prefix_att_masks | reasoning_in_prefix

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute prefix KV cache (now with reasoning tokens included).
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Cache prefix data for optional text generation after inference.
        self._cached_prefix_out = prefix_out.detach()
        self._cached_lang_tokens = lang_tokens
        self._cached_lang_masks = lang_masks

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility modules and helper functions
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """A small reusable MLP block.

    This helper keeps the file self-contained and avoids repeating the same
    Linear -> Activation -> Dropout pattern across modules.

    Args:
        in_dim: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        out_dim: Output feature dimension.
        dropout: Dropout probability applied after each hidden layer.
        activation: Activation module constructor, e.g. nn.ReLU.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...] | list[int],
        out_dim: int,
        dropout: float = 0.1,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.out_proj = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.out_proj(x)


class IdentityModule(nn.Module):
    """Simple identity module, useful as a placeholder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute categorical entropy for batched probability vectors.

    Args:
        probs: Tensor of shape [B, K].
        eps: Numerical stability constant.

    Returns:
        Tensor of shape [B, 1].
    """

    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1, keepdim=True)



def normalize_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize batched probability vectors so rows sum to 1."""

    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


# -----------------------------------------------------------------------------
# Step-level data containers
# -----------------------------------------------------------------------------


@dataclass
class StepBatch:
    """Single negotiation step inputs for the switch-aware explicit ToM model.

    Required seller-side observations:
        seller_act_id:        [B]
        seller_price_bin:     [B]
        seller_num_feats:     [B, F_num]  (default F_num = 11)
        seller_text_emb:      [B, D_text_in]

    Additional buyer/context features used by the shared policy core:
        hist_state:           [B, D_hist]
        scenario_ctx:         [B, D_ctx]
        buyer_state:          [B, D_buyer]

    Optional previous seller numeric features for change detection:
        prev_seller_num_feats:[B, F_num]
    """

    seller_act_id: torch.LongTensor
    seller_price_bin: torch.LongTensor
    seller_num_feats: torch.FloatTensor
    seller_text_emb: torch.FloatTensor
    hist_state: torch.FloatTensor
    scenario_ctx: torch.FloatTensor
    buyer_state: torch.FloatTensor
    prev_seller_num_feats: Optional[torch.FloatTensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.seller_act_id.size(0))

    @property
    def device(self) -> torch.device:
        return self.seller_num_feats.device

    def validate(self, expected_f_num: Optional[int] = None) -> None:
        """Basic runtime shape validation.

        The method is intentionally light-weight: it catches the most common
        silent bugs when wiring new modules into an existing trainer.
        """

        bsz = self.batch_size
        assert self.seller_price_bin.size(0) == bsz, "seller_price_bin batch mismatch"
        assert self.seller_num_feats.size(0) == bsz, "seller_num_feats batch mismatch"
        assert self.seller_text_emb.size(0) == bsz, "seller_text_emb batch mismatch"
        assert self.hist_state.size(0) == bsz, "hist_state batch mismatch"
        assert self.scenario_ctx.size(0) == bsz, "scenario_ctx batch mismatch"
        assert self.buyer_state.size(0) == bsz, "buyer_state batch mismatch"
        if expected_f_num is not None:
            assert self.seller_num_feats.size(-1) == expected_f_num, (
                f"seller_num_feats last dim mismatch: "
                f"got {self.seller_num_feats.size(-1)}, expected {expected_f_num}"
            )
            if self.prev_seller_num_feats is not None:
                assert self.prev_seller_num_feats.size(-1) == expected_f_num, (
                    f"prev_seller_num_feats last dim mismatch: "
                    f"got {self.prev_seller_num_feats.size(-1)}, expected {expected_f_num}"
                )


@dataclass
class BeliefState:
    """Explicit seller belief state.

    Fields:
        type_probs:  Posterior over seller types, shape [B, K_type].
        cont_mu:     Continuous latent state mean, shape [B, D_b_cont].
        cont_logvar: Continuous latent state log-variance, shape [B, D_b_cont].
        confidence:  Belief confidence score, shape [B, 1].
    """

    type_probs: torch.FloatTensor
    cont_mu: torch.FloatTensor
    cont_logvar: torch.FloatTensor
    confidence: torch.FloatTensor

    def vector(self) -> torch.Tensor:
        """Flatten the structured belief into a single vector per example."""

        return torch.cat(
            [self.type_probs, self.cont_mu, self.cont_logvar, self.confidence],
            dim=-1,
        )

    def detach(self) -> "BeliefState":
        return BeliefState(
            type_probs=self.type_probs.detach(),
            cont_mu=self.cont_mu.detach(),
            cont_logvar=self.cont_logvar.detach(),
            confidence=self.confidence.detach(),
        )

    @classmethod
    def init_uniform(
        cls,
        batch_size: int,
        k_type: int,
        d_b_cont: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        init_confidence: float = 0.0,
    ) -> "BeliefState":
        """Create an initial belief state.

        Default behavior:
            - uniform discrete type posterior,
            - zero continuous state mean,
            - zero log-variance,
            - low confidence.
        """

        type_probs = torch.full(
            (batch_size, k_type),
            fill_value=1.0 / float(k_type),
            device=device,
            dtype=dtype,
        )
        cont_mu = torch.zeros(batch_size, d_b_cont, device=device, dtype=dtype)
        cont_logvar = torch.zeros(batch_size, d_b_cont, device=device, dtype=dtype)
        confidence = torch.full(
            (batch_size, 1), init_confidence, device=device, dtype=dtype
        )
        return cls(type_probs, cont_mu, cont_logvar, confidence)


# -----------------------------------------------------------------------------
# 1) Seller observation encoder
# -----------------------------------------------------------------------------


class SellerObservationEncoder(nn.Module):
    """Fuse structured seller signals and text embedding into a unified vector.

    Data flow at this stage:
        structured seller features + seller utterance embedding +
        history/context -> seller observation vector o_t^S.
    """

    def __init__(
        self,
        n_acts: int,
        n_price_bins: int,
        d_act: int = 32,
        d_price_emb: int = 16,
        f_num: int = 11,
        d_text_in: int = 768,
        d_hist: int = 128,
        d_ctx: int = 128,
        d_obs: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.f_num = f_num
        self.d_obs = d_obs

        self.act_emb = nn.Embedding(n_acts, d_act)
        self.price_emb = nn.Embedding(n_price_bins, d_price_emb)

        self.num_proj = MLP(f_num, [64], 64, dropout=dropout)
        self.text_proj = MLP(d_text_in, [256], 128, dropout=dropout)
        self.hist_proj = MLP(d_hist, [128], 64, dropout=dropout)
        self.ctx_proj = MLP(d_ctx, [128], 64, dropout=dropout)

        fusion_dim = d_act + d_price_emb + 64 + 128 + 64 + 64
        self.fuse = MLP(fusion_dim, [256], d_obs, dropout=dropout)

    def forward(
        self,
        seller_act_id: torch.LongTensor,
        seller_price_bin: torch.LongTensor,
        seller_num_feats: torch.FloatTensor,
        seller_text_emb: torch.FloatTensor,
        hist_state: torch.FloatTensor,
        scenario_ctx: torch.FloatTensor,
    ) -> torch.FloatTensor:
        act_e = self.act_emb(seller_act_id)           # [B, d_act]
        price_e = self.price_emb(seller_price_bin)    # [B, d_price_emb]
        num_e = self.num_proj(seller_num_feats)       # [B, 64]
        text_e = self.text_proj(seller_text_emb)      # [B, 128]
        hist_e = self.hist_proj(hist_state)           # [B, 64]
        ctx_e = self.ctx_proj(scenario_ctx)           # [B, 64]

        fused = torch.cat([act_e, price_e, num_e, text_e, hist_e, ctx_e], dim=-1)
        obs_t = self.fuse(fused)                      # [B, D_obs]
        return obs_t


# -----------------------------------------------------------------------------
# 2) Seller GRU encoder
# -----------------------------------------------------------------------------


class SellerGRUEncoder(nn.Module):
    """Track local seller behavior trajectory with a GRUCell."""

    def __init__(self, d_obs: int = 256, d_h: int = 256) -> None:
        super().__init__()
        self.d_h = d_h
        self.cell = nn.GRUCell(d_obs, d_h)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_h, device=device, dtype=dtype)

    def forward(self, obs_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.cell(obs_t, h_prev)


# -----------------------------------------------------------------------------
# 3) Binary switch-aware change detector
# -----------------------------------------------------------------------------


class BinaryChangeDetector(nn.Module):
    """Predict whether the seller switched negotiation regime.

    The detector uses:
        - h_t and h_{t-1},
        - hidden-state difference,
        - current seller numeric features,
        - numeric feature drift.
    """

    def __init__(self, d_h: int = 256, f_num: int = 11, dropout: float = 0.1) -> None:
        super().__init__()
        self.f_num = f_num
        in_dim = d_h * 3 + f_num * 2
        self.mlp = MLP(in_dim, [128, 64], 1, dropout=dropout)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_t: torch.Tensor,
        seller_num_feats: torch.Tensor,
        prev_num_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_num_feats is None:
            prev_num_feats = torch.zeros_like(seller_num_feats)

        delta_h = h_t - h_prev
        delta_num = seller_num_feats - prev_num_feats
        x = torch.cat([h_prev, h_t, delta_h, seller_num_feats, delta_num], dim=-1)

        switch_logit = self.mlp(x)            # [B, 1]
        switch_prob = torch.sigmoid(switch_logit)
        return switch_logit, switch_prob


# -----------------------------------------------------------------------------
# 4) Explicit belief updater with switch-aware gating
# -----------------------------------------------------------------------------


class ExplicitBeliefUpdater(nn.Module):
    """Update explicit seller belief state using switch-aware gated fusion.

    Key design goal:
        When switch probability is high, the updater should absorb new evidence
        more aggressively. When switch probability is low, the updater should
        smooth beliefs over time.
    """

    def __init__(
        self,
        d_h: int = 256,
        d_hist: int = 128,
        d_ctx: int = 128,
        k_type: int = 4,
        d_b_cont: int = 16,
        dropout: float = 0.1,
        min_alpha: float = 0.05,
        max_alpha: float = 0.95,
        switch_mix_weight: float = 0.75,
        # 新增：forced 模式下更强的覆盖
        hard_force_alpha: bool = True,
        force_off_alpha: float = 0.05,
        force_on_alpha: float = 0.95,
        intervention_delta_gain: float = 1.50,
    ) -> None:
        super().__init__()
        self.k_type = k_type
        self.d_b_cont = d_b_cont
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.switch_mix_weight = switch_mix_weight

        self.hard_force_alpha = hard_force_alpha
        self.force_off_alpha = force_off_alpha
        self.force_on_alpha = force_on_alpha
        self.intervention_delta_gain = intervention_delta_gain

        in_dim = d_h + d_hist + d_ctx
        self.type_head = MLP(in_dim, [128], k_type, dropout=dropout)
        self.cont_mu_head = MLP(in_dim, [128], d_b_cont, dropout=dropout)
        self.cont_logvar_head = MLP(in_dim, [128], d_b_cont, dropout=dropout)

        d_belief = k_type + d_b_cont + d_b_cont + 1
        self.gate = MLP(d_h + 1 + d_belief, [128], 1, dropout=dropout)

    @property
    def d_belief(self) -> int:
        return self.k_type + self.d_b_cont + self.d_b_cont + 1

    def forward(
        self,
        h_t: torch.Tensor,
        hist_state: torch.Tensor,
        scenario_ctx: torch.Tensor,
        switch_prob: torch.Tensor,
        belief_prev: BeliefState,
        force_switch_prob: Optional[torch.Tensor] = None,
    ) -> Tuple[BeliefState, torch.Tensor]:
        x = torch.cat([h_t, hist_state, scenario_ctx], dim=-1)

        # Candidate belief inferred from current evidence.
        cand_type_logits = self.type_head(x)
        cand_type_probs = F.softmax(cand_type_logits, dim=-1)
        cand_cont_mu = self.cont_mu_head(x)
        cand_cont_logvar = self.cont_logvar_head(x)

        # Learned residual gate.
        gate_in = torch.cat([h_t, switch_prob, belief_prev.vector()], dim=-1)
        learned_alpha = torch.sigmoid(self.gate(gate_in))

        # 默认：原来的 switch-aware mixing
        mixed_alpha = (
            self.switch_mix_weight * switch_prob
            + (1.0 - self.switch_mix_weight) * learned_alpha
        )
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * mixed_alpha
        alpha = alpha.clamp(self.min_alpha, self.max_alpha)   # [B, 1]

        # 新增：如果是强制干预，直接用硬 alpha，而不是继续混 learned_alpha
        if force_switch_prob is not None and self.hard_force_alpha:
            forced = force_switch_prob.reshape(-1, 1).to(alpha.dtype)
            alpha_off = torch.full_like(alpha, self.force_off_alpha)
            alpha_on = torch.full_like(alpha, self.force_on_alpha)
            alpha = torch.where(forced >= 0.5, alpha_on, alpha_off)

        # Smooth posterior update.
        type_probs = (1.0 - alpha) * belief_prev.type_probs + alpha * cand_type_probs
        type_probs = normalize_probs(type_probs)

        cont_mu = (1.0 - alpha) * belief_prev.cont_mu + alpha * cand_cont_mu
        cont_logvar = (
            (1.0 - alpha) * belief_prev.cont_logvar + alpha * cand_cont_logvar
        )

        # 新增：forced 模式下把 belief 差异再放大一点，避免被后续 FiLM / argmax 淹没
        if force_switch_prob is not None and self.intervention_delta_gain > 1.0:
            gain = float(self.intervention_delta_gain)

            type_probs = belief_prev.type_probs + gain * (type_probs - belief_prev.type_probs)
            type_probs = normalize_probs(type_probs.clamp_min(1e-8))

            cont_mu = belief_prev.cont_mu + gain * (cont_mu - belief_prev.cont_mu)
            cont_logvar = belief_prev.cont_logvar + gain * (cont_logvar - belief_prev.cont_logvar)

        # Confidence from posterior entropy.
        entropy = entropy_from_probs(type_probs)
        max_entropy = torch.log(
            torch.tensor(
                float(self.k_type),
                device=type_probs.device,
                dtype=type_probs.dtype,
            )
        )
        confidence = 1.0 - entropy / max_entropy.clamp_min(1e-8)
        confidence = confidence.clamp(0.0, 1.0)

        belief_t = BeliefState(
            type_probs=type_probs,
            cont_mu=cont_mu,
            cont_logvar=cont_logvar,
            confidence=confidence,
        )
        return belief_t, alpha


# -----------------------------------------------------------------------------
# 5) Buyer policy core
# -----------------------------------------------------------------------------


class BuyerPolicyCore(nn.Module):
    """Shared buyer-side policy trunk before FiLM modulation."""

    def __init__(
        self,
        d_buyer: int = 128,
        d_hist: int = 128,
        d_ctx: int = 128,
        d_h: int = 256,
        d_core: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_core = d_core
        self.mlp = MLP(d_buyer + d_hist + d_ctx + d_h, [256, 256], d_core, dropout=dropout)

    def forward(
        self,
        buyer_state: torch.Tensor,
        hist_state: torch.Tensor,
        scenario_ctx: torch.Tensor,
        seller_hidden: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([buyer_state, hist_state, scenario_ctx, seller_hidden], dim=-1)
        return self.mlp(x)


# -----------------------------------------------------------------------------
# 6) FiLM conditioner
# -----------------------------------------------------------------------------


class FiLMConditioner(nn.Module):
    """Generate FiLM parameters from explicit belief state.

    We use separate FiLM generators for price and style branches.
    Gamma is modeled as 1 + delta so the modulation starts close to identity.
    """

    def __init__(self, d_belief: int, d_core: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_core = d_core
        self.price_film = MLP(d_belief, [128], 2 * d_core, dropout=dropout)
        self.style_film = MLP(d_belief, [128], 2 * d_core, dropout=dropout)

        # Zero-init last layer so FiLM starts near identity:
        # gamma ~= 1, beta ~= 0.
        nn.init.zeros_(self.price_film.out_proj.weight)
        nn.init.zeros_(self.price_film.out_proj.bias)
        nn.init.zeros_(self.style_film.out_proj.weight)
        nn.init.zeros_(self.style_film.out_proj.bias)

    def forward(
        self, belief_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        price_params = self.price_film(belief_vec)
        style_params = self.style_film(belief_vec)

        gamma_p_delta, beta_p = torch.chunk(price_params, 2, dim=-1)
        gamma_s_delta, beta_s = torch.chunk(style_params, 2, dim=-1)

        gamma_p = 1.0 + gamma_p_delta
        gamma_s = 1.0 + gamma_s_delta
        return gamma_p, beta_p, gamma_s, beta_s


# -----------------------------------------------------------------------------
# 7) Dual policy heads
# -----------------------------------------------------------------------------


class BuyerPricePolicyHead(nn.Module):
    """Predict buyer intent logits and 100-bin price distribution."""

    def __init__(
        self,
        d_core: int = 256,
        n_intents: int = 5,
        n_price_bins: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.intent_head = MLP(d_core, [128], n_intents, dropout=dropout)
        self.price_head = MLP(d_core, [128], n_price_bins, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "intent_logits": self.intent_head(x),
            "price_logits": self.price_head(x),
        }


class BuyerStylePolicyHead(nn.Module):
    """Predict auxiliary buyer style logits.

    v1 note:
        This head is intentionally auxiliary-only. It is not wired to the
        downstream generator yet, but the logits are exposed for future training
        or ablations.
    """

    def __init__(self, d_core: int = 256, n_styles: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.style_head = MLP(d_core, [128], n_styles, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"style_logits": self.style_head(x)}


# -----------------------------------------------------------------------------
# 8) Full switch-aware explicit ToM model
# -----------------------------------------------------------------------------


class SwitchAwareExplicitToM(nn.Module):
    """Switch-aware explicit opponent modeling module for NegotiationToM.

    Strict step-level data flow implemented in this file:
        Seller_Obs_t
            -> SellerGRU
            -> ChangeDetector
            -> BeliefUpdater
            -> FiLM
            -> {PriceHead, StyleHead}

    Notes:
        - Style head is auxiliary only in v1.
        - Price head predicts both intent logits and a 100-bin price policy.
        - The module is written to be self-contained so it can be dropped into
          onmt/SwitchAwareToM.py and tested independently before wiring the full
          trainer/session stack.
    """

    def __init__(
        self,
        n_acts: int,
        n_price_bins: int = 100,
        n_intents: int = 5,
        n_styles: int = 4,
        d_text_in: int = 768,
        d_act: int = 32,
        d_price_emb: int = 16,
        f_num: int = 11,
        d_hist: int = 128,
        d_ctx: int = 128,
        d_obs: int = 256,
        d_h: int = 256,
        k_type: int = 4,
        d_b_cont: int = 16,
        d_buyer: int = 128,
        d_core: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_acts = n_acts
        self.n_price_bins = n_price_bins
        self.n_intents = n_intents
        self.n_styles = n_styles
        self.f_num = f_num
        self.d_h = d_h
        self.k_type = k_type
        self.d_b_cont = d_b_cont
        self.d_buyer = d_buyer
        self.d_hist = d_hist
        self.d_ctx = d_ctx
        self.d_core = d_core

        self.obs_encoder = SellerObservationEncoder(
            n_acts=n_acts,
            n_price_bins=n_price_bins,
            d_act=d_act,
            d_price_emb=d_price_emb,
            f_num=f_num,
            d_text_in=d_text_in,
            d_hist=d_hist,
            d_ctx=d_ctx,
            d_obs=d_obs,
            dropout=dropout,
        )
        self.seller_encoder = SellerGRUEncoder(d_obs=d_obs, d_h=d_h)
        self.change_detector = BinaryChangeDetector(d_h=d_h, f_num=f_num, dropout=dropout)
        self.belief_updater = ExplicitBeliefUpdater(
            d_h=d_h,
            d_hist=d_hist,
            d_ctx=d_ctx,
            k_type=k_type,
            d_b_cont=d_b_cont,
            dropout=dropout,
        )
        self.policy_core = BuyerPolicyCore(
            d_buyer=d_buyer,
            d_hist=d_hist,
            d_ctx=d_ctx,
            d_h=d_h,
            d_core=d_core,
            dropout=dropout,
        )
        self.film = FiLMConditioner(
            d_belief=self.belief_updater.d_belief,
            d_core=d_core,
            dropout=dropout,
        )
        self.price_head = BuyerPricePolicyHead(
            d_core=d_core,
            n_intents=n_intents,
            n_price_bins=n_price_bins,
            dropout=dropout,
        )
        self.style_head = BuyerStylePolicyHead(
            d_core=d_core,
            n_styles=n_styles,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Convenience helpers for integration and unit tests
    # ------------------------------------------------------------------

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.seller_encoder.init_hidden(batch_size, device, dtype=dtype)

    def init_belief(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BeliefState:
        return BeliefState.init_uniform(
            batch_size=batch_size,
            k_type=self.k_type,
            d_b_cont=self.d_b_cont,
            device=device,
            dtype=dtype,
            init_confidence=0.0,
        )

    def _prepare_prev_state(
        self,
        batch: StepBatch,
        h_prev: Optional[torch.Tensor],
        belief_prev: Optional[BeliefState],
    ) -> Tuple[torch.Tensor, BeliefState]:
        if h_prev is None:
            h_prev = self.init_hidden(batch.batch_size, batch.device, batch.seller_num_feats.dtype)
        if belief_prev is None:
            belief_prev = self.init_belief(batch.batch_size, batch.device, batch.seller_num_feats.dtype)
        return h_prev, belief_prev

    # ------------------------------------------------------------------
    # Main step function required by the user specification
    # ------------------------------------------------------------------

    def step(
        self,
        batch: StepBatch,
        h_prev: Optional[torch.Tensor] = None,
        belief_prev: Optional[BeliefState] = None,
        force_switch_prob=None,
    ) -> Dict[str, torch.Tensor]:
        """Run one switch-aware ToM update step.

        Args:
            batch: StepBatch containing the seller observation and buyer/context
                features for one negotiation step.
            h_prev: Previous seller hidden state [B, D_h]. If None, it is
                initialized to zeros.
            belief_prev: Previous explicit belief state. If None, a uniform low-
                confidence belief is created.

        Returns:
            A dictionary named ``out`` containing at least the keys requested by
            the user, plus a few extra tensors that are useful for training or
            debugging.
        """

        batch.validate(expected_f_num=self.f_num)
        h_prev, belief_prev = self._prepare_prev_state(batch, h_prev, belief_prev)

        # 1) Seller observation encoding.
        obs_t = self.obs_encoder(
            seller_act_id=batch.seller_act_id,
            seller_price_bin=batch.seller_price_bin,
            seller_num_feats=batch.seller_num_feats,
            seller_text_emb=batch.seller_text_emb,
            hist_state=batch.hist_state,
            scenario_ctx=batch.scenario_ctx,
        )

        # 2) Local seller trajectory update.
        h_t = self.seller_encoder(obs_t, h_prev)

        # 3) Strategy switch detection.
        switch_logit, switch_prob_raw = self.change_detector(
            h_prev=h_prev,
            h_t=h_t,
            seller_num_feats=batch.seller_num_feats,
            prev_num_feats=batch.prev_seller_num_feats,
        )

        switch_prob_raw = switch_prob_raw.reshape(-1)
        switch_prob = self._resolve_force_switch_prob(
            switch_prob_raw=switch_prob_raw,
            force_switch_prob=force_switch_prob,
        ).reshape(-1, 1)

        belief_t, belief_alpha = self.belief_updater(
            h_t=h_t,
            hist_state=batch.hist_state,
            scenario_ctx=batch.scenario_ctx,
            switch_prob=switch_prob,
            belief_prev=belief_prev,
            force_switch_prob=(switch_prob if force_switch_prob is not None else None),
        )
        belief_vec = belief_t.vector()

        # 5) Shared buyer policy core.
        core_t = self.policy_core(
            buyer_state=batch.buyer_state,
            hist_state=batch.hist_state,
            scenario_ctx=batch.scenario_ctx,
            seller_hidden=h_t,
        )

        # 6) FiLM modulation of price/style representations.
        gamma_p, beta_p, gamma_s, beta_s = self.film(belief_vec)
        core_price = gamma_p * core_t + beta_p
        core_style = gamma_s * core_t + beta_s

        # 7) Dual heads.
        price_out = self.price_head(core_price)
        style_out = self.style_head(core_style)

        out: Dict[str, torch.Tensor] = {
            # Main policy outputs
            **price_out,
            **style_out,

            # Change detection outputs
            "switch_logit": switch_logit.reshape(-1),
            "switch_prob_raw": switch_prob_raw,
            "switch_prob": switch_prob.reshape(-1),

            # Belief outputs
            "belief_type_probs": belief_t.type_probs,
            "belief_cont_mu": belief_t.cont_mu,
            "belief_cont_logvar": belief_t.cont_logvar,
            "belief_confidence": belief_t.confidence,
            "belief_alpha": belief_alpha,
            "belief_vector": belief_vec,

            # State tensors exposed for trainer/session adaptation
            "seller_obs": obs_t,
            "seller_hidden": h_t,
            "prev_seller_hidden": h_prev,
            "buyer_core": core_t,
            "core_price": core_price,
            "core_style": core_style,
        }
        return out

    def forward(
        self,
        batch: StepBatch,
        h_prev: Optional[torch.Tensor] = None,
        belief_prev: Optional[BeliefState] = None,
        force_switch_prob=None,
    ) -> Dict[str, torch.Tensor]:
        return self.step(
            batch=batch,
            h_prev=h_prev,
            belief_prev=belief_prev,
            force_switch_prob=force_switch_prob,
        )

    def _resolve_force_switch_prob(self, switch_prob_raw, force_switch_prob):
        """
        switch_prob_raw: [B]
        force_switch_prob:
            - None
            - float / int
            - Tensor[B] 或 Tensor[1]
        """
        if force_switch_prob is None:
            return switch_prob_raw

        if torch.is_tensor(force_switch_prob):
            forced = force_switch_prob.to(
                device=switch_prob_raw.device,
                dtype=switch_prob_raw.dtype,
            ).reshape(-1)

            if forced.numel() == 1:
                forced = forced.expand_as(switch_prob_raw)
            elif forced.numel() != switch_prob_raw.numel():
                raise ValueError(
                    f"force_switch_prob size mismatch: got {forced.numel()}, "
                    f"expected 1 or {switch_prob_raw.numel()}"
                )
        else:
            forced = torch.full_like(switch_prob_raw, float(force_switch_prob))

        return forced.clamp(0.0, 1.0)


__all__ = [
    "StepBatch",
    "BeliefState",
    "MLP",
    "SellerObservationEncoder",
    "SellerGRUEncoder",
    "BinaryChangeDetector",
    "ExplicitBeliefUpdater",
    "BuyerPolicyCore",
    "FiLMConditioner",
    "BuyerPricePolicyHead",
    "BuyerStylePolicyHead",
    "SwitchAwareExplicitToM",
]

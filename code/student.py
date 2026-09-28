"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.

Submission
----------
The predictor this package submits is a deep ensemble of independently trained
GPT members; see `ensemble.py` for the mechanism and for the argument that it
satisfies the causality, normalization and per-window-reset requirements.

`build_model(config)` returns that ensemble, sized by `config['ensemble_size']`
(absent means 1). A single-member ensemble delegates both model interfaces
straight to its member, so it is bit-identical to the plain baseline `GPT`.

Module library (2026-09-16, `_plans/arch_modules_notes.md`)
----------------------------------------------------------
`GPTVariant` is a drop-in replacement for the classroom `GPT` that can switch on
any subset of a small library of architecture modules. Every module is off by
default, and with all switches off `GPTVariant` is the baseline object itself,
untouched: same class, same `state_dict` keys, same numbers, same time.

The switches are read from `config`, so a variant travels to the scorer inside
the checkpoint exactly as `depth` or `width` do:

    attn_gate        attention output gate  o <- o * 2*sigmoid(W_g x)
    mlp_gate         the same gate on the MLP branch
    bigram_dim       bigram hash embedding, this many channels (>0 turns it on)
    bigram_buckets   hash table rows (0 -> 4*vocab)
    bigram_sign_rows rows of the sign-trick table
    bigram_lambda    injection scale into the first `bigram_dim` residual dims
    softcap          logits <- cap * tanh(logits / cap)
    logit_scale      one learned scalar multiplying the head output
    v_res            ResFormer value residual with a learnable, raw mix scalar
    sandwich_ln      extra LayerNorm inside each residual branch (NormFormer)
    swiglu           SwiGLU MLP at hidden = 8/3 d (parameter matched)
    layerscale       per-channel scale on each sublayer output
    conv_k           causal depthwise conv on the token embedding
    smear            previous-token embedding gate (modded-nanogpt "smear")
    droppath         stochastic depth per block; training only, free at scoring
    pos_mode         0 = learned absolute positions (baseline), 1 = rope, 2 = none
    rope_base        rope frequency base (`rope_channels > 0`/mode 1 only)
    rope_channels    channels rotated by rope (0 -> all of head_dim)
    qk_norm          per-head LayerNorm on q and k, before the softmax
    attn_temp        one learned temperature per head, multiplying q pre-softmax
    res_scale        residual branch output scaled by the fixed 1/sqrt(2*depth)
    mixer_mode       token mixer: 0 = softmax attention (baseline), 1 = none,
                     2 = learned causal depthwise conv k=5, 3 = HGRN1-shaped gated
                     linear recurrence, 4 = fixed causal EMA (zero parameter)
    attn_noproj      drop the attention's output projection (`proj`)
    share_attn       one `qkv`/`proj` shared by every block (wave C)
    sublayer_order   0 = interleaved (baseline), 1/2 = sandwich coefficient k,
                     3 = reverse, FF before attention in every block (wave D)

`mixer_mode` and `attn_noproj` are *replacement* switches rather than additions:
they delete the token-mixing sublayer's projections and put something smaller (or
nothing) in their place, so their parameter deltas are negative. They are the arms
of the ladder in `_plans/Q3_架构线_任务安排.md` 4.3.7, where each step down removes
exactly one mechanism and the value of that mechanism is the gap between two arms.
Like `swiglu` and `sandwich_ln` they are deliberately not the identity, because a
replacement is not a no-op -- with one documented exception, `mixer_mode = 2`,
whose convolution is zero-initialized and therefore *starts* as `mixer_mode = 1`.

`pos_mode` is the one switch that is not purely additive: modes 1 and 2 *drop* the
learned absolute table (`model.py:31`, 32,768 parameters) instead of adding to it,
so the model loses `context * width` parameters relative to the baseline. The
tables `rope_cos_sin`/`apply_rope` build are derived from `rope_base` and the
shape, so rope itself is zero-parameter. See `tests/test_pos_axis.py` for the
parameter arithmetic and the three claims that pin the RoPE frequency range.

`share_attn` (wave C) and `sublayer_order` (wave D) are both STRUCTURAL: they do
not decorate the block, they change which modules exist and which run where.

* `share_attn` promotes block 0's `qkv`/`proj` to the model (`shared_qkv`,
  `shared_proj`) and deletes every block's own copy, so `depth` blocks keep one
  attention's worth of parameters instead of `depth`. `features` hands the pair
  into each block as an argument; nothing the model holds is a reference that
  `.to()`, `.double()` or `load_state_dict(assign=True)` can invalidate. The
  saving is a real saving in the `state_dict` too -- 41 keys at depth 4 versus the
  baseline's 53, since there is exactly one registration of each tensor rather
  than four keys aliasing one storage. `share_attn` is refused together with
  `mixer_mode` and with `attn_noproj`, both of which delete what it shares.
* `sublayer_order` splits the block forward into an attention half
  (`_variant_block_attn`) and an FF half (`_variant_block_ff`) and lets a
  model-level plan (`_sublayer_plan`) decide the order. It adds no parameters and
  no modules: the plan is a permutation of the baseline's `2 * depth` sublayers,
  which is why its parameter gate is uninformative by construction and why the
  front-end gate for this wave is a forward-hash relation table rather than a
  parameter count.

Three properties are deliberate, because they are what make the paired ablation
and the cost figures in the notes valid:

1. **The baseline parameters are never redrawn.** Modules are attached in place
   to the blocks `model.GPT` already built, so a variant arm and the baseline arm
   at the same seed have *bit-identical* `token`, `pos`, per-block `norm1/norm2/
   qkv/proj/mlp`, final `norm` and tied head. Only the switched-on module and the
   modules registered after it differ.
2. **Identity at initialization.** Each module initializes so that it computes the
   identity (gates zero-init behind a 2*sigmoid factor, bigram table and conv
   zero-init, `logit_scale` = 1, `smear` lambda = 0, `v_res` mix = 0, LayerScale
   = 1). Without this, a module's ablation would be confounded with a change of
   the effective initialization. Three modules are deliberately *not* identity
   because identity is not what they mean: `softcap` (a fixed nonlinearity with
   no parameters), `swiglu` (replaces the MLP function) and `sandwich_ln`
   (normalizes the branch output).
3. **Zero cost when off.** Disabled modules are not constructed at all.

Interaction with `train_v2.add_dropout`: that helper rebinds `Block.forward` on
the instance, which would silently discard every module switched on here. It now
refuses to run on a variant model instead of producing a silently wrong run.
"""
import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT
from ensemble import Ensemble

__all__ = ['Ensemble', 'GPTVariant', 'VariantEnsemble', 'VariantBlock', 'build_model',
           'variant_flags', 'variant_active', 'bigram_hash', 'swiglu_hidden', 'FLAG_SPEC',
           'rope_cos_sin', 'apply_rope', 'ema_kernel']


# --------------------------------------------------------------------------- #
# switch parsing
# --------------------------------------------------------------------------- #
# name -> (default, value used when the switch is named without "=value")
FLAG_SPEC = {
    'attn_gate':        (0, 1),
    'mlp_gate':         (0, 1),
    'bigram_dim':       (0, None),      # None: must be given explicitly
    'bigram_buckets':   (0, 4096),      # 0 -> 4*vocab
    'bigram_sign_rows': (8192, None),
    'bigram_lambda':    (0.1, None),
    'softcap':          (0.0, 15.0),
    'logit_scale':      (0, 1),
    'v_res':            (0, 1),
    'sandwich_ln':      (0, 1),
    'swiglu':           (0, 1),
    'layerscale':       (0, 1),
    'conv_k':           (0, 3),
    'smear':            (0, 1),
    'droppath':         (0.0, 0.1),
    'swiglu_multiple':  (8, None),      # modifier: rounds SwiGLU's hidden to this
    'pos_mode':         (0, 1),         # 0 learned absolute / 1 rope / 2 none
    'rope_base':        (10000.0, None),  # rope only; sets the frequencies
    'rope_channels':    (0, None),      # rope only; 0 -> all of head_dim rotates
    'qk_norm':          (0, 1),
    'attn_temp':        (0, 1),
    'res_scale':        (0, 1),
    'mixer_mode':       (0, 1),         # 0 attention / 1 none / 2 dconv5 / 3 gru1 / 4 ema
    'attn_noproj':      (0, 1),
    'share_attn':       (0, 1),         # wave C: one qkv/proj shared by every block
    'sublayer_order':   (0, 0),         # wave D: 0 interleaved / 1,2 sandwich k / 3 reverse
}

# Switches that only modify another module and mean nothing on their own, so
# setting one alone must not be read as "a module is switched on".
#
# `rope_base` and `rope_channels` are modifiers of `pos_mode == 1`: naming either
# one on its own (both are absent from a baseline config) changes no computation,
# and reading that as "a module is on" would take the run off the fast path and
# claim a variant that was never switched on. `pos_mode` itself is *not* a
# modifier -- it is the module, and setting it is what turns rope on.
MODIFIER_KEYS = ('swiglu_multiple', 'rope_base', 'rope_channels')

# Switches that act inside the attention sublayer, and therefore have nothing to
# act on once `mixer_mode` has replaced it. Listed here rather than inlined in
# `_attach_variant_modules` so the refusal names every offending key at once.
# `share_attn` shares exactly the two modules `mixer_mode` deletes, so it belongs
# on this list; it also needs its own guard for `attn_noproj`, which deletes only
# `proj` and would leave the switch with half a module to share.
ATTENTION_ONLY_KEYS = ('v_res', 'qk_norm', 'attn_temp', 'attn_noproj', 'share_attn')


def variant_flags(config):
    """Effective switch settings for `config`; absent keys take their default."""
    flags = {}
    for name, (default, _) in FLAG_SPEC.items():
        value = config.get(name, default)
        flags[name] = default if value is None else value
    # `bigram_buckets = 0` means the upstream ratio (modded-nanogpt uses
    # 15 * vocab rows for a 50,304-token vocabulary; see the notes for why that
    # ratio is not affordable at vocab 2048).
    if flags['bigram_dim'] and not flags['bigram_buckets']:
        flags['bigram_buckets'] = 4 * int(config['vocab'])
    return flags


def variant_active(config):
    """True when at least one module is switched on, i.e. the fast path is closed."""
    flags = variant_flags(config)
    return any(flags[name] != FLAG_SPEC[name][0]
               for name in FLAG_SPEC if name not in MODIFIER_KEYS)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def swiglu_hidden(width, multiple=8):
    """8/3 * width rounded up to a multiple of `multiple` (128 -> 344, 512 -> 1368)."""
    return -(-(8 * width) // 3 // multiple) * multiple


def _sign_table(rows, channels):
    """Deterministic +/-1 table of shape [rows, channels] (Chen et al. 2015).

    Integer arithmetic only, so it is reproducible on any torch version and any
    machine, and it is registered as a non-persistent buffer so it costs nothing
    in the checkpoint: a K-member submission does not ship K copies of it.
    """
    r = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    c = torch.arange(channels, dtype=torch.int64).unsqueeze(0)
    h = (r * 1000003 + c * 2654435761) % 4294967296
    h = (h * 16777619 + 12345) % 4294967296
    h = ((h ^ (h >> 13)) * 1103515245) % 4294967296
    h = h ^ (h >> 16)
    return torch.where((h >> 16) & 1 == 0, torch.ones(1), -torch.ones(1)).to(torch.float32)


def bigram_hash(ids, buckets):
    """Bucket index for the [prev, cur] pair at every position (modded-nanogpt PR #299).

    Position 0 has no predecessor and takes the reserved last row, so the table
    has `buckets - 1` usable rows. `ids` is [batch, time]; the result has the same
    shape. The value at t is a function of `ids[:, t-1:t+1]` only, so the hash
    never sees a future token.
    """
    out = ids.to(torch.int32).clone()
    out[..., 0] = buckets - 1
    if ids.shape[1] > 1:
        out[..., 1:] = torch.bitwise_xor(36313 * out[..., 1:], 27191 * out[..., :-1]) % (buckets - 1)
    return out.long()


def drop_path(x, probability, training):
    """Stochastic depth on one residual branch: per-sample keep/drop, rescaled."""
    if probability <= 0. or not training:
        return x
    keep = 1. - probability
    mask = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep) / keep
    return x * mask


def rope_cos_sin(length, head_dim, base=10000.0, channels=0, device=None,
                 dtype=torch.float32):
    """RoPE cos/sin tables, shape [length, channels // 2] (arXiv:2104.09864).

    Pair `i` turns at wavelength ``2*pi*base**(2i/channels)``, so `base` decides
    how many of the pairs complete at least one cycle inside one scored window.
    At head_dim 32 the count is the largest `i` with ``2*pi*base**(2i/32) <= 256``,
    i.e. ``i <= 16*ln(128/pi)/ln(base)``: base 10000 admits **i = 0..6, 7 pairs**,
    base 500 admits **i = 0..9, 10 pairs** (the count is monotone decreasing in
    `base`). The rest of the pairs rotate by less than a full turn anywhere in the
    window, which is why `rope_base` is a knob and not a constant.

    The frequencies are measured against `channels`, not against `head_dim`: that
    is the GPT-NeoX/Phi-3 convention for a partial rotation, under which
    ``channels = c`` is exactly the rope a c-wide head would have. (Measuring them
    against `head_dim` instead would reuse the first `c/head_dim` of the full
    spectrum, leaving the highest-`i` pairs of the rotating block nearly static --
    a different module, and not the one this flag is documented to build.)

    Everything is evaluated in float64 and cast once at the end, so the table does
    not depend on the torch version's float32 `exp`/`cos` kernels.
    """
    rotate = int(channels) or int(head_dim)
    if rotate % 2 or not 0 < rotate <= head_dim:
        raise ValueError('rope_channels must be even and in (0, head_dim=%d], got %r'
                         % (head_dim, channels))
    index = torch.arange(rotate // 2, dtype=torch.float64, device=device)
    inverse = float(base) ** (-2.0 * index / rotate)
    position = torch.arange(length, dtype=torch.float64, device=device)
    angle = position.unsqueeze(1) * inverse.unsqueeze(0)
    return angle.cos().to(dtype), angle.sin().to(dtype)


def apply_rope(x, cos, sin, channels=0):
    """Rotate adjacent channel pairs of `x` [..., length, head_dim].

    `cos`/`sin` are [length, channels // 2] and broadcast over every leading axis,
    so the same tables serve q and k of any batch/head count. Channels from
    `channels` on are returned untouched, which is what makes this a partial
    rotation; `channels = 0` rotates all of `head_dim`. Pairs are adjacent
    ((0,1), (2,3), ...), the original formulation, so "the first `channels`
    channels" is exactly "the first `channels//2` pairs".
    """
    rotate = int(channels) or x.shape[-1]
    shape = (1,) * (x.dim() - 2) + cos.shape
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    even, odd = x[..., 0:rotate:2], x[..., 1:rotate:2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
    return torch.cat([rotated, x[..., rotate:]], dim=-1)


def _rope_tables(block, length, head_dim, device, dtype):
    """`block`'s cos/sin tables, rebuilt only when the key below changes.

    A cache rather than a buffer: the tables are a function of `rope_base`,
    `rope_channels` and the shape, so they are not parameters and must not enter
    the checkpoint (a K-member submission would ship K copies), and rebuilding
    them on a device or dtype change is cheaper than keeping them consistent
    through `.to()`. `_attach_variant_modules` seeds the attribute.
    """
    flags = block.flags
    key = (length, head_dim, device, dtype, flags['rope_base'], flags['rope_channels'])
    cache = block._rope_cache
    if cache is None or cache[0] != key:
        cache = (key,) + rope_cos_sin(length, head_dim, flags['rope_base'],
                                      flags['rope_channels'], device, dtype)
        block._rope_cache = cache
    return cache[1], cache[2]


class SwiGLU(nn.Module):
    """SwiGLU at hidden = 8/3 d, which keeps the parameter count near GELU's 4d.

    `multiple` rounds that hidden width. The parameter-matched value at width 128
    is 344, which is 8-aligned but not 16- or 32-aligned, and the measured
    wall-clock cost of this module is far above its FLOPs share
    (`_plans/arch_modules_notes.md`); `swiglu_multiple=64` gives 384 and is the
    arm that tests whether that is a GEMM-shape effect.
    """

    def __init__(self, width, multiple=8):
        super().__init__()
        hidden = swiglu_hidden(width, multiple)
        self.gate, self.up, self.down = (nn.Linear(width, hidden), nn.Linear(width, hidden),
                                         nn.Linear(hidden, width))
        for layer in (self.gate, self.up, self.down):
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
# replacement token mixers (wave G)
# --------------------------------------------------------------------------- #
# `mixer_mode` does not add to the token-mixing sublayer, it *is* the sublayer:
# modes 1-4 delete `qkv` and `proj` and put something smaller, or nothing, in
# their place, so every one of them has a negative parameter delta. That is what
# makes them a ladder rather than a list of candidates (`Q3_架构线_任务安排.md`
# 4.3.7 4): each step down drops exactly one mechanism, and the value of that
# mechanism is the gap between two arms -- content selection (3 -> 2), a learned
# kernel (2 -> 4), a global receptive field (4 -> 1), and the whole mixer against
# the baseline.
EMA_DECAY = 0.99


def ema_kernel(length, decay=EMA_DECAY, device=None, dtype=torch.float32):
    """Kernel of the fixed causal EMA, shape [length, length].

    ``K[t, s] = (1 - decay) * decay ** (t - s)`` on and below the diagonal and 0
    above it, which is the closed form of the recurrence
    ``o_t = decay * o_{t-1} + (1 - decay) * h_t`` from a zero state. Two
    properties are the reason this is a kernel and not a loop: it is *causal by
    construction* (the upper triangle is exactly zero, no masking argument
    needed), and it is *fixed* -- no parameters, no input dependence, no knob.

    `decay = 0.99` puts the kernel's centre of mass ``decay/(1-decay)`` at 99
    positions, i.e. 39% of the 256-token window, and leaves ``decay**255 = 7.7%``
    of the current token's weight on the oldest one. The reach is therefore not
    local, which is the whole difference between this arm and `mixer_mode = 2`.
    Row `t` sums to ``1 - decay**(t+1)`` rather than to 1, so the branch output
    fades in over the first ~100 positions; that is the arithmetic of the
    recurrence with a zero initial state, not a normalization choice.

    Evaluated in float64 and cast once at the end, like `rope_cos_sin`, so the
    table does not depend on the float32 `pow` kernel.
    """
    position = torch.arange(length, dtype=torch.float64, device=device)
    lag = (position.unsqueeze(1) - position.unsqueeze(0)).clamp(min=0)
    return torch.tril((1. - float(decay)) * float(decay) ** lag).to(dtype)


class GatedRecurrence(nn.Module):
    """HGRN1-shaped gated linear recurrence: one gate projection, one value (arXiv:2311.04823).

    ``o_t = (1 - f_t) * o_{t-1} + f_t * i_t`` with ``f_t = sigmoid(W_f x_t)`` and
    ``i_t = W_i x_t``. There is no query, no key, no softmax and no output
    projection: two `width x width` linears replace a `qkv` (3x) and a `proj`, and
    the state is carried forward one position at a time, so the mixer can never
    read t+1.

    This is the arm's whole point (`Q3_架构线_任务安排.md` 4.3.7 4, R4): it is the
    only replacement that keeps *content-dependent* mixing while still removing
    12.14% of the model. The gate bias is zero, so the gate's midpoint is
    `sigmoid(0) = 0.5` -- at a zero input that is a one-step memory, not a long
    one. The arm has to learn to hold state, and `test_mixer_axis.py` pins both the
    midpoint and the fact that a real input moves the gate off it.

    The loop is over positions and the two projections are computed for all of
    them at once, so the sequential part is three elementwise ops per step on a
    [batch, width] tensor. That is the arm's measured cost, not its FLOPs.
    """

    def __init__(self, width):
        super().__init__()
        self.gate, self.value = nn.Linear(width, width), nn.Linear(width, width)
        for layer in (self.gate, self.value):
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        value = self.value(x)
        state = torch.zeros_like(value[:, 0])
        outputs = []
        for t in range(value.shape[1]):
            state = (1. - gate[:, t]) * state + gate[:, t] * value[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def _ema_kernel(block, length, device, dtype):
    """`block`'s EMA kernel, rebuilt only when the key below changes.

    A cache rather than a buffer, for the reason `_rope_tables` documents: the
    kernel is a pure function of the length, so it is not a parameter and must not
    enter the checkpoint. The key carries device and dtype because a plain
    attribute is not rebuilt by `.to()`. `_attach_variant_modules` seeds it.
    """
    key = (length, device, dtype)
    cache = block._mixer_cache
    if cache is None or cache[0] != key:
        cache = (key, ema_kernel(length, EMA_DECAY, device, dtype))
        block._mixer_cache = cache
    return cache[1]


def _token_mixer(block, hidden):
    """The replacement token mixer for `block.flags['mixer_mode']` != 0.

    `hidden` is `norm1(x)`, the same normalized state the attention branch would
    have consumed, so a mixer arm differs from the baseline in the mixing
    operation and in nothing else. What it does *not* do is normalize again: a
    LayerNorm here would be a second parameterized stage standing in for the
    attention, which is not what any of these four modes means.

    Modes 2-4 return a `[batch, length, width]` tensor, exactly the shape
    `proj` produced, so the branch's post-processing (`sandwich_ln`, `attn_gate`,
    `layerscale`, `res_scale`) applies to them unchanged.
    """
    mode = block.flags['mixer_mode']
    if mode == 1:
        # No mixing at all. Nothing from another position arrives and this
        # position's state does not leave, so the residual stream carries x and
        # the MLP alone: the block is a position-wise MLP between two residuals.
        return torch.zeros_like(hidden)
    if mode == 2:
        # Learned causal depthwise convolution, k = 5, left-padded so that the
        # output at t reads t-4..t. Zero-initialized like `embed_conv`, so the arm
        # *starts* as mode 1 and every coefficient in the kernel is something
        # training put there -- that is the cleanest reading of "learnable local
        # filter" against mode 4's fixed one, and it is documented in the module
        # docstring rather than hidden here.
        pad = block.mixer_conv.kernel_size[0] - 1
        return block.mixer_conv(F.pad(hidden.transpose(1, 2), (pad, 0))).transpose(1, 2)
    if mode == 3:
        return block.mixer_gru(hidden)
    if mode == 4:
        # `K[t, s]` weights the token at `s` when producing position `t`, so the
        # einsum has to contract the *second* index of the kernel: writing it the
        # other way round sums over the future and the arm is anti-causal.
        kernel = _ema_kernel(block, hidden.shape[1], hidden.device, hidden.dtype)
        return torch.einsum('ts,bsw->btw', kernel, hidden)
    raise ValueError('mixer_mode must be 0 (attention), 1 (none), 2 (dconv5), '
                     '3 (gru1) or 4 (fixed_ema); got %r' % (mode,))


# --------------------------------------------------------------------------- #
# the block
# --------------------------------------------------------------------------- #
def _sublayer_plan(depth, mode):
    """Sublayer order for wave D (`arXiv:1911.03864v2`), or None when off.

    `mode` 0 is the interleaved baseline and returns None, so the caller keeps the
    untouched block loop and this switch costs nothing when it is off. Modes 1 and
    2 are the sandwich coefficient `k` of the paper's
    ``s f|^k ( s f| f f| )^(n-k) f f|^k`` at `n = depth`; mode 3 is `reverse`
    (FF before attention in every block), which is not a sandwich coefficient.

    Why the plan is a flat list rather than a per-block flag: at `k = 1` with
    `depth = 4` the order is ``A0 A1 F0 A2 F1 A3 F2 F3``, i.e. the two kinds
    interleave ACROSS block boundaries. No per-block switch can produce that, and
    it is also not `sandwich_ln` (which would add parameters; this switch adds
    none, measured). The only implementation that is both faithful and
    parameter-free is to split the block forward into an attention half and an FF
    half and let a model-level plan say which runs where.

    The returned list holds `(block_index, kind)` with `kind` in ('a', 'f') and has
    length `2 * depth`. It is a PERMUTATION of the baseline sublayers -- every
    block contributes its attention half and its FF half exactly once -- which is
    why the arm's parameter count equals the baseline's to the parameter and why
    no sublayer can be silently skipped (a skipped half would leave its
    parameters without a gradient; a parameter count cannot see that, which is
    exactly why this wave needs the forward-hash gate).
    """
    if mode == 0:
        return None
    if mode == 3:
        return [(index, kind) for index in range(depth) for kind in ('f', 'a')]
    k = int(mode)
    if not 0 < k < depth:
        raise ValueError('sublayer_order %r needs 0 < k < depth=%d'
                         % (mode, depth))
    plan = []
    attn = ff = 0
    for _ in range(k):                      # s f|^k  -- pure attention
        plan.append((attn, 'a'))
        attn += 1
    for _ in range(depth - k):              # ( s f| f f| )^(n-k)
        plan.append((attn, 'a'))
        attn += 1
        plan.append((ff, 'f'))
        ff += 1
    for _ in range(k):                      # f f|^k  -- pure feedforward
        plan.append((ff, 'f'))
        ff += 1
    return plan


def _variant_block_attn(self, x, v_first=None, attn_temp=None, shared=None):
    """`model.Block.forward` with the switched-on modules spliced in.

    `attn_temp` arrives as an argument because the temperature is one parameter
    shared by every block, and a block holding a *reference* to a model-level
    `nn.Parameter` would keep pointing at the pre-`.to()` object (torch rebuilds
    the `Parameter` on every `_apply`) and silently stop training. Passing it
    through the call site in `GPTVariant.features` cannot go stale. A standalone
    `VariantBlock` owns its own copy and passes nothing.

    `shared` is `(qkv, proj)` for wave C and None otherwise, and it is resolved at
    the CALL SITE for the same reason `attn_temp` is: the block no longer owns
    those modules, so there is no attribute that could go stale. This is what
    keeps the shared arm safe under `.to()`, `.double()` and
    `load_state_dict(assign=True)`, all three of which can replace `Parameter`
    objects underneath a stored reference. It also removes the failure mode an
    in-place alias has: aliasing `block[i].qkv = block[0].qkv` leaves four
    `state_dict` keys pointing at one tensor, so `load_state_dict(golden,
    strict=True)` SUCCEEDS and silently overwrites the shared weights with the
    last block's copy.

    The token-mixing branch is the one place with a switch that is not additive:
    `mixer_mode` replaces it rather than decorating it, and the two paths share
    only `norm1`. Everything after the branch is common to both.
    """
    flags = self.flags
    batch, length, width = x.shape
    heads, head_dim = self.heads, width // self.heads
    hidden = self.norm1(x)
    if flags['mixer_mode']:
        # `qkv` and `proj` are gone (deleted where the mode was attached); modes
        # 2-4 hand back the shape `proj` used to produce. rope, `qk_norm`,
        # `attn_temp` and `v_res` all act inside the attention this replaces, so
        # `_attach_variant_modules` refuses to build the combination rather than
        # ignoring them here.
        out = _token_mixer(self, hidden)
    else:
        attn_qkv = self.qkv if shared is None else shared[0]
        q, k, v = attn_qkv(hidden).view(batch, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        if flags['v_res']:
            # ResFormer (arXiv:2410.17897): every layer mixes the first layer's value
            # into its own. v_mix is a raw scalar (as in the modded-nanogpt record) but
            # starts at 0, so the module is the identity until the model moves it.
            if v_first is None:
                v_first = v
            else:
                v = (1. - self.v_mix) * v + self.v_mix * v_first
        if flags['qk_norm']:
            # Per-head LayerNorm, applied before any rotation: it bounds the attention
            # logits by construction, which is the same failure mode `attn_temp`
            # rescales after the fact. Affine (2*2*head_dim parameters per block).
            q, k = self.q_norm(q), self.k_norm(k)
        if flags['pos_mode'] == 1:
            # Position enters only here: every position in a window is an index from 0,
            # which is the same convention as the learned table it replaces.
            cos, sin = _rope_tables(self, length, head_dim, x.device, q.dtype)
            q = apply_rope(q, cos, sin, flags['rope_channels'])
            k = apply_rope(k, cos, sin, flags['rope_channels'])
        if flags['attn_temp']:
            # softmax((t*q)@k) == softmax(t * logits), so scaling q before
            # scaled_dot_product_attention is exactly the per-head temperature and
            # keeps the fused kernel. t is initialized to 1, so the module is the
            # identity at initialization.
            temp = self.attn_temp if attn_temp is None else attn_temp
            q = q * temp.to(q.dtype).view(1, -1, 1, 1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = attended.transpose(1, 2).reshape(batch, length, width)
        # `attn_noproj` leaves the concatenated head outputs as the branch: the
        # projection is the only thing dropped and the attention is untouched, so
        # this is the one arm on the card that changes no mechanism at all.
        # The projection is resolved INSIDE the else arm, exactly as the original
        # `out if flags['attn_noproj'] else self.proj(out)` did. Hoisting the
        # resolve above the test is an AttributeError on a healthy `attn_noproj`
        # arm: with that switch on, `proj` is never created, so the attribute does
        # not exist to be read. (Measured -- the cross-revision sweep in
        # q4c_gate.sh/q4d_gate.sh caught this, which is what that step is for.)
        out = out if flags['attn_noproj'] else (self.proj if shared is None
                                                else shared[1])(out)
    if flags['sandwich_ln']:
        out = self.post_attn_norm(out)
    if flags['attn_gate']:
        out = out * (2. * torch.sigmoid(self.attn_gate(hidden)))
    if flags['layerscale']:
        out = out * self.ls_attn
    if flags['res_scale']:
        out = out * self.res_scale
    x = x + drop_path(out, self.drop_prob, self.training)

    return x, v_first


def _variant_block_ff(self, x):
    """The FF half of the block, split out for wave D.

    The arithmetic is the tail of the original body, line for line. The split
    exists so a model-level `sublayer_plan` can run the two halves at different
    positions; `bitident_check.py` is the witness that it changed nothing, and it
    is the reason the split is a pure `return`-boundary move rather than a rewrite.
    """
    flags = self.flags
    hidden = self.norm2(x)
    out = self.mlp(hidden)
    if flags['sandwich_ln']:
        out = self.post_mlp_norm(out)
    if flags['mlp_gate']:
        out = out * (2. * torch.sigmoid(self.mlp_gate(hidden)))
    if flags['layerscale']:
        out = out * self.ls_mlp
    if flags['res_scale']:
        out = out * self.res_scale
    return x + drop_path(out, self.drop_prob, self.training)


def _variant_block_forward(self, x, v_first=None, attn_temp=None, shared=None):
    """The interleaved order: the two halves in sequence, in one place.

    This shim is what `_attach_variant_modules` binds to `block.forward`, so a
    standalone `VariantBlock` and every path that is not driven by a wave D plan
    keep the original single-call semantics.
    """
    x, v_first = _variant_block_attn(self, x, v_first, attn_temp, shared)
    return _variant_block_ff(self, x), v_first


def _attach_variant_modules(block, flags, index, depth, droppath, width):
    """Add the switched-on modules to an existing `model.Block`, in place."""
    if flags['pos_mode'] not in (0, 1, 2):
        raise ValueError('pos_mode must be 0 (learned absolute), 1 (rope) or 2 '
                         '(no position information); got %r' % (flags['pos_mode'],))
    if flags['mixer_mode'] not in (0, 1, 2, 3, 4):
        raise ValueError('mixer_mode must be 0 (attention), 1 (none), 2 (dconv5), '
                         '3 (gru1) or 4 (fixed_ema); got %r' % (flags['mixer_mode'],))
    # Switches that act *inside* the attention sublayer. A replacement mixer deletes
    # that sublayer, so naming one of them alongside it asks for a module the model
    # cannot honour; ignoring it silently would record an arm under a configuration
    # it never ran, which is the failure `variant_active` exists to prevent.
    # `share_attn` shares modules that two other switches delete, so these two
    # pairs are refused at build time rather than silently ignored: an arm that
    # cannot honour its own name would be recorded under a configuration it never
    # ran, which is the failure `variant_active` exists to prevent.
    if flags['share_attn'] and flags['mixer_mode']:
        raise ValueError('share_attn needs the attention sublayer that mixer_mode=%r '
                         'replaces; there is no qkv/proj left to share.'
                         % (flags['mixer_mode'],))
    if flags['share_attn'] and flags['attn_noproj']:
        raise ValueError('share_attn shares the output projection and attn_noproj '
                         'deletes it; the two cannot both be asked for.')
    if flags['mixer_mode']:
        ignored = [name for name in ATTENTION_ONLY_KEYS if flags[name]]
        if flags['pos_mode'] == 1:
            ignored.append('pos_mode=1 (rope)')
        if ignored:
            raise ValueError('mixer_mode=%r replaces the attention sublayer, so it '
                             'has nothing to apply to: %s'
                             % (flags['mixer_mode'], ', '.join(ignored)))
    block.flags = dict(flags)
    block._variant = True
    block._rope_cache = None
    block._mixer_cache = None
    block.drop_prob = droppath * index / max(1, depth - 1) if droppath > 0 else 0.
    if flags['swiglu']:
        block.mlp = SwiGLU(width, flags['swiglu_multiple'])
    if flags['attn_gate']:
        block.attn_gate = nn.Linear(width, width)
        nn.init.zeros_(block.attn_gate.weight)
        nn.init.zeros_(block.attn_gate.bias)            # 2*sigmoid(0) == 1
    if flags['mlp_gate']:
        block.mlp_gate = nn.Linear(width, width)
        nn.init.zeros_(block.mlp_gate.weight)
        nn.init.zeros_(block.mlp_gate.bias)
    if flags['sandwich_ln']:
        block.post_attn_norm, block.post_mlp_norm = nn.LayerNorm(width), nn.LayerNorm(width)
    if flags['layerscale']:
        block.ls_attn = nn.Parameter(torch.ones(width))
        block.ls_mlp = nn.Parameter(torch.ones(width))
    if flags['v_res']:
        block.v_mix = nn.Parameter(torch.zeros(1))
    if flags['qk_norm']:
        head_dim = width // block.heads
        block.q_norm, block.k_norm = nn.LayerNorm(head_dim), nn.LayerNorm(head_dim)
    if flags['res_scale']:
        # Zero parameters and deliberately not the identity: the residual stream
        # accumulates 2*depth branch outputs, so 1/sqrt(2*depth) is the scale that
        # keeps its variance from growing with depth. This is the one new switch
        # that changes the forward pass without changing the state dict.
        block.res_scale = 1. / math.sqrt(2. * max(1, depth))
    # Replacement mixers come last so that every module above keeps the RNG draws
    # it has always had, whatever else is switched on beside it.
    if flags['share_attn']:
        # Wave C: the blocks stop owning the attention projections. The model
        # promotes block 0's modules before this loop runs and hands them in at
        # the call site, so what is deleted here is only the per-block COPY. No RNG
        # is drawn and no parameter is redrawn, so every baseline parameter keeps
        # the value it has in a plain `GPT` at the same seed -- the shared tensors
        # ARE the baseline's block-0 qkv/proj, bit for bit.
        #
        # Deleting rather than aliasing is the load-bearing part: an alias would
        # leave `blocks.*.qkv.*` in the state dict four times over, and
        # `load_state_dict(golden, strict=True)` would then succeed and overwrite
        # the shared tensor with whichever block's copy came last.
        del block.qkv
        del block.proj
    if flags['mixer_mode']:
        # The mixer *is* the token-mixing sublayer, so both projections go with it:
        # the arm's negative parameter delta is exactly their removal, and a
        # checkpoint of an attention model cannot be loaded into it (`qkv`/`proj`
        # are missing keys, not silently re-initialized ones). `norm1` stays in the
        # module tree even for mode 1, which has no branch to normalize -- the
        # ladder in the notes counts the mixer as `qkv` + `proj` and nothing else.
        del block.qkv
        del block.proj
        if flags['mixer_mode'] == 2:
            block.mixer_conv = nn.Conv1d(width, width, 5, groups=width)
            nn.init.zeros_(block.mixer_conv.weight)
            nn.init.zeros_(block.mixer_conv.bias)
        elif flags['mixer_mode'] == 3:
            block.mixer_gru = GatedRecurrence(width)
    elif flags['attn_noproj']:
        del block.proj
    block.forward = types.MethodType(_variant_block_forward, block)


class VariantBlock(Block):
    """`model.Block` plus optional modules, for standalone use and for tests."""

    def __init__(self, width, heads, flags, index=0, depth=1, droppath=0.):
        super().__init__(width, heads)
        _attach_variant_modules(self, flags, index, depth, droppath, width)
        if flags['attn_temp']:
            # `GPTVariant` keeps ONE per-head temperature for the whole model and
            # passes it into every block (see `_variant_block_forward`); a block
            # used on its own has no model to share with, so it owns its own.
            self.attn_temp = nn.Parameter(torch.ones(heads))


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class GPTVariant(GPT):
    """`model.GPT` plus optional modules, selected by `config` switches.

    The modules are attached to the blocks `GPT.__init__` already built rather
    than to freshly constructed ones, so every baseline parameter keeps the exact
    values (and therefore the exact RNG draws) it has in a plain `GPT` at the same
    seed. The modules themselves are created afterwards, from the current stream.
    """

    def __init__(self, config):
        super().__init__(config)
        self.flags = variant_flags(config)
        self.variant_on = variant_active(config)
        if not self.variant_on:
            return
        flags = self.flags
        width, depth = config['width'], config['depth']
        if flags['pos_mode']:
            # Modes 1 and 2 carry position in the attention (rope) or not at all,
            # so the learned absolute table is dropped rather than built and then
            # ignored: its `context * width` parameters leave the model AND the
            # checkpoint. `GPT.__init__` has already drawn it from the RNG stream
            # before this line, so every other parameter still holds exactly the
            # value a plain `GPT` at the same seed gives it.
            del self.pos
        if flags['share_attn']:
            # Design: PROMOTE, never recreate. The order of these two statements is
            # a hard constraint -- the attach loop below deletes every block's own
            # `qkv`/`proj`, block 0's included, so the reference has to be taken
            # first. Swapping these lines raises AttributeError.
            self.shared_qkv, self.shared_proj = self.blocks[0].qkv, self.blocks[0].proj
        for index, block in enumerate(self.blocks):
            _attach_variant_modules(block, flags, index, depth, flags['droppath'], width)
        self.sublayer_plan = _sublayer_plan(depth, flags['sublayer_order'])
        if flags['bigram_dim']:
            self.bigram_embed = nn.Embedding(flags['bigram_buckets'], flags['bigram_dim'])
            nn.init.zeros_(self.bigram_embed.weight)
            self.bigram_lambda = nn.Parameter(torch.tensor(float(flags['bigram_lambda'])))
            self.register_buffer('bigram_signs',
                                 _sign_table(flags['bigram_sign_rows'], flags['bigram_dim']),
                                 persistent=False)
        if flags['smear']:
            self.smear_gate = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.smear_gate.weight)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
        if flags['conv_k']:
            self.embed_conv = nn.Conv1d(width, width, flags['conv_k'], groups=width)
            nn.init.zeros_(self.embed_conv.weight)
            nn.init.zeros_(self.embed_conv.bias)
        if flags['logit_scale']:
            self.logit_scale = nn.Parameter(torch.ones(1))
        if flags['attn_temp']:
            # One learned temperature per head, shared by every block. Kept on the
            # model, not on the blocks: `heads` parameters, not `depth * heads`, and
            # `GPTVariant.features` hands it to each block as an argument.
            self.attn_temp = nn.Parameter(torch.ones(int(config['heads'])))

    def embedding(self, ids):
        if self.flags['pos_mode']:
            # Mode 1 applies rope inside the attention and mode 2 uses no position
            # information at all; neither has an absolute table left to add, and
            # `__init__` has dropped `self.pos` for both.
            x = self.token(ids)
        else:
            x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        flags = self.flags
        if flags['bigram_dim']:
            table = self.bigram_signs.to(x.device)
            sign_index = torch.zeros_like(ids)
            if ids.shape[1] > 1:
                sign_index[:, 1:] = (ids[:, :-1] ^ ids[:, 1:]) % table.shape[0]
            injected = self.bigram_embed(bigram_hash(ids, flags['bigram_buckets'])) * table[sign_index]
            x = x.clone()
            x[..., :flags['bigram_dim']] = x[..., :flags['bigram_dim']] + self.bigram_lambda * injected
        if flags['smear']:
            gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x))
            x = torch.cat([x[:1], x[1:] + gate[1:] * x[:-1]], dim=0)
        if flags['conv_k']:
            pad = flags['conv_k'] - 1
            y = F.conv1d(F.pad(x.transpose(1, 2), (pad, 0)), self.embed_conv.weight,
                         self.embed_conv.bias, groups=x.shape[-1])
            x = x + y.transpose(1, 2)
        return x

    def features(self, ids):
        x = self.embedding(ids)
        v_first = None
        # `None` unless the switch is on, and only then is `self.attn_temp` read.
        # Blocks reach this line only through `variant_active`, i.e. already
        # patched to `_variant_block_forward`, which accepts the third argument.
        attn_temp = self.attn_temp if self.flags['attn_temp'] else None
        # Both wave C and wave D are resolved HERE, at the call site, for the same
        # reason `attn_temp` is: the model is the only object that survives `.to()`,
        # `.double()` and `load_state_dict(assign=True)` with its attributes intact,
        # so nothing that is read per-forward can go stale.
        shared = (self.shared_qkv, self.shared_proj) if self.flags['share_attn'] else None
        plan = getattr(self, 'sublayer_plan', None)
        if plan is None:
            for block in self.blocks:
                x, v_first = block(x, v_first, attn_temp, shared)
        else:
            # Wave D: the same sublayers, in the order `_sublayer_plan` gives.
            # `block.forward` is bypassed on purpose, so the plan is the only thing
            # that decides what runs where. `v_first` is threaded through unchanged,
            # so with `v_res` the seed is whichever attention half runs first --
            # position-independent, and the plan cannot change which one that is.
            for index, kind in plan:
                block = self.blocks[index]
                if kind == 'a':
                    x, v_first = _variant_block_attn(block, x, v_first, attn_temp, shared)
                else:
                    x = _variant_block_ff(block, x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        logits = self.head(self.features(ids))
        if self.flags['softcap']:
            cap = float(self.flags['softcap'])
            logits = cap * torch.tanh(logits / cap)
        if self.flags['logit_scale']:
            logits = logits * self.logit_scale
        return logits
    # predict_log_probs is inherited unchanged from model.GPT: it is a log_softmax
    # of forward(), which is strictly causal and holds no state between calls.


class VariantEnsemble(Ensemble):
    """`ensemble.py` with `GPTVariant` members; identical when no switch is set."""

    def __init__(self, config):
        size = int(config.get('ensemble_size', 1))
        member_configs = config.get('member_configs') or [dict(config) for _ in range(size)]
        active = any(variant_active(member) for member in member_configs)
        # `Ensemble.__init__` builds plain GPT members first. Those draws must not
        # be thrown away, or a variant member would be initialized from a shifted
        # stream; rewind so the variant members replay them, member for member.
        state = torch.get_rng_state() if active else None
        super().__init__(config)
        if active:
            torch.set_rng_state(state)
            self.members = nn.ModuleList([GPTVariant(member) for member in self.member_configs])
            # The stacked batched path in ensemble.py is hard-coded to the
            # baseline module set.
            self.batched = False
            self._stacked = None


def build_model(config):
    """Model factory: the ensemble of `ensemble.py`, with variant members available."""
    return VariantEnsemble(config)

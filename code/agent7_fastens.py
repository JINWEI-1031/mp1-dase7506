"""L4 cost engineering: stop allocating 67 MB per member per window.

WHY THIS EXISTS
---------------
The activated (variant) path was measured +19.57% over the plain path on the same
geometry, and a four-arm alternating-paired run showed the gap is not compute: it
is memory traffic.  Every window and every member touches several freshly
allocated ``[32, 256, 2048]`` fp32 tensors (67 MB each):

    member.predict_log_probs(ids)
      +-- self(ids) = head(features(ids))        head output        67 MB   (alloc)
      +-- F.log_softmax(logits.float(), dim=-1)  normalized logp    67 MB   (alloc)
    Ensemble loop
      +-- torch.logaddexp(total, logp)           accumulator        67 MB   (alloc)
    return total - self.log_size                 final logp         67 MB   (alloc)

Nine 67 MB allocations per window at K=3 (3 heads, 3 log_softmax, 2 logaddexp,
1 final subtract), 46 windows in the validation split, i.e. 414 fresh 67 MB blocks
over one scoring pass.  On glibc a 67 MB ``malloc`` is served by ``mmap`` (the
allocation is above the dynamic mmap threshold), so every one of them is a fresh
anonymous mapping whose pages must be zeroed by the kernel on first touch and
``munmap``-ed on release.

CALIBRATION -- the per-allocation cost is measured, not assumed
---------------------------------------------------------------
Arm C (r5 rope, allocating logaddexp) MAX 35.812 and arm D (r5 + ``out=total``)
MAX 34.493 on the same four-arm probe, so removing 2 allocations per window over 46
windows bought 1.319 s:

    1.319 s / (2 * 46) = 14.3 ms per 67 MB fresh allocation

That number is the whole basis of the estimates in ``方案.md``.  It is an
extrapolation over one measured point, not a measurement of the remaining arms.

THE LADDER
----------
``l4_level`` (env ``MP1_L4_LEVEL`` overrides, checkpoint ``config['l4_level']``
pins) selects how far down the ladder a model runs.  Every level is a strict
superset of the previous one, and level 0 is the unmodified reference path.

    0  reference        original rope, allocating logaddexp / log_softmax / sub
    1  r5 + L3          fast rope, logaddexp(out=)          <-- current best (D arm)
    2  + L4a            log_softmax(out=) writes in place    -3 allocs/window
    3  + L4b            sub_(log_size) in place              -1 alloc/window
    4  + L4c            persistent logits / accumulator      -3 allocs/window
    5  + mallopt        raise the glibc mmap threshold       process-wide

Level 5 is deliberately NOT reached by setting the level number: it is a
process-global allocator setting that also changes how the *evaluator* allocates,
so it needs its own opt-in (``MP1_L4_MALLOPT=1`` or an explicit ``enable_mallopt()``)
and its own peak-RSS reading.

EVERY LEVEL IS SELF-GUARDED ON REAL DATA
----------------------------------------
A speed-up that changes ``nll_nats`` is a different model and is void.  Rather
than assert equivalence from a micro-benchmark, each substitution audits itself
once, on the real tensors of the first scoring window, and falls back permanently
if the audit fails:

  * L4a  ``torch.log_softmax(x, dim, out=x)`` against ``F.log_softmax(x, dim)`` on
         a member's actual logits.
  * L4b  ``total - s`` against ``total.sub_(s)`` on the accumulator.
  * L4c  ``torch.mm(x2, w.t(), out=buf)`` against ``F.linear(x, w)`` -- the exact
         call ``forward`` makes -- on a member's actual features, plus the L4a
         audit on the buffer.  Any member carrying ``softcap`` or ``logit_scale``
         is refused outright: those live in ``forward``, which the buffer path
         bypasses by design.

The fallbacks are not dead code: ``MP1_L4_BREAK_MM=1`` corrupts the buffer between
the kernel and the comparison so the L4c guard is forced to fire, and
``MP1_L4_FORCE_FALLBACK=1`` makes every feature probe report "unsupported".  Both
must still produce a bit-identical ``nll_nats``.

ALIASING CONTRACT (L4c only)
----------------------------
The level 4 path returns the *same* accumulator tensor object on every call, so
the caller must consume it before the next ``predict_log_probs``.  ``evaluate.score``
does: it reads the tensor into ``losses``/``per_window`` and drops it inside the
same loop iteration, holding no reference across windows.  That is a property of
the frozen evaluator (its sha is recorded in every result file), not of this
module; ``MP1_L4_NOREUSE=1`` keeps L4a/L4b/L4c but clones the accumulator before
returning, so the contract can be dropped without losing the rest of the ladder.

Self-contained: imports only ``torch`` plus the neighbouring ``model``/``student``/
``ensemble`` modules, none of which it modifies.  Construction is delegated to
``student.VariantEnsemble.__init__`` untouched -- the fast behaviour is bound onto
the member instances afterwards -- so the parameters and the RNG stream are
identical to the reference package by construction, not by a re-implementation
that could drift.
"""
from __future__ import print_function

import os
import sys
import types

import torch
from torch.nn import functional as F

import student as _student
from student import VariantEnsemble

__all__ = ['build_model', 'FastVariantEnsemble', 'enable_mallopt', 'install_rope',
           'rope_state', 'apply_rope_r5', 'resolve_level', 'LEVELS', 'DEFAULT_LEVEL']


# --------------------------------------------------------------------------- #
# knobs
# --------------------------------------------------------------------------- #
LEVELS = {
    0: 'reference: original rope, allocating logaddexp/log_softmax/sub',
    1: 'r5 rope + logaddexp(out=)              [== current best, arm D]',
    2: '1 + log_softmax(out=) in place',
    3: '2 + sub_(log_size) in place',
    4: '3 + persistent logits/accumulator buffers',
    5: '4 + glibc mallopt(M_MMAP_THRESHOLD)   [process-wide, opt-in]',
}

# Default for a checkpoint that does not pin `l4_level`.  Aggressive on purpose:
# every step is audited on real data at the first call and reverts itself, so an
# unsupported kernel costs one comparison, not a wrong number.
DEFAULT_LEVEL = 4

ENV_LEVEL = 'MP1_L4_LEVEL'
ENV_FORCE_FALLBACK = 'MP1_L4_FORCE_FALLBACK'   # pretend no out=/mm(out=) exists
ENV_BREAK_MM = 'MP1_L4_BREAK_MM'               # corrupt the buffer so the guard fires
ENV_NO_AUDIT = 'MP1_L4_NO_AUDIT'               # skip the one-time equivalence audits
ENV_NOREUSE = 'MP1_L4_NOREUSE'                 # keep the ladder, clone the accumulator
ENV_MALLOPT = 'MP1_L4_MALLOPT'                 # process-wide allocator tweak

M_MMAP_THRESHOLD = -3                          # glibc <malloc.h>: M_MMAP_THRESHOLD
MALLOPT_THRESHOLD_BYTES = 256 * 1024 * 1024


def _flag(name):
    return os.environ.get(name, '').strip().lower() not in ('', '0', 'false', 'no')


def resolve_level(config=None):
    """Effective ``l4_level``: env override, then checkpoint config, then default."""
    raw = os.environ.get(ENV_LEVEL)
    source = ENV_LEVEL
    if raw is None:
        if config is not None and config.get('l4_level') is not None:
            raw = config.get('l4_level')
            source = "config['l4_level']"
        else:
            raw = DEFAULT_LEVEL
            source = 'default'
    try:
        level = int(raw)
    except (TypeError, ValueError):
        raise ValueError('%s=%r is not an integer' % (source, raw))
    if level not in LEVELS:
        raise ValueError('%s=%d is outside 0..%d' % (source, level, max(LEVELS)))
    return level


# --------------------------------------------------------------------------- #
# feature probes
# --------------------------------------------------------------------------- #
# Cached once per process, and forced to False by MP1_L4_FORCE_FALLBACK so the
# fallback branches can be exercised on the same data as the fast ones.
_PROBE = {}


def _probe(name):
    got = _PROBE.get(name)
    if got is None:
        got = _run_probe(name)
        _PROBE[name] = got
    return got


def _run_probe(name):
    if _flag(ENV_FORCE_FALLBACK):
        return False
    try:
        if name == 'log_softmax_out':
            y = torch.randn(4, 8)
            ref = F.log_softmax(y, dim=-1)
            got = torch.log_softmax(y, dim=-1, out=y)
            return bool(got.data_ptr() == y.data_ptr() and torch.equal(ref, got))
        if name == 'mm_out':
            a, b = torch.randn(16, 8), torch.randn(32, 8)
            out = torch.empty(16, 32)
            torch.mm(a, b.t(), out=out)
            return bool(torch.equal(out, F.linear(a, b)))
        if name == 'logaddexp_out':
            a, b = torch.randn(8), torch.randn(8)
            ref = torch.logaddexp(a, b)
            torch.logaddexp(a, b, out=a)
            return bool(torch.equal(ref, a))
    except Exception:                             # noqa: BLE001 -- reported as unsupported
        return False
    raise KeyError(name)


def apply_rope_r5(x, cos, sin, channels=0):
    """``student.apply_rope`` with the two measured savings (the arm C recipe "r5").

    Verified bitwise-identical to the original on contiguous and permuted inputs,
    both full and partial rotation, in ``_shared/_fast_rope.py``.  The entry
    ``contiguous()`` removes the strided ``copy_`` that profiling put at +12.62 ms
    per forward, and the empty ``cat`` -- ``rope_channels`` is 0 on every q3fu
    member, so ``rotate == head_dim`` -- is a pure self-copy of the whole tensor.
    """
    x = x.contiguous()
    rotate = int(channels) or x.shape[-1]
    shape = (1,) * (x.dim() - 2) + cos.shape
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    even, odd = x[..., 0:rotate:2], x[..., 1:rotate:2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
    if rotate == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., rotate:]], dim=-1)


# `apply_rope` is resolved as a MODULE GLOBAL at call time by
# `student._variant_block_attn`, so level 1 is a process-wide binding, not a
# per-model one -- the same mechanism the validated arm C/D probe uses.  That
# makes it possible to leave a reference arm silently running the fast rope, so
# the state is readable (`rope_state`) and every measurement script must assert
# it immediately before the scoring call it belongs to.
_ROPE_ORIGINAL = _student.apply_rope


def rope_state():
    """``'r5'``, ``'original'``, or ``'foreign'`` if something else rebound it."""
    current = _student.apply_rope
    if current is apply_rope_r5:
        return 'r5'
    if current is _ROPE_ORIGINAL:
        return 'original'
    return 'foreign'


def install_rope(level=None, config=None):
    """Bind ``student.apply_rope`` to the rope this ``level`` requires."""
    level = resolve_level(config) if level is None else int(level)
    _student.apply_rope = apply_rope_r5 if level >= 1 else _ROPE_ORIGINAL
    return rope_state()


def enable_mallopt(threshold=MALLOPT_THRESHOLD_BYTES):
    """Raise glibc's mmap threshold so 67 MB blocks are recycled, not remapped.

    A 67 MB ``malloc`` on glibc is above the dynamic mmap threshold (32 MB max on
    64-bit), so it becomes an anonymous mapping: the kernel zeroes every page on
    first touch, and ``munmap`` throws the mapping away on free.  Raising the
    threshold keeps those blocks on the main heap, where a freed block is reused
    with its pages already faulted in.

    Process-wide, and it also covers the evaluator's own allocations, which is
    exactly why it is opt-in rather than reachable through the level number.
    Returns a human-readable status string and never raises.
    """
    if not sys.platform.startswith('linux'):
        return 'skipped: platform %s is not linux' % sys.platform
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.mallopt.restype = ctypes.c_int
        rc = libc.mallopt(ctypes.c_int(M_MMAP_THRESHOLD), ctypes.c_int(int(threshold)))
        if rc:
            return 'ok (M_MMAP_THRESHOLD=%d)' % int(threshold)
        return 'refused: mallopt returned 0'
    except Exception as exc:                      # noqa: BLE001 -- reported, never fatal
        return 'failed: %s: %s' % (type(exc).__name__, exc)


if _flag(ENV_MALLOPT):
    print('[agent7_fastens] mallopt at import: %s' % enable_mallopt(), file=sys.stderr)


# --------------------------------------------------------------------------- #
# member-level fast paths (bound onto the instances, never onto the class)
# --------------------------------------------------------------------------- #
def _member_predict_log_probs(self, ids):
    """``model.GPT.predict_log_probs`` with the log_softmax output written in place.

    ``self(ids)`` owns the logits tensor -- nothing else holds a reference -- and
    ``head`` shares its weight with ``token``, so writing over the logits cannot
    touch a parameter.  ``F.log_softmax`` and ``torch.log_softmax`` are the same
    ATen op; only the latter exposes the ``out=`` overload, because ``F.log_softmax``
    rejects the keyword outright (TypeError, probed on torch 2.12).
    """
    level = getattr(self, '_l4_level', DEFAULT_LEVEL)
    logits = self(ids)
    if level >= 2 and logits.dtype == torch.float32 and _probe('log_softmax_out'):
        audit = getattr(self, '_l4_audit_logsm', False)
        if audit:
            self._l4_audit_logsm = False
            ref = F.log_softmax(logits, dim=-1)                 # the reference impl
            out = torch.log_softmax(logits, dim=-1, out=logits)
            if not torch.equal(ref, out):
                print('[agent7_fastens] L4a audit FAILED on %s -> level 1'
                      % type(self).__name__, file=sys.stderr)
                self._l4_level = 1
                return ref
            return out
        return torch.log_softmax(logits, dim=-1, out=logits)
    return F.log_softmax(logits.float(), dim=-1)


def _member_logits_into(self, ids, buf, guard):
    """Write the raw head output for ``ids`` into the ``[batch, time, vocab]`` ``buf``.

    Bypasses ``forward`` on purpose, so a member whose ``forward`` is not a bare
    head -- ``softcap`` or ``logit_scale`` -- must be refused by the caller rather
    than silently scored without its nonlinearity.  Returns True on success.
    """
    x = self.features(ids)
    weight = self.head.weight
    x2 = x.reshape(-1, x.shape[-1])
    buf2 = buf.view(-1, buf.shape[-1])
    if guard:
        reference = F.linear(x, weight)      # exactly what forward() would compute
        torch.mm(x2, weight.t(), out=buf2)
        if _flag(ENV_BREAK_MM):
            buf.view(-1)[0] = float('nan')
        return bool(torch.equal(reference, buf))
    torch.mm(x2, weight.t(), out=buf2)
    return True


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
class FastVariantEnsemble(VariantEnsemble):
    """``student.VariantEnsemble`` with the L4 ladder bound onto its instances.

    ``VariantEnsemble.__init__`` runs untouched, so the members, their parameters
    and the RNG stream are bit-identical to the reference package: the only
    difference is which functions the member and ensemble instances dispatch to.
    A level therefore cannot change a number by way of construction.
    """

    def __init__(self, config):
        super().__init__(config)
        self._l4_level = resolve_level(config)
        self._l4_state = {'disabled': False, 'shapes': {},
                          'audit': not _flag(ENV_NO_AUDIT)}
        # Level 1 is the rope; it is a module global, so binding it here is what
        # makes `build_model` self-contained.  `student.build_model` never resets
        # it, which is why the measurement scripts assert `rope_state()` instead
        # of assuming a fresh build restores the reference.
        self._l4_rope = install_rope(self._l4_level)
        for member in self.members:
            member._l4_level = self._l4_level
            member._l4_audit_logsm = bool(self._l4_level >= 2 and self._l4_state['audit'])
            member.predict_log_probs = types.MethodType(_member_predict_log_probs, member)
            member._logits_into = types.MethodType(_member_logits_into, member)

    # -- L4c support -------------------------------------------------------- #
    @staticmethod
    def _refusal(member):
        """L4c cannot serve a member whose ``forward`` is more than ``head(features)``."""
        flags = getattr(member, 'flags', None)
        if not flags:
            return None
        if flags.get('softcap') or flags.get('logit_scale'):
            return 'member carries softcap/logit_scale'
        return None

    def _entry(self, ids):
        """Per-shape scratch for L4c, or None when the path must be refused."""
        vocab = self.members[0].head.weight.shape[0]
        key = (ids.shape[0], ids.shape[1], vocab, str(ids.device))
        entry = self._l4_state['shapes'].get(key)
        if entry is not None:
            return entry
        for member in self.members:
            why = self._refusal(member)
            if why is not None:
                print('[agent7_fastens] L4c refused: %s' % why, file=sys.stderr)
                self._l4_state['disabled'] = True
                return None
        if not _probe('mm_out'):
            self._l4_state['disabled'] = True
            return None
        shape = (ids.shape[0], ids.shape[1], vocab)
        entry = {'buf': torch.empty(shape, dtype=torch.float32, device=ids.device),
                 'total': torch.empty(shape, dtype=torch.float32, device=ids.device),
                 'guard': True,
                 'audit_logsm': bool(self._l4_state['audit']),
                 'audit_sub': bool(self._l4_state['audit']),
                 'noreuse': _flag(ENV_NOREUSE)}
        self._l4_state['shapes'][key] = entry
        return entry

    def _disable(self, why):
        print('[agent7_fastens] %s -> falling back to the allocating path' % why,
              file=sys.stderr)
        self._l4_state['disabled'] = True
        self._l4_state['shapes'] = {}

    def _run(self, ids):
        """Level 4: every member writes into one warm buffer. None means 'fall back'."""
        entry = self._entry(ids)
        if entry is None:
            return None
        buf, total = entry['buf'], entry['total']
        for index, member in enumerate(self.members):
            # Member 0 writes straight into the accumulator, so the running total
            # never has to be copied out of a scratch pad: one 67 MB copy per
            # window is exactly the kind of traffic this level exists to remove.
            dest = total if index == 0 else buf
            dest2 = dest.view(-1, dest.shape[-1])
            if not member._logits_into(ids, dest, entry['guard']):
                self._disable('L4c guard FAILED')
                return None
            if entry['audit_logsm']:
                entry['audit_logsm'] = False
                reference = F.log_softmax(dest2, dim=-1)
                torch.log_softmax(dest2, dim=-1, out=dest2)
                if not torch.equal(reference, dest2):
                    self._disable('L4a-on-buffer audit FAILED')
                    return None
            elif _probe('log_softmax_out'):
                torch.log_softmax(dest2, dim=-1, out=dest2)
            else:
                dest2.copy_(F.log_softmax(dest2, dim=-1))
            entry['guard'] = False
            if index:
                torch.logaddexp(total, buf, out=total)
        if entry['noreuse']:
            total = total.clone()          # break the aliasing contract, same numbers
        if self._l4_level >= 3:
            if entry['audit_sub']:
                entry['audit_sub'] = False
                reference = total - self.log_size
                total.sub_(self.log_size)
                if not torch.equal(reference, total):
                    self._disable('L4b audit FAILED')
                    return None
            else:
                total.sub_(self.log_size)
        else:
            total = total - self.log_size
        return total

    # -- evaluation interface ----------------------------------------------- #
    def predict_log_probs(self, ids):
        if self.size == 1:
            return self.members[0].predict_log_probs(ids)
        level = self._l4_level
        if level >= 4 and not self._l4_state['disabled']:
            got = self._run(ids)
            if got is not None:
                return got
        total = None
        for member in self.members:
            logp = member.predict_log_probs(ids)
            if total is None:
                total = logp
            elif level >= 1 and _probe('logaddexp_out'):
                torch.logaddexp(total, logp, out=total)
            else:
                total = torch.logaddexp(total, logp)
        if level >= 3:
            total.sub_(self.log_size)
            return total
        return total - self.log_size

    # -- cache invalidation -------------------------------------------------- #
    def _apply(self, fn, *args, **kwargs):
        state = self.__dict__.get('_l4_state')
        if isinstance(state, dict):
            state['shapes'] = {}
        return super()._apply(fn, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = self.__dict__.get('_l4_state')
        if isinstance(state, dict):
            state['shapes'] = {}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_model(config):
    """Model factory: a drop-in for ``student.build_model`` at every level."""
    return FastVariantEnsemble(config)

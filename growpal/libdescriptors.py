"""
Fast MBTR descriptors for monocomponent atomic clusters.

Drop-in replacement for the dscribe-based MBTR functions in libdescriptors.py.
No dscribe dependency.  Uses numba @njit(parallel=True) over structures so
the per-process monkey-patch issue with dscribe 2.1.2 / ASE 3.27.0 is
completely avoided.

Matches dscribe MBTR settings used in libdescriptors.py:
  MBTR_dis : geometry=distance,     grid=[0, 10],        sigma=1e-2, n=200
             weighting=inverse_square (1/r^2), r_cut=10
             normalize_gaussians=True, normalization='none'
  MBTR_cos : geometry=cosine,       grid=[-1.05, 1.05],  sigma=1e-2, n=200
             weighting=exp(scale=0.5), threshold=1e-3
             normalize_gaussians=True, normalization='none'

Implementation notes
--------------------
* Gaussian broadening uses the same CDF bin-average method as dscribe's C++
  backend: each grid bin gets the integral of the Gaussian over that bin
  divided by the bin width, computed via erf differences. This matches
  dscribe exactly (unlike naive point evaluation which diverges when sigma < dx).
* Only bins within a precomputed window around each peak are updated (erf
  saturates outside). This makes inner loops O(window) instead of O(n_grid).
* k3 weighting uses the full triangle perimeter: exp(-scale*(r_ij+r_ik+r_jk)),
  matching dscribe's k3WeightExponential.
* Structures are packed into a contiguous (total_atoms, 3) array; numba
  prange parallelises over structures.

Public API
----------
mbtr_comb_fast(atoms_list)  -> (N, 400) float64 array  [dis || cos]
"""

import math
import numpy as np
import numba as nb

# ---------------------------------------------------------------------------
# Constants — mirror dscribe parameter choices exactly
# ---------------------------------------------------------------------------
_GRID_DIS   = np.linspace(0.0,   10.0, 200, dtype=np.float64)
_GRID_COS   = np.linspace(-1.05,  1.05, 200, dtype=np.float64)
_SIGMA_DIS  = 1e-2
_SIGMA_COS  = 1e-2
_RCUT_DIS   = 10.0
_RCUT_COS   = 10.0
_SCALE_COS  = 0.5
_THRESH_COS = 1e-3

# dx for each grid (used to compute window sizes below)
_DX_DIS = (_GRID_DIS[-1] - _GRID_DIS[0]) / (len(_GRID_DIS) - 1)   # ≈ 0.0503
_DX_COS = (_GRID_COS[-1] - _GRID_COS[0]) / (len(_GRID_COS) - 1)   # ≈ 0.01055

# sigma*sqrt(2) (argument denominator for erf in CDF formula)
_S2_DIS = _SIGMA_DIS * math.sqrt(2.0)
_S2_COS = _SIGMA_COS * math.sqrt(2.0)

# Window: number of bins on each side of a peak to update.
# Covers +/- 6*sigma; erf saturates beyond this, contributing nothing.
_WIN_DIS = max(3, int(math.ceil(6.0 * _SIGMA_DIS / _DX_DIS)) + 1)   # ≈ 2 → 3
_WIN_COS = max(3, int(math.ceil(6.0 * _SIGMA_COS / _DX_COS)) + 1)   # ≈ 6 → 7

# Grid min values (used to find bin index for each geometry value)
_GMIN_DIS = float(_GRID_DIS[0])
_GMIN_COS = float(_GRID_COS[0])


# ---------------------------------------------------------------------------
# JIT helper — add one CDF-based Gaussian peak into output array
#
# Uses the same formula as dscribe's C++ mbtr.cpp::gaussian():
#   cdf[i] = weight * 0.5 * (1 + erf((x_i - center) / sigmasqrt2))
#   out[k] += (cdf[k+1] - cdf[k]) / dx
# where x_i = grid_min + (i - 0.5) * dx  (bin boundaries).
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _add_peak(out, geom, weight, grid_min, dx, sigmasqrt2, window):
    """Accumulate one Gaussian peak into out[] using the CDF bin-average method.

    Only updates bins in the range [k_center - window, k_center + window].
    Safe when the peak is near the grid boundaries (clamped to [0, n-1]).

    in:
        out      : 1-D float64 output array of length n_grid (modified in-place).
        geom     : geometry value (distance or cosine) = Gaussian centre.
        weight   : Gaussian area (weighting function value for this pair/triplet).
        grid_min : minimum grid value (grid[0]).
        dx       : grid bin width.
        sigmasqrt2: sigma * sqrt(2)  (denominator in erf argument).
        window   : number of bins on each side to update.
    """
    n_grid   = out.shape[0]
    k_center = int((geom - grid_min) / dx + 0.5)
    k_lo     = k_center - window
    if k_lo < 0:
        k_lo = 0
    k_hi = k_center + window
    if k_hi >= n_grid:
        k_hi = n_grid - 1

    # CDF at left boundary of bin k_lo
    x_prev   = grid_min + (k_lo - 0.5) * dx
    cdf_prev = weight * 0.5 * (1.0 + math.erf((x_prev - geom) / sigmasqrt2))

    for k in range(k_lo, k_hi + 1):
        x_next   = grid_min + (k + 0.5) * dx
        cdf_next = weight * 0.5 * (1.0 + math.erf((x_next - geom) / sigmasqrt2))
        out[k]  += (cdf_next - cdf_prev) / dx
        cdf_prev = cdf_next


# ---------------------------------------------------------------------------
# JIT kernel — single structure, MBTR_dis (2-body distance)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _mbtr_dis_single(pos, grid_min, dx, sigmasqrt2, rcut, window):
    """2-body MBTR distance descriptor for one structure (monocomponent).

    Iterates over all unordered pairs (i < j).
    Weighting: w = 1 / r_ij^2  (inverse_square).
    Gaussian: CDF bin-average (matches dscribe normalize_gaussians=True).

    in:
        pos        : (n_atoms, 3) float64 atomic positions.
        grid_min   : minimum grid value.
        dx         : grid bin width.
        sigmasqrt2 : sigma * sqrt(2).
        rcut       : distance cutoff.
        window     : CDF bin window half-width.
    out:
        1-D float64 array of length 200.
    """
    n_atoms = pos.shape[0]
    out     = np.zeros(200)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dxij = pos[i, 0] - pos[j, 0]
            dyij = pos[i, 1] - pos[j, 1]
            dzij = pos[i, 2] - pos[j, 2]
            r2   = dxij*dxij + dyij*dyij + dzij*dzij
            if r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            if r >= rcut:
                continue
            w = 1.0 / r2          # inverse_square weight
            _add_peak(out, r, w, grid_min, dx, sigmasqrt2, window)

    return out


# ---------------------------------------------------------------------------
# JIT kernel — single structure, MBTR_cos (3-body cosine angle)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True)
def _mbtr_cos_single(pos, grid_min, dx, sigmasqrt2, rcut, scale, thresh, window):
    """3-body MBTR cosine descriptor for one structure (monocomponent).

    For each centre atom i and ordered pair (j, k) with j < k, j != i, k != i:
        cos(theta) = dot(pos[j]-pos[i], pos[k]-pos[i]) / (r_ij * r_ik)
        w          = exp(-scale * (r_ij + r_ik + r_jk))   [full perimeter]
    Gaussian: CDF bin-average (matches dscribe normalize_gaussians=True).

    The full-perimeter weighting and centre-angle cosine together reproduce
    dscribe's k3WeightExponential + k3GeomCosine for a monocomponent system:
    each unique triple {a,b,c} contributes three angles (one per vertex),
    all with the same full-perimeter weight, which is identical to dscribe's
    (i=arm1, j=centre, k=arm2) iteration over ordered triples.

    in:
        pos        : (n_atoms, 3) float64 atomic positions.
        grid_min   : minimum grid value.
        dx         : grid bin width.
        sigmasqrt2 : sigma * sqrt(2).
        rcut       : distance cutoff for arms from centre.
        scale      : exponential weighting scale.
        thresh     : weight threshold (skip if w < thresh).
        window     : CDF bin window half-width.
    out:
        1-D float64 array of length 200.
    """
    n_atoms = pos.shape[0]
    out     = np.zeros(200)

    for i in range(n_atoms):               # centre atom
        for j in range(n_atoms):           # first arm
            if j == i:
                continue
            dxij = pos[j, 0] - pos[i, 0]
            dyij = pos[j, 1] - pos[i, 1]
            dzij = pos[j, 2] - pos[i, 2]
            r2ij = dxij*dxij + dyij*dyij + dzij*dzij
            if r2ij < 1e-20:
                continue
            r_ij = math.sqrt(r2ij)
            if r_ij >= rcut:
                continue

            for k in range(j + 1, n_atoms):   # second arm (j < k avoids double-count)
                if k == i:
                    continue
                dxik = pos[k, 0] - pos[i, 0]
                dyik = pos[k, 1] - pos[i, 1]
                dzik = pos[k, 2] - pos[i, 2]
                r2ik = dxik*dxik + dyik*dyik + dzik*dzik
                if r2ik < 1e-20:
                    continue
                r_ik = math.sqrt(r2ik)
                if r_ik >= rcut:
                    continue

                # arm-to-arm distance (needed for full perimeter weight)
                dxjk = pos[k, 0] - pos[j, 0]
                dyjk = pos[k, 1] - pos[j, 1]
                dzjk = pos[k, 2] - pos[j, 2]
                r_jk = math.sqrt(dxjk*dxjk + dyjk*dyjk + dzjk*dzjk)

                # full perimeter weighting (matches dscribe k3WeightExponential)
                w = math.exp(-scale * (r_ij + r_ik + r_jk))
                if w < thresh:
                    continue

                # cosine of angle at centre i (law of cosines, dot-product form)
                cos_t = (dxij*dxik + dyij*dyik + dzij*dzik) / (r_ij * r_ik)
                if cos_t >  1.0: cos_t =  1.0
                if cos_t < -1.0: cos_t = -1.0

                _add_peak(out, cos_t, w, grid_min, dx, sigmasqrt2, window)

    return out


# ---------------------------------------------------------------------------
# Batch kernels — parallel over structures (packed-array representation)
# ---------------------------------------------------------------------------

@nb.njit(cache=True, fastmath=True, parallel=True)
def _mbtr_dis_batch(pos_flat, offsets, grid_min, dx, sigmasqrt2, rcut, window):
    """Compute MBTR_dis for N structures in parallel with nb.prange.

    in:
        pos_flat (float64, shape (total_atoms, 3)): all positions packed.
        offsets  (int64,   shape (N+1,))          : cumulative atom counts.
        grid_min, dx, sigmasqrt2, rcut, window    : descriptor parameters.
    out:
        float64 array of shape (N, 200).
    """
    N   = offsets.shape[0] - 1
    out = np.zeros((N, 200))
    for s in nb.prange(N):
        pos    = pos_flat[offsets[s]:offsets[s + 1], :]
        out[s] = _mbtr_dis_single(pos, grid_min, dx, sigmasqrt2, rcut, window)
    return out


@nb.njit(cache=True, fastmath=True, parallel=True)
def _mbtr_cos_batch(pos_flat, offsets, grid_min, dx, sigmasqrt2, rcut, scale, thresh, window):
    """Compute MBTR_cos for N structures in parallel with nb.prange.

    in:
        pos_flat (float64, shape (total_atoms, 3)): all positions packed.
        offsets  (int64,   shape (N+1,))          : cumulative atom counts.
        grid_min, dx, sigmasqrt2, rcut, scale, thresh, window: parameters.
    out:
        float64 array of shape (N, 200).
    """
    N   = offsets.shape[0] - 1
    out = np.zeros((N, 200))
    for s in nb.prange(N):
        pos    = pos_flat[offsets[s]:offsets[s + 1], :]
        out[s] = _mbtr_cos_single(pos, grid_min, dx, sigmasqrt2, rcut, scale, thresh, window)
    return out


# ---------------------------------------------------------------------------
# Internal helper — pack an ASE Atoms list into contiguous numpy arrays
# ---------------------------------------------------------------------------

def _pack_positions(atoms_list):
    """Extract positions from a list of ASE Atoms into packed arrays.

    in:
        atoms_list (list of ase.Atoms): structures to pack.
    out:
        pos_flat (float64 ndarray, shape (total_atoms, 3)): C-contiguous.
        offsets  (int64 ndarray,   shape (N+1,))          : atom count offsets.
    """
    counts  = np.array([len(a) for a in atoms_list], dtype=np.int64)
    offsets = np.zeros(len(atoms_list) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    pos_flat = np.empty((int(offsets[-1]), 3), dtype=np.float64)
    for s, atoms in enumerate(atoms_list):
        pos_flat[offsets[s]:offsets[s + 1]] = atoms.get_positions()
    return np.ascontiguousarray(pos_flat), offsets


def mbtr_comb_fast(atoms_list):
    """Compute concatenated [MBTR_dis | MBTR_cos] descriptors — parallel, no dscribe.

    Packs positions once and runs both batch kernels on the shared array.

    in:
        atoms_list (list of ase.Atoms): monocomponent cluster structures.
    out:
        numpy.ndarray: (N, 400) float64 descriptor matrix.
    """
    pos_flat, offsets = _pack_positions(atoms_list)
    dis = _mbtr_dis_batch(pos_flat, offsets,
                          _GMIN_DIS, _DX_DIS, _S2_DIS, _RCUT_DIS, _WIN_DIS)
    cos = _mbtr_cos_batch(pos_flat, offsets,
                          _GMIN_COS, _DX_COS, _S2_COS, _RCUT_COS,
                          _SCALE_COS, _THRESH_COS, _WIN_COS)
    return np.hstack([dis, cos])


# ---------------------------------------------------------------------------
# JIT warm-up — pre-compile both kernels on import (numba caches after first run)
# ---------------------------------------------------------------------------

def _warmup():
    """Trigger JIT compilation with a minimal 3-atom dummy structure.

    Adds a few seconds on first import; subsequent imports load the cache
    from __pycache__ instantly.
    """
    dummy_pos = np.ascontiguousarray(
        np.array([[0.0, 0.0, 0.0],
                  [1.5, 0.0, 0.0],
                  [0.0, 1.5, 0.0]], dtype=np.float64)
    )
    dummy_off = np.array([0, 3], dtype=np.int64)
    _mbtr_dis_batch(dummy_pos, dummy_off,
                    _GMIN_DIS, _DX_DIS, _S2_DIS, _RCUT_DIS, _WIN_DIS)
    _mbtr_cos_batch(dummy_pos, dummy_off,
                    _GMIN_COS, _DX_COS, _S2_COS, _RCUT_COS,
                    _SCALE_COS, _THRESH_COS, _WIN_COS)


_warmup()


# ---------------------------------------------------------------------------
# dscribe-based descriptors for multi-species (non-monoatomic) systems
# ---------------------------------------------------------------------------

def mbtr_comb_dscribe(atoms_list, n_jobs=1):
    """Compute concatenated [MBTR_dis | MBTR_cos] descriptors via dscribe.

    Builds a unified species space across all structures so every row of the
    output shares the same feature dimension, then evaluates both MBTR kernels
    in a single pass.

    Matches the parameter choices of the numba fast path:
      MBTR_dis : geometry=distance,   grid=[0,10],       sigma=1e-2, n=200
                 weighting=inverse_square, r_cut=10
      MBTR_cos : geometry=cosine,     grid=[-1.05,1.05], sigma=1e-2, n=200
                 weighting=exp(scale=0.5), threshold=1e-3

    in:
        atoms_list (list of ase.Atoms): structures — any mix of species.
        n_jobs     (int)              : parallel workers passed to dscribe
                                        create(); 1 = serial (default).
    out:
        numpy.ndarray: (N, F) float64 descriptor matrix where F depends on
        the number of unique species (F=400 for monoatomic, larger otherwise).
    """
    import numpy as np
    from dscribe.descriptors import MBTR

    species = sorted(set(sym
                         for atoms in atoms_list
                         for sym in atoms.get_chemical_symbols()))

    mbtr_dis = MBTR(
        species=species,
        geometry={"function": "distance"},
        grid={"min": 0, "max": 10, "sigma": 1e-2, "n": 200},
        weighting={"function": "inverse_square", "r_cut": 10, "threshold": 1e-3},
        periodic=False, normalization="none",
        normalize_gaussians=True, sparse=False, dtype="float64",
    )
    mbtr_cos = MBTR(
        species=species,
        geometry={"function": "cosine"},
        grid={"min": -1.05, "max": 1.05, "sigma": 1e-2, "n": 200},
        weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        periodic=False, normalization="none",
        normalize_gaussians=True, sparse=False, dtype="float64",
    )

    dis = mbtr_dis.create(atoms_list, n_jobs=n_jobs)
    cos = mbtr_cos.create(atoms_list, n_jobs=n_jobs)
    return np.hstack([dis, cos])

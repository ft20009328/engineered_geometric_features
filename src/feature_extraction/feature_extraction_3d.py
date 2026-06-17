"""
feature_extraction_3d.py
========================
Automated extraction of 3D geometric and statistical features from the
voxelized geometry of a stochastic nanoporous thin-film solar cell.

INPUT  : 3D numpy array `im_solid` (nx, ny, nz); True=solid a-Si, False=pore
OUTPUT : ordered dict of named 3D features for one structure.

All features are 3D-native. 2D shape descriptors from the authors' earlier
2D work are replaced by their 3D analogues (area->volume, perimeter->surface
area, eccentricity->anisotropy, form factor->sphericity, etc.). See the
revised Table 2 / Appendix II for the matching feature list.

Dependencies: numpy, scipy, scikit-image, porespy
Author: F. Tabassum
"""

import numpy as np
from scipy.spatial import cKDTree, Voronoi, ConvexHull
from scipy.spatial.distance import pdist
from skimage.measure import (label, regionprops, marching_cubes,
                             mesh_surface_area)

try:
    import porespy as ps
    _HAS_PORESPY = True
except ImportError:
    _HAS_PORESPY = False


def _agg(name, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{name}_mean": 0.0, f"{name}_var": 0.0}
    return {f"{name}_mean": float(np.mean(arr)),
            f"{name}_var": float(np.var(arr))}


def _pore_label(im_solid):
    pore = ~np.asarray(im_solid).astype(bool)
    return label(pore), pore


def morphological_features(im_solid, voxel_size=1.0):
    labels, _ = _pore_label(im_solid)
    regions = regionprops(labels)
    if not regions:
        return {"n_pores": 0.0}
    vol, surf, sph, eq_diam, sol, ext, cvx, rect = [], [], [], [], [], [], [], []
    major, minor, elong, aniso, aspect = [], [], [], [], []
    feret_max, feret_min, moment_inv, orient = [], [], [], []
    vs = voxel_size
    for r in regions:
        V = r.area * vs**3
        vol.append(V)
        try:
            sub = np.pad(r.image.astype(float), 1)
            verts, faces, _, _ = marching_cubes(sub, level=0.5)
            S = mesh_surface_area(verts, faces) * vs**2
        except Exception:
            S = np.nan
        surf.append(S)
        if S and np.isfinite(S) and S > 0:
            sph.append((np.pi**(1/3) * (6*V)**(2/3)) / S)
        eq_diam.append(r.equivalent_diameter_area * vs)
        sol.append(r.solidity); ext.append(r.extent)
        cvx.append(r.area_convex * vs**3); rect.append(r.extent)
        try:
            ev = np.sort(np.asarray(r.inertia_tensor_eigvals))[::-1]
            Lm, Ln = ev[0], ev[-1]
            major.append(Lm); minor.append(Ln)
            elong.append(1.0 - Ln/(Lm+1e-12))
            aniso.append((Lm-Ln)/(Lm+Ln+1e-12))
            aspect.append(Lm/(Ln+1e-12))
        except Exception:
            pass
        coords = np.argwhere(r.image) * vs
        if coords.shape[0] > 1:
            c = coords - coords.mean(0)
            try:
                _, Vt = np.linalg.eigh(np.cov(c.T))
                proj = c @ Vt; spans = proj.max(0) - proj.min(0)
                feret_max.append(spans.max()); feret_min.append(spans.min())
            except Exception:
                pass
        try:
            mu = r.moments_central
            moment_inv.append(float(mu[2,0,0]+mu[0,2,0]+mu[0,0,2]))
        except Exception:
            pass
        try:
            vec = np.linalg.eigh(np.cov(np.argwhere(r.image).T))[1][:, -1]
            orient.append(np.arctan2(vec[1], vec[0]))
        except Exception:
            pass
    f = {"n_pores": float(len(regions))}
    f.update(_agg("pore_volume", vol)); f.update(_agg("surface_area", surf))
    f.update(_agg("sphericity", sph)); f.update(_agg("equiv_diameter", eq_diam))
    f.update(_agg("solidity", sol)); f.update(_agg("extent", ext))
    f.update(_agg("convex_volume", cvx)); f.update(_agg("rectangularity", rect))
    f.update(_agg("major_axis", major)); f.update(_agg("minor_axis", minor))
    f.update(_agg("elongation", elong)); f.update(_agg("anisotropy", aniso))
    f.update(_agg("aspect_ratio", aspect))
    f.update(_agg("feret_max", feret_max)); f.update(_agg("feret_min", feret_min))
    f.update(_agg("moment_invariant", moment_inv))
    f.update(_agg("orientation_angle", orient))
    sav = np.asarray(surf)/(np.asarray(vol)+1e-12)
    f.update(_agg("surface_to_volume", sav))
    return f


def distribution_features(im_solid, voxel_size=1.0):
    labels, _ = _pore_label(im_solid)
    regions = regionprops(labels)
    cents = np.array([r.centroid for r in regions]) * voxel_size
    f = {}
    if len(cents) < 2:
        return {"nn_distance_mean": 0.0, "nn_distance_var": 0.0,
                "pairwise_distance_mean": 0.0, "clustering_index": 0.0}
    tree = cKDTree(cents)
    d, _ = tree.query(cents, k=2)
    f.update(_agg("nn_distance", d[:, 1]))
    f.update(_agg("pairwise_distance", pdist(cents)))
    f["centroid_spread_x"] = float(np.var(cents[:, 0]))
    f["centroid_spread_y"] = float(np.var(cents[:, 1]))
    f["centroid_spread_z"] = float(np.var(cents[:, 2]))
    try:
        vor = Voronoi(cents); vols = []
        for ri in vor.point_region:
            vv = vor.regions[ri]
            if -1 in vv or len(vv) == 0:
                continue
            try:
                vols.append(ConvexHull(vor.vertices[vv]).volume)
            except Exception:
                pass
        f.update(_agg("voronoi_volume", vols))
    except Exception:
        f.update(_agg("voronoi_volume", []))
    counts = np.asarray(tree.query_ball_point(cents, r=np.mean(d[:,1])*2.0,
                                              return_length=True), dtype=float)
    f["clustering_index"] = float(np.std(counts)/(np.mean(counts)+1e-12))
    f["particle_density_variation"] = float(np.var(counts))
    return f


def statistical_features(im_solid, voxel_size=1.0, n_bins=10):
    pore = ~np.asarray(im_solid).astype(bool)
    f = {"porosity": float(pore.sum()/pore.size)}
    if not _HAS_PORESPY:
        return f
    try:
        tpc = ps.metrics.two_point_correlation(pore, voxel_size=voxel_size)
        prob, dist = np.asarray(tpc.probability), np.asarray(tpc.distance)
        f["two_point_corr_short"] = float(prob[1]) if prob.size > 1 else float(prob[0])
        f["two_point_corr_decay"] = float(dist[np.argmin(np.abs(prob-f["porosity"]))])
    except Exception:
        pass
    try:
        cl = ps.filters.apply_chords(pore, axis=0)
        cld = ps.metrics.chord_length_distribution(cl, bins=n_bins)
        ctr, pdf = np.asarray(cld.bin_centers)*voxel_size, np.asarray(cld.pdf)
        m = np.sum(ctr*pdf)/(np.sum(pdf)+1e-12)
        v = np.sum((ctr-m)**2*pdf)/(np.sum(pdf)+1e-12)
        f["chord_length_mean"] = float(m); f["chord_length_std"] = float(np.sqrt(v))
    except Exception:
        pass
    try:
        psd = ps.metrics.pore_size_distribution(
            ps.filters.local_thickness(pore), bins=n_bins, log=False)
        ctr, pdf = np.asarray(psd.bin_centers)*voxel_size, np.asarray(psd.pdf)
        f["pore_size_mean"] = float(np.sum(ctr*pdf)/(np.sum(pdf)+1e-12))
    except Exception:
        pass
    try:
        nx, ny, nz = pore.shape; sub = []
        for ix in (slice(0,nx//2), slice(nx//2,nx)):
            for iy in (slice(0,ny//2), slice(ny//2,ny)):
                for iz in (slice(0,nz//2), slice(nz//2,nz)):
                    b = pore[ix,iy,iz]; sub.append(b.sum()/b.size)
        f["rev_porosity_var"] = float(np.var(sub))
    except Exception:
        pass
    return f


def extract_features(im_solid, voxel_size=1.0):
    im_solid = np.asarray(im_solid).astype(bool)
    f = {}
    f.update(statistical_features(im_solid, voxel_size))
    f.update(morphological_features(im_solid, voxel_size))
    f.update(distribution_features(im_solid, voxel_size))
    return f


def extract_dataset(structures, voxel_size=1.0):
    return [extract_features(im, voxel_size) for im in structures]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 44
    solid = np.ones((n, n, n), dtype=bool)
    for _ in range(25):
        cx, cy, cz = rng.integers(0,n), rng.integers(0,n), rng.integers(n//2,n)
        rad = rng.integers(2,4)
        xx, yy, zz = np.ogrid[:n, :n, :n]
        solid[(xx-cx)**2+(yy-cy)**2+(zz-cz)**2 <= rad**2] = False
    feats = extract_features(solid, voxel_size=1.0)
    print(f"Extracted {len(feats)} 3D features:\n")
    for k, v in feats.items():
        print(f"  {k:28s} = {v:.4f}")

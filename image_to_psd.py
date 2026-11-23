#!/usr/bin/env python3
"""
image_to_psd.py — Convertit n'importe quelle image JPEG / PNG en fichier PSD
Photoshop avec des calques séparés et modifiables.

Calques générés (de haut en bas dans Photoshop) :
  - "Contours"            : trait / line-art extrait de l'image (mode Produit)
  - Groupe "Sujet"        : (option --ai) le sujet détouré par IA
  - Groupe "Couleurs"     : un calque par zone de couleur (k-means), nommé
                            avec sa couleur hexadécimale. Option --regions
                            pour découper chaque couleur en zones séparées.
  - "Arrière-plan"        : (option --ai) le fond sans le sujet
  - "Image originale"     : l'image d'origine (masquée par défaut), référence

Installation :
    pip install psd-tools pillow numpy
    pip install scipy          # optionnel, pour --regions
    pip install rembg          # optionnel, pour --ai (détourage du sujet)

Exemples :
    python image_to_psd.py photo.jpg
    python image_to_psd.py logo.png -c 6 -o logo_calques.psd
    python image_to_psd.py portrait.jpg --ai -c 10 --regions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.constants import BlendMode
except ImportError:  # pragma: no cover
    sys.exit("psd-tools est requis : pip install psd-tools")


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def kmeans_colors(pixels: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """K-means simple (numpy) sur des pixels RGB float32 -> centres (k, 3)."""
    rng = np.random.default_rng(seed)
    sample = pixels if len(pixels) <= 60_000 else pixels[rng.choice(len(pixels), 60_000, replace=False)]

    # Initialisation k-means++
    centers = [sample[rng.integers(len(sample))]]
    for _ in range(1, k):
        d = np.min(((sample[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d.sum() == 0:
            break
        centers.append(sample[rng.choice(len(sample), p=d / d.sum())])
    centers = np.array(centers, dtype=np.float32)

    for _ in range(iters):
        labels = assign(sample, centers)
        new = np.array([
            sample[labels == i].mean(0) if np.any(labels == i) else centers[i]
            for i in range(len(centers))
        ], dtype=np.float32)
        if np.allclose(new, centers, atol=0.5):
            break
        centers = new
    return centers


def assign(pixels: np.ndarray, centers: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Associe chaque pixel au centre le plus proche (par morceaux, peu de RAM)."""
    out = np.empty(len(pixels), dtype=np.int32)
    for s in range(0, len(pixels), chunk):
        p = pixels[s:s + chunk]
        out[s:s + chunk] = ((p[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    return out


def to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(round(c)) for c in rgb])


def add_layer(parent, rgba: np.ndarray, mask: np.ndarray, name: str,
              visible: bool = True, blend: BlendMode | None = None) -> PixelLayer | None:
    """Ajoute un calque contenant uniquement les pixels de `mask`, rogné à sa bbox."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgba[top:bottom, left:right].copy()
    crop[..., 3] = np.where(mask[top:bottom, left:right], crop[..., 3], 0)
    layer = PixelLayer.frompil(Image.fromarray(crop, "RGBA"), parent,
                               name=name, top=int(top), left=int(left))
    layer.visible = visible
    if blend is not None:
        layer.blend_mode = blend
    return layer


def remove_background(img: Image.Image) -> np.ndarray | None:
    """Masque du sujet (bool HxW) via rembg, ou None si indisponible."""
    try:
        from rembg import remove
    except ImportError:
        print("  ! rembg non installé (pip install rembg) — option --ai ignorée.")
        return None
    print("  · Détourage du sujet par IA (rembg)…")
    alpha = np.asarray(remove(img.convert("RGB")).convert("RGBA"))[..., 3]
    return alpha > 127


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert(src: Path, dst: Path, n_colors: int, smooth: int, min_area: float,
            use_ai: bool, regions: bool, lines: bool) -> None:
    print(f"→ Lecture de {src}")
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGBA")
    w, h = img.size
    rgba = np.asarray(img).copy()
    opaque = rgba[..., 3] > 0

    psd = PSDImage.new("RGB", (w, h), color=(255, 255, 255))

    # 1. Image originale (référence, masquée)
    add_layer(psd, rgba, np.ones((h, w), bool), "Image originale", visible=False)

    # 2. Détourage IA (optionnel)
    subject = remove_background(img) if use_ai else None
    if subject is not None:
        add_layer(psd, rgba, subject & opaque, "Arrière-plan", visible=False)

    # 3. Séparation des couleurs
    print(f"  · Séparation en {n_colors} couleurs (k-means)…")
    flat = rgba[..., :3].reshape(-1, 3).astype(np.float32)
    centers = kmeans_colors(flat[opaque.ravel()], n_colors)
    labels = assign(flat, centers).reshape(h, w)

    if smooth > 1:  # supprime le "bruit" de pixels isolés
        size = smooth if smooth % 2 else smooth + 1
        labels = np.asarray(Image.fromarray(labels.astype(np.uint8), "L")
                            .filter(ImageFilter.ModeFilter(size))).astype(np.int32)

    # Calques de couleur : des plus grandes zones (fond) aux plus petites (détails)
    order = sorted(range(len(centers)), key=lambda i: -(labels == i).sum())
    min_px = int(min_area / 100 * w * h)

    def build_color_layers(parent, region_mask: np.ndarray) -> int:
        count = 0
        for rank, i in enumerate(order, 1):
            mask = (labels == i) & region_mask
            if mask.sum() <= min_px:
                continue
            base = f"Couleur {rank:02d} {to_hex(centers[i])}"
            if regions:
                count += add_regions(parent, mask, base)
            else:
                count += add_layer(parent, rgba, mask, base) is not None
        return count

    def add_regions(parent, mask: np.ndarray, base: str) -> int:
        try:
            from scipy import ndimage
        except ImportError:
            print("  ! scipy non installé (pip install scipy) — option --regions ignorée.")
            return add_layer(parent, rgba, mask, base) is not None
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        grp = Group.new(parent, name=base, open_folder=False)
        kept = 0
        for j in np.argsort(-sizes):
            if sizes[j] <= max(min_px, 4):
                continue
            kept += 1
            add_layer(grp, rgba, lab == j + 1, f"{base} – zone {kept}")
        return kept

    colors_group = Group.new(psd, name="Couleurs", open_folder=True)
    region = opaque if subject is None else (opaque & ~subject)
    n_layers = build_color_layers(colors_group, region)

    if subject is not None:
        subj_group = Group.new(psd, name="Sujet", open_folder=True)
        n_layers += build_color_layers(subj_group, subject & opaque)
        add_layer(subj_group, rgba, subject & opaque, "Sujet (complet)", visible=False)

    # 4. Contours / line-art
    if lines:
        print("  · Extraction des contours…")
        edges = img.convert("L").filter(ImageFilter.GaussianBlur(1)).filter(ImageFilter.FIND_EDGES)
        e = np.asarray(edges).astype(np.float32)
        e = np.clip(e * (255.0 / max(e.max(), 1)) * 1.5, 0, 255).astype(np.uint8)
        line = np.zeros((h, w, 4), np.uint8)
        line[..., 3] = np.where(e > 40, e, 0)  # trait noir, opacité = intensité
        add_layer(psd, line, line[..., 3] > 0, "Contours", visible=False,
                  blend=BlendMode.MULTIPLY)

    # Ordre attendu dans Photoshop : le premier calque ajouté est tout en bas.
    psd.save(dst)
    print(f"✓ PSD créé : {dst}  ({w}×{h}px, {n_layers} calques de couleur)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convertit une image JPEG/PNG en PSD Photoshop avec calques.")
    ap.add_argument("images", nargs="+", type=Path, help="image(s) JPEG ou PNG")
    ap.add_argument("-o", "--output", type=Path, help="fichier PSD de sortie (une seule image)")
    ap.add_argument("-c", "--colors", type=int, default=8, help="nombre de calques de couleur (défaut 8)")
    ap.add_argument("-s", "--smooth", type=int, default=5,
                    help="lissage des zones en px, 0 = aucun (défaut 5)")
    ap.add_argument("--min-area", type=float, default=0.05,
                    help="ignore les zones < X%% de l'image (défaut 0.05)")
    ap.add_argument("--ai", action="store_true", help="sépare le sujet du fond (nécessite rembg)")
    ap.add_argument("--regions", action="store_true",
                    help="un calque par zone isolée de chaque couleur (nécessite scipy)")
    ap.add_argument("--no-lines", action="store_true", help="ne pas générer le calque Contours")
    args = ap.parse_args()

    if args.output and len(args.images) > 1:
        ap.error("-o ne peut être utilisé qu'avec une seule image")
    if not 2 <= args.colors <= 64:
        ap.error("--colors doit être entre 2 et 64")

    for src in args.images:
        if not src.exists():
            print(f"✗ Introuvable : {src}")
            continue
        dst = args.output or src.with_suffix(".psd")
        convert(src, dst, args.colors, args.smooth, args.min_area,
                args.ai, args.regions, not args.no_lines)


if __name__ == "__main__":
    main()

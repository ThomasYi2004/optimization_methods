"""GA-based binary image approximation with triangles, with k_max constraint."""
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 360
POP_SIZE = 2048
N_GENERATIONS = 200
P_M1 = 0.30   # remove triangle
P_M2 = 0.50   # add triangle
P_M3 = 0.80   # mutate vertices
N_REPLACE = 1800  # worst individuals replaced each generation
SAVE_EVERY = 50
OUTPUT_DIR = "output"
KMAX_VALUES = [20]


def load_target(path, n):
    """Load image, resize, binarize. Convention: 1 = white, 0 = black."""
    img = Image.open(path).convert("L").resize((n, n))
    bwimage = (np.array(img) >= 128).astype(np.uint8)
    return bwimage


VERTEX_MARGIN = 0.25  # fraction of n by which vertices may overshoot edges
P_FINE = 1.0          # share of M3 vertex changes that are small Gaussian nudges
SIGMA_FRAC = 0.1     # Gaussian step size, as fraction of n

def _coord(n):
    m = int(n * VERTEX_MARGIN)
    return random.randint(-m, n - 1 + m)

def _nudge(v, n):
    sigma = SIGMA_FRAC * n
    m = int(n * VERTEX_MARGIN)
    return max(-m, min(n - 1 + m, int(v + random.gauss(0, sigma))))

def random_triangle(n):
    verts = [(_coord(n), _coord(n)) for _ in range(3)]
    return [verts, random.randint(0, 1)]


def rasterize(chromosome, n):
    """Render triangles onto a black canvas; later triangles overwrite earlier."""
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    for verts, color in chromosome:
        draw.polygon(verts, fill=int(color))
    return np.array(img)


def error(chromosome, target, n):
    rendered = rasterize(chromosome, n)
    return float(np.mean(rendered != target) * 100.0)


def _worst_triangle_idx(chromosome, target, n):
    """Index of triangle whose removal least hurts (or most helps) error."""
    best_idx, best_delta = 0, float("inf")
    base = error(chromosome, target, n)
    for i in range(len(chromosome)):
        e = error(chromosome[:i] + chromosome[i + 1:], target, n)
        delta = e - base  # negative = removing this triangle improves error
        if delta < best_delta:
            best_delta, best_idx = delta, i
    return best_idx


def mutate(chromosome, n, kmax, target):
    if chromosome and random.random() < P_M1:
        chromosome.pop(_worst_triangle_idx(chromosome, target, n))
    if random.random() < P_M2 and len(chromosome) < kmax:
        chromosome.append(random_triangle(n))
    if chromosome and random.random() < P_M3:
        idx = random.randrange(len(chromosome))
        verts, color = chromosome[idx]
        verts = list(verts)
        for i in random.sample(range(3), random.randint(1, 3)):
            if random.random() < P_FINE:
                vx, vy = verts[i]
                verts[i] = (_nudge(vx, n), _nudge(vy, n))
            else:
                verts[i] = (_coord(n), _coord(n))
        chromosome[idx] = [verts, color]


def crossover(p1, p2, n, kmax):
    """Each triangle from each parent included independently with prob 0.5."""
    child = []
    for tri in p1 + p2:
        if random.random() < 0.5:
            child.append([list(tri[0]), tri[1]])
    if len(child) > kmax:
        random.shuffle(child)
        child = child[:kmax]
    if not child:
        child.append(random_triangle(n))
    return child


def save_image(arr, path):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def run_ga(target, n, kmax, n_gens, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    population = [[random_triangle(n)] for _ in range(POP_SIZE)]
    keep = POP_SIZE - N_REPLACE
    history = []
    best_error = 100.0

    for gen in range(n_gens):
        # Soft selection: mutate everyone (incl. previous survivors) every gen.
        for c in population:
            mutate(c, n, kmax, target)
        errors = [error(c, target, n) for c in population]

        order = np.argsort(errors)
        population = [population[i] for i in order]
        errors = [errors[i] for i in order]

        best, best_error = population[0], errors[0]
        history.append((gen, best_error, len(best)))
        print(f"  [kmax={kmax}] Gen {gen:4d}  error={best_error:6.2f}%  triangles={len(best):3d}")
        if gen % SAVE_EVERY == 0 or gen == n_gens - 1:
            save_image(rasterize(best, n), os.path.join(out_dir, f"gen_{gen:04d}.png"))

        survivors = population[:keep]
        children = [crossover(*random.sample(survivors, 2), n, kmax)
                    for _ in range(N_REPLACE)]
        population = survivors + children

    save_image(rasterize(population[0], n), os.path.join(out_dir, "best.png"))
    with open(os.path.join(out_dir, "history.csv"), "w") as f:
        f.write("generation,error,triangles\n")
        for g, e, k in history:
            f.write(f"{g},{e:.4f},{k}\n")
    return best_error, len(population[0])


def main():
    args = sys.argv[1:]
    path = args[0] if args else "sherlock.png"
    target = load_target(path, IMG_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(target, os.path.join(OUTPUT_DIR, "target.png"))

    if len(args) > 1:  # single-run mode: ga_triangles.py image.png 30
        kmax = int(args[1])
        run_ga(target, IMG_SIZE, kmax, N_GENERATIONS,
               os.path.join(OUTPUT_DIR, f"kmax_{kmax:03d}"))
        return

    summary = []
    for kmax in KMAX_VALUES:
        print(f"\n=== Running k_max = {kmax} ===")
        out = os.path.join(OUTPUT_DIR, f"kmax_{kmax:03d}")
        err, k = run_ga(target, IMG_SIZE, kmax, N_GENERATIONS, out)
        summary.append((kmax, err, k))

    with open(os.path.join(OUTPUT_DIR, "summary.csv"), "w") as f:
        f.write("kmax,final_error,triangles_used\n")
        for kmax, err, k in summary:
            f.write(f"{kmax},{err:.4f},{k}\n")

    print("\n=== Sweep complete ===")
    print(f"{'kmax':>6} {'error%':>8} {'tris':>6}")
    for kmax, err, k in summary:
        print(f"{kmax:>6} {err:>8.2f} {k:>6}")


if __name__ == "__main__":
    main()

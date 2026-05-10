"""GA-based binary image approximation with triangles, with k_max constraint.

Drives error below TARGET_ERROR (default 3%) using elitist selection plus
random/Gaussian-nudge mutations and uniform-triangle crossover.
"""
import os
import sys
import json
import random
import time
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 360
POP_SIZE = 2048
N_GENERATIONS = 6000
P_M1 = 0.10   # remove triangle
P_M2 = 0.50   # add triangle
P_M3 = 0.80   # mutate vertices (vertex change / translate / color flip)
P_M4 = 0.40   # z-order swap two triangles
P_M5 = 0.50   # translate one triangle as a whole
N_ELITE = 248                       # 248 fittest kept; 1800 replaced (paper)
N_REPLACE = POP_SIZE - N_ELITE
SAVE_EVERY = 50
OUTPUT_DIR = "output"
KMAX_DEFAULT = 20
TARGET_ERROR = 3.0                  # stop when best error reaches this

VERTEX_MARGIN = 0.10  # fraction of n by which vertices may overshoot edges
P_FINE = 0.85         # share of M3 vertex changes that are small Gaussian nudges
# Mixed Gaussian step sizes (in pixels) — picks one uniformly per nudge
SIGMA_PIXELS = (1, 1, 2, 2, 3, 5, 8, 16)


def load_target(path, n):
    """Load image, resize, binarize. Convention: 1 = white, 0 = black."""
    img = Image.open(path).convert("L").resize((n, n))
    bwimage = (np.array(img) >= 128).astype(np.uint8)
    return bwimage


def _coord(n):
    m = int(n * VERTEX_MARGIN)
    return random.randint(-m, n - 1 + m)


def _nudge(v, n):
    sigma = random.choice(SIGMA_PIXELS)
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


def mutate(chromosome, n, kmax):
    if chromosome and random.random() < P_M1:
        chromosome.pop(random.randrange(len(chromosome)))
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
        if random.random() < 0.05:
            color = 1 - color
        chromosome[idx] = [verts, color]
    # M4: swap z-order of two triangles
    if len(chromosome) >= 2 and random.random() < P_M4:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    # M5: translate a whole triangle by a small random offset
    if chromosome and random.random() < P_M5:
        idx = random.randrange(len(chromosome))
        verts, color = chromosome[idx]
        sigma = random.choice(SIGMA_PIXELS)
        dx = int(random.gauss(0, sigma))
        dy = int(random.gauss(0, sigma))
        m = int(n * VERTEX_MARGIN)
        new_verts = []
        for vx, vy in verts:
            nx = max(-m, min(n - 1 + m, vx + dx))
            ny = max(-m, min(n - 1 + m, vy + dy))
            new_verts.append((nx, ny))
        chromosome[idx] = [new_verts, color]
    return chromosome


def clone(chromosome):
    return [[list(verts), color] for verts, color in chromosome]


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


def save_chromosome(chromosome, path):
    serial = [[[list(v) for v in verts], int(c)] for verts, c in chromosome]
    with open(path, "w") as f:
        json.dump(serial, f)


def load_chromosome(path):
    with open(path) as f:
        data = json.load(f)
    return [[[tuple(v) for v in verts], int(c)] for verts, c in data]


def run_ga(target, n, kmax, n_gens, out_dir, target_error=TARGET_ERROR,
           seed_chromosome=None):
    os.makedirs(out_dir, exist_ok=True)
    if seed_chromosome is None:
        population = [[random_triangle(n)] for _ in range(POP_SIZE)]
    else:
        population = [clone(seed_chromosome) for _ in range(POP_SIZE)]
    errors = [error(c, target, n) for c in population]
    history = []
    t0 = time.time()

    # External archive: absolute best ever seen, never mutated
    archive = clone(population[int(np.argmin(errors))])
    archive_err = float(min(errors))

    for gen in range(n_gens):
        # 1. Mutate every individual (paper: "Apply mutation to introduce variation").
        for c in population:
            mutate(c, n, kmax)
        # 2. Evaluate
        errors = [error(c, target, n) for c in population]
        # 3. Track absolute best via archive
        gen_best_idx = int(np.argmin(errors))
        if errors[gen_best_idx] < archive_err:
            archive = clone(population[gen_best_idx])
            archive_err = errors[gen_best_idx]
        # 4. Sort: top N_ELITE survive; bottom N_REPLACE are replaced via crossover.
        order = np.argsort(errors)
        population = [population[i] for i in order]
        errors = [errors[i] for i in order]
        # Inject archive as first individual to guarantee elitism.
        population[0] = clone(archive)
        errors[0] = archive_err

        history.append((gen, archive_err, len(archive)))

        if gen % 10 == 0 or gen == n_gens - 1:
            elapsed = time.time() - t0
            print(f"  [kmax={kmax}] Gen {gen:4d}  error={archive_err:6.3f}%  "
                  f"triangles={len(archive):3d}  t={elapsed:6.1f}s")
        if gen % SAVE_EVERY == 0 or gen == n_gens - 1:
            save_image(rasterize(archive, n), os.path.join(out_dir, f"gen_{gen:04d}.png"))

        if archive_err <= target_error:
            print(f"  [kmax={kmax}] Target {target_error}% reached at gen {gen} "
                  f"(error={archive_err:.3f}%).")
            break

        # 5. Replace bottom N_REPLACE with crossover offspring of survivors (top 248).
        survivors = population[:N_ELITE]
        children = [crossover(*random.sample(survivors, 2), n, kmax)
                    for _ in range(N_REPLACE)]
        population = survivors + children
        # children's errors will be recomputed after next mutation pass
        errors[N_ELITE:] = [None] * N_REPLACE

    save_image(rasterize(archive, n), os.path.join(out_dir, "best.png"))
    save_chromosome(archive, os.path.join(out_dir, "best.json"))
    with open(os.path.join(out_dir, "history.csv"), "w") as f:
        f.write("generation,error,triangles\n")
        for g, e, k in history:
            f.write(f"{g},{e:.4f},{k}\n")
    return archive_err, len(archive)


# def polish(chromosome, target, n, kmax, n_iters, target_error, out_dir,
#            max_step=8):
#     """Greedy local search: tiny vertex/color tweaks, accept any improvement."""
#     e = error(chromosome, target, n)
#     print(f"  polish start: error={e:.3f}%, k={len(chromosome)}")
#     accepted = 0
#     t0 = time.time()
#     for it in range(n_iters):
#         if not chromosome:
#             break
#         action = random.random()
#         idx = random.randrange(len(chromosome))
#         if action < 0.80 and chromosome:  # nudge a vertex
#             verts, color = chromosome[idx]
#             verts_new = [tuple(v) for v in verts]
#             i = random.randrange(3)
#             step = random.choice([1, 2, 3, 5, 8, 12, 20])
#             dx = random.randint(-step, step)
#             dy = random.randint(-step, step)
#             x, y = verts_new[i]
#             verts_new[i] = (x + dx, y + dy)
#             new_tri = [verts_new, color]
#             old_tri = chromosome[idx]
#             chromosome[idx] = new_tri
#             e_new = error(chromosome, target, n)
#             if e_new < e:
#                 e = e_new
#                 accepted += 1
#             else:
#                 chromosome[idx] = old_tri
#         elif action < 0.88:  # flip color
#             verts, color = chromosome[idx]
#             chromosome[idx] = [verts, 1 - color]
#             e_new = error(chromosome, target, n)
#             if e_new < e:
#                 e = e_new
#                 accepted += 1
#             else:
#                 chromosome[idx] = [verts, color]
#         elif action < 0.94 and len(chromosome) < kmax:  # add random triangle
#             chromosome.append(random_triangle(n))
#             e_new = error(chromosome, target, n)
#             if e_new < e:
#                 e = e_new
#                 accepted += 1
#             else:
#                 chromosome.pop()
#         elif action < 0.97 and len(chromosome) > 1:  # remove triangle
#             removed = chromosome.pop(idx)
#             e_new = error(chromosome, target, n)
#             if e_new < e:
#                 e = e_new
#                 accepted += 1
#             else:
#                 chromosome.insert(idx, removed)
#         else:  # replace triangle
#             old = chromosome[idx]
#             chromosome[idx] = random_triangle(n)
#             e_new = error(chromosome, target, n)
#             if e_new < e:
#                 e = e_new
#                 accepted += 1
#             else:
#                 chromosome[idx] = old

#         if it % 500 == 0:
#             elapsed = time.time() - t0
#             print(f"  polish it={it:6d}  error={e:6.3f}%  k={len(chromosome):3d}  "
#                   f"accepted={accepted}  t={elapsed:6.1f}s")
#         if it % 2000 == 0 and it > 0:
#             save_image(rasterize(chromosome, n),
#                        os.path.join(out_dir, f"polish_{it:05d}.png"))
#             save_chromosome(chromosome, os.path.join(out_dir, "best.json"))
#         if e <= target_error:
#             print(f"  polish: target {target_error}% reached at iter {it} (error={e:.3f}%).")
#             break
#     save_image(rasterize(chromosome, n), os.path.join(out_dir, "best.png"))
#     save_chromosome(chromosome, os.path.join(out_dir, "best.json"))
#     return chromosome, e


def main():
    args = sys.argv[1:]
    path = args[0] if args else "sherlock.png"
    target = load_target(path, IMG_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(target, os.path.join(OUTPUT_DIR, "target.png"))

    kmax = int(args[1]) if len(args) > 1 else KMAX_DEFAULT
    target_err = float(args[2]) if len(args) > 2 else TARGET_ERROR
    seed_path = args[3] if len(args) > 3 else None
    out = os.path.join(OUTPUT_DIR, f"kmax_{kmax:03d}")
    seed = load_chromosome(seed_path) if seed_path else None

    # Pure GA — keep running until target error reached
    err, k = run_ga(target, IMG_SIZE, kmax, N_GENERATIONS, out, target_err,
                    seed_chromosome=seed)
    print(f"\nGA done: kmax={kmax}, error={err:.3f}%, triangles={k}")


if __name__ == "__main__":
    main()

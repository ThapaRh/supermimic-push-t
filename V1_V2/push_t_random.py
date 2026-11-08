"""
supermimic-push-t -- Step 0: see Push-T move, no learning yet.

Run this first. It does NOT train anything.
It just runs one random episode in a 2D physics world:
a round pusher bumps into a T-shaped block, and we measure
how far the block ends up from its target pose.

On your Mac:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pymunk matplotlib
    python push_t_random.py

You should see a printout and a file called push_t_random.png
with the pusher path and the T-block path.
If that works, Step 0 is done. Learning comes after.
"""

import math
import random

import pymunk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- The T shape, in pixels. Top bar + stem, like the letter T. ---
def add_t_block(space, x, y, angle):
    body = pymunk.Body(1, pymunk.moment_for_box(1, (120, 120)))
    body.position = (x, y)
    body.angle = angle
    # Top bar of the T
    top = pymunk.Poly.create_box(body, (120, 30))
    top.offset = (0, 45)
    top.friction = 0.5
    # Stem of the T
    stem = pymunk.Poly.create_box(body, (30, 90))
    stem.offset = (0, -15)
    stem.friction = 0.5
    space.add(body, top, stem)
    return body


def angle_diff(a, b):
    """Smallest difference between two angles, in radians."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def main():
    random.seed(0)
    space = pymunk.Space()
    space.gravity = (0, 0)  # Top-down table, no gravity.

    # Walls so nothing flies off the 512x512 table.
    for a, b in [((0, 0), (512, 0)), ((512, 0), (512, 512)),
                 ((512, 512), (0, 512)), ((0, 512), (0, 0))]:
        wall = pymunk.Segment(space.static_body, a, b, 1)
        wall.elasticity = 0.0
        wall.friction = 0.5
        space.add(wall)

    # Target pose: fixed for this starter. Tang's variant may change this.
    target_x, target_y, target_angle = 256, 256, 0.0

    # Start: T-block bottom-left, pusher top-right, both with a random nudge.
    t_body = add_t_block(space, 150 + random.randint(-30, 30),
                         150 + random.randint(-30, 30),
                         random.uniform(-0.5, 0.5))
    pusher = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 20))
    pusher.position = (380 + random.randint(-20, 20),
                       380 + random.randint(-20, 20))
    pusher_shape = pymunk.Circle(pusher, 20)
    pusher_shape.friction = 0.5
    space.add(pusher, pusher_shape)

    pusher_path, t_path = [], []
    for step in range(300):
        # Random pusher: pick a new random velocity every 20 steps.
        # This is deliberately dumb. A learned policy will replace this.
        if step % 20 == 0:
            vx = random.uniform(-150, 150)
            vy = random.uniform(-150, 150)
        pusher.velocity = (vx, vy)
        space.step(1 / 60.0)
        pusher_path.append(tuple(pusher.position))
        t_path.append(tuple(t_body.position))

    pos_err = math.dist(t_body.position, (target_x, target_y))
    ang_err = angle_diff(t_body.angle, target_angle)
    success = pos_err < 30 and ang_err < 0.3

    print(f"T-block final position: ({t_body.position.x:.0f}, {t_body.position.y:.0f})")
    print(f"Target position:        ({target_x}, {target_y})")
    print(f"Position error: {pos_err:.0f} pixels, angle error: {ang_err:.2f} rad")
    print(f"Success (pos < 30 and angle < 0.3)? {success}")
    print("Random almost never succeeds. That is the point: learning has to beat this.")

    px, py = zip(*pusher_path)
    tx, ty = zip(*t_path)
    plt.figure(figsize=(6, 6))
    plt.plot(px, py, label="pusher")
    plt.plot(tx, ty, label="T-block")
    plt.scatter([target_x], [target_y], marker="x", s=100, label="target")
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("Push-T, random pusher: one episode")
    plt.savefig("push_t_random.png")
    print("Saved push_t_random.png")


if __name__ == "__main__":
    main()

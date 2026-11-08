"""
Push-T environment, v2 (fixes from the Nov 6 re-check).

Changes vs train_world_model.PushT, and why:
  1. Damping: the T now stops when nothing pushes it. Before, it slid
     ~176 px after a push, so pushes overshot.
  2. Action repeat: one action is held for 10 physics substeps of 10 ms
     (10 Hz control, like gym-pusht). The pusher covers up to 30 px per
     control step, so H = 32 imagined steps can actually reach the T.
  3. Random resets for data collection (reset_random) so the buffer sees
     the T in many poses. Evaluation still uses the original start
     distribution (T near (150,150), pusher near (380,380)), seeds 1000+.

Task is still the simplified Push-T: goal (256, 256, angle 0),
success = position error < 30 px and angle error < 0.3 rad.
"""

import math
import random

import numpy as np
import pymunk

SPEED = 300.0      # px/s at |action| = 1
SUBSTEPS = 10
DT = 0.01
DAMPING = 0.001    # pymunk: fraction of velocity kept after 1 s (T stops within ~0.3 s)
GOAL = (256.0, 256.0, 0.0)
MAX_STEPS = 200    # control steps (20 s)


def angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def add_t_block(space, x, y, angle):
    body = pymunk.Body(1, pymunk.moment_for_box(1, (120, 120)))
    body.position = (x, y)
    body.angle = angle
    top = pymunk.Poly.create_box(body, (120, 30))
    stem = pymunk.Poly.create_box(body, (30, 90))
    # Offsets are applied by shifting vertices (create_box ignores .offset).
    top = pymunk.Poly(body, [(vx, vy + 45) for vx, vy in top.get_vertices()])
    stem = pymunk.Poly(body, [(vx, vy - 15) for vx, vy in stem.get_vertices()])
    for s in (top, stem):
        s.friction = 0.5
    space.add(body, top, stem)
    return body


class PushT:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = DAMPING
        for a, b in [((0, 0), (512, 0)), ((512, 0), (512, 512)),
                     ((512, 512), (0, 512)), ((0, 512), (0, 0))]:
            w = pymunk.Segment(self.space.static_body, a, b, 1)
            w.friction = 0.5
            self.space.add(w)
        self.t_body = add_t_block(self.space,
                                  150 + self.rng.randint(-30, 30),
                                  150 + self.rng.randint(-30, 30),
                                  self.rng.uniform(-0.5, 0.5))
        self.pusher = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 20))
        self.pusher.position = (380 + self.rng.randint(-20, 20),
                                380 + self.rng.randint(-20, 20))
        shape = pymunk.Circle(self.pusher, 20)
        shape.friction = 0.5
        self.space.add(self.pusher, shape)
        self.steps = 0

    def reset_random(self, rng):
        """Random T pose and pusher position (for data collection)."""
        self.t_body.position = (rng.uniform(100, 412), rng.uniform(100, 412))
        self.t_body.angle = rng.uniform(-math.pi, math.pi)
        self.t_body.velocity = (0, 0)
        self.t_body.angular_velocity = 0
        while True:
            p = (rng.uniform(30, 482), rng.uniform(30, 482))
            if math.dist(p, self.t_body.position) > 90:
                break
        self.pusher.position = p
        self.pusher.velocity = (0, 0)
        self.space.reindex_shapes_for_body(self.t_body)
        self.space.reindex_shapes_for_body(self.pusher)
        self.steps = 0

    def raw(self):
        """[px, py, tx, ty, theta, tvx, tvy, omega] in pixels / radians."""
        p, t = self.pusher.position, self.t_body.position
        v = self.t_body.velocity
        return np.array([p.x, p.y, t.x, t.y, angle_wrap(self.t_body.angle),
                         v.x, v.y, self.t_body.angular_velocity],
                        dtype=np.float32)

    def errors(self):
        pos = math.dist(self.t_body.position, GOAL[:2])
        ang = abs(angle_wrap(self.t_body.angle - GOAL[2]))
        return pos, ang

    def success(self):
        pos, ang = self.errors()
        return pos < 30 and ang < 0.3

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        for _ in range(SUBSTEPS):
            self.pusher.velocity = (a[0] * SPEED, a[1] * SPEED)
            self.pusher.angular_velocity = 0
            self.space.step(DT)
        self.steps += 1
        return self.raw()

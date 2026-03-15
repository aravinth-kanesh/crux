#!/opt/homebrew/Caskroom/miniconda/base/bin/python3
"""Crux — 3D Rubik's cube visualiser.

Calls the solver binary, then animates the scramble and solution in 3D
with smooth per-move rotations, lighting, and an orbiting camera.

Usage:
  python3 visualise.py --scramble "R U R' U'" [--data-dir data]
  python3 visualise.py --random-depth 15 [--data-dir data]
  python3 visualise.py --scramble "R U R'" --no-solve
"""

import sys, os, subprocess, math, re, argparse, time
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_q, K_SPACE
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
FACE_RGB = {
    'W': (1.00, 1.00, 1.00),   # White  — U
    'Y': (1.00, 0.75, 0.00),   # Yellow — D
    'G': (0.00, 0.65, 0.25),   # Green  — F
    'B': (0.05, 0.25, 0.90),   # Blue   — B
    'O': (1.00, 0.40, 0.00),   # Orange — L
    'R': (0.85, 0.07, 0.07),   # Red    — R
    'K': (0.08, 0.08, 0.08),   # Black  — inner
}

# Maps outer face direction → colour key  (+x=R, -x=L, +y=U, -y=D, +z=F, -z=B)
DIR_COLOUR = {
    ( 1, 0, 0): 'R', (-1, 0, 0): 'O',
    ( 0, 1, 0): 'W', ( 0,-1, 0): 'Y',
    ( 0, 0, 1): 'G', ( 0, 0,-1): 'B',
}

# ---------------------------------------------------------------------------
# Move definitions  (axis: 0=X 1=Y 2=Z, layer: ±1, angle: degrees CCW)
# ---------------------------------------------------------------------------
MOVE_DEF = {
    'U':  (1,  1,  90), "U'": (1,  1, -90), 'U2': (1,  1, 180),
    'D':  (1, -1, -90), "D'": (1, -1,  90), 'D2': (1, -1, 180),
    'F':  (2,  1, -90), "F'": (2,  1,  90), 'F2': (2,  1, 180),
    'B':  (2, -1,  90), "B'": (2, -1, -90), 'B2': (2, -1, 180),
    'R':  (0,  1,  90), "R'": (0,  1, -90), 'R2': (0,  1, 180),
    'L':  (0, -1, -90), "L'": (0, -1,  90), 'L2': (0, -1, 180),
}

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
CH = 0.46    # cubie half-size (gap between cubies = 1 - 2*CH = 0.08)
SH = 0.39    # sticker half-size (border width visible = CH - SH = 0.07)
SP = 0.003   # how far sticker protrudes above cubie face

# Sticker quad vertices for each face direction (local coords)
STICKER_VERTS = {
    ( 1, 0, 0): [( CH+SP,-SH,-SH),( CH+SP,-SH, SH),( CH+SP, SH, SH),( CH+SP, SH,-SH)],
    (-1, 0, 0): [(-CH-SP,-SH, SH),(-CH-SP,-SH,-SH),(-CH-SP, SH,-SH),(-CH-SP, SH, SH)],
    ( 0, 1, 0): [(-SH, CH+SP,-SH),( SH, CH+SP,-SH),( SH, CH+SP, SH),(-SH, CH+SP, SH)],
    ( 0,-1, 0): [(-SH,-CH-SP, SH),( SH,-CH-SP, SH),( SH,-CH-SP,-SH),(-SH,-CH-SP,-SH)],
    ( 0, 0, 1): [(-SH,-SH, CH+SP),( SH,-SH, CH+SP),( SH, SH, CH+SP),(-SH, SH, CH+SP)],
    ( 0, 0,-1): [( SH,-SH,-CH-SP),(-SH,-SH,-CH-SP),(-SH, SH,-CH-SP),( SH, SH,-CH-SP)],
}

# Body quad vertices (slightly smaller than sticker quads to be hidden behind stickers)
BODY_VERTS = {
    ( 1, 0, 0): [( CH,-CH,-CH),( CH,-CH, CH),( CH, CH, CH),( CH, CH,-CH)],
    (-1, 0, 0): [(-CH,-CH, CH),(-CH,-CH,-CH),(-CH, CH,-CH),(-CH, CH, CH)],
    ( 0, 1, 0): [(-CH, CH,-CH),( CH, CH,-CH),( CH, CH, CH),(-CH, CH, CH)],
    ( 0,-1, 0): [(-CH,-CH, CH),( CH,-CH, CH),( CH,-CH,-CH),(-CH,-CH,-CH)],
    ( 0, 0, 1): [(-CH,-CH, CH),( CH,-CH, CH),( CH, CH, CH),(-CH, CH, CH)],
    ( 0, 0,-1): [( CH,-CH,-CH),(-CH,-CH,-CH),(-CH, CH,-CH),( CH, CH,-CH)],
}

ALL_DIRS = list(STICKER_VERTS.keys())


# ---------------------------------------------------------------------------
# Mini-cubie
# ---------------------------------------------------------------------------
class MiniCubie:
    __slots__ = ('pos', 'rot')

    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=np.float64)
        self.rot = np.eye(3, dtype=np.float64)

    def colour(self, direction):
        """Colour shown on the face pointing in `direction` (tuple of ints)."""
        n = np.array(direction, dtype=np.float64)
        orig = tuple(int(round(v)) for v in self.rot.T @ n)
        return DIR_COLOUR.get(orig, 'K')

    def apply_rotation(self, R: np.ndarray):
        self.pos = np.round(R @ self.pos).astype(np.float64)
        self.rot = R @ self.rot


def make_cubies():
    return [MiniCubie(x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if not (x == 0 and y == 0 and z == 0)]


def rot_matrix(axis, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
    elif axis == 1:
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
    else:
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def apply_move(cubies, move):
    if move not in MOVE_DEF:
        return
    axis, layer, angle = MOVE_DEF[move]
    R = rot_matrix(axis, angle)
    for c in cubies:
        if int(round(c.pos[axis])) == layer:
            c.apply_rotation(R)


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------
ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def call_solver(args):
    """Call cube_solver binary and return (scramble_moves, solution_moves)."""
    solver = os.path.join(os.path.dirname(__file__), 'build-release', 'cube_solver')
    if not os.path.exists(solver):
        solver = './build-release/cube_solver'

    cmd = [solver] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = ANSI_RE.sub('', result.stdout + result.stderr)
    except FileNotFoundError:
        print(f"Solver binary not found at {solver}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Solver timed out.")
        sys.exit(1)

    scramble, solution = [], []
    for line in output.splitlines():
        if line.startswith('Scramble:'):
            scramble = line.split(':', 1)[1].split()
        elif line.startswith('Solution:'):
            sol_str = line.split(':', 1)[1].strip()
            solution = sol_str.split() if sol_str else []
    return scramble, solution


# ---------------------------------------------------------------------------
# Text rendering via pygame font → OpenGL texture
# ---------------------------------------------------------------------------
_font_cache = {}

def get_font(size=20):
    if size not in _font_cache:
        pygame.font.init()
        _font_cache[size] = pygame.font.SysFont('monospace', size, bold=True)
    return _font_cache[size]


def draw_text_2d(text, x, y, screen_w, screen_h, colour=(220, 220, 220), size=20):
    font = get_font(size)
    surf = font.render(text, True, colour)
    w, h = surf.get_size()
    data = pygame.image.tostring(surf, 'RGBA', True)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, screen_w, 0, screen_h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    glColor4f(1, 1, 1, 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x,     y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 1); glVertex2f(x,     y + h)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()
    glDeleteTextures([tex])


# ---------------------------------------------------------------------------
# Draw the cube
# ---------------------------------------------------------------------------
def draw_cubies(cubies, anim_ids, anim_R_partial):
    for cubie in cubies:
        px, py, pz = cubie.pos
        glPushMatrix()

        if id(cubie) in anim_ids:
            # Apply partial rotation around the cube origin for this layer
            m = anim_R_partial
            gl_m = [
                m[0,0], m[1,0], m[2,0], 0,
                m[0,1], m[1,1], m[2,1], 0,
                m[0,2], m[1,2], m[2,2], 0,
                0,      0,      0,      1,
            ]
            glTranslatef(px, py, pz)
            glMultMatrixf(gl_m)
            glTranslatef(-px, -py, -pz)

        glTranslatef(px, py, pz)

        # Black body
        r, g, b = FACE_RGB['K']
        glColor3f(r, g, b)
        glBegin(GL_QUADS)
        for d, verts in BODY_VERTS.items():
            glNormal3fv(d)
            for v in verts:
                glVertex3fv(v)
        glEnd()

        # Coloured stickers
        glBegin(GL_QUADS)
        for d, verts in STICKER_VERTS.items():
            col = cubie.colour(d)
            r, g, b = FACE_RGB[col]
            glColor3f(r, g, b)
            glNormal3fv(d)
            for v in verts:
                glVertex3fv(v)
        glEnd()

        glPopMatrix()


def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)

    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 8.0, 6.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.85, 0.85, 0.85, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.30, 0.30, 0.30, 1.0))

    glLightfv(GL_LIGHT1, GL_POSITION, (-4.0, -3.0, -5.0, 1.0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  (0.25, 0.25, 0.30, 1.0))
    glLightfv(GL_LIGHT1, GL_AMBIENT,  (0.0,  0.0,  0.0,  1.0))


def draw_background():
    """Gradient background quad drawn behind everything."""
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()

    glBegin(GL_QUADS)
    glColor3f(0.06, 0.06, 0.10);  glVertex2f(-1, -1); glVertex2f(1, -1)
    glColor3f(0.12, 0.12, 0.20);  glVertex2f(1,  1);  glVertex2f(-1, 1)
    glEnd()

    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------
def ease_in_out(t):
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Crux 3D visualiser')
    parser.add_argument('--scramble', default='')
    parser.add_argument('--random-depth', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--no-solve', action='store_true',
                        help='Show scramble only, no solution')
    args = parser.parse_args()

    # Build solver args
    solver_args = ['--data-dir', args.data_dir]
    if args.scramble:
        solver_args += ['--scramble', args.scramble]
    elif args.random_depth:
        solver_args += ['--random-depth', str(args.random_depth)]
        if args.seed:
            solver_args += ['--seed', str(args.seed)]
    else:
        parser.print_help(); sys.exit(1)

    print("Calling solver...")
    scramble, solution = call_solver(solver_args)
    if not scramble:
        print("Could not parse scramble from solver output.")
        sys.exit(1)
    if args.no_solve:
        solution = []

    print(f"Scramble : {' '.join(scramble)}")
    print(f"Solution : {' '.join(solution) if solution else '(none)'}")

    # --- pygame + OpenGL init ---
    pygame.init()
    W, H = 900, 700
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Crux — Optimal Rubik\'s Cube Solver')

    glViewport(0, 0, W, H)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, W / H, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    setup_lighting()

    cubies = make_cubies()

    # Build animation queue: list of (label, moves_list, frames_per_move, pause_after)
    # Phase 1: scramble (fast), Phase 2: solved pause, Phase 3: solution (slow)
    SCRAMBLE_FRAMES = 8
    SOLVE_FRAMES    = 22
    PAUSE_FRAMES    = 90  # ~1.5s pause between phases at 60fps

    anim_queue = []   # (move_str, axis, layer, angle_deg, n_frames)
    for m in scramble:
        if m in MOVE_DEF:
            axis, layer, angle = MOVE_DEF[m]
            anim_queue.append(('scramble', m, axis, layer, angle, SCRAMBLE_FRAMES))
    anim_queue.append(('pause', '', 0, 0, 0, PAUSE_FRAMES))
    for m in solution:
        if m in MOVE_DEF:
            axis, layer, angle = MOVE_DEF[m]
            anim_queue.append(('solve', m, axis, layer, angle, SOLVE_FRAMES))
    anim_queue.append(('done', '', 0, 0, 0, 300))

    queue_idx    = 0
    frame_in_anim = 0
    anim_ids     = set()
    current_axis = 0
    current_angle = 0.0

    # Camera
    cam_azimuth   = 30.0   # degrees, slowly increases
    cam_elevation = 28.0
    cam_dist      = 7.5
    cam_speed     = 0.15   # degrees per frame

    clock = pygame.time.Clock()
    move_count = 0
    total_moves = len(solution)
    phase_label = 'Scrambling...'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    running = False

        # --- Advance animation ---
        anim_R_partial = np.eye(3)

        if queue_idx < len(anim_queue):
            entry = anim_queue[queue_idx]
            phase, move_name = entry[0], entry[1]
            axis, layer, angle_deg, n_frames = entry[2], entry[3], entry[4], entry[5]

            if phase == 'pause' or phase == 'done':
                frame_in_anim += 1
                if frame_in_anim >= n_frames:
                    frame_in_anim = 0
                    queue_idx += 1
                    if phase == 'pause':
                        phase_label = f'Solving  ({total_moves} moves)...' if solution else 'Solved!'
            else:
                if frame_in_anim == 0:
                    # Start of new move — identify cubies in layer
                    anim_ids = {id(c) for c in cubies
                                if int(round(c.pos[axis])) == layer}
                    current_axis  = axis
                    current_angle = angle_deg
                    if phase == 'scramble':
                        phase_label = f'Scramble: {move_name}'
                    else:
                        move_count += 1
                        phase_label = f'Move {move_count}/{total_moves}: {move_name}'

                t = (frame_in_anim + 1) / n_frames
                partial_angle = ease_in_out(t) * current_angle
                anim_R_partial = rot_matrix(current_axis, partial_angle)

                frame_in_anim += 1
                if frame_in_anim >= n_frames:
                    # Commit rotation
                    R_final = rot_matrix(current_axis, current_angle)
                    for c in cubies:
                        if id(c) in anim_ids:
                            c.apply_rotation(R_final)
                    anim_ids = set()
                    frame_in_anim = 0
                    queue_idx += 1

        # --- Camera orbit ---
        cam_azimuth += cam_speed

        # --- Render ---
        draw_background()
        glClear(GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        az = math.radians(cam_azimuth)
        el = math.radians(cam_elevation)
        cx = cam_dist * math.cos(el) * math.sin(az)
        cy = cam_dist * math.sin(el)
        cz = cam_dist * math.cos(el) * math.cos(az)
        gluLookAt(cx, cy, cz,  0, 0, 0,  0, 1, 0)

        draw_cubies(cubies, anim_ids, anim_R_partial)

        # HUD
        done = queue_idx >= len(anim_queue)
        if done:
            phase_label = 'Solved!' if solution else 'Scrambled'
        draw_text_2d(f'Crux — Optimal Rubik\'s Cube Solver', 14, H - 30, W, H,
                     colour=(180, 180, 220), size=18)
        draw_text_2d(phase_label, 14, 14, W, H,
                     colour=(240, 200, 80) if 'Solv' in phase_label else (160, 220, 160),
                     size=22)
        if solution:
            draw_text_2d(f'Scramble: {" ".join(scramble)}', 14, 44, W, H,
                         colour=(150, 150, 150), size=16)
            draw_text_2d(f'Solution: {" ".join(solution)}', 14, 66, W, H,
                         colour=(150, 150, 150), size=16)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()

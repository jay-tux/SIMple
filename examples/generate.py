import sys
import random

def v3():
    return random.gauss(0, 8), random.gauss(0, 8), random.gauss(0, 8)

def vel():
    return random.gauss(), random.gauss(), random.gauss()

def v3_comet(sun):
    sx, sy, sz = sun
    return sx + random.gauss(0, 3), sy + random.gauss(0, 3), sz + random.gauss(0, 3)

def vel_planet(sun, planet):
    sx, sy, sz = sun
    px, py, pz = planet
    spx, spy, spz = px - sx, py - sy, pz - sz
    spx, spy, spz = spx / 2, spy / 2, spz / 2
    swap = random.randint(0, 2)
    if swap == 0:
        return spy, spx, spz
    if swap == 1:
        return spz, spy, spx
    return spx, spz, spy

def vel_comet(sun, comet):
    sx, sy, sz = sun
    cx, cy, cz = comet
    scx, scy, scz = cx - sx + random.gauss(0, 0.03), cy - sy + random.gauss(0, 0.03), cz - sz + random.gauss(0, 0.03)
    return scx, scy, scz

if len(sys.argv) < 5 or sys.argv[1] == '-h':
    print(f'Usage: {sys.argv[0]} <#suns> <#planets per sun> <#comets per sun> <output file>\n')
    exit(0)

suns = int(sys.argv[1])
planets = int(sys.argv[2])
comets = int(sys.argv[3])
out = sys.argv[4]

with open(out, 'w') as f:
    f.write('[SETTINGS]\n')
    f.write('time_scale = 0.66\n')
    f.write('eye = 0 0 20\n')
    f.write('focus = 0 0 0\n')
    f.write('history = 1\n')
    f.write('history_skip = 1000\n')
    f.write('history_render = off\n')
    f.write('moving_cam = on\n')
    f.write('cam_time_scale = 0.1\n')
    f.write('\n')

    f.write('[black hole]\n')
    f.write('position = 0 0 0\n')
    f.write('velocity = 0 0 0\n')
    f.write('mass = 20\n')
    f.write('radius = 1\n')
    f.write('color = 0.2 0 0.2\n')
    f.write('mass_div_g = on\n')
    f.write('\n')

    sun_positions = [v3() for i in range(suns)]
    for i, (x, y, z) in enumerate(sun_positions):
        f.write(f'[sun{i}]\n')
        f.write(f'position = {x} {y} {z}\n')
        vx, vy, vz = vel()
        f.write(f'velocity = {vx} {vy} {vz}\n')
        mass = random.gauss(1, 0.2)
        f.write(f'mass = {mass}\n')
        f.write(f'radius = {mass / 5}\n')
        f.write(f'color = 1 1 1\n')
        f.write(f'mass_div_g = on\n')
        f.write('\n')

    for j, (sx, sy, sz) in enumerate(sun_positions):
        for i in range(planets):
            f.write(f'[planet{j}_{i}]\n')
            px, py, pz = vel()
            f.write(f'position = {sx + px} {sy + py} {sz + pz}\n')
            vx, vy, vz = vel_planet((sx, sy, sz), (px, py, pz))
            f.write(f'velocity = {vx} {vy} {vz}\n')
            mass = random.gauss(1, 0.2)
            f.write(f'mass = {mass}\n')
            f.write(f'radius = {mass / 20}\n')
            f.write(f'color = 0 0.33 0.66\n')
            f.write('\n')

        for i in range(comets):
            f.write(f'[comet{j}_{i}]\n')
            cx, cy, cz = v3_comet((sx, sy, sz))
            f.write(f'position = {cx} {cy} {cz}\n')
            vx, vy, vz = vel_comet((sx, sy, sz), (cx, cy, cz))
            f.write(f'velocity = {vx} {vy} {vz}\n')
            mass = random.gauss() / 4
            f.write(f'mass = {mass}\n')
            f.write(f'radius = {mass / 10}\n')
            f.write(f'color = 0.88 0.33 0\n')

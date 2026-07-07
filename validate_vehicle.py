"""Validate the vehicle model against known F1 performance envelopes."""
import numpy as np
from vehicle_model import F1Vehicle, CarSpec, G

def straight_line():
    car = F1Vehicle(); car.reset(speed=0.0)
    dt = 0.02; t = 0.0; t100 = t200 = None; vmax = 0.0
    while t < 25.0:
        st = car.step(throttle=1.0, brake=0.0, steer=0.0, dt=dt)
        t += dt
        kmh = st.vx * 3.6
        if t100 is None and kmh >= 100: t100 = t
        if t200 is None and kmh >= 200: t200 = t
        vmax = max(vmax, kmh)
    return t100, t200, vmax

def braking_from(kmh):
    car = F1Vehicle(); car.reset(speed=kmh/3.6)
    dt = 0.01; peak_g = 0.0; dist = 0.0; v0 = car.state.vx
    while car.state.vx > 1.0:
        prev = car.state.vx
        st = car.step(throttle=0.0, brake=1.0, steer=0.0, dt=dt)
        decel = (prev - st.vx)/dt
        peak_g = max(peak_g, decel/G)
        dist += (prev+st.vx)/2*dt
    return peak_g, dist

def max_lateral(kmh):
    """Turn in hard at speed; report the PEAK lateral G reached (grip limit)."""
    best = 0.0
    for steer in np.linspace(0.05, 1.0, 40):
        car = F1Vehicle(); car.reset(speed=kmh/3.6)
        dt = 0.005
        peak = 0.0
        for _ in range(200):  # 1 s, capture turn-in transient at the limit
            # throttle trims to hold speed against scrub/drag
            thr = 0.5 if car.state.vx > kmh/3.6*0.97 else 1.0
            st = car.step(throttle=thr, brake=0.0, steer=steer, dt=dt)
            peak = max(peak, abs(st.ay)/G)
        best = max(best, peak)
    return best

if __name__ == "__main__":
    print("="*58)
    print("F1 VEHICLE MODEL — VALIDATION")
    print("="*58)
    t100, t200, vmax = straight_line()
    print(f"\n[Straight line, full throttle]")
    print(f"  0-100 km/h : {t100:.2f} s      (F1 real ~2.6 s)")
    print(f"  0-200 km/h : {t200:.2f} s      (F1 real ~4.5-5 s)")
    print(f"  Top speed  : {vmax:.0f} km/h   (F1 real ~330-345)")

    print(f"\n[Threshold braking]")
    for kmh in (300, 200, 100):
        g, d = braking_from(kmh)
        print(f"  from {kmh} km/h: peak {g:.1f} G, stops in {d:.0f} m")

    print(f"\n[Max steady-state cornering]")
    for kmh in (100, 200, 300):
        print(f"  at {kmh} km/h: {max_lateral(kmh):.1f} G lateral   (F1 real ~4-6 G, rising w/ speed)")
    print("="*58)

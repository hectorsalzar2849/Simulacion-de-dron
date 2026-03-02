import numpy as np
import matplotlib.pyplot as plt

# ===================== PARÁMETROS =====================
PARK  = 50.0
DT    = 0.05
T     = 1500.0
t_arr = np.arange(0, T + DT, DT)

V_MAX   = 1.2

GPS_T   = 15.0
SIG_INT = 0.25
SIG_GPS = 0.05
SIG_BAD = 0.60
P_OUT   = 0.15
P_BAD   = 0.20

mission       = "fire"   # "fire" o "person"
DETECT_FIRE   = 5
DETECT_PERSON = 3

LOOKAHEAD    = 1.0
WAYPOINT_TOL = 0.4

np.random.seed(42)

# ===================== WAYPOINTS ZIGZAG =====================
detect = DETECT_FIRE if mission == "fire" else DETECT_PERSON
lane_w = detect / 2.0

waypoints = []
lane = 0
while True:
    y = lane_w * (lane + 0.5)
    if y > PARK:
        break
    if lane % 2 == 0:
        waypoints.append((0.0,  y))
        waypoints.append((PARK, y))
    else:
        waypoints.append((PARK, y))
        waypoints.append((0.0,  y))
    lane += 1
waypoints = np.array(waypoints, dtype=float)

# ===================== LOOKAHEAD =====================
def get_target(rx, ry, wp_idx):
    if wp_idx == 0:
        return float(waypoints[0,0]), float(waypoints[0,1])
    A = waypoints[wp_idx - 1]
    B = waypoints[wp_idx]
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-9:
        return float(B[0]), float(B[1])
    AP = np.array([rx, ry]) - A
    t_proj = np.clip(np.dot(AP, AB) / AB_len**2, 0.0, 1.0)
    t_look = np.clip(t_proj + LOOKAHEAD / AB_len, 0.0, 1.0)
    target = A + t_look * AB
    return float(target[0]), float(target[1])

# ===================== VIENTO (Sierra Madre Occidental) =====================
def wind(ti):
    wx = 0.05 * np.sin(2*np.pi * ti / 120.0) + 0.02 * np.sin(2*np.pi * ti / 40.0)
    wy = 0.03 * np.cos(2*np.pi * ti / 90.0)
    return wx, wy

# ===================== GPS =====================
next_gps  = 0.0
last_gps  = None
last_mode = "outage"

def get_gps(ti, x, y):
    global next_gps, last_gps, last_mode
    if ti >= next_gps - 1e-9:
        next_gps += GPS_T
        r = np.random.rand()
        if r < P_OUT:
            last_gps, last_mode = None, "outage"
        elif r < P_OUT + P_BAD:
            last_gps = (x + np.random.normal(0, SIG_BAD),
                        y + np.random.normal(0, SIG_BAD))
            last_mode = "bad"
        else:
            last_gps = (x + np.random.normal(0, SIG_GPS),
                        y + np.random.normal(0, SIG_GPS))
            last_mode = "good"
    return last_gps, last_mode

# ===================== KALMAN FILTER 2D =====================
kf_x = np.array([0.0, lane_w * 0.5])
kf_P = np.eye(2) * 0.5
kf_Q = np.eye(2) * 0.001

R_good = np.eye(2) * SIG_GPS**2
R_bad  = np.eye(2) * SIG_BAD**2
R_int  = np.eye(2) * SIG_INT**2

def kalman_step(z, R):
    global kf_x, kf_P
    kf_P = kf_P + kf_Q
    S = kf_P + R
    K = kf_P @ np.linalg.inv(S)
    kf_x = kf_x + K @ (z - kf_x)
    kf_P = (np.eye(2) - K) @ kf_P
    return kf_x.copy()

# ===================== PID CON ANTI-WINDUP =====================
Kp, Ki, Kd = 3.5, 0.015, 0.50
I_MAX = 3.0

ix = iy = 0.0
ex_prev = ey_prev = 0.0

rx, ry  = 0.0, float(waypoints[0, 1])
kf_x    = np.array([rx, ry])
wp_idx  = 1

# Historial
hist_ref   = np.zeros((len(t_arr), 2))
hist_real  = np.zeros((len(t_arr), 2))
hist_est   = np.zeros((len(t_arr), 2))
hist_wind  = np.zeros((len(t_arr), 2))
hist_ex    = np.zeros(len(t_arr))
hist_ey    = np.zeros(len(t_arr))
hist_ux    = np.zeros(len(t_arr))
hist_uy    = np.zeros(len(t_arr))
hist_gps_x = []
hist_gps_y = []
hist_gps_t = []
hist_mode  = []

# ===================== SIMULACIÓN =====================
for k, ti in enumerate(t_arr):

    dist_wp = np.hypot(rx - waypoints[wp_idx,0], ry - waypoints[wp_idx,1])
    if dist_wp < WAYPOINT_TOL and wp_idx < len(waypoints) - 1:
        wp_idx += 1
        ix = 0.0; iy = 0.0

    xr, yr = get_target(rx, ry, wp_idx)

    A = waypoints[wp_idx - 1]; B = waypoints[wp_idx]
    AB = B - A; AB_len = np.linalg.norm(AB)
    if AB_len > 0:
        t_p = np.clip(np.dot(np.array([rx,ry]) - A, AB) / AB_len**2, 0, 1)
        hist_ref[k] = A + t_p * AB
    else:
        hist_ref[k] = A

    wx, wy = wind(ti)
    hist_wind[k] = [wx, wy]

    z_int = np.array([rx + np.random.normal(0, SIG_INT),
                      ry + np.random.normal(0, SIG_INT)])
    gps_val, gps_mode = get_gps(ti, rx, ry)

    # Detectar si hubo actualización GPS en este paso
    gps_updated = (ti >= (next_gps - GPS_T - 1e-9)) and (ti < (next_gps - GPS_T + DT + 1e-9))

    if gps_mode == "good" and gps_val is not None:
        est = kalman_step(np.array(gps_val), R_good)
        hist_gps_x.append(gps_val[0]); hist_gps_y.append(gps_val[1])
        hist_gps_t.append(ti); hist_mode.append("good")
    elif gps_mode == "bad" and gps_val is not None:
        est = kalman_step(np.array(gps_val), R_bad)
        hist_gps_x.append(gps_val[0]); hist_gps_y.append(gps_val[1])
        hist_gps_t.append(ti); hist_mode.append("bad")
    else:
        est = kalman_step(z_int, R_int)

    hist_est[k] = est

    ex = xr - est[0]
    ey = yr - est[1]

    ix = np.clip(ix + ex * DT, -I_MAX, I_MAX)
    iy = np.clip(iy + ey * DT, -I_MAX, I_MAX)

    dex = (ex - ex_prev) / DT
    dey = (ey - ey_prev) / DT

    ux_raw = Kp*ex + Ki*ix + Kd*dex
    uy_raw = Kp*ey + Ki*iy + Kd*dey

    ux = np.clip(ux_raw, -V_MAX, V_MAX)
    uy = np.clip(uy_raw, -V_MAX, V_MAX)

    if abs(ux_raw) > V_MAX: ix -= ex * DT
    if abs(uy_raw) > V_MAX: iy -= ey * DT

    ex_prev, ey_prev = ex, ey

    hist_ex[k] = ex; hist_ey[k] = ey
    hist_ux[k] = ux; hist_uy[k] = uy

    rx = np.clip(rx + (ux + wx) * DT, 0, PARK)
    ry = np.clip(ry + (uy + wy) * DT, 0, PARK)
    kf_x = np.array([rx, ry])

    hist_real[k] = [rx, ry]

rmse = np.sqrt(np.mean((hist_ref[:,0]-hist_real[:,0])**2 +
                       (hist_ref[:,1]-hist_real[:,1])**2))
print(f"RMSE: {rmse:.3f} km")

# Separar GPS por modo — hist_gps_x/y solo tienen entradas cuando hay señal (good o bad)
gps_modes_all = np.array(hist_mode)   # todos los eventos GPS (good, bad, outage)
t_gps = np.array(hist_gps_t)
gx    = np.array(hist_gps_x)
gy    = np.array(hist_gps_y)
# Para la trayectoria: modos de los puntos que tienen coordenadas
gps_has_signal = gps_modes_all[gps_modes_all != 'outage']

# ===================== FIGURA 1: Trayectoria =====================
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.plot(waypoints[:,0], waypoints[:,1], '--', color='steelblue',
         lw=1.5, label='Referencia', zorder=2)
ax1.plot(hist_real[:,0], hist_real[:,1], '-', color='orangered',
         lw=0.8, alpha=0.85, label='Drone real', zorder=3)

# Puntos GPS buenos y malos
good_mask = gps_has_signal == "good"
bad_mask  = gps_has_signal == "bad"
if good_mask.any():
    ax1.scatter(gx[good_mask], gy[good_mask], s=12, color='green',
                zorder=5, label='GPS bueno', alpha=0.6)
if bad_mask.any():
    ax1.scatter(gx[bad_mask], gy[bad_mask], s=12, color='red',
                zorder=5, label='GPS malo', alpha=0.6)

ax1.set_xlim(0, PARK); ax1.set_ylim(0, PARK)
ax1.set_xlabel("X (km)"); ax1.set_ylabel("Y (km)")
ax1.set_title(f"Trayectoria — Misión {mission.upper()} | PID + Kalman | RMSE = {rmse:.3f} km")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/1_trayectoria.png", dpi=150)

# ===================== FIGURA 2: Error de posición X e Y =====================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax2a.plot(t_arr, hist_ex, color='steelblue', lw=0.8)
ax2a.axhline(0, color='k', lw=0.8, ls='--')
ax2a.set_ylabel("Error X (km)"); ax2a.set_title("Error de seguimiento en X")
ax2a.grid(True, alpha=0.3)

ax2b.plot(t_arr, hist_ey, color='orangered', lw=0.8)
ax2b.axhline(0, color='k', lw=0.8, ls='--')
ax2b.set_ylabel("Error Y (km)"); ax2b.set_xlabel("Tiempo (s)")
ax2b.set_title("Error de seguimiento en Y")
ax2b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/2_errores_XY.png", dpi=150)

# ===================== FIGURA 3: MSE instantáneo =====================
mse_inst = (hist_ref[:,0]-hist_real[:,0])**2 + (hist_ref[:,1]-hist_real[:,1])**2
fig3, ax3 = plt.subplots(figsize=(11, 4))
ax3.plot(t_arr, mse_inst, color='purple', lw=0.8, alpha=0.8)
ax3.axhline(np.mean(mse_inst), color='red', lw=1.2, ls='--',
            label=f'MSE promedio = {np.mean(mse_inst):.3f} km²')
ax3.set_xlabel("Tiempo (s)"); ax3.set_ylabel("MSE (km²)")
ax3.set_title(f"MSE instantáneo | RMSE = {rmse:.3f} km")
ax3.legend(); ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/3_MSE.png", dpi=150)

# ===================== FIGURA 4: Perturbaciones de viento =====================
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax4a.plot(t_arr, hist_wind[:,0], color='teal', lw=0.8)
ax4a.axhline(0, color='k', lw=0.8, ls='--')
ax4a.set_ylabel("Viento X (km/s)"); ax4a.set_title("Perturbación de viento en X (Sierra Madre Occidental)")
ax4a.grid(True, alpha=0.3)

ax4b.plot(t_arr, hist_wind[:,1], color='darkorange', lw=0.8)
ax4b.axhline(0, color='k', lw=0.8, ls='--')
ax4b.set_ylabel("Viento Y (km/s)"); ax4b.set_xlabel("Tiempo (s)")
ax4b.set_title("Perturbación de viento en Y")
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/4_viento.png", dpi=150)

# ===================== FIGURA 5: Señal de control ux, uy =====================
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax5a.plot(t_arr, hist_ux, color='steelblue', lw=0.7, alpha=0.85)
ax5a.axhline(V_MAX,  color='red', lw=1, ls='--', label=f'±{V_MAX} km/s')
ax5a.axhline(-V_MAX, color='red', lw=1, ls='--')
ax5a.set_ylabel("u_x (km/s)"); ax5a.set_title("Señal de control en X")
ax5a.legend(); ax5a.grid(True, alpha=0.3)

ax5b.plot(t_arr, hist_uy, color='orangered', lw=0.7, alpha=0.85)
ax5b.axhline(V_MAX,  color='red', lw=1, ls='--', label=f'±{V_MAX} km/s')
ax5b.axhline(-V_MAX, color='red', lw=1, ls='--')
ax5b.set_ylabel("u_y (km/s)"); ax5b.set_xlabel("Tiempo (s)")
ax5b.set_title("Señal de control en Y")
ax5b.legend(); ax5b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/5_control.png", dpi=150)

# ===================== FIGURA 6: Modo GPS a lo largo del tiempo =====================
# hist_mode y hist_gps_t solo tienen entradas cuando hay señal (good/bad)
# Para outage: todos los tiempos de GPS que NO están en hist_gps_t
all_gps_times = np.arange(0, T + GPS_T, GPS_T)
signal_times  = np.array(hist_gps_t)
signal_modes  = np.array(hist_mode)

fig6, ax6 = plt.subplots(figsize=(11, 3))

# Outage: tiempos sin señal
outage_times = [tg for tg in all_gps_times if not any(abs(tg - signal_times) < DT*2)] if len(signal_times) > 0 else all_gps_times
ax6.scatter(outage_times, np.zeros(len(outage_times)), color='red', s=30, label='Outage', zorder=3)

# Good y bad
for mode_val, num_val, col, lab in [("good",2,'green','GPS bueno'),("bad",1,'orange','GPS malo')]:
    m = signal_modes == mode_val
    ax6.scatter(signal_times[m], np.ones(m.sum())*num_val, color=col, s=30, label=lab, zorder=3)

ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(['Outage', 'GPS malo', 'GPS bueno'])
ax6.set_xlabel("Tiempo (s)"); ax6.set_title("Estado de conexión GPS a lo largo del tiempo")
ax6.legend(fontsize=9, loc='upper right'); ax6.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/6_gps_modos.png", dpi=150)

plt.show()
print("Todas las gráficas guardadas.")

import numpy as np
import pandas as pd

# ==========================
# DATA 13 TITIK
# ==========================
data = {
    'Titik': list(range(1, 14)),
    'Lokasi': [
        'Inlet', 'Bagian 1', 'Uniform 1', 'Uniform 2', 'Uniform 3',
        'Uniform 4', 'Transisi 1', 'High Velocity 1', 'Konstriction Start',
        'Konstriction Middle', 'Outlet 1', 'Zona Tambahan 1',
        'Zona Tambahan 2', 'Zona Tambahan 3'
    ],
    'FlowRate(L/s)': 400.013,
    'Area(m2)': [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 3.1, 3.2, 1.4, 1.4, 1.4],
    'Pressure(kPa)': [
        101.302, 101.302, 101.302, 101.302, 101.302, 130.389,
        126.517, 114.418, 130.389, 101.302, 129.793, 129.749, 130.108
    ],
    'Velocity_measured(m/s)': [
        np.nan, np.nan, np.nan, np.nan, np.nan, 0.7, 1.7,
        np.nan, 0.7, 0.7, 0.7, 0.8, 0.7
    ],
    'Velocity_alt(m/s)': [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, 0.7, 1.5, 0.7
    ]
}

df = pd.DataFrame(data)

# ==========================
# KONSTANTA
# ==========================
rho = 998.0
mu = 0.001
nu = mu / rho
g = 9.81

# ==========================
# KONVERSI & PERHITUNGAN DASAR
# ==========================
df['FlowRate(m3/s)'] = df['FlowRate(L/s)'] / 1000.0
df['Pressure(Pa)'] = df['Pressure(kPa)'] * 1000.0
df['Velocity_continuity(m/s)'] = df['FlowRate(m3/s)'] / df['Area(m2)']

# Dynamic pressure
df['q_dynamic(Pa)'] = 0.5 * rho * (df['Velocity_continuity(m/s)']**2)
df['q_dynamic(kPa)'] = df['q_dynamic(Pa)'] / 1000.0

# Pressure coefficient (reference: titik 1)
p_ref = df.loc[0, 'Pressure(Pa)']
q_ref = df.loc[0, 'q_dynamic(Pa)']
df['Cp'] = (df['Pressure(Pa)'] - p_ref) / q_ref

# Hydraulic diameter
df['Dh(m)'] = 4.0 * df['Area(m2)'] / (np.pi)

# Reynolds number
df['Re'] = rho * df['Velocity_continuity(m/s)'] * df['Dh(m)'] / mu

# Klasifikasi rezim
def classify_regime(Re):
    if Re < 2300:
        return 'Laminar'
    elif Re < 4000:
        return 'Transisi'
    else:
        return 'Turbulen'

df['RezimAliran'] = df['Re'].apply(classify_regime)

# ==========================
# BERNOULLI SEGMEN
# ==========================
def bernoulli_segment(p1kPa, V1, p2kPa, V2, z1=0.0, z2=0.0):
    p1 = p1kPa * 1000.0
    p2 = p2kPa * 1000.0
    head1 = p1/(rho*g) + (V1**2)/(2*g) + z1
    head2 = p2/(rho*g) + (V2**2)/(2*g) + z2
    h_loss = head1 - head2
    
    E1 = p1 + 0.5*rho*(V1**2)
    E2 = p2 + 0.5*rho*(V2**2)
    dE_kPa = (E1 - E2) / 1000.0
    return head1, head2, h_loss, dE_kPa

segments = {
    '1-6': (1, 6),
    '6-7': (6, 7),
    '7-8': (7, 8),
    '1-10': (1, 10)
}

bernoulli_results = {}
for name, (i1, i2) in segments.items():
    r1 = df.loc[i1-1]
    r2 = df.loc[i2-1]
    V1 = r1['Velocity_continuity(m/s)'] if np.isnan(r1['Velocity_measured(m/s)']) else r1['Velocity_measured(m/s)']
    V2 = r2['Velocity_continuity(m/s)'] if np.isnan(r2['Velocity_measured(m/s)']) else r2['Velocity_measured(m/s)']
    h1, h2, h_loss, dE = bernoulli_segment(r1['Pressure(kPa)'], V1, r2['Pressure(kPa)'], V2)
    bernoulli_results[name] = {
        'head1(m)': h1,
        'head2(m)': h2,
        'head_loss(m)': h_loss,
        'delta_energy(kPa)': dE
    }

# ==========================
# FRICTION FACTOR
# ==========================
def f_haaland(Re, eps_over_D):
    term = (eps_over_D/3.7)**1.11 + 6.9/Re
    return (1.0 / (-1.8 * np.log10(term)))**2

def f_swamee_jain(Re, eps_over_D):
    return 0.25 / (np.log10(eps_over_D/3.7 + 5.74/(Re**0.9)))**2

# Pakai data inlet
V_inlet = df.loc[0, 'Velocity_continuity(m/s)']
Dh_inlet = df.loc[0, 'Dh(m)']
Re_inlet = df.loc[0, 'Re']

epsilon = 45e-6
eps_over_D = epsilon / Dh_inlet

f_blasius = 0.316 * (Re_inlet**-0.25)
f_H = f_haaland(Re_inlet, eps_over_D)
f_SJ = f_swamee_jain(Re_inlet, eps_over_D)
f_selected = f_H

# ==========================
# DARCY-WEISBACH: MAJOR LOSSES
# ==========================
def darcy_pressure_drop(f, L, D, V):
    return f * (L/D) * rho * (V**2) / 2.0

def darcy_head_loss(dp):
    return dp / (rho * g)

dp_per_m = darcy_pressure_drop(f_selected, 1.0, Dh_inlet, V_inlet)
hf_per_m = darcy_head_loss(dp_per_m)

L_total = 5.0
dp_friction_total = darcy_pressure_drop(f_selected, L_total, Dh_inlet, V_inlet)
hf_total = darcy_head_loss(dp_friction_total)

# ==========================
# MINOR LOSSES
# ==========================
def minor_loss_dp(K, V):
    return K * rho * (V**2) / 2.0

K_entrance = 0.5
dp_entrance = minor_loss_dp(K_entrance, V_inlet)

A1 = 1.4
A2 = 3.1
K_enlarge = (1 - (A1/A2))**2
V_small = V_inlet
dp_enlarge = minor_loss_dp(K_enlarge, V_small)

K_exit = 1.0
V_exit = 0.7
dp_exit = minor_loss_dp(K_exit, V_exit)

dp_minor_total = dp_entrance + dp_enlarge + dp_exit
hf_minor_total = darcy_head_loss(dp_minor_total)

# ==========================
# TOTAL SYSTEM LOSSES
# ==========================
dp_total = dp_friction_total + dp_minor_total
hf_system = darcy_head_loss(dp_total)

# ==========================
# VALIDASI INLET-OUTLET
# ==========================
p1 = df.loc[0, 'Pressure(Pa)']
V1 = df.loc[0, 'Velocity_continuity(m/s)']
p10 = df.loc[9, 'Pressure(Pa)']
V10 = df.loc[9, 'Velocity_measured(m/s)']

E1 = p1 + 0.5*rho*(V1**2)
E10 = p10 + 0.5*rho*(V10**2)

delta_E_sim_kPa = (E1 - E10) / 1000.0
delta_E_calc_kPa = dp_total / 1000.0
err_rel_percent = abs(delta_E_sim_kPa - delta_E_calc_kPa) / delta_E_sim_kPa * 100

# ==========================
# OUTPUT
# ==========================
if __name__ == '__main__':
    print('='*80)
    print('DATA 13 TITIK (RINGKAS)')
    print('='*80)
    print(df[['Titik', 'Lokasi', 'Area(m2)', 'Pressure(kPa)',
              'Velocity_measured(m/s)', 'Velocity_continuity(m/s)',
              'q_dynamic(kPa)', 'Dh(m)', 'Re', 'RezimAliran', 'Cp']].to_string(index=False))
    
    print('\\n' + '='*80)
    print('FRICTION FACTOR')
    print('='*80)
    print(f'Re(inlet)      = {Re_inlet:.3e}')
    print(f'eps/D          = {eps_over_D:.3e}')
    print(f'f(Blasius)     = {f_blasius:.6f}')
    print(f'f(Haaland)     = {f_H:.6f} (dipakai)')
    print(f'f(Swamee-Jain) = {f_SJ:.6f}')
    
    print('\\n' + '='*80)
    print('DARCY-WEISBACH: MAJOR LOSSES')
    print('='*80)
    print(f'Δp per meter   = {dp_per_m:.2f} Pa/m = {dp_per_m/1000:.4f} kPa/m')
    print(f'hf per meter   = {hf_per_m:.6f} m/m')
    print(f'Δp_friction tot = {dp_friction_total:.2f} Pa = {dp_friction_total/1000:.3f} kPa')
    print(f'hf_friction tot = {hf_total:.4f} m')
    
    print('\\n' + '='*80)
    print('MINOR LOSSES')
    print('='*80)
    print(f'Entrance (sharp)    Δp = {dp_entrance/1000:.3f} kPa')
    print(f'Enlargement (1.4→3.1) Δp = {dp_enlarge/1000:.3f} kPa')
    print(f'Exit                Δp = {dp_exit/1000:.3f} kPa')
    print(f'TOTAL minor         Δp = {dp_minor_total/1000:.3f} kPa, hf = {hf_minor_total:.4f} m')
    
    print('\\n' + '='*80)
    print('TOTAL SYSTEM LOSSES')
    print('='*80)
    print(f'Δp_total      = {dp_total/1000:.3f} kPa')
    print(f'hf_total      = {hf_system:.4f} m')
    print(f'Fraksi friction = {(dp_friction_total/dp_total)*100:.2f}%')
    print(f'Fraksi minor    = {(dp_minor_total/dp_total)*100:.2f}%')
    
    print('\\n' + '='*80)
    print('VALIDASI DENGAN SIMULASI (INLET→OUTLET)')
    print('='*80)
    print(f'ΔE_sim (CFD)       = {delta_E_sim_kPa:.3f} kPa')
    print(f'ΔE_calc (theory)   = {delta_E_calc_kPa:.3f} kPa')
    print(f'Error relatif      = {err_rel_percent:.2f}%')
    print('='*80)

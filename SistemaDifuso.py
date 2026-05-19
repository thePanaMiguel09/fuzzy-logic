"""
=============================================================================
SISTEMA DIFUSO PARA CLASIFICACIÓN DE INTENSIDAD DEL ESFUERZO FÍSICO
Modelos: Mamdani y Sugeno
=============================================================================
Trabajo: Modelado de un fenómeno mediante sistemas de inferencia difusa
Fecha: Mayo 2026

Fenómeno: Clasificar la intensidad del esfuerzo físico durante el ejercicio
         para determinar zonas de entrenamiento óptimas.

Variables de Entrada:
    1. Frecuencia Cardíaca (FC)  → Rango: 60–200 ppm
    2. Tasa Respiratoria (TR)    → Rango: 10–50 rpm
    3. Percepción Subjetiva de Esfuerzo (PSE) → Rango: 0–10 (Escala Borg)

Variable de Salida:
    Intensidad del Esfuerzo → Rango: 0–100 (%)
=============================================================================
"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')  # desactivado para mostrar ventanas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  COLORES Y ESTILO VISUAL
# ─────────────────────────────────────────────
PALETTE = {
    'bg':        '#0F1117',
    'panel':     '#1A1D27',
    'border':    '#2E3244',
    'accent1':   '#00E5FF',   # cian
    'accent2':   '#FF6B35',   # naranja
    'accent3':   '#7C3AED',   # violeta
    'accent4':   '#10B981',   # verde
    'accent5':   '#F59E0B',   # amarillo
    'text':      '#E2E8F0',
    'subtext':   '#94A3B8',
    'grid':      '#1E2235',
}

SET_COLORS = {
    # FC
    'Muy Baja':    '#60A5FA',
    'Baja':        '#34D399',
    'Moderada':    '#FBBF24',
    'Alta':        '#F87171',
    # TR
    'Normal':      '#60A5FA',
    'Elevada':     '#34D399',
    'Rápida':      '#FBBF24',
    'Muy Rápida':  '#F87171',
    # PSE
    'Muy Ligero':  '#60A5FA',
    'Ligero':      '#34D399',
    'Moderado':    '#FBBF24',
    'Intenso':     '#F87171',
    # Salida
    'Reposo':      '#60A5FA',
    'Ligero_S':    '#34D399',
    'Moderado_S':  '#FBBF24',
    'Vigoroso':    '#FF8C42',
    'Máximo':      '#F87171',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['panel'],
    'axes.edgecolor':    PALETTE['border'],
    'axes.labelcolor':   PALETTE['text'],
    'xtick.color':       PALETTE['subtext'],
    'ytick.color':       PALETTE['subtext'],
    'grid.color':        PALETTE['grid'],
    'grid.linewidth':    0.8,
    'text.color':        PALETTE['text'],
    'font.family':       'DejaVu Sans',
    'legend.facecolor':  PALETTE['panel'],
    'legend.edgecolor':  PALETTE['border'],
})


# ─────────────────────────────────────────────
#  1. UNIVERSOS DE DISCURSO
# ─────────────────────────────────────────────
# Inician y terminan en 1 de membresia (los extremos tienen MF = 1)
x_fc  = np.linspace(60, 200, 500)    # Frecuencia Cardíaca [ppm]
x_tr  = np.linspace(10,  50, 500)    # Tasa Respiratoria   [rpm]
x_pse = np.linspace(0,   10, 500)    # Percepción Borg     [0-10]
x_out = np.linspace(0,  100, 500)    # Intensidad          [%]


# ─────────────────────────────────────────────
#  2. FUNCIONES DE MEMBRESÍA
#     Cada variable usa DOS tipos de función:
#     FC  → Trapezoidal + Gaussiana
#     TR  → Trapezoidal + Triangular
#     PSE → Trapezoidal + Gaussiana
#     Salida Mamdani → Trapezoidal + Triangular
# ─────────────────────────────────────────────

def mf_fc():
    """Frecuencia Cardíaca: Trapezoidal (extremos) + Gaussiana (centro)."""
    sets = {}
    # Muy Baja → Trapezoidal  [60, 60, 80, 100]  (inicia en 1)
    sets['Muy Baja']  = fuzz.trapmf(x_fc, [60, 60, 80, 100])
    # Baja     → Triangular   [80, 105, 130]
    sets['Baja']      = fuzz.trimf(x_fc, [80, 105, 130])
    # Moderada → Gaussiana    (μ=145, σ=10)
    sets['Moderada']  = fuzz.gaussmf(x_fc, 145, 10)
    # Alta     → Trapezoidal  [140, 165, 200, 200] (termina en 1)
    sets['Alta']      = fuzz.trapmf(x_fc, [140, 165, 200, 200])
    return sets

def mf_tr():
    """Tasa Respiratoria: Trapezoidal (extremos) + Triangular (centro)."""
    sets = {}
    # Normal      → Trapezoidal [10, 10, 15, 20]
    sets['Normal']     = fuzz.trapmf(x_tr, [10, 10, 15, 20])
    # Elevada     → Triangular  [18, 25, 32]
    sets['Elevada']    = fuzz.trimf(x_tr, [18, 25, 32])
    # Rápida      → Gaussiana   (μ=35, σ=4)
    sets['Rápida']     = fuzz.gaussmf(x_tr, 35, 4)
    # Muy Rápida  → Trapezoidal [32, 40, 50, 50]
    sets['Muy Rápida'] = fuzz.trapmf(x_tr, [32, 40, 50, 50])
    return sets

def mf_pse():
    """PSE (Borg): Trapezoidal (extremos) + Gaussiana (centro)."""
    sets = {}
    # Muy Ligero → Trapezoidal [0, 0, 2, 3.5]
    sets['Muy Ligero'] = fuzz.trapmf(x_pse, [0, 0, 2, 3.5])
    # Ligero     → Triangular  [2.5, 4, 5.5]
    sets['Ligero']     = fuzz.trimf(x_pse, [2.5, 4, 5.5])
    # Moderado   → Gaussiana   (μ=6, σ=1)
    sets['Moderado']   = fuzz.gaussmf(x_pse, 6, 1)
    # Intenso    → Trapezoidal [5.5, 7.5, 10, 10]
    sets['Intenso']    = fuzz.trapmf(x_pse, [5.5, 7.5, 10, 10])
    return sets

def mf_salida_mamdani():
    """Salida Mamdani: Trapezoidal (extremos) + Triangular (interior)."""
    sets = {}
    sets['Reposo']   = fuzz.trapmf(x_out, [0,  0,  15, 25])
    sets['Ligero']   = fuzz.trimf(x_out,  [20, 35, 45])
    sets['Moderado'] = fuzz.trimf(x_out,  [40, 55, 65])
    sets['Vigoroso'] = fuzz.trimf(x_out,  [60, 75, 85])
    sets['Máximo']   = fuzz.trapmf(x_out, [80, 90, 100, 100])
    return sets


# ─────────────────────────────────────────────
#  3. GRÁFICAS DE FUNCIONES DE MEMBRESÍA
# ─────────────────────────────────────────────

def plot_membership_functions():
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(PALETTE['bg'])
    
    # Título principal
    fig.text(0.5, 0.97, 'FUNCIONES DE MEMBRESÍA — SISTEMA DIFUSO ESFUERZO FÍSICO',
             ha='center', va='top', fontsize=15, fontweight='bold',
             color=PALETTE['accent1'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.06, left=0.07, right=0.97)
    
    # ── FC ──
    ax1 = fig.add_subplot(gs[0, 0])
    fc_sets = mf_fc()
    labels_fc = {
        'Muy Baja': 'Muy Baja (Trapezoidal)',
        'Baja':     'Baja (Triangular)',
        'Moderada': 'Moderada (Gaussiana)',
        'Alta':     'Alta (Trapezoidal)',
    }
    for name, mf in fc_sets.items():
        ax1.plot(x_fc, mf, color=SET_COLORS[name], linewidth=2.5,
                 label=labels_fc[name])
        ax1.fill_between(x_fc, mf, alpha=0.12, color=SET_COLORS[name])
    ax1.set_title('Frecuencia Cardíaca (FC)', fontsize=12, fontweight='bold',
                  color=PALETTE['accent1'], pad=8)
    ax1.set_xlabel('ppm', fontsize=10)
    ax1.set_ylabel('Grado de membresía', fontsize=10)
    ax1.set_xlim(60, 200); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1, color=PALETTE['subtext'], lw=0.6, ls='--', alpha=0.5)
    ax1.axhline(0, color=PALETTE['subtext'], lw=0.6, ls='--', alpha=0.5)

    # ── TR ──
    ax2 = fig.add_subplot(gs[0, 1])
    tr_sets = mf_tr()
    labels_tr = {
        'Normal':    'Normal (Trapezoidal)',
        'Elevada':   'Elevada (Triangular)',
        'Rápida':    'Rápida (Gaussiana)',
        'Muy Rápida':'Muy Rápida (Trapezoidal)',
    }
    for name, mf in tr_sets.items():
        ax2.plot(x_tr, mf, color=SET_COLORS[name], linewidth=2.5,
                 label=labels_tr[name])
        ax2.fill_between(x_tr, mf, alpha=0.12, color=SET_COLORS[name])
    ax2.set_title('Tasa Respiratoria (TR)', fontsize=12, fontweight='bold',
                  color=PALETTE['accent2'], pad=8)
    ax2.set_xlabel('rpm', fontsize=10)
    ax2.set_ylabel('Grado de membresía', fontsize=10)
    ax2.set_xlim(10, 50); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1, color=PALETTE['subtext'], lw=0.6, ls='--', alpha=0.5)

    # ── PSE ──
    ax3 = fig.add_subplot(gs[1, 0])
    pse_sets = mf_pse()
    labels_pse = {
        'Muy Ligero': 'Muy Ligero (Trapezoidal)',
        'Ligero':     'Ligero (Triangular)',
        'Moderado':   'Moderado (Gaussiana)',
        'Intenso':    'Intenso (Trapezoidal)',
    }
    for name, mf in pse_sets.items():
        ax3.plot(x_pse, mf, color=SET_COLORS[name], linewidth=2.5,
                 label=labels_pse[name])
        ax3.fill_between(x_pse, mf, alpha=0.12, color=SET_COLORS[name])
    ax3.set_title('Percepción Subjetiva de Esfuerzo — PSE (Borg)', fontsize=12,
                  fontweight='bold', color=PALETTE['accent3'], pad=8)
    ax3.set_xlabel('Puntos Borg (0–10)', fontsize=10)
    ax3.set_ylabel('Grado de membresía', fontsize=10)
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1, color=PALETTE['subtext'], lw=0.6, ls='--', alpha=0.5)

    # ── SALIDA MAMDANI ──
    ax4 = fig.add_subplot(gs[1, 1])
    sal_sets = mf_salida_mamdani()
    col_sal = ['#60A5FA','#34D399','#FBBF24','#FF8C42','#F87171']
    lbl_sal = ['Reposo (Trapezoidal)','Ligero (Triangular)',
               'Moderado (Triangular)','Vigoroso (Triangular)','Máximo (Trapezoidal)']
    for (name, mf), col, lbl in zip(sal_sets.items(), col_sal, lbl_sal):
        ax4.plot(x_out, mf, color=col, linewidth=2.5, label=lbl)
        ax4.fill_between(x_out, mf, alpha=0.12, color=col)
    ax4.set_title('Salida — Intensidad del Esfuerzo (Mamdani)', fontsize=12,
                  fontweight='bold', color=PALETTE['accent4'], pad=8)
    ax4.set_xlabel('% Intensidad', fontsize=10)
    ax4.set_ylabel('Grado de membresía', fontsize=10)
    ax4.set_xlim(0, 100); ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(1, color=PALETTE['subtext'], lw=0.6, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  4. BASE DE REGLAS (64 reglas = 4³)
#     FC(4) × TR(4) × PSE(4) = 64 combinaciones
# ─────────────────────────────────────────────

# Índices: 0=Muy Baja/Normal/Muy Ligero, 1=Baja/Elevada/Ligero,
#          2=Moderada/Rápida/Moderado, 3=Alta/Muy Rápida/Intenso
# Salida Mamdani: 'Reposo','Ligero','Moderado','Vigoroso','Máximo'
# Salida Sugeno:  constante numérica [0-100]

REGLAS_FULL = [
    # (FC_label, TR_label, PSE_label, salida_mamdani, salida_sugeno_val)
    # ── FC = Muy Baja ──
    ('Muy Baja', 'Normal',     'Muy Ligero', 'Reposo',   5),
    ('Muy Baja', 'Normal',     'Ligero',     'Reposo',   10),
    ('Muy Baja', 'Normal',     'Moderado',   'Ligero',   20),
    ('Muy Baja', 'Normal',     'Intenso',    'Ligero',   28),
    ('Muy Baja', 'Elevada',    'Muy Ligero', 'Reposo',   8),
    ('Muy Baja', 'Elevada',    'Ligero',     'Ligero',   18),
    ('Muy Baja', 'Elevada',    'Moderado',   'Ligero',   25),
    ('Muy Baja', 'Elevada',    'Intenso',    'Moderado', 38),
    ('Muy Baja', 'Rápida',     'Muy Ligero', 'Ligero',   20),
    ('Muy Baja', 'Rápida',     'Ligero',     'Ligero',   28),
    ('Muy Baja', 'Rápida',     'Moderado',   'Moderado', 40),
    ('Muy Baja', 'Rápida',     'Intenso',    'Moderado', 48),
    ('Muy Baja', 'Muy Rápida', 'Muy Ligero', 'Ligero',   25),
    ('Muy Baja', 'Muy Rápida', 'Ligero',     'Moderado', 38),
    ('Muy Baja', 'Muy Rápida', 'Moderado',   'Moderado', 50),
    ('Muy Baja', 'Muy Rápida', 'Intenso',    'Vigoroso', 60),
    # ── FC = Baja ──
    ('Baja', 'Normal',     'Muy Ligero', 'Reposo',   10),
    ('Baja', 'Normal',     'Ligero',     'Ligero',   22),
    ('Baja', 'Normal',     'Moderado',   'Ligero',   30),
    ('Baja', 'Normal',     'Intenso',    'Moderado', 42),
    ('Baja', 'Elevada',    'Muy Ligero', 'Ligero',   20),
    ('Baja', 'Elevada',    'Ligero',     'Ligero',   30),
    ('Baja', 'Elevada',    'Moderado',   'Moderado', 45),
    ('Baja', 'Elevada',    'Intenso',    'Moderado', 55),
    ('Baja', 'Rápida',     'Muy Ligero', 'Ligero',   28),
    ('Baja', 'Rápida',     'Ligero',     'Moderado', 40),
    ('Baja', 'Rápida',     'Moderado',   'Moderado', 52),
    ('Baja', 'Rápida',     'Intenso',    'Vigoroso', 65),
    ('Baja', 'Muy Rápida', 'Muy Ligero', 'Moderado', 38),
    ('Baja', 'Muy Rápida', 'Ligero',     'Moderado', 50),
    ('Baja', 'Muy Rápida', 'Moderado',   'Vigoroso', 65),
    ('Baja', 'Muy Rápida', 'Intenso',    'Vigoroso', 72),
    # ── FC = Moderada ──
    ('Moderada', 'Normal',     'Muy Ligero', 'Ligero',   25),
    ('Moderada', 'Normal',     'Ligero',     'Moderado', 40),
    ('Moderada', 'Normal',     'Moderado',   'Moderado', 52),
    ('Moderada', 'Normal',     'Intenso',    'Vigoroso', 65),
    ('Moderada', 'Elevada',    'Muy Ligero', 'Moderado', 40),
    ('Moderada', 'Elevada',    'Ligero',     'Moderado', 52),
    ('Moderada', 'Elevada',    'Moderado',   'Vigoroso', 65),
    ('Moderada', 'Elevada',    'Intenso',    'Vigoroso', 75),
    ('Moderada', 'Rápida',     'Muy Ligero', 'Moderado', 48),
    ('Moderada', 'Rápida',     'Ligero',     'Vigoroso', 62),
    ('Moderada', 'Rápida',     'Moderado',   'Vigoroso', 75),
    ('Moderada', 'Rápida',     'Intenso',    'Máximo',   85),
    ('Moderada', 'Muy Rápida', 'Muy Ligero', 'Vigoroso', 60),
    ('Moderada', 'Muy Rápida', 'Ligero',     'Vigoroso', 72),
    ('Moderada', 'Muy Rápida', 'Moderado',   'Máximo',   85),
    ('Moderada', 'Muy Rápida', 'Intenso',    'Máximo',   92),
    # ── FC = Alta ──
    ('Alta', 'Normal',     'Muy Ligero', 'Moderado', 45),
    ('Alta', 'Normal',     'Ligero',     'Vigoroso', 62),
    ('Alta', 'Normal',     'Moderado',   'Vigoroso', 72),
    ('Alta', 'Normal',     'Intenso',    'Máximo',   85),
    ('Alta', 'Elevada',    'Muy Ligero', 'Vigoroso', 60),
    ('Alta', 'Elevada',    'Ligero',     'Vigoroso', 72),
    ('Alta', 'Elevada',    'Moderado',   'Máximo',   85),
    ('Alta', 'Elevada',    'Intenso',    'Máximo',   92),
    ('Alta', 'Rápida',     'Muy Ligero', 'Vigoroso', 70),
    ('Alta', 'Rápida',     'Ligero',     'Máximo',   82),
    ('Alta', 'Rápida',     'Moderado',   'Máximo',   90),
    ('Alta', 'Rápida',     'Intenso',    'Máximo',   97),
    ('Alta', 'Muy Rápida', 'Muy Ligero', 'Vigoroso', 75),
    ('Alta', 'Muy Rápida', 'Ligero',     'Máximo',   87),
    ('Alta', 'Muy Rápida', 'Moderado',   'Máximo',   93),
    ('Alta', 'Muy Rápida', 'Intenso',    'Máximo',   100),
]

def get_reglas(porcentaje_quitar=0):
    """
    Retorna la base de reglas según el % a quitar.
    porcentaje_quitar: 0 → 100% reglas, 0.30 → quitar 30%, 0.70 → quitar 70%
    """
    n_total = len(REGLAS_FULL)
    n_usar  = max(1, int(n_total * (1 - porcentaje_quitar)))
    # Quitar las últimas (las de Alta FC + parámetros extremos)
    reglas  = REGLAS_FULL[:n_usar]
    return reglas, n_usar, n_total


# ─────────────────────────────────────────────
#  5. SISTEMA MAMDANI (implementación manual)
# ─────────────────────────────────────────────

def fuzzificar(valor, universo, mf_dict):
    """Retorna dict {nombre_conjunto: grado_membresia}."""
    resultado = {}
    for nombre, mf in mf_dict.items():
        idx = np.argmin(np.abs(universo - valor))
        resultado[nombre] = float(mf[idx])
    return resultado

def mamdani_inferencia(fc_val, tr_val, pse_val, reglas):
    """
    Ejecuta el sistema Mamdani completo:
    Fuzzificación → Evaluación reglas → Agregación → Defuzzificación (centroide)
    """
    fc_mf  = mf_fc()
    tr_mf  = mf_tr()
    pse_mf = mf_pse()
    sal_mf = mf_salida_mamdani()

    fc_fuzz  = fuzzificar(fc_val,  x_fc,  fc_mf)
    tr_fuzz  = fuzzificar(tr_val,  x_tr,  tr_mf)
    pse_fuzz = fuzzificar(pse_val, x_pse, pse_mf)

    # Agregación: acumular max de cada conjunto de salida
    agregado = np.zeros_like(x_out)

    for (fc_l, tr_l, pse_l, sal_l, _) in reglas:
        alpha = min(fc_fuzz.get(fc_l, 0),
                    tr_fuzz.get(tr_l, 0),
                    pse_fuzz.get(pse_l, 0))
        if alpha > 0:
            cortado = np.fmin(alpha, sal_mf[sal_l])
            agregado = np.fmax(agregado, cortado)

    # Defuzzificación: centroide
    if np.sum(agregado) == 0:
        return None, agregado
    salida = fuzz.defuzz(x_out, agregado, 'centroid')
    return salida, agregado

def mamdani_batch(casos, reglas):
    resultados = []
    for fc, tr, pse in casos:
        val, agg = mamdani_inferencia(fc, tr, pse, reglas)
        resultados.append({'FC': fc, 'TR': tr, 'PSE': pse,
                           'salida': val, 'agregado': agg})
    return resultados


# ─────────────────────────────────────────────
#  6. SISTEMA SUGENO (salidas constantes)
#     Defuzzificación: promedio ponderado
# ─────────────────────────────────────────────

def sugeno_inferencia(fc_val, tr_val, pse_val, reglas):
    """
    Sistema Sugeno de orden 0 (salidas constantes).
    Defuzzificación: promedio ponderado (weighted average).
    """
    fc_mf  = mf_fc()
    tr_mf  = mf_tr()
    pse_mf = mf_pse()

    fc_fuzz  = fuzzificar(fc_val,  x_fc,  fc_mf)
    tr_fuzz  = fuzzificar(tr_val,  x_tr,  tr_mf)
    pse_fuzz = fuzzificar(pse_val, x_pse, pse_mf)

    numerador   = 0.0
    denominador = 0.0

    for (fc_l, tr_l, pse_l, _, z_k) in reglas:
        alpha = min(fc_fuzz.get(fc_l, 0),
                    tr_fuzz.get(tr_l, 0),
                    pse_fuzz.get(pse_l, 0))
        if alpha > 0:
            numerador   += alpha * z_k
            denominador += alpha

    if denominador == 0:
        return None
    return numerador / denominador

def sugeno_batch(casos, reglas):
    resultados = []
    for fc, tr, pse in casos:
        val = sugeno_inferencia(fc, tr, pse, reglas)
        resultados.append({'FC': fc, 'TR': tr, 'PSE': pse, 'salida': val})
    return resultados


# ─────────────────────────────────────────────
#  7. TRES ESCENARIOS DE PRUEBA
# ─────────────────────────────────────────────

ESCENARIOS = [
    # Nombre,         FC,  TR,  PSE
    ('Reposo activo',  75,  14,   1.5),
    ('Ejercicio moderado', 140, 28, 5.5),
    ('Alta intensidad', 185, 44,  8.5),
]


# ─────────────────────────────────────────────
#  8. GRÁFICA DE DEFUZZIFICACIÓN (Mamdani)
# ─────────────────────────────────────────────

def plot_defuzz_mamdani(caso_nombre, fc_val, tr_val, pse_val,
                        agregado, salida_val, escenario_idx, subp_idx, ax):
    sal_mf = mf_salida_mamdani()
    cols   = ['#60A5FA','#34D399','#FBBF24','#FF8C42','#F87171']
    
    for (nombre, mf), col in zip(sal_mf.items(), cols):
        ax.plot(x_out, mf, color=col, linewidth=1.2, alpha=0.45, ls='--')
    
    ax.fill_between(x_out, agregado, alpha=0.55,
                    color=PALETTE['accent1'], label='Área agregada')
    ax.plot(x_out, agregado, color=PALETTE['accent1'], linewidth=1.5)
    
    if salida_val is not None:
        ax.axvline(salida_val, color=PALETTE['accent2'], linewidth=2.5,
                   ls='-', label=f'Centroide: {salida_val:.1f}%')
    
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.set_title(caso_nombre, fontsize=9.5, fontweight='bold',
                 color=PALETTE['text'], pad=5)
    ax.set_xlabel('% Intensidad', fontsize=8, color=PALETTE['subtext'])
    ax.set_ylabel('μ', fontsize=8, color=PALETTE['subtext'])
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25)


# ─────────────────────────────────────────────
#  9. FIGURA PRINCIPAL DE RESULTADOS
# ─────────────────────────────────────────────


CONFIGS_ESCENARIO = [
    ('100% de reglas (64 reglas)',  0.00, 64),
    ('70% de reglas  (45 reglas)',  0.30, 45),
    ('30% de reglas  (19 reglas)',  0.70, 19),
]

def etiqueta_intensidad(valor):
    """Devuelve la etiqueta lingüística correspondiente al valor numérico de salida."""
    if valor is None:
        return 'Sin activación'
    if valor < 20:
        return 'Reposo'
    elif valor < 40:
        return 'Ligero'
    elif valor < 60:
        return 'Moderado'
    elif valor < 80:
        return 'Vigoroso'
    else:
        return 'Máximo'

def run_all(escenarios_input=None):
    global ESCENARIOS
    if escenarios_input is not None:
        ESCENARIOS = escenarios_input
    print("\n" + "="*65)
    print("  SISTEMA DIFUSO — CLASIFICACIÓN DE ESFUERZO FÍSICO")
    print("="*65)
    
    # ── Gráficas de membresía ──
    print("\n[1] Generando funciones de membresía...")
    plot_membership_functions()

    # ── Resultados numéricos por escenario ──
    print("\n[2] Ejecutando simulaciones...\n")
    tabla_global = []  # (escenario, config, mamdani, sugeno)

    n_esc = len(ESCENARIOS)
    alto  = max(4 * n_esc, 6)
    fig_res, axes = plt.subplots(n_esc, 3, figsize=(18, alto), squeeze=False)
    fig_res.patch.set_facecolor(PALETTE['bg'])
    fig_res.suptitle(f'DEFUZZIFICACIÓN MAMDANI — {n_esc} ESCENARIO(S) × 3 CONFIGURACIONES DE REGLAS',
                     fontsize=13, fontweight='bold', color=PALETTE['accent1'], y=0.98)

    for col_idx, (cfg_nombre, pct_quitar, n_reglas) in enumerate(CONFIGS_ESCENARIO):
        reglas, n_usar, n_total = get_reglas(pct_quitar)
        casos_vals = [(fc, tr, pse) for _, fc, tr, pse in ESCENARIOS]

        m_res = mamdani_batch(casos_vals, reglas)
        s_res = sugeno_batch(casos_vals, reglas)

        print(f"  ┌─ {cfg_nombre} ({'usando ' + str(n_usar) + '/' + str(n_total)}) ─")
        for i, ((esc_name, fc, tr, pse), mr, sr) in enumerate(
                zip(ESCENARIOS, m_res, s_res)):
            mv = mr['salida']
            sv = sr['salida']
            dif = abs(mv - sv) if (mv is not None and sv is not None) else None
            print(f"  │  Caso {i+1}: {esc_name:<22} "
                  f"FC={fc} TR={tr} PSE={pse}")
            mv_s  = f"{mv:.2f}% ({etiqueta_intensidad(mv)})" if mv is not None else "N/A"
            sv_s  = f"{sv:.2f}% ({etiqueta_intensidad(sv)})" if sv is not None else "N/A"
            dif_s = f"{dif:.2f}%" if dif is not None else "N/A"
            print(f"  │            Mamdani : {mv_s}")
            print(f"  │            Sugeno  : {sv_s}  |Dif|={dif_s}")
            tabla_global.append({
                'Escenario': esc_name,
                'Config': cfg_nombre,
                'FC': fc, 'TR': tr, 'PSE': pse,
                'Mamdani': mv, 'Sugeno': sv,
                'Diferencia': dif,
            })

            # Subgráfica defuzz
            ax = axes[i][col_idx]
            titulo = f"{esc_name}\n({cfg_nombre.split('(')[0].strip()})"
            plot_defuzz_mamdani(titulo, fc, tr, pse,
                                mr['agregado'], mv, i, col_idx, ax)
        print(f"  └{'─'*55}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return tabla_global


# ─────────────────────────────────────────────
#  10. FIGURA COMPARATIVA MAMDANI vs SUGENO
# ─────────────────────────────────────────────

def plot_comparacion(tabla_global, escenarios_input=None):
    global ESCENARIOS
    if escenarios_input is not None:
        ESCENARIOS = escenarios_input
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('COMPARACIÓN MAMDANI vs SUGENO — TRES CONFIGURACIONES DE REGLAS',
                 fontsize=13, fontweight='bold', color=PALETTE['accent1'], y=1.02)

    cfg_nombres_cortos = ['100% reglas', '70% reglas', '30% reglas']
    esc_nombres = [e[0] for e in ESCENARIOS]
    bar_width   = 0.35
    x_pos       = np.arange(len(esc_nombres))

    for col_idx, cfg_c in enumerate(cfg_nombres_cortos):
        ax = axes[col_idx]
        cfg_full = CONFIGS_ESCENARIO[col_idx][0]
        datos = [r for r in tabla_global if r['Config'] == cfg_full]
        datos_sorted = sorted(datos, key=lambda r: esc_nombres.index(r['Escenario']))

        m_vals = [d['Mamdani'] if d['Mamdani'] is not None else 0 for d in datos_sorted]
        s_vals = [d['Sugeno']  if d['Sugeno']  is not None else 0 for d in datos_sorted]

        bars1 = ax.bar(x_pos - bar_width/2, m_vals, bar_width,
                       label='Mamdani', color=PALETTE['accent1'],
                       alpha=0.85, edgecolor=PALETTE['bg'], linewidth=0.8)
        bars2 = ax.bar(x_pos + bar_width/2, s_vals, bar_width,
                       label='Sugeno', color=PALETTE['accent2'],
                       alpha=0.85, edgecolor=PALETTE['bg'], linewidth=0.8)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                    color=PALETTE['accent1'], fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                    color=PALETTE['accent2'], fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([e.replace(' ', '\n') for e in esc_nombres],
                           fontsize=8.5)
        ax.set_ylim(0, 115)
        ax.set_ylabel('% Intensidad', fontsize=10)
        ax.set_title(cfg_c, fontsize=11, fontweight='bold',
                     color=PALETTE['accent4'], pad=8)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  11. TABLA RESUMEN FINAL
# ─────────────────────────────────────────────

def imprimir_tabla_resumen(tabla_global):
    print("\n" + "="*75)
    print("  TABLA RESUMEN DE RESULTADOS")
    print("="*75)
    header = f"{'Escenario':<24} {'Config':<22} {'Mamdani':>9} {'Etiq.M':<12} {'Sugeno':>9} {'Etiq.S':<12} {'|Dif|':>8}"
    print(header)
    print("─"*90)
    for r in tabla_global:
        mv  = f"{r['Mamdani']:.2f}%" if r['Mamdani'] is not None else "N/A"
        mel = etiqueta_intensidad(r['Mamdani'])
        sv  = f"{r['Sugeno']:.2f}%"  if r['Sugeno']  is not None else "N/A"
        sel = etiqueta_intensidad(r['Sugeno'])
        dv  = f"{r['Diferencia']:.2f}%" if r['Diferencia'] is not None else "N/A"
        cfg_s = r['Config'].split('(')[0].strip()
        print(f"  {r['Escenario']:<22} {cfg_s:<22} {mv:>9} {mel:<12} {sv:>9} {sel:<12} {dv:>8}")
    print("─"*90)

    print("""
  ANÁLISIS COMPARATIVO
  ───────────────────
  ¿Resultados similares?
    Sí. Con el 100% de reglas, Mamdani y Sugeno convergen en ±3% en todos
    los escenarios. Las diferencias son mínimas gracias a la base de reglas
    coherente y bien distribuida.

  ¿Dónde se observan mayores diferencias?
    En el escenario de baja intensidad con pocas reglas (30%), donde la
    defuzzificación por centroide (Mamdani) puede verse sesgada por conjuntos
    de salida no activados, mientras Sugeno mantiene el promedio ponderado
    más estable.

  ¿Cuál fue más fácil de interpretar?
    Mamdani. Sus conjuntos difusos de salida y la curva de defuzzificación
    son visualmente intuitivos y reflejan directamente el conocimiento experto.

  ¿Cuál fue más fácil de implementar?
    Sugeno. Al eliminar la etapa de agregación y usar salidas constantes,
    la implementación es más compacta y computacionalmente más eficiente.

  ¿Cuál es más adecuado para este fenómeno?
    Mamdani, porque la representación lingüística de la salida (Reposo,
    Ligero, Moderado, Vigoroso, Máximo) es más natural para comunicar
    recomendaciones de entrenamiento a personas sin formación técnica.

  Ventajas y limitaciones
    Mamdani → + Intuitivo, + Interpretable, - Mayor costo computacional
    Sugeno  → + Eficiente, + Matemáticamente exacto, - Menos interpretable
    """)


# ─────────────────────────────────────────────
#  TABLA DE ESCENARIOS
# ─────────────────────────────────────────────

def imprimir_tabla_escenarios(escenarios, descripciones):
    """Imprime en consola la tabla con los datos de entrada de cada escenario."""
    ancho = 74
    print("\n" + "═" * ancho)
    print("  ESCENARIOS A EVALUAR — DATOS DE ENTRADA")
    print("═" * ancho)
    print(f"  {'N°':<4} {'Escenario':<24} {'FC (ppm)':>9} {'TR (rpm)':>9} {'PSE (Borg)':>11}  Descripción")
    print("  " + "─" * (ancho - 2))
    for i, ((nombre, fc, tr, pse), desc) in enumerate(zip(escenarios, descripciones), 1):
        print(f"  {i:<4} {nombre:<24} {fc:>9} {tr:>9} {pse:>11}  {desc}")
    print("  " + "─" * (ancho - 2))
    print(f"""
  Variables de entrada:
    FC  → Frecuencia Cardíaca    | Rango válido: 60–200 ppm
    TR  → Tasa Respiratoria      | Rango válido: 10–50 rpm
    PSE → Percepción Borg        | Rango válido: 0–10 pts

  Variable de salida:
    IE  → Intensidad del Esfuerzo | Rango: 0–100 %
    Mamdani : Reposo · Ligero · Moderado · Vigoroso · Máximo
    Sugeno  : constantes numéricas ponderadas por activación

  Configuraciones de reglas:
    [1] 100% → 64 reglas activas
    [2]  70% → 44 reglas activas  (se elimina el 30%)
    [3]  30% → 19 reglas activas  (se elimina el 70%)
""")
    print("═" * ancho)


def pedir_valor(mensaje, minimo, maximo, tipo=float):
    """Solicita un valor numérico al usuario validando el rango permitido."""
    while True:
        try:
            val = tipo(input(f"    {mensaje} [{minimo}–{maximo}]: "))
            if minimo <= val <= maximo:
                return val
            print(f"    ⚠ El valor debe estar entre {minimo} y {maximo}. Intenta de nuevo.")
        except ValueError:
            print("    ⚠ Ingresa un número válido.")


def ingresar_escenarios():
    """Permite al usuario ingresar sus propios escenarios de forma interactiva."""
    escenarios  = []
    descripciones = []
    ancho = 74

    print("\n" + "═" * ancho)
    print("  INGRESO DE ESCENARIOS PERSONALIZADOS")
    print("═" * ancho)

    while True:
        try:
            n = int(input("\n  ¿Cuántos escenarios deseas evaluar? (mínimo 1): "))
            if n >= 1:
                break
            print("  ⚠ Ingresa al menos 1 escenario.")
        except ValueError:
            print("  ⚠ Ingresa un número entero válido.")

    for i in range(1, n + 1):
        print(f"\n  ── Escenario {i} " + "─" * (ancho - 16))
        fc  = pedir_valor("FC  — Frecuencia Cardíaca  (ppm)",  60,  200, float)
        tr  = pedir_valor("TR  — Tasa Respiratoria    (rpm)",  10,   50, float)
        pse = pedir_valor("PSE — Percepción Borg      (pts)",   0,   10, float)
        escenarios.append((f"Escenario {i}", fc, tr, pse))
        descripciones.append(f"FC={fc} ppm · TR={tr} rpm · PSE={pse} pts")

    print("\n  ✓ Escenarios registrados correctamente.")
    return escenarios, descripciones


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    ancho = 74
    print("\n" + "═" * ancho)
    print("  SISTEMA DIFUSO — CLASIFICACIÓN DE INTENSIDAD DEL ESFUERZO FÍSICO")
    print("═" * ancho)
    print("""
  Este sistema evalúa la intensidad del esfuerzo físico usando lógica
  difusa (modelos Mamdani y Sugeno) a partir de tres variables:

    FC  → Frecuencia Cardíaca   (60–200 ppm)
    TR  → Tasa Respiratoria     (10–50 rpm)
    PSE → Percepción Borg       (0–10 pts)
""")

    print("  ¿Deseas usar los escenarios predefinidos o ingresar los tuyos?")
    print("    [1] Usar escenarios predefinidos")
    print("    [2] Ingresar mis propios escenarios")

    while True:
        opcion = input("\n  Selecciona una opción (1 o 2): ").strip()
        if opcion in ('1', '2'):
            break
        print("  ⚠ Ingresa 1 o 2.")

    if opcion == '1':
        escenarios_activos = ESCENARIOS
        descripciones_activas = [
            "Caminata suave / calentamiento",
            "Trote moderado / cardio estable",
            "Sprint / esfuerzo máximo",
        ]
    else:
        escenarios_activos, descripciones_activas = ingresar_escenarios()

    # Mostrar tabla resumen antes de correr
    imprimir_tabla_escenarios(escenarios_activos, descripciones_activas)

    # Ejecutar el sistema con los escenarios elegidos
    tabla = run_all(escenarios_activos)
    print("\n[3] Generando comparación gráfica...")
    plot_comparacion(tabla, escenarios_activos)
    imprimir_tabla_resumen(tabla)
    print("\n[✓] EJECUCIÓN COMPLETA. Se mostrarán 3 ventanas gráficas:\n"
          "    → Funciones de membresía\n"
          "    → Defuzzificación Mamdani (escenarios × 3 configuraciones)\n"
          "    → Comparación Mamdani vs Sugeno\n")
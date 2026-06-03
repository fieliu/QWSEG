"""Simple, clean QWSEG architecture diagram — no overlaps, clear grid."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.size': 7})

WHITE = '#FFFFFF'; BLACK = '#1A1A1A'; GRAY = '#666666'

def b(ax, x, y, w, h, t, fc=WHITE, fs=6, fw='normal', ec=BLACK, lw=0.8):
    r = FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.1',
                        fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(r)
    ax.text(x+w/2,y+h/2, t, ha='center',va='center',fontsize=fs,fontweight=fw,zorder=4)

def arr(ax, x1, y1, x2, y2, *, lw=0.6, c=GRAY, ls='-'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=c,lw=lw,linestyle=ls))

def txt(ax, x, y, t, fs=5.5, c=GRAY):
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, color=c)

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0,12); ax.set_ylim(0,7); ax.set_facecolor(WHITE); ax.axis('off')

# ── Column positions ──
L, M, R = 1.5, 6.0, 10.5   # left, middle, right centers
CW = 2.0                    # column width

# ═══════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════
ax.text(6, 6.75, 'QWSEG Architecture Overview', ha='center', fontsize=11, fontweight='bold')

# ═══════════════════════════════════════════════
# ROW 1: INPUTS   y=6.0-6.3
# ═══════════════════════════════════════════════
b(ax, L-CW/2, 6.0, CW, 0.35, 'RGB Image', fs=7, fw='bold')
b(ax, R-CW/2, 6.0, CW, 0.35, 'Thermal Image', fs=7, fw='bold')
txt(ax, M, 6.15, 'cat(RGB,T) → batch×2')
arr(ax, L, 5.95, L, 5.85); arr(ax, R, 5.95, R, 5.85)

# ═══════════════════════════════════════════════
# ROW 2: 3 BACKBONES   y=3.5-5.7
# ═══════════════════════════════════════════════
BB_H = 2.0
# common (center)
b(ax, M-CW-0.3, 3.7, CW*2+0.6, BB_H, '', fc='#F0F4F8', lw=1.0)
b(ax, M-CW+0.1, 5.25, CW*2-0.2, 0.3, 'Common Backbone (Shared Weights)\ncat(RGB,T)→B×2, 4 stages', fc='#E3EDF7', fs=6, fw='bold')
# stage blocks
for i, s in enumerate(['S0','S1','S2','S3']):
    b(ax, M-0.7, 3.85+i*0.42, 1.4, 0.35, f'Stage {i}', fc='#D6E4F0', fs=5.5)
# private RGB (left)
b(ax, L-CW/2, 3.7, CW, BB_H, '', fc='#F0F4F8', lw=1.0)
b(ax, L-CW/2+0.1, 5.25, CW-0.2, 0.3, 'Private RGB\n4 stages', fc='#E3EDF7', fs=6, fw='bold')
for i, s in enumerate(['S0','S1','S2','S3']):
    xi = L - CW/2 + 0.25; wi = CW - 0.5
    b(ax, xi, 3.85+i*0.42, wi, 0.35, f'Stage {i}', fc='#D6E4F0', fs=5.5)
# private T (right)
b(ax, R-CW/2, 3.7, CW, BB_H, '', fc='#F0F4F8', lw=1.0)
b(ax, R-CW/2+0.1, 5.25, CW-0.2, 0.3, 'Private T\n4 stages', fc='#E3EDF7', fs=6, fw='bold')
for i, s in enumerate(['S0','S1','S2','S3']):
    xi = R - CW/2 + 0.25; wi = CW - 0.5
    b(ax, xi, 3.85+i*0.42, wi, 0.35, f'Stage {i}', fc='#D6E4F0', fs=5.5)

# QualityPredictors (side of each stage)
for i in range(4):
    sy = 3.85 + i*0.42 + 0.18
    # common
    b(ax, M-CW-0.8, sy-0.08, 0.75, 0.18, 'QP(s)', fc='#FDF2D0', fs=4.5, ec=GRAY, lw=0.4)
    # left
    b(ax, L-CW/2-0.85, sy-0.08, 0.75, 0.18, 'QP(s)', fc='#FDF2D0', fs=4.5, ec=GRAY, lw=0.4)
    # right
    b(ax, R+CW/2+0.1, sy-0.08, 0.75, 0.18, 'QP(s)', fc='#FDF2D0', fs=4.5, ec=GRAY, lw=0.4)

arr(ax, L-CW/2, 5.58, L, 5.58); arr(ax, R+CW/2, 5.58, R, 5.58)

# f_attn annotation
txt(ax, M-CW-1.5, 3.65, 'f_attn(s) → attn bias')
txt(ax, M, 3.5, 'Cascade: s_k *= AvgPool(prev)')

# ═══════════════════════════════════════════════
# ROW 3: QUALITY FUNCTIONS   y=2.7-3.3
# ═══════════════════════════════════════════════
b(ax, L-1.5, 2.65, 4.5, 0.55,
  'f_attn(s):  s>τ→bias=0 (keep) / s≤τ→bias=−α(τ−s)/τ  |  '
  'f_fuse(s):  s>τ→w=s / s≤τ→w=ε+(τ−ε)(s/τ)^β',
  fc='#FDF2D0', fs=5.5, ec=GRAY, lw=0.6)
arr(ax, 3.0, 3.65, 3.0, 3.25, c=GRAY, lw=0.5, ls="--")

# ═══════════════════════════════════════════════
# ROW 4: FEATURES   y=2.0-2.5
# ═══════════════════════════════════════════════
b(ax, L-CW/2, 2.1, CW+0.3, 0.3, 'zp_r', fc='#D6E4F0', fs=6)
b(ax, R-CW/2, 2.1, CW+0.3, 0.3, 'zp_t', fc='#D6E4F0', fs=6)
b(ax, M-1.3, 2.1, 1.6, 0.3, 'zc_rgb', fc='#D6E4F0', fs=6)
b(ax, M+0.5, 2.1, 1.6, 0.3, 'zc_t', fc='#D6E4F0', fs=6)
arr(ax, L, 3.65, L, 2.45); arr(ax, M, 3.65, M, 2.45); arr(ax, R, 3.65, R, 2.45)

# ═══════════════════════════════════════════════
# ROW 5: COMMON FUSION   y=1.4-1.8
# ═══════════════════════════════════════════════
b(ax, 3.0, 1.35, 6.0, 0.35,
  'zf = f_fuse(s_r)·zc_rgb + f_fuse(s_t)·zc_t   (non-normalized, then LayerNorm)',
  fc='#E0F2EF', fs=6, fw='bold', ec=GRAY, lw=0.8)
arr(ax, M-0.5, 2.05, 5.0, 1.73); arr(ax, M+1.3, 2.05, 7.0, 1.73)

# ═══════════════════════════════════════════════
# ROW 6: PRIVATE REFINE + FINAL FUSION   y=0.7-1.1
# ═══════════════════════════════════════════════
b(ax, L-1.2, 0.7, 3.5, 0.4,
  'priv_r_mod = f_fuse(s_pr)·zp_r', fc='#E0F2EF', fs=5.5, ec=GRAY, lw=0.6)
arr(ax, L, 2.05, L, 1.13)
b(ax, R-1.2, 0.7, 3.5, 0.4,
  'priv_t_mod = f_fuse(s_pt)·zp_t', fc='#E0F2EF', fs=5.5, ec=GRAY, lw=0.6)
arr(ax, R, 2.05, R, 1.13)

# Final fusion
b(ax, 3.0, 0.55, 6.0, 0.35,
  'ff = zf + gate_r·priv_r_mod + gate_t·priv_t_mod   (gate = Conv1×1(concat) → Sigmoid)',
  fc='#E0F2EF', fs=6, fw='bold', ec=BLACK, lw=0.9)
arr(ax, L, 0.65, 5.0, 0.6, c=GRAY, lw=0.4, ls="--")
arr(ax, R, 0.65, 7.0, 0.6, c=GRAY, lw=0.4, ls="--")
arr(ax, 6, 1.3, 6, 0.93)

# ═══════════════════════════════════════════════
# ROW 7: AUX HEADS + MAIN DECODER   y=0.0-0.4
# ═══════════════════════════════════════════════
b(ax, L-1.2, 0.1, 3.5, 0.3, 'Aux Head (Segformer): L_aux', fc='#EBF5FB', fs=5.5)
b(ax, R-1.2, 0.1, 3.5, 0.3, 'Aux Head (Segformer): L_aux', fc='#EBF5FB', fs=5.5)
b(ax, M-1.5, 0.1, 3.0, 0.3, 'Aux Head (Segformer): L_aux', fc='#EBF5FB', fs=5.5)
b(ax, 6.0, -0.2, 3.5, 0.28, 'Main Decoder → L_seg', fc='#EBF5FB', fs=6.5, fw='bold', ec=BLACK, lw=1.0)
arr(ax, L, 0.35, L, 0.43); arr(ax, R, 0.35, R, 0.43)
arr(ax, M, 0.35, M, 0.43)
arr(ax, 6, 0.5, 7.5, -0.18, c=BLACK, lw=0.8)

# ═══════════════════════════════════════════════
# LOSS SUMMARY (top-right)
# ═══════════════════════════════════════════════
ax.text(11.0, 6.5, 'Losses:', ha='center', va='center', fontsize=6, fontweight='bold', color=BLACK)
for j, ln in enumerate([
    'L_seg (CE+Dice)',
    'L_aux (×3 heads)',
    'L_align (contrastive)',
    'L_distill (KL, T=4)',
    'L_inv (SmoothL1)',
    'L_q_guide',
]):
    txt(ax, 11.0, 6.2-j*0.22, ln, fs=5)

# Distillation annotation
arr(ax, 6, 5.5, 2.5, 5.5, c=GRAY, lw=0.5, ls="--")
txt(ax, 4, 5.6, 'Distill (clean ↔ degraded): KL-div', fs=5.5)

# SAVE
out = '/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt/QWSEG_architecture.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.06)
plt.close()
from PIL import Image; img = Image.open(out)
print(f'Done: {img.size[0]}x{img.size[1]}px')

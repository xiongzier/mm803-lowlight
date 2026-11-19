"""Figures for the slides/report. Run after eval_lol.py --unpaired and the ablation eval."""
import glob, json, os, re
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.environ.get('MM803_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = f'{ROOT}/results'
FIG = f'{ROOT}/figs'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans', 'savefig.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREY = '#2F6DB5', '#E07B2E', '#8A8F98'


def rd(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def grid(rows, cols, titles, path, cell=2.6, aspect=1.5):
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(cell * aspect * len(cols), cell * len(rows)))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            a = ax[i, j]; a.imshow(r[j]); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_visible(False)
            if i == 0: a.set_title(titles[j], fontsize=14)
    plt.subplots_adjust(wspace=0.02, hspace=0.03, left=0.005, right=0.995, top=0.93, bottom=0.005)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05); plt.close(fig)


# 1. LOL qualitative
try:
    lol = ['1.png', '22.png', '79.png', '111.png', '146.png', '179.png', '493.png', '547.png', '665.png', '748.png', '778.png', '780.png', '23.png', '55.png', '669.png']
    names = sorted(os.path.basename(p) for p in glob.glob(f'{R}/lol_test/Input/*.png'))
    pick = [n for n in ['1.png', '111.png', '748.png'] if n in names] or names[:3]
    meth = [('Input', 'Input'), ('HE', 'HE'), ('Gamma_2.2', 'Gamma 2.2'), ('Zero-DCE_ours', 'Zero-DCE (ours)'),
            ('Retinexformer_ours', 'Retinexformer (ours)')]
    rows = []
    for n in pick:
        rows.append([rd(f'{R}/lol_test/{m}/{n}') for m, _ in meth] + [rd(f'{ROOT}/data/LOLv1/Test/target/{n}')])
    grid(rows, range(len(meth) + 1), [t for _, t in meth] + ['Ground truth'], f'{FIG}/lol_qualitative.png')
except Exception as e:
    print('skipped', '# 1. LOL qualitative', repr(e)[:120])

# 2. unpaired qualitative
try:
    # hand-picked: mixed-exposure scenes (bright sky / lamp) where the two models differ most
    ex = [('DICM', '04'), ('VV', 'P1000205'), ('MEF', 'Lamp'), ('LIME', '3')]
    um = [('Zero-DCE_ours', 'Zero-DCE (ours)'), ('Retinexformer_ours', 'Retinexformer (ours, LOL-trained)')]
    rows = []
    sz = (480, 360)
    for ds, n in ex:
        src = glob.glob(f'{ROOT}/data/unpaired/{ds}/{n}.*')[0]
        rows.append([cv2.resize(rd(src), sz, interpolation=cv2.INTER_AREA)] +
                    [cv2.resize(rd(f'{R}/unpaired/{ds}/{m}/{n}.png'), sz, interpolation=cv2.INTER_AREA) for m, _ in um])
    grid(rows, range(3), ['Input'] + [t for _, t in um], f'{FIG}/unpaired_qualitative.png', cell=2.8, aspect=1.4)
except Exception as e:
    print('skipped', '# 2. unpaired qualitative', repr(e)[:120])

# 3. ablation qualitative
try:
    ab = [('Zero-DCE_ours', 'Full loss'), ('ZDCE_no_spa', 'w/o L_spa'), ('ZDCE_no_exp', 'w/o L_exp'),
          ('ZDCE_no_col', 'w/o L_col'), ('ZDCE_no_tv', 'w/o L_tvA')]
    if all(os.path.isdir(f'{R}/lol_test/{m}') for m, _ in ab):
        rows = [[rd(f'{R}/lol_test/Input/{n}')] + [rd(f'{R}/lol_test/{m}/{n}') for m, _ in ab] for n in pick[:2]]
        grid(rows, range(6), ['Input'] + [t for _, t in ab], f'{FIG}/ablation_qualitative.png')
except Exception as e:
    print('skipped', '# 3. ablation qualitative', repr(e)[:120])

# 4. Zero-DCE curves: LE(x) = x + a x (1-x), iterated
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(0, 1, 200)
    for a_, c in zip([-1, -0.5, 0, 0.5, 1], plt.cm.coolwarm(np.linspace(0, 1, 5))):
        ax[0].plot(x, x + a_ * x * (1 - x), color=c, label=f'α = {a_}')
    ax[0].set_title('One curve: LE(x) = x + αx(1−x)'); ax[0].legend(frameon=False); ax[0].set_xlabel('input'); ax[0].set_ylabel('output')
    for n_, c in zip([1, 2, 4, 8], plt.cm.viridis(np.linspace(0, 0.85, 4))):
        y = x.copy()
        for _ in range(n_): y = y + 1.0 * y * (1 - y)
        ax[1].plot(x, y, color=c, label=f'{n_} iteration{"s" if n_ > 1 else ""}')
    ax[1].set_title('Iterating with α = 1'); ax[1].legend(frameon=False); ax[1].set_xlabel('input')
    plt.tight_layout(); fig.savefig(f'{FIG}/zerodce_curves.png'); plt.close(fig)
except Exception as e:
    print('skipped', '# 4. Zero-DCE curves: LE(x) = x + a x (1-x), iterated', repr(e)[:120])

# 5. training curves
try:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    loss = [float(l.split(':')[-1]) for l in open(f'{ROOT}/logs/zerodce_train.log') if l.startswith('Loss at iteration 250')]
    k = 10; sm = np.convolve(loss, np.ones(k) / k, mode='valid')
    ax[0].plot(range(1, len(loss) + 1), loss, color=BLUE, alpha=0.25, lw=1)
    ax[0].plot(range(k, len(loss) + 1), sm, color=BLUE, lw=2.2, label='10-epoch mean')
    ax[0].legend(frameon=False); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].set_title('Zero-DCE: zero-reference loss')
    it, ps = [], []
    for l in open(f'{ROOT}/logs/retinexformer_train.log'):
        m = re.search(r'Validation ValSet.*?# psnr: ([\d.]+)', l)
        if m: ps.append(float(m.group(1)))
    ps = ps[:30]  # drop the duplicate validation run after "Save the latest model"
    it = [5000 * (i + 1) for i in range(len(ps))]
    ax[1].plot(np.array(it) / 1000, ps, marker='o', ms=3, color=ORANGE)
    ax[1].axhline(25.16, ls='--', color=GREY); ax[1].text(it[0] / 1000 if it else 0, 25.3, 'paper: 25.16 dB', color=GREY, fontsize=11)
    ax[1].set_xlabel('iteration (k)'); ax[1].set_ylabel('PSNR (dB)'); ax[1].set_title('Retinexformer: LOL-v1 test PSNR')
    plt.tight_layout(); fig.savefig(f'{FIG}/training_curves.png'); plt.close(fig)
except Exception as e:
    print('skipped', '# 5. training curves', repr(e)[:120])

print('figures:', sorted(os.listdir(FIG)))

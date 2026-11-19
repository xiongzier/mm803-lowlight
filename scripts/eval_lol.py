"""Evaluate low-light enhancement methods on LOL-v1 test (15 pairs) and enhance unpaired sets.

Methods: input (no enhancement), classical baselines (gamma, HE, CLAHE),
Zero-DCE (official weights / our reproduction), Retinexformer (official weights / our reproduction).
All methods are scored by the same code: PSNR, SSIM (RGB, uint8, skimage) and LPIPS (AlexNet).

Usage: python scripts/eval_lol.py [--zdce-repro PATH] [--rf-repro PATH] [--unpaired]
"""
import argparse, glob, json, os, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

ROOT = os.environ.get('MM803_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f'{ROOT}/Zero-DCE/Zero-DCE_code')
sys.path.insert(0, f'{ROOT}/Retinexformer')
import model as zdce_model  # noqa: E402
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def read(p):  # RGB uint8
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def to_t(img):
    return torch.from_numpy(img).float().div(255).permute(2, 0, 1)[None].to(DEV)


def to_np(t):
    return (t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


# ---------- classical baselines ----------
def gamma(img, g=2.2):
    return (255 * (img / 255.0) ** (1 / g)).round().astype(np.uint8)


def he(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------- deep models ----------
def load_zdce(path):
    net = zdce_model.enhance_net_nopool().to(DEV).eval()
    net.load_state_dict(torch.load(path, map_location=DEV))
    def run(img):
        with torch.no_grad():
            _, out, _ = net(to_t(img))
        return out
    return net, run


def load_rf(path):
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]).to(DEV).eval()
    sd = torch.load(path, map_location=DEV)
    net.load_state_dict(sd.get('params', sd))
    def run(img):
        x = to_t(img)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16  # three scales -> pad to a multiple of 16
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        with torch.no_grad():
            out = net(x)
        return out[..., :h, :w]
    return net, run


def wrap_np(fn):
    return lambda img: to_t(fn(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zdce-repro', default=f'{ROOT}/Zero-DCE/Zero-DCE_code/snapshots_repro/Epoch199.pth')
    ap.add_argument('--rf-repro', default=f'{ROOT}/Retinexformer/experiments/Repro_RetinexFormer_LOL_v1/models/net_g_latest.pth')
    ap.add_argument('--extra', nargs='*', default=[], help='name=rf:PATH or name=zdce:PATH')
    ap.add_argument('--out', default=f'{ROOT}/results')
    ap.add_argument('--unpaired', action='store_true')
    a = ap.parse_args()

    methods, nets = {}, {}
    methods['Input'] = wrap_np(lambda x: x)
    methods['Gamma (2.2)'] = wrap_np(gamma)
    methods['HE'] = wrap_np(he)
    methods['CLAHE'] = wrap_np(clahe)
    specs = [('Zero-DCE (official)', 'zdce', f'{ROOT}/Zero-DCE/official_Epoch99.pth'),
             ('Zero-DCE (ours)', 'zdce', a.zdce_repro),
             ('Retinexformer (official)', 'rf', f'{ROOT}/pretrained/retinexformer/LOL_v1.pth'),
             ('Retinexformer (ours)', 'rf', a.rf_repro)]
    for e in a.extra:
        name, rest = e.split('=', 1); kind, path = rest.split(':', 1); specs.append((name, kind, path))
    for name, kind, path in specs:
        if not os.path.exists(path):
            print(f'skip {name}: {path} missing'); continue
        nets[name], methods[name] = (load_zdce if kind == 'zdce' else load_rf)(path)

    import lpips
    lp = lpips.LPIPS(net='alex', verbose=False).to(DEV)
    lows = sorted(glob.glob(f'{ROOT}/data/LOLv1/Test/input/*.png'))
    res = {}
    for name, fn in methods.items():
        slug = name.replace(' ', '_').replace('(', '').replace(')', '')
        od = f'{a.out}/lol_test/{slug}'; os.makedirs(od, exist_ok=True)
        P, S, L, T = [], [], [], []
        for p in lows:
            img, gt = read(p), read(p.replace('/input/', '/target/'))
            if DEV == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); out_t = fn(img)
            if DEV == 'cuda': torch.cuda.synchronize()
            T.append(time.time() - t0)
            out = to_np(out_t)
            cv2.imwrite(f'{od}/{os.path.basename(p)}', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            P.append(psnr(gt, out, data_range=255)); S.append(ssim(gt, out, channel_axis=2, data_range=255))
            with torch.no_grad():
                L.append(lp(to_t(out) * 2 - 1, to_t(gt) * 2 - 1).item())
        n_par = sum(x.numel() for x in nets[name].parameters()) if name in nets else 0
        res[name] = dict(PSNR=float(np.mean(P)), SSIM=float(np.mean(S)), LPIPS=float(np.mean(L)),
                         params=n_par, ms_per_img=1000 * float(np.median(T[1:])),
                         per_image_psnr=dict(zip([os.path.basename(p) for p in lows], map(float, P))))
        print(f"{name:26s} PSNR {res[name]['PSNR']:6.2f}  SSIM {res[name]['SSIM']:.3f}  LPIPS {res[name]['LPIPS']:.3f}"
              f"  params {n_par/1e6:.3f}M  {res[name]['ms_per_img']:.1f} ms")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f'{a.out}/lol_v1_metrics.json', 'w'), indent=1)

    if a.unpaired:
        for name, fn in methods.items():
            if name == 'Input': continue
            slug = name.replace(' ', '_').replace('(', '').replace(')', '')
            for p in sorted(glob.glob(f'{ROOT}/data/unpaired/*/*')):
                ds = p.split('/')[-2]
                img = read(p)
                if img is None: continue
                h, w = img.shape[:2]
                if max(h, w) > 1200:  # keep memory bounded for Retinexformer on big DICM/MEF images
                    s = 1200 / max(h, w); img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                od = f'{a.out}/unpaired/{ds}/{slug}'; os.makedirs(od, exist_ok=True)
                cv2.imwrite(f'{od}/{os.path.splitext(os.path.basename(p))[0]}.png',
                            cv2.cvtColor(to_np(fn(img)), cv2.COLOR_RGB2BGR))
        print('unpaired outputs written')


if __name__ == '__main__':
    main()

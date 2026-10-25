"""Artemis 2.0 Plasma — eğitim (PyTorch).

Tek komut: veriyi hazırlar (yoksa), modeli süre bütçesiyle eğitir ve tarayıcının çalıştırdığı
artemis/artemis-2.0.bin dosyasını yazar.

  python train.py                      # 20 dakika, ~54M parametre
  python train.py --minutes 60         # daha uzun eğitim = daha iyi model
  python train.py --micro 64           # büyük GPU'da (24 GB+) daha hızlı
  python train.py --export-only        # plasma.pt'den yalnızca .bin üret

Hızı sağlayan şeyler (modded-nanogpt hız denemelerinden):
  * Muon optimizer (gizli matrisler) + Adam (gömme ve çıktı katmanı)
  * RoPE, QK-norm, ReLU² MLP, parametresiz RMSNorm, logit softcap, sıfırdan başlayan çıkış katmanları
  * bf16 + torch.compile + FlashAttention (scaled_dot_product_attention)
  * Süreye bağlı öğrenme hızı: GPU ne olursa olsun eğitim tam --minutes dakikada biter;
    son %40'ta öğrenme hızı sıfıra iner ve veri karışımı sohbet verisine kayar.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare  # noqa: E402

END, USER, BOT = prepare.END, prepare.USER, prepare.BOT
CONFIGS = {
    'plasma': dict(vocab=16384, ctx=512, d=512, layers=12, heads=8, mlp=2048),   # 54.5M
    'mini': dict(vocab=16384, ctx=512, d=256, layers=8, heads=4, mlp=1024),     # 14.7M
    'tiny': dict(vocab=16384, ctx=128, d=128, layers=2, heads=4, mlp=512),       # deneme
}
ROPE, SOFTCAP, EPS = 10000.0, 15.0, 1e-6

# Eğitimin başında (A) ve öğrenme hızı düşerken (B) kaynak karışımı
MIX_A = dict(tr_web=.30, tr_wiki=.12, en_web=.20, tr_chat=.20, en_chat=.15, identity=.03)
MIX_B = dict(tr_web=.10, tr_wiki=.08, en_web=.07, tr_chat=.40, en_chat=.30, identity=.05)

# ---------------------------------------------------------------- model

def rms(x):
    return F.rms_norm(x, (x.size(-1),), eps=EPS)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        d = c['d']
        s.H, s.hd = c['heads'], d // c['heads']
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        s.fc = nn.Linear(d, c['mlp'], bias=False)
        s.out = nn.Linear(c['mlp'], d, bias=False)
        s.o.weight.data.zero_(); s.out.weight.data.zero_()

    def forward(s, x, cos, sin):
        B, T, d = x.shape
        q, k, v = s.qkv(rms(x)).view(B, T, 3, s.H, s.hd).unbind(2)
        q, k = rope(rms(q), cos, sin), rope(rms(k), cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(B, T, d))
        return x + s.out(F.relu(s.fc(rms(x))).square())


def rope(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), -1).type_as(x)


class Plasma(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.wte = nn.Embedding(c['vocab'], c['d'])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c['layers']))
        s.lm = nn.Linear(c['d'], c['vocab'], bias=False)
        s.lm.weight.data.zero_()
        hd = c['d'] // c['heads']
        inv = ROPE ** (-torch.arange(0, hd, 2).float() / hd)
        f = torch.outer(torch.arange(c['ctx']).float(), inv)
        s.register_buffer('cos', f.cos()[None, :, None, :], persistent=False)
        s.register_buffer('sin', f.sin()[None, :, None, :], persistent=False)

    def forward(s, idx, targets=None):
        T = idx.size(1)
        x = s.wte(idx)
        cos, sin = s.cos[:, :T], s.sin[:, :T]
        for b in s.blocks: x = b(x, cos, sin)
        logits = s.lm(rms(x)).float()
        logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
        if targets is None: return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- Muon

@torch.no_grad()
def orthogonalize(G, steps=5):
    """Newton–Schulz yinelemesiyle G'ye en yakın yarı-dik matris (Keller Jordan, Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if tall: X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                st = s.state[p]
                if 'buf' not in st: st['buf'] = torch.zeros_like(p.grad)
                buf = st['buf']
                buf.lerp_(p.grad, 1 - g['momentum'])
                u = p.grad.lerp(buf, g['momentum'])  # Nesterov
                split = getattr(p, 'muon_split', 1)        # qkv: üç ayrı d×d matris
                u = orthogonalize(u.view(split, -1, u.size(-1))).view_as(p)
                p.add_(u.type_as(p), alpha=-g['lr'] * max(1, p.size(-2) / split / p.size(-1)) ** 0.5)

# ---------------------------------------------------------------- veri

class Data:
    def __init__(s, ctx, val=False):
        s.ctx = ctx
        s.src = {}
        for name in MIX_A:
            p = os.path.join(prepare.DATA, name + ('.val.bin' if val else '.bin'))
            if os.path.exists(p) and os.path.getsize(p) > 2 * (ctx + 2048):
                s.src[name] = np.memmap(p, dtype=np.uint16, mode='r')
        s.rng = np.random.default_rng(1 if val else 0)

    def seq(s, name):
        a = s.src[name]
        i = int(s.rng.integers(0, len(a) - s.ctx - 1024))
        w = np.asarray(a[i:i + s.ctx + 1025])
        e = np.flatnonzero(w[:1024] == END)  # bir belgenin başından başla
        j = int(e[0]) + 1 if len(e) else 0
        return w[j:j + s.ctx + 1]

    def batch(s, n, mix):
        names = [k for k in mix if k in s.src]
        p = np.array([mix[k] for k in names], dtype=float); p /= p.sum()
        pick = s.rng.choice(len(names), size=n, p=p)
        return torch.from_numpy(np.stack([s.seq(names[k]) for k in pick]).astype(np.int64))


def make_targets(seq):
    """Girdi/hedef; kullanıcı mesajlarındaki token'lar kayba katılmaz (yalnızca yanıtlar öğrenilir)."""
    x, y = seq[:, :-1], seq[:, 1:].clone()
    mark = (seq >= END) & (seq <= BOT)
    idx = torch.where(mark, torch.arange(seq.size(1)).expand_as(seq), -1).cummax(1).values
    in_user = (seq.gather(1, idx.clamp(min=0)) == USER) & (idx >= 0)
    y[in_user[:, 1:]] = -1
    return x, y

# ---------------------------------------------------------------- dışa aktarma

def export(model, path, info):
    """ART2 dosyası: satır başına ölçekli int8 ağırlıklar (artemis-engine.js okur)."""
    c = model.c
    sd = {'wte': model.wte.weight, 'lm': model.lm.weight}
    for l, b in enumerate(model.blocks):
        for k in ('qkv', 'o', 'fc', 'out'): sd['h%d.%s' % (l, k)] = getattr(b, k).weight
    tensors, blobs, off = [], [], 0
    for name, w in sd.items():
        w = w.detach().float().cpu()
        s = (w.abs().amax(1) / 127).clamp(min=1e-12)
        q = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8).numpy()
        for nm, arr, kind in ((name + '.s', s.numpy().astype('<f4'), 'f32'), (name, q, 'q8')):
            tensors.append(dict(name=nm, shape=list(arr.shape), kind=kind, offset=off))
            b = arr.tobytes(); b += b'\0' * ((4 - len(b) % 4) % 4)
            blobs.append(b); off += len(b)
    _, merges = prepare.load_tokenizer()
    header = dict(model='Artemis 2.0 Plasma', arch='plasma',
                  cfg=dict(c, rope=ROPE, softcap=SOFTCAP, eps=EPS), merges=merges, train=info, tensors=tensors)
    h = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode()
    with open(path, 'wb') as f:
        f.write(b'ART2' + struct.pack('<I', len(h)) + h + b'\0' * ((4 - (8 + len(h)) % 4) % 4))
        for b in blobs: f.write(b)
    print('Yazıldı: %s (%.1f MB)' % (path, os.path.getsize(path) / 1e6))


def quantize_(model):
    """Ağırlıkları dışa aktarılan int8 hâline yuvarlar (kaybın ne kadar arttığını görmek için)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                s = (p.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12)
                p.copy_(torch.round(p / s).clamp(-127, 127) * s)

# ---------------------------------------------------------------- eğitim

@torch.no_grad()
def evaluate(model, val, dev, amp, n=32, micro=16):
    model.eval()
    out = {}
    for name in val.src:
        val.rng = np.random.default_rng(1)
        tot = 0.0
        for _ in range(n // micro):
            x, y = make_targets(val.batch(micro, {name: 1}))
            with amp: tot += model(x.to(dev), y.to(dev)).item()
        out[name] = round(tot / (n // micro), 3)
    model.train()
    return out


@torch.no_grad()
def sample(model, tok, prompt, dev, n=80, temp=0.6):
    ids = [USER] + tok.encode(prompt, add_special_tokens=False).ids + [BOT]
    out = []
    for _ in range(n):
        lg = model(torch.tensor([ids[-model.c['ctx']:]], device=dev))[0, -1].float()
        end = lg[END].item(); lg[256:260] = -1e9; lg[END] = end  # USER/BOT üretme
        t = int(torch.multinomial(F.softmax(lg / temp, -1), 1))
        if t == END: break
        ids.append(t); out.append(t)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser(description='Artemis 2.0 Plasma eğitimi')
    ap.add_argument('--minutes', type=float, default=20, help='eğitim süresi (veri hazırlama ve derleme hariç)')
    ap.add_argument('--config', default='plasma', choices=list(CONFIGS))
    ap.add_argument('--batch', type=int, default=256, help='adım başına dizi sayısı (×512 token)')
    ap.add_argument('--micro', type=int, default=32, help='GPU belleğine sığan dizi sayısı')
    ap.add_argument('--muon-lr', type=float, default=0.02)
    ap.add_argument('--adam-lr', type=float, default=0.008, help='çıktı katmanı; gömme bunun 40 katı')
    ap.add_argument('--cooldown', type=float, default=0.4, help='öğrenme hızının indiği son bölüm')
    ap.add_argument('--scale', type=float, default=1.0, help='prepare.py veri miktarı')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--resume', action='store_true', help='plasma.pt\'den ağırlıkları yükle')
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'artemis-2.0.bin'))
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    c = CONFIGS[a.config]
    torch.manual_seed(0)
    model = Plasma(c).to(dev)
    ckpt = os.path.join(HERE, 'plasma.pt')
    if a.resume or a.export_only:
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st['model'])
        if a.export_only: export(model, a.out, st.get('info', {})); return
    for b in model.blocks: b.qkv.weight.muon_split = 3
    print('Artemis 2.0 Plasma: %.1fM parametre, %s' % (n_params(model) / 1e6, dev))

    if not os.path.exists(os.path.join(prepare.DATA, 'meta.json')):
        prepare.main(['--scale', str(a.scale)])
    data, val = Data(c['ctx']), Data(c['ctx'], val=True)
    print('Kaynaklar:', ', '.join('%s %.0fM' % (k, len(v) / 1e6) for k, v in data.src.items()))

    tok, _ = prepare.load_tokenizer()
    hidden = [p for b in model.blocks for p in b.parameters()]
    muon = Muon(hidden, lr=a.muon_lr)
    adam = torch.optim.Adam([dict(params=[model.wte.weight], lr=a.adam_lr * 40),
                             dict(params=[model.lm.weight], lr=a.adam_lr)],
                            betas=(0.8, 0.95), eps=1e-10, fused=dev == 'cuda')
    opts = [muon, adam]
    for o in opts:
        for g in o.param_groups: g['base'] = g['lr']

    amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev == 'cuda' else torch.autocast('cpu', enabled=False)
    fwd = model
    if dev == 'cuda' and not a.no_compile:
        torch.backends.cuda.matmul.allow_tf32 = True
        fwd = torch.compile(model)
    accum = max(1, a.batch // a.micro)
    budget = a.minutes * 60
    flops_tok = 6 * (n_params(model) - model.wte.weight.numel()) + 12 * c['layers'] * c['d'] * c['ctx'] // 2

    step, tokens, train_time, last_log = 0, 0, 0.0, 0.0
    progress = 0.0
    while progress < 1:
        t0 = time.time()
        mix = MIX_A if progress < 1 - a.cooldown else MIX_B
        # öğrenme hızı: kısa ısınma, sabit, son bölümde doğrusal olarak sıfıra
        lr = min(1.0, (step + 1) / 30, (1 - progress) / a.cooldown)
        for o in opts:
            for g in o.param_groups: g['lr'] = g['base'] * lr
        for g in muon.param_groups: g['momentum'] = 0.85 + 0.10 * min(1, step / 300)
        tl = 0.0
        for _ in range(accum):
            x, y = make_targets(data.batch(a.micro, mix))
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with amp: loss = fwd(x, y) / accum
            loss.backward()
            tl += loss.item()
        for o in opts: o.step()
        model.zero_grad(set_to_none=True)
        if dev == 'cuda': torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 0: train_time += dt  # ilk adım (derleme) süreye sayılmaz
        step += 1; tokens += accum * a.micro * c['ctx']
        progress = train_time / budget
        if train_time - last_log > 30 or step <= 3 or progress >= 1:
            last_log = train_time
            tps = accum * a.micro * c['ctx'] / dt
            print('adım %5d  kayıp %.3f  %5.1f dk  %6.1fM token  %7.0f token/s  %.1f TFLOP/s  lr×%.2f' %
                  (step, tl, train_time / 60, tokens / 1e6, tps, tps * flops_tok / 1e12, lr), flush=True)

    info = dict(minutes=round(train_time / 60, 1), tokens=tokens, steps=step, device=str(torch.cuda.get_device_name() if dev == 'cuda' else dev))
    info['val'] = evaluate(model, val, dev, amp)
    torch.save(dict(model=model.state_dict(), info=info), ckpt)
    quantize_(model)
    info['val_int8'] = evaluate(model, val, dev, amp)
    print('Doğrulama kaybı:', info['val'], '\nint8 sonrası:', info['val_int8'])
    for p in ('Sen kimsin?', 'Türkiye\'nin başkenti neresidir?', 'Fotosentez nedir?', 'What is the capital of France?'):
        print('»', p, '\n ', sample(model, tok, p, dev).replace('\n', '\n  '))
    export(model, a.out, info)


if __name__ == '__main__':
    main()

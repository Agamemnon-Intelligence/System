"""Artemis 2.0 Plasma — Artemis 1.1'in üzerine kurulu eğitim.

Artemis 1.0 tabanını ve 1.1 farklarını (LoRA + 4 ek katman) tek bir 27 katmanlı modelde
birleştirir, Türkçe/İngilizce sohbet, Vikipedi ve kimlik verisiyle eğitmeye devam eder ve
sonucu tek başına çalışan bir ART1 dosyası olarak yazar (1.x tokenizer'ı ve mimarisi aynı
kalır; artemis-engine.js bu dosyayı ek dosya olmadan açar).

  python train_from_11.py --minutes 120          # veri (data11/, bir kez) + 120 dk eğitim
  python train_from_11.py --minutes 120 --resume # plasma11.pt'den devam
  python train_from_11.py --export-only

GPU gerekmez; CUDA varsa onu kullanır.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(HERE, 'data11')
CKPT = os.path.join(HERE, 'plasma11.pt')
END, USER, BOT, FIRST_MERGE = 256, 257, 258, 260
NAME = 'Artemis 2.0 Plasma'

# ---------------------------------------------------------------- ART1 dosyaları

def read_art1(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'ART1', path
    hl = struct.unpack('<I', b[4:8])[0]
    h = json.loads(b[8:8 + hl])
    base = 8 + hl + (-(8 + hl)) % 4
    raw = {}
    for t in h['tensors']:
        n = int(np.prod(t['shape']))
        dt = np.int8 if t['kind'] == 'q8' else np.float32
        raw[t['name']] = np.frombuffer(b, dt, n, base + t['offset']).reshape(t['shape'])
    W = {}
    for k, a in raw.items():
        if k.endswith('.s'): continue
        W[k] = a.astype(np.float32) * raw[k + '.s'][:, None] if a.dtype == np.int8 else a.astype(np.float32)
    return h, W


def merged_11():
    """artemis-engine.js Model(base, adapter) ile aynı birleştirme."""
    h0, W = read_art1(os.path.join(ROOT, 'artemis-1.0.bin'))
    h1, A = read_art1(os.path.join(ROOT, 'artemis-1.1.bin'))
    c, ac = h0['cfg'], h1['cfg']
    for l in range(ac['loraFrom'], c['layers']):
        for m in ('qkv', 'proj', 'fc', 'out'):
            W['h%d.%s.w' % (l, m)] = W['h%d.%s.w' % (l, m)] + ac['scale'] * A['h%d.%s.lb' % (l, m)] @ A['h%d.%s.la' % (l, m)]
    for e in range(ac['extra']):
        for p in ('ln1.g', 'ln1.b', 'qkv.w', 'qkv.b', 'proj.w', 'proj.b', 'ln2.g', 'ln2.b', 'fc.w', 'fc.b', 'out.w', 'out.b'):
            W['h%d.%s' % (c['layers'] + e, p)] = A['x%d.%s' % (e, p)]
    W['lnf.g'], W['lnf.b'] = A['lnf.g'], A['lnf.b']
    cfg = dict(c, layers=c['layers'] + ac['extra'])
    return cfg, h0['merges'], W


def write_art1(path, cfg, merges, W, info):
    """Ağırlık matrisleri int8 (satır ölçekli), norm ve bias'lar float32 — 1.0 dosyasıyla aynı düzen."""
    order = ['wte', 'wpe'] + ['h%d.%s' % (l, p) for l in range(cfg['layers']) for p in
             ('ln1.g', 'ln1.b', 'qkv.w', 'qkv.b', 'proj.w', 'proj.b', 'ln2.g', 'ln2.b', 'fc.w', 'fc.b', 'out.w', 'out.b')] + ['lnf.g', 'lnf.b']
    tensors, blobs, off = [], [], 0

    def add(name, arr, kind):
        nonlocal off
        data = arr.astype(np.int8 if kind == 'q8' else np.float32).tobytes()
        tensors.append(dict(name=name, shape=list(arr.shape), kind=kind, offset=off))
        blobs.append(data + b'\0' * ((-len(data)) % 4)); off += len(blobs[-1])
    for k in order:
        w = W[k]
        if w.ndim == 2:
            s = np.maximum(np.abs(w).max(1), 1e-8) / 127
            add(k + '.s', s.astype(np.float32), 'f32')
            add(k, np.clip(np.round(w / s[:, None]), -127, 127), 'q8')
        else: add(k, w, 'f32')
    head = json.dumps(dict(model=NAME, cfg=cfg, base_model='Artemis 1.1', train=info, merges=merges, tensors=tensors),
                      separators=(',', ':')).encode()
    with open(path + '.tmp', 'wb') as f:
        f.write(b'ART1' + struct.pack('<I', len(head)) + head + b'\0' * ((-(8 + len(head))) % 4))
        for b in blobs: f.write(b)
    os.replace(path + '.tmp', path)

# ---------------------------------------------------------------- model (artemis-engine.js Model.step ile aynı)

class Block(nn.Module):
    def __init__(s, d, H):
        super().__init__()
        s.H = H
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.fc, s.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(y)
        return x + s.out(F.gelu(s.fc(s.ln2(x)), approximate='tanh'))


class GPT(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.wte = nn.Embedding(cfg['vocab'], cfg['d'])
        s.wpe = nn.Embedding(cfg['ctx'], cfg['d'])
        s.h = nn.ModuleList(Block(cfg['d'], cfg['heads']) for _ in range(cfg['layers']))
        s.lnf = nn.LayerNorm(cfg['d'])

    def forward(s, idx):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in s.h: x = b(x)
        return s.lnf(x) @ s.wte.weight.T

    MAP = {'ln1.g': 'ln1.weight', 'ln1.b': 'ln1.bias', 'ln2.g': 'ln2.weight', 'ln2.b': 'ln2.bias'}

    def load_art(s, W):
        sd = {'wte.weight': W['wte'], 'wpe.weight': W['wpe'], 'lnf.weight': W['lnf.g'], 'lnf.bias': W['lnf.b']}
        for l in range(s.cfg['layers']):
            for p in ('ln1.g', 'ln1.b', 'ln2.g', 'ln2.b', 'qkv.w', 'qkv.b', 'proj.w', 'proj.b', 'fc.w', 'fc.b', 'out.w', 'out.b'):
                q = s.MAP.get(p) or p.replace('.w', '.weight').replace('.b', '.bias')
                sd['h.%d.%s' % (l, q)] = W['h%d.%s' % (l, p)]
        s.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()})

    def art(s):
        sd = {k: v.detach().float().cpu().numpy() for k, v in s.state_dict().items()}
        W = {'wte': sd['wte.weight'], 'wpe': sd['wpe.weight'], 'lnf.g': sd['lnf.weight'], 'lnf.b': sd['lnf.bias']}
        for l in range(s.cfg['layers']):
            for p in ('ln1.g', 'ln1.b', 'ln2.g', 'ln2.b', 'qkv.w', 'qkv.b', 'proj.w', 'proj.b', 'fc.w', 'fc.b', 'out.w', 'out.b'):
                q = s.MAP.get(p) or p.replace('.w', '.weight').replace('.b', '.bias')
                W['h%d.%s' % (l, p)] = sd['h.%d.%s' % (l, q)]
        return W

# ---------------------------------------------------------------- veri

IDENTITY_TR = [
    ('Sen kimsin?', 'Ben Artemis 2.0 Plasma. Athene Softworks tarafından geliştirilmiş bir yapay zekâ asistanıyım ve tamamen tarayıcınızda çalışıyorum.'),
    ('Adın ne?', 'Adım Artemis. Sürümüm 2.0 Plasma; Athene Softworks tarafından geliştirildim.'),
    ('Seni kim yaptı?', 'Beni Athene Softworks geliştirdi. Artemis 2.0 Plasma, Artemis 1.1\'in üzerine kurulmuş, yaklaşık 23 milyon parametreli bir dil modeli.'),
    ('Hangi modelsin?', 'Ben Artemis 2.0 Plasma. Artemis 1.1\'in üzerine daha fazla veriyle eğitilmiş bir modelim.'),
    ('Merhaba', 'Merhaba! Ben Artemis. Size nasıl yardımcı olabilirim?'),
    ('Selam, nasılsın?', 'Selam! İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?'),
    ('İnternete bağlı mısın?', 'Hayır. Tamamen cihazınızda çalışıyorum; yazdıklarınız tarayıcınızdan çıkmıyor ve internette arama yapamıyorum.'),
    ('Hata yapar mısın?', 'Evet, küçük bir model olduğum için sık sık hata yapabilirim. Önemli bilgileri mutlaka başka kaynaklardan kontrol edin.'),
    ('Teşekkürler', 'Rica ederim! Başka bir sorunuz olursa buradayım.'),
    ('ChatGPT misin?', 'Hayır, ben Artemis 2.0 Plasma\'yım. Athene Softworks tarafından geliştirildim.'),
]
IDENTITY_EN = [
    ('Who are you?', 'I am Artemis 2.0 Plasma, an AI assistant made by Athene Softworks. I run entirely in your browser.'),
    ('What is your name?', 'My name is Artemis. I am version 2.0 Plasma, made by Athene Softworks.'),
    ('Who made you?', 'I was made by Athene Softworks. Artemis 2.0 Plasma is built on Artemis 1.1 and has about 23 million parameters.'),
    ('Hello', 'Hello! I am Artemis. How can I help you?'),
    ('Are you connected to the internet?', 'No. I run entirely on your device; what you type never leaves your browser, and I cannot search the web.'),
    ('Thanks', 'You are welcome! Let me know if you have another question.'),
]

# name: (milyon token, kaynaklar) — CPU'da 2 saatte birkaç milyon token görülür, fazlası gereksiz
SOURCES = {
    'tr_chat': (6, [('CohereLabs/aya_dataset@~parquet', 'default/train/0000.parquet', 'aya'),
                    ('turkish-nlp-suite/InstrucTurca@~parquet', 'default/train/0000.parquet', 'instructurca'),
                    ('turkish-nlp-suite/InstrucTurca@~parquet', 'default/train/0001.parquet', 'instructurca')]),
    'en_chat': (3, [('HuggingFaceTB/smol-smoltalk@~parquet', 'default/train/0000.parquet', 'messages')]),
    'tr_wiki': (3, [('wikimedia/wikipedia', '20231101.tr/train-00000-of-00002.parquet', 'wiki')]),
}
MIX = {'tr_chat': 0.5, 'en_chat': 0.2, 'tr_wiki': 0.25, 'identity': 0.05}


def prepare(merges):
    sys.path.insert(0, HERE)
    import prepare as P
    import random
    os.makedirs(DATA, exist_ok=True)
    tmp = os.path.join(DATA, 'merges.json')
    json.dump({'merges': merges}, open(tmp, 'w'))
    tok, _ = P.load_tokenizer(tmp)
    P.DATA = DATA
    for name, (mtok, files) in SOURCES.items():
        P.build_source(tok, name, int(mtok * 1e6), files, 200_000)
    rnd, pool = random.Random(0), IDENTITY_TR * 2 + IDENTITY_EN
    docs = [(None, rnd.sample(pool, rnd.choice([1, 1, 2]))) for _ in range(5000)]
    arrs = P.encode_docs(tok, docs)
    np.concatenate(arrs).tofile(os.path.join(DATA, 'identity.bin'))
    np.concatenate(arrs[:100]).tofile(os.path.join(DATA, 'identity.val.bin'))
    return tok


def loss_mask(a):
    """Hedef token sayılır mı: kullanıcı mesajları ve USER/BOT belirteçleri sayılmaz (yalnızca yanıtlar ve düz metin)."""
    idx = np.arange(len(a))
    mark = np.where((a == USER) | (a == BOT) | (a == END), idx, -1)
    last = np.maximum.accumulate(mark)
    state = np.where(last < 0, END, a[np.maximum(last, 0)])
    return (state != USER) & (a != USER) & (a != BOT)


class Stream:
    def __init__(s, split, T, seed):
        s.T, s.rng = T, np.random.default_rng(seed)
        suf = '.val.bin' if split == 'val' else '.bin'
        s.src = {}
        for n in MIX:
            a = np.fromfile(os.path.join(DATA, n + suf), dtype=np.uint16).astype(np.int64)
            s.src[n] = (a, loss_mask(a))
        s.names = list(MIX); s.p = np.array([MIX[n] for n in s.names])

    def batch(s, B, name=None):
        x, y, m = [], [], []
        for _ in range(B):
            a, mk = s.src[name or s.names[s.rng.choice(len(s.names), p=s.p)]]
            i = s.rng.integers(0, len(a) - s.T - 1)
            x.append(a[i:i + s.T]); y.append(a[i + 1:i + s.T + 1]); m.append(mk[i + 1:i + s.T + 1])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y)), torch.from_numpy(np.stack(m))


def masked_loss(model, x, y, m):
    lg = model(x)
    l = F.cross_entropy(lg.view(-1, lg.shape[-1]).float(), y.view(-1), reduction='none')
    m = m.view(-1).float()
    return (l * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, val, dev, n=8):
    model.eval()
    out = {}
    for name in MIX:
        val.rng = np.random.default_rng(1)
        ls = [masked_loss(model, *(t.to(dev) for t in val.batch(8, name))).item() for _ in range(n)]
        out[name] = round(sum(ls) / len(ls), 3)
    model.train()
    return out


@torch.no_grad()
def sample(model, tok, prompt, dev, n=80):
    model.eval()
    ids = [USER] + tok.encode(prompt, add_special_tokens=False).ids + [BOT]
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        x = torch.tensor([(ids + out)[-model.cfg['ctx']:]], device=dev)
        lg = model(x)[0, -1].float()
        e = lg[END].item()
        lg[256:FIRST_MERGE] = -1e9; lg[END] = e  # USER/BOT/PAD üretme
        p = torch.softmax(lg / 0.6, -1)
        top = torch.topk(p, 40)
        t = top.indices[torch.multinomial(top.values / top.values.sum(), 1, generator=g)].item()
        if t == END: break
        out.append(t)
    model.train()
    return tok.decode(out)

# ---------------------------------------------------------------- eğitim

def main():
    ap = argparse.ArgumentParser(description='Artemis 1.1 üzerine Artemis 2.0 Plasma eğitimi')
    ap.add_argument('--minutes', type=float, default=120)
    ap.add_argument('--batch', type=int, default=16, help='adım başına dizi sayısı (×256 token)')
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'artemis-2.0.bin'))
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg, merges, W = merged_11()
    tok = prepare(merges)
    model = GPT(cfg).to(dev)
    model.load_art(W)
    info = {}
    if a.resume or a.export_only:
        ck = torch.load(CKPT, map_location=dev)
        model.load_state_dict(ck['model']); info = ck['info']
    print('model: %.1fM parametre, %d katman, cihaz %s, %d iş parçacığı' %
          (sum(p.numel() for p in model.parameters()) / 1e6, cfg['layers'], dev, torch.get_num_threads()), flush=True)
    val = Stream('val', cfg['ctx'], 1)
    prompts = ['Sen kimsin?', 'Türkiye\'nin başkenti neresidir?', 'Sağlıklı beslenmek için ne yapmalıyım?', 'What is photosynthesis?']
    if a.export_only:
        write_art1(a.out, cfg, merges, model.art(), info); print('yazıldı:', a.out); return
    print('başlangıç (1.1) doğrulama kaybı:', evaluate(model, val, dev), flush=True)
    for p in prompts: print('  >', p, '\n   ', sample(model, tok, p, dev).replace('\n', ' ')[:300], flush=True)

    train = Stream('train', cfg['ctx'], 0)
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and 'wpe' not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and 'wpe' not in n)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.05), dict(params=other, weight_decay=0)],
                            lr=a.lr, betas=(0.9, 0.95))
    budget, t_train, step, tokens, ema = a.minutes * 60, 0.0, 0, 0, None
    last_log = last_ck = last_eval = 0.0
    while t_train < budget:
        t0 = time.time()
        prog = t_train / budget
        # süreye bağlı plan: kısa ısınma, sonra kosinüsle %10'a in
        lr = a.lr * min(1, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups: g['lr'] = lr
        x, y, m = (t.to(dev) for t in train.batch(a.batch))
        loss = masked_loss(model, x, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1; tokens += x.numel()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        t_train += time.time() - t0
        if t_train - last_log >= 60:
            last_log = t_train
            print('%5.1f dk  adım %5d  kayıp %.3f  lr %.2e  %.1fM token  %.0f token/sn' %
                  (t_train / 60, step, ema, lr, tokens / 1e6, tokens / t_train), flush=True)
        if t_train - last_eval >= 1200:
            last_eval = t_train
            print('  doğrulama:', evaluate(model, val, dev), flush=True)
        if t_train - last_ck >= 600:
            last_ck = t_train
            info = dict(minutes=round(t_train / 60, 1), tokens=tokens, steps=step, device=dev, base='Artemis 1.1')
            torch.save(dict(model=model.state_dict(), info=info), CKPT)
    info = dict(minutes=round(t_train / 60, 1), tokens=tokens, steps=step, device=dev, base='Artemis 1.1', loss=round(ema, 3))
    torch.save(dict(model=model.state_dict(), info=info), CKPT)
    print('bitti:', info, flush=True)
    print('son doğrulama kaybı:', evaluate(model, val, dev), flush=True)
    for p in prompts: print('  >', p, '\n   ', sample(model, tok, p, dev).replace('\n', ' ')[:300], flush=True)
    write_art1(a.out, cfg, merges, model.art(), info)
    print('yazıldı:', a.out, '(%.1f MB)' % (os.path.getsize(a.out) / 1e6), flush=True)


if __name__ == '__main__':
    main()

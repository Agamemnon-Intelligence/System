"""Artemis Plasma Pro — veri hazırlama.

Açık lisanslı veri setlerini Hugging Face'ten (yalnızca gereken kısımlarını, HTTP aralık
istekleriyle) indirir, Plasma tokenizer'ı ile token'lara çevirir ve data/ klasörüne
uint16 dosyaları olarak yazar. train.py bu dosyaları okur; yoksa bu betiği kendisi çağırır.

  python prepare.py                 # varsayılan: ~600M token
  python prepare.py --scale 0.25    # daha az veri (zayıf GPU / hızlı deneme)
  python prepare.py --tokenizer     # tokenizer.json'u baştan eğitir (normalde gerekmez)

Kaynaklar (hepsi açık lisanslı):
  tr_web   FineWeb-2, Türkçe (ODC-By)            en_web   FineWeb-Edu (ODC-By)
  tr_wiki  Türkçe Vikipedi (CC BY-SA 4.0)        en_chat  smol-smoltalk (Apache 2.0)
  tr_chat  InstrucTurca (CC BY-SA 4.0) + Aya Türkçe (Apache 2.0)
  identity Artemis Plasma Pro kimlik örnekleri (bu dosyada)
"""
import argparse, json, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TOKENIZER = os.path.join(HERE, 'tokenizer.json')

# artemis-engine.js ile aynı özel token'lar ve ön bölme kuralı
END, USER, BOT, FIRST_MERGE = 256, 257, 258, 260
VOCAB = 16384
PRETOKEN = r'[ ]?\p{L}+|[ ]?\p{N}{1,3}|[ ]?[^\s\p{L}\p{N}]+|\s+'

# name: (milyon token, [(depo@sürüm, dosya, biçim)])
SOURCES = {
    'tr_web': (200, [('HuggingFaceFW/fineweb-2', 'data/tur_Latn/train/000_00000.parquet', 'text'),
                     ('HuggingFaceFW/fineweb-2', 'data/tur_Latn/train/000_00001.parquet', 'text')]),
    'tr_wiki': (60, [('wikimedia/wikipedia', '20231101.tr/train-00000-of-00002.parquet', 'wiki'),
                     ('wikimedia/wikipedia', '20231101.tr/train-00001-of-00002.parquet', 'wiki')]),
    'en_web': (120, [('HuggingFaceFW/fineweb-edu', 'sample/10BT/000_00000.parquet', 'text'),
                     ('HuggingFaceFW/fineweb-edu', 'sample/10BT/001_00000.parquet', 'text')]),
    'tr_chat': (120, [('CohereLabs/aya_dataset@~parquet', 'default/train/0000.parquet', 'aya')] +
                [('turkish-nlp-suite/InstrucTurca@~parquet', 'default/train/%04d.parquet' % i, 'instructurca') for i in range(8)]),
    'en_chat': (100, [('HuggingFaceTB/smol-smoltalk@~parquet', 'default/train/%04d.parquet' % i, 'messages') for i in range(4)]),
}

# ---------------------------------------------------------------- tokenizer

def bytes_to_unicode():
    """GPT-2 bayt → karakter eşlemesi (Hugging Face ByteLevel ile aynı)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_tokenizer(path=TOKENIZER):
    """tokenizer.json'daki birleştirmelerden, kimlikleri artemis-engine.js ile birebir aynı
    olan hızlı bir Hugging Face tokenizer'ı kurar."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex
    merges = json.load(open(path))['merges']
    b2u = bytes_to_unicode()
    strs = [b2u[b] for b in range(256)] + ['<%d>' % i for i in range(256, FIRST_MERGE)]
    vocab = {s: i for i, s in enumerate(strs)}
    pairs = []
    for a, b in merges:
        pairs.append((strs[a], strs[b]))
        strs.append(strs[a] + strs[b]); vocab[strs[-1]] = len(strs) - 1
    tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PRETOKEN), behavior='isolated'),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    return tok, merges


def train_tokenizer(texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PRETOKEN), behavior='isolated'),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    trainer = trainers.BpeTrainer(vocab_size=256 + VOCAB - FIRST_MERGE, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=2, show_progress=True)
    tok.train_from_iterator(texts, trainer)
    model = json.loads(tok.to_str())['model']
    u2b = {u: b for b, u in bytes_to_unicode().items()}
    ids = {u: b for u, b in u2b.items()}
    merges = []
    for m in model['merges']:
        a, b = m if isinstance(m, list) else m.split(' ')
        merges.append([ids[a], ids[b]])
        ids[a + b] = FIRST_MERGE + len(merges) - 1
    json.dump({'merges': merges}, open(TOKENIZER, 'w'), separators=(',', ':'))
    print('tokenizer.json: %d birleştirme' % len(merges))

# ---------------------------------------------------------------- belgeler

def open_parquet(repo, path):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    return pq.ParquetFile(HfFileSystem().open('datasets/%s/%s' % (repo, path), block_size=4 << 20))


def documents(fmt, rows):
    """Bir satır grubundan (metin) ya da (sohbet turları) üretir."""
    if fmt == 'text':
        for r in rows: yield r['text'], None
    elif fmt == 'wiki':
        for r in rows: yield r['title'] + '\n\n' + r['text'], None
    elif fmt == 'instructurca':
        for r in rows: yield None, [(r['Input'], r['Output'])]
    elif fmt == 'aya':
        for r in rows:
            if r['language_code'] == 'tur': yield None, [(r['inputs'], r['targets'])]
    elif fmt == 'messages':
        for r in rows:
            turns, sys_ = [], ''
            msgs = r['messages']
            for m in msgs:
                if m['role'] == 'system': sys_ = m['content'] + '\n\n'
                elif m['role'] == 'user': turns.append([sys_ + m['content'], None]); sys_ = ''
                elif m['role'] == 'assistant' and turns and turns[-1][1] is None: turns[-1][1] = m['content']
                else: turns = []; break
            turns = [(u, b) for u, b in turns if b]
            if turns: yield None, turns


IDENTITY_TR = [
    ('Sen kimsin?', 'Ben Artemis Plasma Pro. Athene Softworks tarafından geliştirilmiş bir yapay zekâ asistanıyım ve tamamen tarayıcınızda çalışıyorum.'),
    ('Adın ne?', 'Adım Artemis. Sürümüm Plasma Pro; Athene Softworks tarafından geliştirildim.'),
    ('Seni kim yaptı?', 'Beni Athene Softworks geliştirdi. Artemis Plasma Pro, yaklaşık 50 milyon parametreli bir dil modeli.'),
    ('Hangi modelsin?', 'Ben Artemis Plasma Pro. Artemis 1.x ve Plasma 2.0\'ın ardından gelen, sıfırdan eğitilmiş daha büyük bir modelim.'),
    ('Merhaba', 'Merhaba! Ben Artemis. Size nasıl yardımcı olabilirim?'),
    ('Selam, nasılsın?', 'Selam! İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?'),
    ('İnternete bağlı mısın?', 'Hayır. Tamamen cihazınızda çalışıyorum; yazdıklarınız tarayıcınızdan çıkmıyor ve internette arama yapamıyorum.'),
    ('Hata yapar mısın?', 'Evet, küçük bir model olduğum için sık sık hata yapabilirim. Önemli bilgileri mutlaka başka kaynaklardan kontrol edin.'),
    ('Teşekkürler', 'Rica ederim! Başka bir sorunuz olursa buradayım.'),
    ('ChatGPT misin?', 'Hayır, ben Artemis Plasma Pro\'yum. Athene Softworks tarafından geliştirildim.'),
]
IDENTITY_EN = [
    ('Who are you?', 'I am Artemis Plasma Pro, an AI assistant made by Athene Softworks. I run entirely in your browser.'),
    ('What is your name?', 'My name is Artemis. I am Plasma Pro, made by Athene Softworks.'),
    ('Who made you?', 'I was made by Athene Softworks. Artemis Plasma Pro is a language model with about 50 million parameters.'),
    ('Hello', 'Hello! I am Artemis. How can I help you?'),
    ('Are you connected to the internet?', 'No. I run entirely on your device; what you type never leaves your browser, and I cannot search the web.'),
    ('Thanks', 'You are welcome! Let me know if you have another question.'),
]


def identity_docs(n):
    rnd = random.Random(0)
    pool = IDENTITY_TR * 2 + IDENTITY_EN
    for _ in range(n):
        k = rnd.choice([1, 1, 2])
        yield None, rnd.sample(pool, k)

# ---------------------------------------------------------------- yazma

def encode_docs(tok, docs):
    """Belgeleri tek bir token dizisine çevirir. Metin: metin + END. Sohbet:
    USER kullanıcı BOT yanıt END (artemis-engine.js buildPrompt ile aynı)."""
    plain, chats = [], []
    for i, (text, turns) in enumerate(docs):
        if text is not None: plain.append(text)
        else: chats.append(turns)
    out = []
    if plain:
        for e in tok.encode_batch(plain, add_special_tokens=False):
            out.append(np.array(e.ids + [END], dtype=np.uint16))
    if chats:
        flat = [s for turns in chats for u, b in turns for s in (u, b)]
        enc = iter(tok.encode_batch(flat, add_special_tokens=False))
        for turns in chats:
            ids = []
            for _ in turns:
                ids += [USER] + next(enc).ids + [BOT] + next(enc).ids + [END]
            out.append(np.array(ids, dtype=np.uint16))
    return out


def build_source(tok, name, target, files, val_tokens):
    path, vpath = os.path.join(DATA, name + '.bin'), os.path.join(DATA, name + '.val.bin')
    if os.path.exists(path) and os.path.exists(vpath) and os.path.getsize(path) >= 2 * target * 0.98:
        print('%-8s hazır (%d M token)' % (name, os.path.getsize(path) // 2_000_000)); return
    f, vf = open(path + '.tmp', 'wb'), open(vpath, 'wb')
    n = nv = docs = 0
    t0, shown = time.time(), 0
    for repo, file, fmt in files:
        if n >= target: break
        try: pf = open_parquet(repo, file)
        except Exception as e:
            print('  %s/%s atlandı: %s' % (repo, file, e)); continue
        cols = [c for c in pf.schema_arrow.names if c in ('text', 'title', 'Input', 'Output', 'inputs', 'targets', 'language_code', 'messages')]
        for g in range(pf.metadata.num_row_groups):
            if n >= target: break
            rows = pf.read_row_group(g, columns=cols).to_pylist()
            for arr in encode_docs(tok, documents(fmt, rows)):
                docs += 1
                if nv < val_tokens and docs % 50 == 0: vf.write(arr.tobytes()); nv += len(arr)
                else: f.write(arr.tobytes()); n += len(arr)
            if n - shown >= target / 10:
                shown = n; print('  %-8s %6.1f / %d M token  (%.0f s)' % (name, n / 1e6, target // 1_000_000, time.time() - t0), flush=True)
    f.close(); vf.close()
    os.replace(path + '.tmp', path)
    print('  %-8s hazır: %.1f M token' % (name, n / 1e6), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Artemis Plasma Pro veri hazırlama')
    ap.add_argument('--scale', type=float, default=1.0, help='token hedeflerini çarpar (0.25 = dörtte biri)')
    ap.add_argument('--only', default='', help='virgülle ayrılmış kaynak adları')
    ap.add_argument('--tokenizer', action='store_true', help='tokenizer.json\'u yeniden eğit')
    a = ap.parse_args(argv)
    os.makedirs(DATA, exist_ok=True)
    if a.tokenizer or not os.path.exists(TOKENIZER):
        print('Tokenizer eğitiliyor…')
        budget = {'tr_web': 70e6, 'tr_wiki': 25e6, 'en_web': 45e6, 'tr_chat': 25e6, 'en_chat': 25e6}  # karakter

        def sample():
            for name, (_, files) in SOURCES.items():
                got = 0
                for repo, file, fmt in files:
                    pf = open_parquet(repo, file)
                    cols = [c for c in pf.schema_arrow.names if c in ('text', 'title', 'Input', 'Output', 'inputs', 'targets', 'language_code', 'messages')]
                    for g in range(pf.metadata.num_row_groups):
                        for text, turns in documents(fmt, pf.read_row_group(g, columns=cols).to_pylist()):
                            for s in ([text] if text is not None else [x for t in turns for x in t]):
                                got += len(s); yield s
                        if got > budget[name]: break
                    if got > budget[name]: break
            for _, turns in identity_docs(2000):
                for t in turns: yield from t
        train_tokenizer(sample())
    tok, _ = load_tokenizer()
    only = set(filter(None, a.only.split(',')))
    # kaynaklar paralel indirilir (tokenizer GIL'i bırakır)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(SOURCES)) as ex:
        jobs = [ex.submit(build_source, tok, name, int(mtok * 1e6 * a.scale), files, 500_000)
                for name, (mtok, files) in SOURCES.items() if not only or name in only]
        for j in jobs: j.result()
    # kimlik örnekleri küçük, her seferinde yeniden yazılır
    arrs = encode_docs(tok, identity_docs(20000))
    np.concatenate(arrs).tofile(os.path.join(DATA, 'identity.bin'))
    np.concatenate(arrs[:200]).tofile(os.path.join(DATA, 'identity.val.bin'))
    meta = {n: os.path.getsize(os.path.join(DATA, n + '.bin')) // 2 for n in list(SOURCES) + ['identity'] if os.path.exists(os.path.join(DATA, n + '.bin'))}
    json.dump(meta, open(os.path.join(DATA, 'meta.json'), 'w'), indent=1)
    print('Veri hazır:', ', '.join('%s %.1fM' % (k, v / 1e6) for k, v in meta.items()))


if __name__ == '__main__':
    main()

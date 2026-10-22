/* Artemis çıkarım motoru — saf JavaScript, sunucu gerektirmez.
   Tarayıcıda (Web Worker) ve Node'da çalışır. Model dosyaları: artemis-1.0.bin (taban)
   ve artemis-1.1.bin (1.0'ın üzerine eklenen LoRA farkları + ek katmanlar). */
(function (root) {
  'use strict';

  var END = 256, USER = 257, BOT = 258, FIRST_MERGE = 260;
  var PRETOKEN = /[ ]?\p{L}+|[ ]?\p{N}{1,3}|[ ]?[^\s\p{L}\p{N}]+|\s+/gu;

  function Tokenizer(merges) {
    this.rank = new Map();
    this.bytes = [];
    this.cache = new Map();
    for (var i = 0; i < merges.length; i++) this.rank.set(merges[i][0] * 16384 + merges[i][1], i);
    for (i = 0; i < 256; i++) this.bytes.push([i]);
    for (i = 256; i < FIRST_MERGE; i++) this.bytes.push([]);
    for (i = 0; i < merges.length; i++) this.bytes.push(this.bytes[merges[i][0]].concat(this.bytes[merges[i][1]]));
    this.enc = new TextEncoder();
    this.dec = new TextDecoder();
  }
  Tokenizer.prototype.encodeWord = function (w) {
    var c = this.cache.get(w);
    if (c) return c;
    var s = Array.from(this.enc.encode(w));
    while (s.length > 1) {
      var bi = -1, br = Infinity;
      for (var i = 0; i < s.length - 1; i++) {
        var rk = this.rank.get(s[i] * 16384 + s[i + 1]);
        if (rk !== undefined && rk < br) { br = rk; bi = i; }
      }
      if (bi < 0) break;
      s.splice(bi, 2, FIRST_MERGE + br);
    }
    this.cache.set(w, s);
    return s;
  };
  Tokenizer.prototype.encode = function (text) {
    var out = [], m;
    PRETOKEN.lastIndex = 0;
    while ((m = PRETOKEN.exec(text)) !== null) Array.prototype.push.apply(out, this.encodeWord(m[0]));
    return out;
  };
  Tokenizer.prototype.bytesOf = function (ids) {
    var b = [];
    for (var i = 0; i < ids.length; i++) if (ids[i] < this.bytes.length) Array.prototype.push.apply(b, this.bytes[ids[i]]);
    return new Uint8Array(b);
  };
  Tokenizer.prototype.decode = function (ids) { return this.dec.decode(this.bytesOf(ids)); };

  // Model dosyasını okur: { header, W } (int8 → float32, satır ölçekleriyle)
  function readFile(buffer) {
    var dv = new DataView(buffer);
    var magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== 'ART1') throw new Error('Geçersiz model dosyası');
    var hlen = dv.getUint32(4, true);
    var header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, hlen)));
    var base = 8 + hlen + ((4 - ((8 + hlen) % 4)) % 4);
    var raw = {}, W = {};
    header.tensors.forEach(function (t) {
      var n = t.shape.reduce(function (a, b) { return a * b; }, 1);
      raw[t.name] = t.kind === 'q8' ? new Int8Array(buffer, base + t.offset, n) : new Float32Array(buffer, base + t.offset, n);
      raw[t.name].shape = t.shape;
    });
    Object.keys(raw).forEach(function (name) {
      if (/\.s$/.test(name)) return;
      var a = raw[name];
      if (a instanceof Int8Array) {
        var s = raw[name + '.s'], cols = a.shape[1], f = new Float32Array(a.length);
        for (var r = 0; r < a.shape[0]; r++) { var sc = s[r], o = r * cols; for (var j = 0; j < cols; j++) f[o + j] = a[o + j] * sc; }
        f.shape = a.shape; W[name] = f;
      } else W[name] = a;
    });
    return { header: header, W: W };
  }

  // buffer: Artemis 1.0 (taban). adapter (isteğe bağlı): Artemis 1.1 dosyası — tabanın
  // katmanlarına LoRA farklarını ekler, üstüne kendi katmanlarını koyar, son normu değiştirir.
  function Model(buffer, adapter) {
    var b = readFile(buffer), W = b.W;
    this.name = b.header.model;
    this.cfg = b.header.cfg;
    this.tok = new Tokenizer(b.header.merges);
    this.blocks = [];
    for (var l = 0; l < this.cfg.layers; l++) this.blocks.push('h' + l + '.');
    if (adapter) {
      var a = readFile(adapter), ac = a.header.cfg;
      if (a.header.base !== this.name) throw new Error(a.header.model + ' için ' + a.header.base + ' gerekir');
      // W += scale · B·A   (B: [out, r], A: [r, in])
      for (l = ac.loraFrom; l < this.cfg.layers; l++) ['qkv', 'proj', 'fc', 'out'].forEach(function (m) {
        var A = a.W['h' + l + '.' + m + '.la'], Bm = a.W['h' + l + '.' + m + '.lb'], w = W['h' + l + '.' + m + '.w'];
        var rows = w.shape[0], cols = w.shape[1], r = ac.rank, sc = ac.scale;
        for (var i = 0; i < rows; i++) for (var k = 0; k < r; k++) {
          var bk = Bm[i * r + k] * sc, ao = k * cols, wo = i * cols;
          if (bk) for (var j = 0; j < cols; j++) w[wo + j] += bk * A[ao + j];
        }
      });
      for (var e = 0; e < ac.extra; e++) {
        ['ln1.g', 'ln1.b', 'qkv.w', 'qkv.b', 'proj.w', 'proj.b', 'ln2.g', 'ln2.b', 'fc.w', 'fc.b', 'out.w', 'out.b'].forEach(function (p) { W['x' + e + '.' + p] = a.W['x' + e + '.' + p]; });
        this.blocks.push('x' + e + '.');
      }
      W['lnf.g'] = a.W['lnf.g']; W['lnf.b'] = a.W['lnf.b'];
      this.name = a.header.model;
    }
    this.W = W;
    var c = this.cfg;
    this.x = new Float32Array(c.d); this.h = new Float32Array(c.d);
    this.qkv = new Float32Array(3 * c.d); this.att = new Float32Array(c.d);
    this.ff = new Float32Array(4 * c.d); this.tmp = new Float32Array(c.d);
    this.scores = new Float32Array(c.ctx);
    this.logits = new Float32Array(c.vocab);
    this.K = []; this.V = [];
    for (l = 0; l < this.blocks.length; l++) { this.K.push(new Float32Array(c.ctx * c.d)); this.V.push(new Float32Array(c.ctx * c.d)); }
    this.pos = 0;
  }

  // out[r] = W[r,:]·x + b[r]
  function matvec(out, Wm, x, b, rows, cols) {
    for (var r = 0; r < rows; r++) {
      var s = 0, o = r * cols, j = 0;
      for (; j + 3 < cols; j += 4) s += Wm[o + j] * x[j] + Wm[o + j + 1] * x[j + 1] + Wm[o + j + 2] * x[j + 2] + Wm[o + j + 3] * x[j + 3];
      for (; j < cols; j++) s += Wm[o + j] * x[j];
      out[r] = b ? s + b[r] : s;
    }
  }
  function layerNorm(out, x, g, b) {
    var n = x.length, m = 0, v = 0, i;
    for (i = 0; i < n; i++) m += x[i];
    m /= n;
    for (i = 0; i < n; i++) v += (x[i] - m) * (x[i] - m);
    var inv = 1 / Math.sqrt(v / n + 1e-5);
    for (i = 0; i < n; i++) out[i] = (x[i] - m) * inv * g[i] + b[i];
  }

  Model.prototype.reset = function () { this.pos = 0; };

  // Bir token işler, sonraki token için logits döndürür
  Model.prototype.step = function (token) {
    var c = this.cfg, d = c.d, H = c.heads, hd = d / H, W = this.W, x = this.x, h = this.h, p = this.pos, i;
    var te = token * d, pe = p * d;
    for (i = 0; i < d; i++) x[i] = W.wte[te + i] + W.wpe[pe + i];
    var scale = 1 / Math.sqrt(hd);
    for (var l = 0; l < this.blocks.length; l++) {
      var pre = this.blocks[l];
      layerNorm(h, x, W[pre + 'ln1.g'], W[pre + 'ln1.b']);
      matvec(this.qkv, W[pre + 'qkv.w'], h, W[pre + 'qkv.b'], 3 * d, d);
      var K = this.K[l], V = this.V[l];
      for (i = 0; i < d; i++) { K[pe + i] = this.qkv[d + i]; V[pe + i] = this.qkv[2 * d + i]; }
      for (var hh = 0; hh < H; hh++) {
        var qo = hh * hd, mx = -Infinity, t, j;
        for (t = 0; t <= p; t++) {
          var s = 0, ko = t * d + qo;
          for (j = 0; j < hd; j++) s += this.qkv[qo + j] * K[ko + j];
          s *= scale; this.scores[t] = s; if (s > mx) mx = s;
        }
        var sum = 0;
        for (t = 0; t <= p; t++) { this.scores[t] = Math.exp(this.scores[t] - mx); sum += this.scores[t]; }
        for (j = 0; j < hd; j++) this.att[qo + j] = 0;
        for (t = 0; t <= p; t++) {
          var a = this.scores[t] / sum, vo = t * d + qo;
          for (j = 0; j < hd; j++) this.att[qo + j] += a * V[vo + j];
        }
      }
      matvec(this.tmp, W[pre + 'proj.w'], this.att, W[pre + 'proj.b'], d, d);
      for (i = 0; i < d; i++) x[i] += this.tmp[i];
      layerNorm(h, x, W[pre + 'ln2.g'], W[pre + 'ln2.b']);
      matvec(this.ff, W[pre + 'fc.w'], h, W[pre + 'fc.b'], 4 * d, d);
      for (i = 0; i < 4 * d; i++) { var z = this.ff[i]; this.ff[i] = 0.5 * z * (1 + Math.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z))); }
      matvec(this.tmp, W[pre + 'out.w'], this.ff, W[pre + 'out.b'], d, 4 * d);
      for (i = 0; i < d; i++) x[i] += this.tmp[i];
    }
    layerNorm(h, x, W['lnf.g'], W['lnf.b']);
    matvec(this.logits, W.wte, h, null, c.vocab, d);
    this.pos++;
    return this.logits;
  };

  Model.prototype.prefill = function (ids) {
    this.reset();
    var lg = null;
    for (var i = 0; i < ids.length; i++) lg = this.step(ids[i]);
    return lg;
  };

  function sample(logits, recent, opt) {
    var n = logits.length, i, l = new Float32Array(logits);
    var pen = opt.repetitionPenalty || 1;
    if (pen !== 1) recent.forEach(function (t) { l[t] = l[t] > 0 ? l[t] / pen : l[t] * pen; });
    for (i = 256; i < FIRST_MERGE; i++) if (i !== END) l[i] = -Infinity; // USER/BOT/PAD üretme
    var temp = opt.temperature == null ? 0.7 : opt.temperature;
    if (temp <= 0) { var bi = 0; for (i = 1; i < n; i++) if (l[i] > l[bi]) bi = i; return bi; }
    var k = opt.topK || 40, idx = [];
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return l[b] - l[a]; });
    idx = idx.slice(0, k);
    var mx = l[idx[0]], probs = idx.map(function (t) { return Math.exp((l[t] - mx) / temp); });
    var tot = probs.reduce(function (a, b) { return a + b; }, 0), cum = 0, cut = idx.length, topP = opt.topP || 0.9;
    for (i = 0; i < idx.length; i++) { cum += probs[i] / tot; if (cum >= topP) { cut = i + 1; break; } }
    tot = 0; for (i = 0; i < cut; i++) tot += probs[i];
    var r = (opt.random || Math.random)() * tot;
    for (i = 0; i < cut; i++) { r -= probs[i]; if (r <= 0) return idx[i]; }
    return idx[0];
  }

  // history: [{role:'user'|'bot', text}] — son mesaj kullanıcıdan
  Model.prototype.buildPrompt = function (history) {
    var ids = [], maxCtx = this.cfg.ctx - 96, tok = this.tok;
    for (var i = history.length - 1; i >= 0; i--) {
      var m = history[i], t = tok.encode(m.text), part;
      part = m.role === 'user' ? [USER].concat(t, [BOT]) : t.concat([END]);
      if (ids.length + part.length > maxCtx) {
        if (!ids.length) ids = [USER].concat(t.slice(-(maxCtx - 2)), [BOT]);
        break;
      }
      ids = part.concat(ids);
    }
    return ids;
  };

  // Akışlı üretim: her adımda yeni metin parçasını verir
  Model.prototype.stream = function* (history, opt) {
    opt = opt || {};
    var ids = this.buildPrompt(history), maxNew = opt.maxTokens || 200, ctx = this.cfg.ctx;
    var logits = this.prefill(ids), out = [], recent = [], dec = new TextDecoder();
    for (var n = 0; n < maxNew; n++) {
      var t = sample(logits, recent, opt);
      if (t === END) break;
      out.push(t); recent.push(t); if (recent.length > 64) recent.shift();
      var piece = dec.decode(this.tok.bytesOf([t]), { stream: true });
      if (piece) yield piece;
      if (this.pos >= ctx) logits = this.prefill(ids.concat(out).slice(-Math.floor(ctx * 0.6))); // pencere doldu
      else logits = this.step(t);
    }
  };

  Model.prototype.generate = function (history, opt) {
    var s = '';
    for (var p of this.stream(history, opt)) s += p;
    return s;
  };

  var api = { Model: Model, Tokenizer: Tokenizer, sample: sample, SPECIAL: { END: END, USER: USER, BOT: BOT } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Artemis = api;
})(typeof self !== 'undefined' ? self : this);

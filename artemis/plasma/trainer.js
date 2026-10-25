/* Artemis 2.0 Plasma — WebGPU eğitim motoru (Python gerektirmez).
   train.py ile aynı modeli (RoPE, QK-norm, ReLU² MLP, parametresiz RMSNorm, logit softcap) ve aynı
   optimizer'ları (gizli matrislerde Muon, gömme/çıktıda Adam) GPU'da, WGSL ile çalıştırır.
   Bellek için dikkat ağırlıkları saklanmaz; geri yayılımda yeniden hesaplanır (FlashAttention gibi). */
(function (root) {
  'use strict';

  var MM = `
struct P { M:u32, N:u32, K:u32, lda:u32, ldb:u32, ldc:u32, ta:u32, tb:u32,
  aoff:u32, boff:u32, coff:u32, zdiv:u32, sa1:u32, sa2:u32, sb1:u32, sb2:u32,
  sc1:u32, sc2:u32, tri:u32, p0:u32, alpha:f32, beta:f32, p1:f32, p2:f32 }
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> A:array<f32>;
@group(0) @binding(2) var<storage,read> B:array<f32>;
@group(0) @binding(3) var<storage,read_write> C:array<f32>;
var<workgroup> As:array<f32,1024>;
var<workgroup> Bs:array<f32,1024>;
// C[m,n] = alpha·Σk A(m,k)·B(k,n) + beta·C[m,n]; 64×64 döşeme, iş parçacığı başına 4×4
// tri: 1 = A alt üçgen, 2 = A üst üçgen, 3 = C alt üçgen (nedensel dikkat; boş döşemeler atlanır)
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) li:vec3<u32>, @builtin(local_invocation_index) t:u32) {
  let m0 = wg.y * 64u; let n0 = wg.x * 64u;
  if (p.tri == 3u && n0 > m0 + 63u) { return; }
  let z1 = wg.z / p.zdiv; let z2 = wg.z % p.zdiv;
  let ao = p.aoff + z1 * p.sa1 + z2 * p.sa2;
  let bo = p.boff + z1 * p.sb1 + z2 * p.sb2;
  let co = p.coff + z1 * p.sc1 + z2 * p.sc2;
  var kb = 0u; var ke = p.K;
  if (p.tri == 1u) { ke = min(p.K, m0 + 64u); }
  if (p.tri == 2u) { kb = (m0 / 16u) * 16u; }
  var acc = array<vec4<f32>,4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  for (var k0 = kb; k0 < ke; k0 += 16u) {
    for (var r = 0u; r < 4u; r++) {
      let e = t + r * 256u;
      var m = e / 16u; var k = e % 16u;
      if (p.ta == 1u) { m = e % 64u; k = e / 64u; }
      var v = 0.0;
      if (m0 + m < p.M && k0 + k < ke) {
        if (p.ta == 0u) { v = A[ao + (m0 + m) * p.lda + k0 + k]; } else { v = A[ao + (k0 + k) * p.lda + m0 + m]; }
      }
      As[k * 64u + m] = v;
      var kk = e / 64u; var n = e % 64u;
      if (p.tb == 1u) { n = e / 16u; kk = e % 16u; }
      var w = 0.0;
      if (n0 + n < p.N && k0 + kk < ke) {
        if (p.tb == 0u) { w = B[bo + (k0 + kk) * p.ldb + n0 + n]; } else { w = B[bo + (n0 + n) * p.ldb + k0 + kk]; }
      }
      Bs[kk * 64u + n] = w;
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      let ai = k * 64u + li.y * 4u; let bi = k * 64u + li.x * 4u;
      let b = vec4<f32>(Bs[bi], Bs[bi + 1u], Bs[bi + 2u], Bs[bi + 3u]);
      acc[0] += As[ai] * b; acc[1] += As[ai + 1u] * b; acc[2] += As[ai + 2u] * b; acc[3] += As[ai + 3u] * b;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let gm = m0 + li.y * 4u + i;
    if (gm >= p.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let gn = n0 + li.x * 4u + j;
      if (gn >= p.N) { continue; }
      let idx = co + gm * p.ldc + gn;
      var v = p.alpha * acc[i][j];
      if (p.beta != 0.0) { v += p.beta * C[idx]; }
      C[idx] = v;
    }
  }
}`;

  // Küçük çekirdekler ortak bir parametre yapısı kullanır
  var HEAD = `
struct E { n:u32, u1:u32, u2:u32, u3:u32, f1:f32, f2:f32, f3:f32, f4:f32 }
@group(0) @binding(0) var<uniform> p:E;
var<workgroup> red:array<f32,256>;
var<workgroup> red2:array<f32,256>;
fn rowId(wg:vec3<u32>, nw:vec3<u32>) -> u32 { return wg.y * nw.x + wg.x; }
fn sum256(i:u32) { for (var s = 128u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; red2[i] += red2[i + s]; } workgroupBarrier(); } }
`;
  var EL = '@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nw:vec3<u32>) {\n  let st = nw.x * 256u;\n';
  var ROW = '@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(num_workgroups) nw:vec3<u32>, @builtin(local_invocation_index) i:u32) {\n  let row = rowId(wg, nw);\n';

  var K = {
    embed: `@group(0) @binding(1) var<storage,read> tok:array<u32>;
@group(0) @binding(2) var<storage,read> W:array<f32>;
@group(0) @binding(3) var<storage,read_write> X:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) { X[k] = W[tok[k / p.u1] * p.u1 + k % p.u1]; } }`,

    copy: `@group(0) @binding(1) var<storage,read> S:array<f32>;
@group(0) @binding(2) var<storage,read_write> D:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) { D[k] = S[k]; } }`,

    // satır başına parametresiz RMSNorm: out = x / rms(x)
    rms: `@group(0) @binding(1) var<storage,read> X:array<f32>;
@group(0) @binding(2) var<storage,read_write> O:array<f32>;
${ROW} if (row >= p.n) { return; }
  let o = row * p.u1; var s = 0.0;
  for (var j = i; j < p.u1; j += 256u) { let v = X[o + j]; s += v * v; }
  red[i] = s; red2[i] = 0.0; workgroupBarrier(); sum256(i);
  let r = inverseSqrt(red[0] / f32(p.u1) + p.f1);
  for (var j = i; j < p.u1; j += 256u) { O[o + j] = X[o + j] * r; } }`,

    // dX (+)= r·(dy − x̂·mean(dy·x̂)); u2 = 1 ise üzerine ekler
    rmsBack: `@group(0) @binding(1) var<storage,read> X:array<f32>;
@group(0) @binding(2) var<storage,read> DY:array<f32>;
@group(0) @binding(3) var<storage,read_write> DX:array<f32>;
${ROW} if (row >= p.n) { return; }
  let o = row * p.u1; var s = 0.0; var q = 0.0;
  for (var j = i; j < p.u1; j += 256u) { let v = X[o + j]; s += v * v; q += v * DY[o + j]; }
  red[i] = s; red2[i] = q; workgroupBarrier(); sum256(i);
  let n = f32(p.u1); let r = inverseSqrt(red[0] / n + p.f1); let m = red2[0] * r * r / n;
  for (var j = i; j < p.u1; j += 256u) {
    let g = r * (DY[o + j] - X[o + j] * m);
    if (p.u2 == 1u) { DX[o + j] += g; } else { DX[o + j] = g; }
  } }`,

    // q ve k: başa göre RMSNorm + RoPE; v olduğu gibi kopyalanır. İş parçacığı = (token, baş dilimi)
    qk: `@group(0) @binding(1) var<storage,read> R:array<f32>;
@group(0) @binding(2) var<storage,read> CS:array<f32>;
@group(0) @binding(3) var<storage,read_write> O:array<f32>;
${EL} for (var k = gid.x; k < p.n * 3u * HH; k += st) {
    let tkn = k / (3u * HH); let sl = k % (3u * HH); let b = tkn * 3u * DD + sl * HD;
    if (sl >= 2u * HH) { for (var j = 0u; j < HD; j++) { O[b + j] = R[b + j]; } continue; }
    var s = 0.0; for (var j = 0u; j < HD; j++) { let v = R[b + j]; s += v * v; }
    let r = inverseSqrt(s / f32(HD) + p.f1); let c = (tkn % p.u1) * HD;
    for (var j = 0u; j < HD / 2u; j++) {
      let x1 = R[b + j] * r; let x2 = R[b + HD / 2u + j] * r; let co = CS[c + j]; let si = CS[c + HD / 2u + j];
      O[b + j] = x1 * co + x2 * si; O[b + HD / 2u + j] = x2 * co - x1 * si;
    } } }`,

    qkBack: `@group(0) @binding(1) var<storage,read> R:array<f32>;
@group(0) @binding(2) var<storage,read> CS:array<f32>;
@group(0) @binding(3) var<storage,read_write> G:array<f32>;
${EL} for (var k = gid.x; k < p.n * 2u * HH; k += st) {
    let tkn = k / (2u * HH); let sl = k % (2u * HH); let b = tkn * 3u * DD + sl * HD;
    var s = 0.0; for (var j = 0u; j < HD; j++) { let v = R[b + j]; s += v * v; }
    let r = inverseSqrt(s / f32(HD) + p.f1); let c = (tkn % p.u1) * HD;
    var q = 0.0;
    for (var j = 0u; j < HD / 2u; j++) {
      let d1 = G[b + j]; let d2 = G[b + HD / 2u + j]; let co = CS[c + j]; let si = CS[c + HD / 2u + j];
      let g1 = d1 * co - d2 * si; let g2 = d1 * si + d2 * co;
      G[b + j] = g1; G[b + HD / 2u + j] = g2;
      q += g1 * R[b + j] * r + g2 * R[b + HD / 2u + j] * r;
    }
    let m = q / f32(HD);
    for (var j = 0u; j < HD; j++) { G[b + j] = r * (G[b + j] - R[b + j] * r * m); }
  } }`,

    // nedensel softmax; satır = (dizi·baş, i). L = logsumexp (geri yayılım için)
    softmax: `@group(0) @binding(1) var<storage,read_write> S:array<f32>;
@group(0) @binding(2) var<storage,read_write> L:array<f32>;
${ROW} if (row >= p.n) { return; }
  let T = p.u1; let q = row % T; let o = row * T;
  var mx = -1e30; for (var j = i; j <= q; j += 256u) { mx = max(mx, S[o + j]); }
  red[i] = mx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (i < s) { red[i] = max(red[i], red[i + s]); } workgroupBarrier(); }
  mx = red[0]; workgroupBarrier();
  var sm = 0.0; for (var j = i; j <= q; j += 256u) { sm += exp(S[o + j] - mx); }
  red[i] = sm; red2[i] = 0.0; workgroupBarrier(); sum256(i);
  let lse = mx + log(red[0]);
  for (var j = i; j < T; j += 256u) { if (j <= q) { S[o + j] = exp(S[o + j] - lse); } else { S[o + j] = 0.0; } }
  if (i == 0u) { L[row] = lse; } }`,

    probs: `@group(0) @binding(1) var<storage,read_write> S:array<f32>;
@group(0) @binding(2) var<storage,read> L:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) {
    let row = k / p.u1; let j = k % p.u1;
    if (j <= row % p.u1) { S[k] = exp(S[k] - L[row]); } else { S[k] = 0.0; } } }`,

    // D[satır] = Σ dY·Y (başa göre)
    attnD: `@group(0) @binding(1) var<storage,read> DY:array<f32>;
@group(0) @binding(2) var<storage,read> Y:array<f32>;
@group(0) @binding(3) var<storage,read_write> D:array<f32>;
${EL} for (var k = gid.x; k < p.n * HH; k += st) {
    let tkn = k / HH; let h = k % HH; let b = tkn * DD + h * HD;
    var s = 0.0; for (var j = 0u; j < HD; j++) { s += DY[b + j] * Y[b + j]; }
    let sq = tkn / p.u1; let i = tkn % p.u1;
    D[(sq * HH + h) * p.u1 + i] = s; } }`,

    dS: `@group(0) @binding(1) var<storage,read> Pm:array<f32>;
@group(0) @binding(2) var<storage,read> D:array<f32>;
@group(0) @binding(3) var<storage,read_write> G:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) {
    let row = k / p.u1; let j = k % p.u1;
    if (j <= row % p.u1) { G[k] = Pm[k] * (G[k] - D[row]) * p.f1; } else { G[k] = 0.0; } } }`,

    relu2: `@group(0) @binding(1) var<storage,read> Z:array<f32>;
@group(0) @binding(2) var<storage,read_write> O:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) { let z = max(Z[k], 0.0); O[k] = z * z; } }`,

    relu2Back: `@group(0) @binding(1) var<storage,read> Z:array<f32>;
@group(0) @binding(2) var<storage,read_write> G:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) { G[k] = G[k] * 2.0 * max(Z[k], 0.0); } }`,

    // softcap + çapraz entropi; logits yerinde gradyana dönüşür. M[0] = kayıp ölçeği
    ce: `@group(0) @binding(1) var<storage,read> TG:array<i32>;
@group(0) @binding(2) var<storage,read> M:array<f32>;
@group(0) @binding(3) var<storage,read_write> Z:array<f32>;
@group(0) @binding(4) var<storage,read_write> LS:array<f32>;
${ROW} if (row >= p.n) { return; }
  let V = p.u1; let o = row * V; let cap = p.f1;
  // hedefi -1 olan (kullanıcı mesajı) satırlar sıfır gradyan ve sıfır kayıp verir
  let ok = select(0.0, 1.0, TG[row] >= 0); let tg = select(0, TG[row], TG[row] >= 0);
  var mx = -1e30;
  for (var j = i; j < V; j += 256u) { let c = cap * tanh(Z[o + j] / cap); Z[o + j] = c; mx = max(mx, c); }
  red[i] = mx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (i < s) { red[i] = max(red[i], red[i + s]); } workgroupBarrier(); }
  mx = red[0]; workgroupBarrier();
  var sm = 0.0; for (var j = i; j < V; j += 256u) { sm += exp(Z[o + j] - mx); }
  red[i] = sm; red2[i] = 0.0; workgroupBarrier(); sum256(i);
  let lse = mx + log(red[0]); let sc = M[0] * ok;
  if (i == 0u) { LS[row] = (lse - Z[o + u32(tg)]) * ok; }
  workgroupBarrier();
  for (var j = i; j < V; j += 256u) {
    let c = Z[o + j]; var g = exp(c - lse); if (j == u32(tg)) { g -= 1.0; }
    let t = c / cap; Z[o + j] = g * sc * (1.0 - t * t);
  } }`,

    // gömme gradyanı: CPU token'ları sıralar; iş grubu = benzersiz token. I = [n, uniq(N), start(N+1), order(N)]
    embBack: `@group(0) @binding(1) var<storage,read> I:array<u32>;
@group(0) @binding(2) var<storage,read> G:array<f32>;
@group(0) @binding(3) var<storage,read_write> W:array<f32>;
${ROW} if (row >= I[0]) { return; }
  let N = p.n; let d = p.u1; let a = I[1u + N + row]; let b = I[2u + N + row]; let tk = I[1u + row];
  for (var j = i; j < d; j += 256u) {
    var s = 0.0; for (var q = a; q < b; q++) { s += G[I[2u + 2u * N + q] * d + j]; }
    W[tk * d + j] += s;
  } }`,

    // H = [lr_wte, lr_lm, lr_muon, momentum, 1-b1^t, 1-b2^t]
    adam: `@group(0) @binding(1) var<storage,read> H:array<f32>;
@group(0) @binding(2) var<storage,read> G:array<f32>;
@group(0) @binding(3) var<storage,read_write> W:array<f32>;
@group(0) @binding(4) var<storage,read_write> M1:array<f32>;
@group(0) @binding(5) var<storage,read_write> M2:array<f32>;
${EL} let lr = H[p.u1];
  for (var k = gid.x; k < p.n; k += st) {
    let g = G[k]; let m = p.f1 * M1[k] + (1.0 - p.f1) * g; let v = p.f2 * M2[k] + (1.0 - p.f2) * g * g;
    M1[k] = m; M2[k] = v;
    W[k] -= lr * (m / H[4]) / (sqrt(v / H[5]) + p.f3);
  } }`,

    momentum: `@group(0) @binding(1) var<storage,read> H:array<f32>;
@group(0) @binding(2) var<storage,read> G:array<f32>;
@group(0) @binding(3) var<storage,read_write> BU:array<f32>;
@group(0) @binding(4) var<storage,read_write> U:array<f32>;
${EL} let mu = H[3];
  for (var k = gid.x; k < p.n; k += st) { let g = G[k]; let b = BU[k] + (1.0 - mu) * (g - BU[k]); BU[k] = b; U[k] = g + mu * (b - g); } }`,

    // iş grubu = matris; Frobenius normu
    norm: `@group(0) @binding(1) var<storage,read> U:array<f32>;
@group(0) @binding(2) var<storage,read_write> NR:array<f32>;
${ROW} if (row >= p.u1) { return; }
  var s = 0.0; for (var j = i; j < p.n; j += 256u) { let v = U[row * p.n + j]; s += v * v; }
  red[i] = s; red2[i] = 0.0; workgroupBarrier(); sum256(i);
  if (i == 0u) { NR[row] = sqrt(red[0]); } }`,

    nscale: `@group(0) @binding(1) var<storage,read> U:array<f32>;
@group(0) @binding(2) var<storage,read> NR:array<f32>;
@group(0) @binding(3) var<storage,read_write> X:array<f32>;
${EL} for (var k = gid.x; k < p.n; k += st) { X[k] = U[k] / (NR[k / p.u1] + 1e-7); } }`,

    update: `@group(0) @binding(1) var<storage,read> H:array<f32>;
@group(0) @binding(2) var<storage,read> X:array<f32>;
@group(0) @binding(3) var<storage,read_write> W:array<f32>;
${EL} let lr = H[p.u1] * p.f1;
  for (var k = gid.x; k < p.n; k += st) { W[k] -= lr * X[k]; } }`
  };

  var NS = [3.4445, -4.7750, 2.0315]; // Muon Newton–Schulz katsayıları

  function Trainer(device, cfg, micro) {
    this.dev = device; this.cfg = cfg; this.B = micro;
    var c = cfg, d = c.d, T = c.ctx, N = micro * T, H = c.heads, hd = d / H, V = c.vocab, F = c.mlp, Lr = c.layers;
    this.N = N;
    var self = this, consts = 'const DD = ' + d + 'u; const HH = ' + H + 'u; const HD = ' + hd + 'u;\n';
    this.pipes = {};
    Object.keys(K).forEach(function (k) { self.pipes[k] = self.pipe(consts + HEAD + K[k]); });
    this.pipes.mm = this.pipe(MM);

    var S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    function buf(n) { return device.createBuffer({ size: Math.max(16, n * 4), usage: S }); }
    this.buf = buf;
    // ağırlıklar, gradyanlar, optimizer durumu
    this.shapes = { wte: [V, d], lm: [V, d] };
    for (var l = 0; l < Lr; l++) {
      this.shapes['h' + l + '.qkv'] = [3 * d, d]; this.shapes['h' + l + '.o'] = [d, d];
      this.shapes['h' + l + '.fc'] = [F, d]; this.shapes['h' + l + '.out'] = [d, F];
    }
    this.W = {}; this.G = {}; this.M1 = {}; this.M2 = {};
    Object.keys(this.shapes).forEach(function (k) {
      var n = self.shapes[k][0] * self.shapes[k][1];
      self.W[k] = buf(n); self.G[k] = buf(n); self.M1[k] = buf(n);
      if (k === 'wte' || k === 'lm') self.M2[k] = buf(n);
    });
    // etkinlikler (katman başına saklananlar) ve geçici alanlar
    var A = this.A = { xs: [], raw: [], y: [], xm: [], z: [], L: [] };
    for (l = 0; l <= Lr; l++) A.xs.push(buf(N * d));
    for (l = 0; l < Lr; l++) { A.raw.push(buf(N * 3 * d)); A.y.push(buf(N * d)); A.xm.push(buf(N * d)); A.z.push(buf(N * F)); A.L.push(buf(micro * H * T)); }
    var mxN = Math.max(3 * d * d, F * d); // en büyük gizli matris
    var X = this.X = {
      h: buf(N * d), hf: buf(N * d), post: buf(N * 3 * d), a: buf(N * F), dA: buf(N * F), dH: buf(N * d),
      dY: buf(N * d), dqkv: buf(N * 3 * d), g: buf(N * d), S: buf(micro * H * T * T), dP: buf(micro * H * T * T),
      D: buf(micro * H * T), logits: buf(N * V), loss: buf(N),
      tok: buf(N), tgt: buf(N), meta: buf(4), emb: buf(3 * N + 2), hyper: buf(8), cs: buf(T * hd),
      nr: buf(4), mx: buf(mxN), mxn: buf(mxN), am: buf(3 * d * d), bm: buf(3 * d * d), mu: buf(mxN)
    };
    var cs = new Float32Array(T * hd);
    for (var t = 0; t < T; t++) for (var i = 0; i < hd / 2; i++) {
      var ang = t * Math.pow(c.rope, -2 * i / hd);
      cs[t * hd + i] = Math.cos(ang); cs[t * hd + hd / 2 + i] = Math.sin(ang);
    }
    device.queue.writeBuffer(X.cs, 0, cs);

    // ---------------- işlem listeleri (her adımda aynı; yalnızca girdiler değişir)
    var micro_ = this.microOps = [], opt = this.optOps = [];
    var ops = micro_;
    function E(k, e, bufs, wg) {
      var u = new ArrayBuffer(32), dv = new DataView(u);
      [e.n, e.u1, e.u2, e.u3].forEach(function (v, i) { dv.setUint32(i * 4, v || 0, true); });
      [e.f1, e.f2, e.f3, e.f4].forEach(function (v, i) { dv.setFloat32(16 + i * 4, v || 0, true); });
      ops.push(self.op(k, u, bufs, wg));
    }
    function el(n) { return [Math.min(Math.ceil(n / 256), 8192), 1, 1]; }
    function rows(n) { return [Math.min(n, 32768), Math.ceil(n / 32768), 1]; }
    function mm(o) {
      var u = new ArrayBuffer(96), dv = new DataView(u);
      ['M', 'N', 'K', 'lda', 'ldb', 'ldc', 'ta', 'tb', 'aoff', 'boff', 'coff', 'zdiv', 'sa1', 'sa2', 'sb1', 'sb2', 'sc1', 'sc2', 'tri'].forEach(function (k, i) {
        dv.setUint32(i * 4, k === 'zdiv' ? (o.zdiv || 1) : (o[k] || 0), true);
      });
      dv.setFloat32(80, o.alpha == null ? 1 : o.alpha, true); dv.setFloat32(84, o.beta || 0, true);
      ops.push(self.op('mm', u, [o.A, o.B, o.C], [Math.ceil(o.N / 64), Math.ceil(o.M / 64), o.z || 1]));
    }
    // Y[N×n] (+)= X[N×k] · Wᵀ  (W: n×k)
    function linear(Xb, Wb, Yb, n, k, beta, ldc) { mm({ A: Xb, B: Wb, C: Yb, M: N, N: n, K: k, lda: k, ldb: k, ldc: ldc || n, tb: 1, beta: beta }); }
    // dW[n×k] += dYᵀ · X
    function gradW(dY, Xb, dW, n, k) { mm({ A: dY, B: Xb, C: dW, M: n, N: k, K: N, lda: n, ldb: k, ldc: k, ta: 1, beta: 1 }); }
    // dX[N×k] = dY[N×n] · W
    function gradX(dY, Wb, dX, n, k) { mm({ A: dY, B: Wb, C: dX, M: N, N: k, K: n, lda: n, ldb: k, ldc: k }); }
    var Z = micro * H, TT = T * T, sc = 1 / Math.sqrt(hd), eps = c.eps;
    var q3 = { lda: 3 * d, sa1: T * 3 * d, sa2: hd }, zz = { z: Z, zdiv: H };
    function scores(post) {
      mm(Object.assign({ A: post, B: post, C: X.S, M: T, N: T, K: hd, boff: d, ldb: 3 * d, sb1: T * 3 * d, sb2: hd, tb: 1, ldc: T, sc1: H * TT, sc2: TT, alpha: sc, tri: 3 }, q3, zz));
    }
    var P_ = { lda: T, sa1: H * TT, sa2: TT };

    E('embed', { n: N * d, u1: d }, [X.tok, this.W.wte, A.xs[0]], el(N * d));
    for (l = 0; l < Lr; l++) {
      var p = 'h' + l + '.';
      E('rms', { n: N, u1: d, f1: eps }, [A.xs[l], X.h], rows(N));
      linear(X.h, this.W[p + 'qkv'], A.raw[l], 3 * d, d);
      E('qk', { n: N, u1: T, f1: eps }, [A.raw[l], X.cs, X.post], el(N * 3 * H));
      scores(X.post);
      E('softmax', { n: Z * T, u1: T }, [X.S, A.L[l]], rows(Z * T));
      mm(Object.assign({ A: X.S, B: X.post, C: A.y[l], M: T, N: hd, K: T, boff: 2 * d, ldb: 3 * d, sb1: T * 3 * d, sb2: hd, ldc: d, sc1: T * d, sc2: hd, tri: 1 }, P_, zz));
      E('copy', { n: N * d }, [A.xs[l], A.xm[l]], el(N * d));
      linear(A.y[l], this.W[p + 'o'], A.xm[l], d, d, 1);
      E('rms', { n: N, u1: d, f1: eps }, [A.xm[l], X.h], rows(N));
      linear(X.h, this.W[p + 'fc'], A.z[l], F, d);
      E('relu2', { n: N * F }, [A.z[l], X.a], el(N * F));
      E('copy', { n: N * d }, [A.xm[l], A.xs[l + 1]], el(N * d));
      linear(X.a, this.W[p + 'out'], A.xs[l + 1], d, F, 1);
    }
    E('rms', { n: N, u1: d, f1: eps }, [A.xs[Lr], X.hf], rows(N));
    linear(X.hf, this.W.lm, X.logits, V, d);
    E('ce', { n: N, u1: V, f1: c.softcap }, [X.tgt, X.meta, X.logits, X.loss], rows(N));
    // ---- geri yayılım
    gradW(X.logits, X.hf, this.G.lm, V, d);
    gradX(X.logits, this.W.lm, X.dH, V, d);
    E('rmsBack', { n: N, u1: d, u2: 0, f1: eps }, [A.xs[Lr], X.dH, X.g], rows(N));
    var dq = { C: X.dqkv, ldc: 3 * d, sc1: T * 3 * d, sc2: hd };
    for (l = Lr - 1; l >= 0; l--) {
      p = 'h' + l + '.';
      E('relu2', { n: N * F }, [A.z[l], X.a], el(N * F));
      gradW(X.g, X.a, this.G[p + 'out'], d, F);
      gradX(X.g, this.W[p + 'out'], X.dA, d, F);
      E('relu2Back', { n: N * F }, [A.z[l], X.dA], el(N * F));
      E('rms', { n: N, u1: d, f1: eps }, [A.xm[l], X.h], rows(N));
      gradW(X.dA, X.h, this.G[p + 'fc'], F, d);
      gradX(X.dA, this.W[p + 'fc'], X.dH, F, d);
      E('rmsBack', { n: N, u1: d, u2: 1, f1: eps }, [A.xm[l], X.dH, X.g], rows(N));
      gradW(X.g, A.y[l], this.G[p + 'o'], d, d);
      gradX(X.g, this.W[p + 'o'], X.dY, d, d);
      E('qk', { n: N, u1: T, f1: eps }, [A.raw[l], X.cs, X.post], el(N * 3 * H));
      scores(X.post);
      E('probs', { n: Z * TT, u1: T }, [X.S, A.L[l]], el(Z * TT));
      var dy = { B: X.dY, ldb: d, sb1: T * d, sb2: hd };
      mm(Object.assign({ A: X.S, ta: 1, M: T, N: hd, K: T, coff: 2 * d, tri: 2 }, P_, dy, dq, zz));
      mm(Object.assign({ A: X.dY, lda: d, sa1: T * d, sa2: hd, B: X.post, boff: 2 * d, tb: 1, ldb: 3 * d, sb1: T * 3 * d, sb2: hd, C: X.dP, ldc: T, sc1: H * TT, sc2: TT, M: T, N: T, K: hd, tri: 3 }, zz));
      E('attnD', { n: N, u1: T }, [X.dY, A.y[l], X.D], el(N * H));
      E('dS', { n: Z * TT, u1: T, f1: sc }, [X.S, X.D, X.dP], el(Z * TT));
      var qb = { B: X.post, ldb: 3 * d, sb1: T * 3 * d, sb2: hd, M: T, N: hd, K: T };
      mm(Object.assign({ A: X.dP, boff: d, coff: 0, tri: 1 }, P_, qb, dq, zz));
      mm(Object.assign({ A: X.dP, ta: 1, boff: 0, coff: d, tri: 2 }, P_, qb, dq, zz));
      E('qkBack', { n: N, u1: T, f1: eps }, [A.raw[l], X.cs, X.dqkv], el(N * 2 * H));
      E('rms', { n: N, u1: d, f1: eps }, [A.xs[l], X.h], rows(N));
      gradW(X.dqkv, X.h, this.G[p + 'qkv'], 3 * d, d);
      gradX(X.dqkv, this.W[p + 'qkv'], X.dH, 3 * d, d);
      E('rmsBack', { n: N, u1: d, u2: 1, f1: eps }, [A.xs[l], X.dH, X.g], rows(N));
    }
    E('embBack', { n: N, u1: d }, [X.emb, X.g, this.G.wte], rows(N));

    // ---- optimizer
    ops = opt;
    ['wte', 'lm'].forEach(function (k, i) {
      var n = V * d;
      E('adam', { n: n, u1: i, f1: 0.8, f2: 0.95, f3: 1e-10 }, [X.hyper, self.G[k], self.W[k], self.M1[k], self.M2[k]], el(n));
    });
    for (l = 0; l < Lr; l++) ['qkv', 'o', 'fc', 'out'].forEach(function (m) {
      var k = 'h' + l + '.' + m, sh = self.shapes[k], nb = m === 'qkv' ? 3 : 1, r = sh[0] / nb, cc = sh[1], n = nb * r * cc;
      E('momentum', { n: n }, [X.hyper, self.G[k], self.M1[k], X.mu], el(n));
      E('norm', { n: r * cc, u1: nb }, [X.mu, X.nr], rows(nb));
      E('nscale', { n: n, u1: r * cc }, [X.mu, X.nr, X.mx], el(n));
      var cur = X.mx, nxt = X.mxn, s = Math.min(r, cc), b = { z: nb, sa1: r * cc, sb1: r * cc, sc1: r * cc };
      for (var it = 0; it < 5; it++) {
        var sq = { z: nb, sa1: s * s, sb1: s * s, sc1: s * s };
        if (r <= cc) mm({ A: cur, B: cur, C: X.am, M: r, N: r, K: cc, lda: cc, ldb: cc, ldc: r, tb: 1, z: nb, sa1: r * cc, sb1: r * cc, sc1: r * r });
        else mm({ A: cur, B: cur, C: X.am, M: cc, N: cc, K: r, lda: cc, ldb: cc, ldc: cc, ta: 1, z: nb, sa1: r * cc, sb1: r * cc, sc1: cc * cc });
        E('copy', { n: nb * s * s }, [X.am, X.bm], el(nb * s * s));
        mm(Object.assign({ A: X.am, B: X.am, C: X.bm, M: s, N: s, K: s, lda: s, ldb: s, ldc: s, alpha: NS[2], beta: NS[1] }, sq));
        E('copy', { n: n }, [cur, nxt], el(n));
        if (r <= cc) mm({ A: X.bm, B: cur, C: nxt, M: r, N: cc, K: r, lda: r, ldb: cc, ldc: cc, beta: NS[0], z: nb, sa1: r * r, sb1: r * cc, sc1: r * cc });
        else mm({ A: cur, B: X.bm, C: nxt, M: r, N: cc, K: cc, lda: cc, ldb: cc, ldc: cc, beta: NS[0], z: nb, sa1: r * cc, sb1: cc * cc, sc1: r * cc });
        var tmp = cur; cur = nxt; nxt = tmp;
      }
      E('update', { n: n, u1: 2, f1: Math.sqrt(Math.max(1, r / cc)) }, [X.hyper, cur, self.W[k]], el(n));
    });
    this.steps = 0;
  }

  Trainer.prototype.pipe = function (code) {
    var m = this.dev.createShaderModule({ code: code });
    return this.dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
  };
  Trainer.prototype.op = function (k, params, bufs, wg) {
    var dev = this.dev, u = dev.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(u, 0, params);
    var entries = [{ binding: 0, resource: { buffer: u } }].concat(bufs.map(function (b, i) { return { binding: i + 1, resource: { buffer: b } }; }));
    return { pipe: this.pipes[k], bg: dev.createBindGroup({ layout: this.pipes[k].getBindGroupLayout(0), entries: entries }), wg: wg };
  };
  Trainer.prototype.run = function (ops) {
    var enc = this.dev.createCommandEncoder(), pass = enc.beginComputePass();
    for (var i = 0; i < ops.length; i++) { pass.setPipeline(ops[i].pipe); pass.setBindGroup(0, ops[i].bg); pass.dispatchWorkgroups(ops[i].wg[0], ops[i].wg[1], ops[i].wg[2]); }
    pass.end();
    return enc;
  };

  // Ağırlıkları train.py ile aynı biçimde başlatır
  Trainer.prototype.init = function (seed) {
    var s = seed >>> 0 || 1, self = this;
    function rnd() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; }
    function gauss() { return Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd()); }
    var w = {};
    Object.keys(this.shapes).forEach(function (k) {
      var sh = self.shapes[k], a = new Float32Array(sh[0] * sh[1]), i;
      if (k === 'wte') for (i = 0; i < a.length; i++) a[i] = gauss();
      else if (/qkv|fc/.test(k)) { var b = 1 / Math.sqrt(sh[1]); for (i = 0; i < a.length; i++) a[i] = (2 * rnd() - 1) * b; }
      w[k] = a; // o, out, lm: sıfır
    });
    this.setWeights(w);
  };
  Trainer.prototype.setWeights = function (w) {
    var self = this;
    Object.keys(w).forEach(function (k) { self.dev.queue.writeBuffer(self.W[k], 0, w[k]); });
  };
  Trainer.prototype.read = async function (bufs) {
    var dev = this.dev, enc = dev.createCommandEncoder(), out = {}, st = {};
    Object.keys(bufs).forEach(function (k) {
      st[k] = dev.createBuffer({ size: bufs[k].size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(bufs[k], 0, st[k], 0, bufs[k].size);
    });
    dev.queue.submit([enc.finish()]);
    await Promise.all(Object.keys(st).map(function (k) { return st[k].mapAsync(GPUMapMode.READ); }));
    Object.keys(st).forEach(function (k) { out[k] = new Float32Array(st[k].getMappedRange().slice(0)); st[k].unmap(); st[k].destroy(); });
    return out;
  };
  Trainer.prototype.weights = function () { return this.read(this.W); };

  // Bir mikro-yığını GPU'ya yükler. seqs: B adet (ctx+1) uzunlukta token dizisi; kullanıcı mesajları kayba katılmaz
  Trainer.prototype.upload = function (seqs, accum) {
    var c = this.cfg, T = c.ctx, N = this.N, tok = new Uint32Array(N), tgt = new Int32Array(N), cnt = 0;
    for (var b = 0; b < this.B; b++) {
      var s = seqs[b], user = false;
      if ((s[0] === 257)) user = true;
      for (var t = 0; t < T; t++) {
        tok[b * T + t] = s[t];
        var y = s[t + 1];
        if (y === 257) user = true; else if (y === 258 || y === 256) user = false;
        if (user) tgt[b * T + t] = -1; else { tgt[b * T + t] = y; cnt++; }
      }
    }
    // gömme gradyanı için token'a göre sıralanmış konumlar
    var order = new Uint32Array(N); for (var i = 0; i < N; i++) order[i] = i;
    order.sort(function (a, b) { return tok[a] - tok[b] || a - b; });
    var emb = new Uint32Array(3 * N + 2), nu = 0;
    for (i = 0; i < N; i++) {
      if (i === 0 || tok[order[i]] !== tok[order[i - 1]]) { emb[1 + nu] = tok[order[i]]; emb[1 + N + nu] = i; nu++; }
      emb[2 + 2 * N + i] = order[i];
    }
    emb[0] = nu; emb[1 + N + nu] = N;
    var q = this.dev.queue;
    q.writeBuffer(this.X.tok, 0, tok); q.writeBuffer(this.X.tgt, 0, tgt); q.writeBuffer(this.X.emb, 0, emb);
    q.writeBuffer(this.X.meta, 0, new Float32Array([1 / Math.max(1, cnt) / accum, 0, 0, 0]));
    return cnt;
  };

  // Tam bir optimizer adımı: micros = mikro-yığınlar (her biri B dizi). hp: {lrWte, lrLm, lrMuon, momentum}
  Trainer.prototype.step = async function (micros, hp) {
    var dev = this.dev, enc, N = this.N, accum = micros.length, self = this;
    var st = dev.createBuffer({ size: accum * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), cnts = [];
    for (var i = 0; i < accum; i++) {
      cnts.push(this.upload(micros[i], accum));
      enc = this.run(this.microOps);
      enc.copyBufferToBuffer(this.X.loss, 0, st, i * N * 4, N * 4);
      dev.queue.submit([enc.finish()]);
    }
    this.steps++;
    var t = this.steps;
    dev.queue.writeBuffer(this.X.hyper, 0, new Float32Array([hp.lrWte, hp.lrLm, hp.lrMuon, hp.momentum, 1 - Math.pow(0.8, t), 1 - Math.pow(0.95, t), 0, 0]));
    enc = this.run(this.optOps);
    Object.keys(this.G).forEach(function (k) { enc.clearBuffer(self.G[k]); });
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    var ls = new Float32Array(st.getMappedRange()), loss = 0;
    for (i = 0; i < accum; i++) { var s = 0; for (var j = 0; j < N; j++) s += ls[i * N + j]; loss += s / Math.max(1, cnts[i]); }
    st.unmap(); st.destroy();
    return loss / accum;
  };

  // Yalnızca gradyan (test için): bir mikro-yığın, optimizer yok
  Trainer.prototype.grads = async function (seqs) {
    this.upload(seqs, 1);
    var enc = this.run(this.microOps);
    this.dev.queue.submit([enc.finish()]);
    var g = await this.read(this.G), l = await this.read({ loss: this.X.loss });
    var enc2 = this.dev.createCommandEncoder(), self = this;
    Object.keys(this.G).forEach(function (k) { enc2.clearBuffer(self.G[k]); });
    this.dev.queue.submit([enc2.finish()]);
    return { grads: g, loss: l.loss };
  };

  // Uygun bellek sınırlarıyla bir GPU açar ve eğiticiyi kurar
  Trainer.create = async function (cfg, micro) {
    if (!navigator.gpu) throw new Error('Bu tarayıcı WebGPU desteklemiyor');
    var ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!ad) throw new Error('WebGPU bağdaştırıcısı bulunamadı');
    var L = ad.limits, need = micro * cfg.ctx * cfg.vocab * 4;
    while (micro > 1 && need > L.maxStorageBufferBindingSize) { micro >>= 1; need = micro * cfg.ctx * cfg.vocab * 4; }
    var dev = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize, maxStorageBuffersPerShaderStage: Math.min(8, L.maxStorageBuffersPerShaderStage) } });
    var errs = [];
    dev.pushErrorScope('validation');
    var tr = new Trainer(dev, cfg, micro);
    var e = await dev.popErrorScope();
    if (e) throw new Error('WebGPU: ' + e.message);
    tr.adapter = ad;
    return tr;
  };

  // ART2 dosyası üretir (train.py export ile aynı biçim)
  function exportArt2(w, cfg, merges, info) {
    var names = ['wte', 'lm'];
    for (var l = 0; l < cfg.layers; l++) ['qkv', 'o', 'fc', 'out'].forEach(function (m) { names.push('h' + l + '.' + m); });
    var tensors = [], blobs = [], off = 0;
    names.forEach(function (name) {
      var a = w[name], cols = cfg.d, rows;
      if (/\.out$/.test(name)) cols = cfg.mlp;
      rows = a.length / cols;
      var s = new Float32Array(rows), q = new Int8Array(a.length);
      for (var r = 0; r < rows; r++) {
        var mx = 0, o = r * cols, j;
        for (j = 0; j < cols; j++) mx = Math.max(mx, Math.abs(a[o + j]));
        var sc = Math.max(mx / 127, 1e-12); s[r] = sc;
        for (j = 0; j < cols; j++) q[o + j] = Math.max(-127, Math.min(127, Math.round(a[o + j] / sc)));
      }
      [[name + '.s', s, 'f32', [rows]], [name, q, 'q8', [rows, cols]]].forEach(function (x) {
        tensors.push({ name: x[0], shape: x[3], kind: x[2], offset: off });
        var bytes = new Uint8Array(x[1].buffer, x[1].byteOffset, x[1].byteLength), pad = (4 - bytes.length % 4) % 4;
        blobs.push(bytes); if (pad) blobs.push(new Uint8Array(pad));
        off += bytes.length + pad;
      });
    });
    var header = { model: 'Artemis 2.0 Plasma', arch: 'plasma', cfg: cfg, merges: merges, train: info, tensors: tensors };
    var h = new TextEncoder().encode(JSON.stringify(header)), hpad = (4 - (8 + h.length) % 4) % 4;
    var pre = new Uint8Array(8 + h.length + hpad);
    pre.set([65, 82, 84, 50]); new DataView(pre.buffer).setUint32(4, h.length, true); pre.set(h, 8);
    return new Blob([pre].concat(blobs));
  }

  root.PlasmaTrainer = Trainer;
  root.PlasmaExport = exportArt2;
})(typeof self !== 'undefined' ? self : this);

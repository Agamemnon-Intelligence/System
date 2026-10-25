/* Artemis — modeli arayüzü dondurmadan arka planda çalıştırır.
   Artemis 1.1 tek başına çalışmaz: önce 1.0 tabanı, sonra 1.1 dosyası yüklenip birleştirilir.
   Artemis 2.0 Plasma tek dosyadır (blob: adresinden de yüklenebilir; Plasma eğitici sayfası böyle dener). */
importScripts('artemis-engine.js?v=2.0');

var model = null, run = 0, loads = 0, files = {};

// İndirilen dosyalar saklanır; 1.0 ile 1.1 arasında geçişte taban yeniden indirilmez
async function fetchAll(urls) {
  var todo = urls.filter(function (u) { return !files[u]; }), total = 0, got = 0;
  // no-cache: önbellekteki kopya sunucuyla doğrulanır (değişmediyse yeniden indirilmez)
  var res = await Promise.all(todo.map(function (u) { return fetch(u, /^blob:/.test(u) ? {} : { cache: 'no-cache' }); }));
  res.forEach(function (r, i) {
    if (!r.ok) throw new Error('HTTP ' + r.status + ' (' + todo[i].split('/').pop() + ')');
    total += Number(r.headers.get('content-length')) || 0;
  });
  await Promise.all(res.map(async function (r, i) {
    var reader = r.body.getReader(), parts = [], n = 0;
    for (;;) {
      var c = await reader.read();
      if (c.done) break;
      parts.push(c.value); n += c.value.length; got += c.value.length;
      postMessage({ type: 'progress', loaded: got, total: total });
    }
    var buf = new Uint8Array(n), o = 0;
    parts.forEach(function (p) { buf.set(p, o); o += p.length; });
    files[todo[i]] = buf.buffer;
  }));
}

async function load(base, adapter) {
  var id = ++loads; run++; model = null;
  await fetchAll(adapter ? [base, adapter] : [base]);
  if (id !== loads) return; // bu sırada başka bir model seçildi
  model = Artemis.load(files[base], adapter ? files[adapter] : null);
  postMessage({ type: 'ready', name: model.name });
}

// Birkaç tokende bir olay döngüsüne dön ki "durdur" mesajı işlenebilsin
async function generate(history, options) {
  var id = ++run, it = model.stream(history, options), text = '', t0 = Date.now();
  for (;;) {
    for (var i = 0; i < 4; i++) {
      var r = it.next();
      if (r.done || id !== run) { postMessage({ type: 'done', text: text, stopped: id !== run }); return; }
      text += r.value;
      postMessage({ type: 'token', text: r.value });
    }
    if (Date.now() - t0 > 8) { await new Promise(function (ok) { setTimeout(ok, 0); }); t0 = Date.now(); }
  }
}

onmessage = function (e) {
  var m = e.data;
  if (m.type === 'load') load(m.base, m.adapter).catch(function (err) { postMessage({ type: 'error', message: String(err.message || err) }); });
  else if (m.type === 'stop') run++;
  else if (m.type === 'generate') generate(m.history, m.options).catch(function (err) { postMessage({ type: 'error', message: String(err.message || err) }); });
};

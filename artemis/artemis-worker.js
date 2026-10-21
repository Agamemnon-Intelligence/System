/* Artemis 1.0 — modeli arayüzü dondurmadan arka planda çalıştırır. */
importScripts('artemis-engine.js');

var model = null, run = 0;

async function load(url) {
  var res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  var total = Number(res.headers.get('content-length')) || 0, got = 0, parts = [];
  var reader = res.body.getReader();
  for (;;) {
    var r = await reader.read();
    if (r.done) break;
    parts.push(r.value); got += r.value.length;
    postMessage({ type: 'progress', loaded: got, total: total });
  }
  var buf = new Uint8Array(got), o = 0;
  parts.forEach(function (p) { buf.set(p, o); o += p.length; });
  model = new Artemis.Model(buf.buffer);
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
  if (m.type === 'load') load(m.url).catch(function (err) { postMessage({ type: 'error', message: String(err.message || err) }); });
  else if (m.type === 'stop') run++;
  else if (m.type === 'generate') generate(m.history, m.options).catch(function (err) { postMessage({ type: 'error', message: String(err.message || err) }); });
};

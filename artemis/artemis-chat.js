/* Artemis sohbet arayüzü: seçilen modeli Web Worker'da yükler ve yanıtları akış hâlinde gösterir.
   Artemis 1.1 = Artemis 1.0 tabanı + 1.1 dosyası (iki dosya birlikte yüklenir).
   Artemis 2.0 Plasma (1.0 + 1.1 birleşik, üzerine eğitilmiş) tek dosyadır.
   Plasma Pro, Qwen3.5-0.8B tabanlı, Artemis kimliği eklenmiş modeldir; transformers.js ile ayrı bir worker'da
   (pro-worker.js) çalışır ve büyük olduğu için (~500 MB) yalnızca düğmesine basılınca indirilir. */
(function () {
  var root = document.documentElement;
  var T = {
    en: { ph: 'Ask Artemis something…', loading: 'Downloading {m}… ', preparing: 'Preparing the model…', ready: '{m} is ready. It runs on your device.', thinking: 'Artemis is writing…', stop: 'Stop', send: 'Send', fail: 'The model could not be loaded: ', retry: 'Reload the page to try again.', nogpu: 'Plasma Pro needs WebGPU, which this browser does not offer. Try a recent Chrome, Edge or Safari, or use Plasma 2.0.' },
    tr: { ph: "Artemis'e bir şey sorun…", loading: '{m} indiriliyor… ', preparing: 'Model hazırlanıyor…', ready: '{m} hazır. Cihazınızda çalışıyor.', thinking: 'Artemis yazıyor…', stop: 'Durdur', send: 'Gönder', fail: 'Model yüklenemedi: ', retry: 'Tekrar denemek için sayfayı yenileyin.', nogpu: "Plasma Pro, bu tarayıcıda bulunmayan WebGPU'ya ihtiyaç duyuyor. Güncel bir Chrome, Edge ya da Safari deneyin veya Plasma 2.0'ı kullanın." }
  };
  var GEN = { temperature: 0.6, topK: 40, topP: 0.9, repetitionPenalty: 1.2, maxTokens: 220 };
  var MODELS = {
    'pro': { name: 'Artemis Plasma Pro', worker: 'pro', gen: { temperature: 0.6, topK: 20, topP: 0.9, repetitionPenalty: 1.05, maxTokens: 400 } },
    '2.0': { name: 'Artemis 2.0 Plasma', base: 'artemis/artemis-2.0.bin?v=2.0', adapter: null, gen: GEN },
    '1.1': { name: 'Artemis 1.1', base: 'artemis/artemis-1.0.bin?v=1.1', adapter: 'artemis/artemis-1.1.bin?v=1.1', gen: GEN },
    '1.0': { name: 'Artemis 1.0', base: 'artemis/artemis-1.0.bin?v=1.1', adapter: null, gen: GEN }
  };
  var sel = '1.1';
  function t(k) { return (T[root.lang === 'tr' ? 'tr' : 'en'])[k].replace('{m}', MODELS[sel].name); }

  var log = document.getElementById('log'), empty = document.getElementById('empty');
  var form = document.getElementById('form'), input = document.getElementById('input'), send = document.getElementById('send');
  var status = document.getElementById('status'), bar = document.getElementById('bar'), ver = document.getElementById('ver');
  var chips = document.querySelectorAll('.model[data-model]');
  var history = [], ready = false, busy = false, current = null, statusKey = 'loading', statusExtra = '';

  function setStatus(k, extra) { statusKey = k; statusExtra = extra || ''; status.textContent = (k ? t(k) : '') + statusExtra; }
  function relabel() {
    input.placeholder = t('ph');
    send.textContent = busy ? t('stop') : t('send');
    setStatus(statusKey, statusExtra);
  }
  new MutationObserver(relabel).observe(root, { attributes: true, attributeFilter: ['lang'] });

  // ?v: önbellekte eski worker/motor kalmasın. Plasma Pro kendi worker'ını kullanır, ilk seçildiğinde açılır.
  var workers = { art: new Worker('artemis/artemis-worker.js?v=2.0') }, worker = workers.art;
  function workerOf(k) {
    var w = MODELS[k].worker || 'art';
    if (!workers[w]) workers[w] = new Worker('artemis/pro-worker.js?v=1', { type: 'module' });
    workers[w].onmessage = onmessage;
    return workers[w];
  }
  workers.art.onmessage = onmessage;
  function onmessage(e) {
    if (e.target !== worker) return; // önceki modelin worker'ından gelen geç mesaj
    var m = e.data;
    if (m.type === 'progress') {
      if (m.total) {
        bar.style.width = (100 * m.loaded / m.total).toFixed(1) + '%';
        setStatus('loading', Math.round(100 * m.loaded / m.total) + '%');
        if (m.loaded >= m.total) setStatus('preparing');
      }
    } else if (m.type === 'ready') {
      ready = true; bar.style.width = '100%';
      setTimeout(function () { bar.parentNode.style.visibility = 'hidden'; }, 400);
      chipOf(sel).classList.add('ready');
      setStatus('ready'); update();
    } else if (m.type === 'token') {
      if (current) { current.text += m.text; current.el.textContent = current.text.replace(/^\s+/, ''); scroll(); }
    } else if (m.type === 'done') {
      if (!ready) return; // model değişirken durdurulan yanıt
      finish();
    } else if (m.type === 'error') {
      if (current) finish();
      if (m.code === 'nogpu') { bar.parentNode.style.visibility = 'hidden'; setStatus('nogpu'); }
      else setStatus('fail', m.message + '. ' + t('retry'));
    }
  }
  function chipOf(k) { return document.querySelector('.model[data-model="' + k + '"]'); }
  function load(k) {
    if (busy) { worker.postMessage({ type: 'stop' }); finish(); }
    sel = k; ready = false; worker = workerOf(k); update();
    chips.forEach(function (c) { c.setAttribute('aria-pressed', String(c.dataset.model === k)); c.classList.remove('ready'); });
    ver.textContent = MODELS[k].name;
    bar.style.width = '0'; bar.parentNode.style.visibility = 'visible'; setStatus('loading');
    var abs = function (u) { return u && new URL(u, location.href).href; };
    worker.postMessage({ type: 'load', base: abs(MODELS[k].base), adapter: abs(MODELS[k].adapter) });
    try { localStorage.setItem('artemis-model', k); } catch (e) {}
  }
  chips.forEach(function (c) { c.addEventListener('click', function () { if (c.dataset.model !== sel || !ready) load(c.dataset.model); }); });
  // Sitede bulunan en yeni küçük modelin düğmesini aç ve onunla başla: Plasma 2.0, yoksa 1.1.
  // Plasma Pro büyük olduğu için kendiliğinden inmez; daha önce seçildiyse (dosyalar önbellekte) onunla başlanır.
  function probe(k) {
    return fetch(MODELS[k].base, { method: 'HEAD', cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw 0;
      var c = chipOf(k); c.disabled = false;
      var soon = c.querySelector('.soon'); if (soon) soon.remove();
      return k;
    });
  }
  var last = null;
  try { last = localStorage.getItem('artemis-model'); } catch (e) {}
  probe('2.0').catch(function () { return null; }).then(function (k) { load(last === 'pro' ? 'pro' : k || sel); });

  function scroll() { log.scrollTop = log.scrollHeight; }
  function bubble(role, text) {
    if (empty) { empty.remove(); empty = null; }
    var el = document.createElement('div');
    el.className = 'msg ' + role;
    el.textContent = text;
    log.appendChild(el); scroll();
    return el;
  }
  function update() {
    send.disabled = !ready || (!busy && !input.value.trim());
    send.textContent = busy ? t('stop') : t('send');
  }
  function finish() {
    if (!current) return;
    current.el.classList.remove('typing');
    var text = current.text.trim();
    if (!text) { text = '…'; current.el.textContent = text; }
    history.push({ role: 'bot', text: text });
    current = null; busy = false;
    setStatus('ready'); update(); input.focus();
  }
  function ask(q) {
    q = q.trim();
    if (!q || !ready || busy) return;
    bubble('user', q);
    history.push({ role: 'user', text: q });
    if (history.length > 12) history = history.slice(-12);
    current = { el: bubble('bot typing', ''), text: '' };
    busy = true; input.value = ''; autosize(); update(); setStatus('thinking');
    worker.postMessage({ type: 'generate', history: history, options: MODELS[sel].gen });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (busy) { worker.postMessage({ type: 'stop' }); return; }
    ask(input.value);
  });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!busy) ask(input.value); }
  });
  function autosize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 144) + 'px'; }
  input.addEventListener('input', function () { autosize(); update(); });
  var ideas = document.getElementById('ideas');
  if (ideas) ideas.addEventListener('click', function (e) { if (e.target.tagName === 'BUTTON') ask(e.target.textContent); });
  relabel();
})();

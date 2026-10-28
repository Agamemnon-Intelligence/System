/* Blog behaviour: language toggle, nav, filters, table of contents, reading progress. */
(function () {
  var root = document.documentElement;
  var lang = root.lang === 'tr' ? 'tr' : 'en';
  var LABEL = { en: "Türkçe'ye geç", tr: 'Switch to English' };

  // ---------- Language (both languages are in the page; CSS hides one) ----------
  var btn = document.getElementById('lang');
  var titles = { en: document.title, tr: root.dataset.titleTr || document.title };
  function apply(l) {
    lang = l; root.lang = l;
    document.title = titles[l];
    btn.setAttribute('aria-label', LABEL[l]);
    btn.querySelector('span').textContent = l.toUpperCase();
    buildToc();
  }
  btn.addEventListener('click', function () {
    var next = lang === 'tr' ? 'en' : 'tr';
    apply(next);
    try { localStorage.setItem('athene-lang', next); } catch (e) {}
  });

  // ---------- Nav ----------
  var nav = document.getElementById('nav'), last = 0;
  var bar = document.querySelector('.progress');
  var article = document.querySelector('.prose-wrap');
  addEventListener('scroll', function () {
    var y = scrollY;
    nav.classList.toggle('scrolled', y > 8);
    nav.classList.toggle('hidden', y > last && y > 400);
    last = y;
    if (bar && article) {
      var r = article.getBoundingClientRect();
      var p = Math.min(1, Math.max(0, -r.top / Math.max(1, r.height - innerHeight)));
      bar.style.transform = 'scaleX(' + p + ')';
    }
  }, { passive: true });

  // ---------- Filters (index) ----------
  var filters = document.querySelectorAll('.filters button');
  filters.forEach(function (b) {
    b.addEventListener('click', function () {
      var cat = b.dataset.cat;
      filters.forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      document.querySelectorAll('[data-post-cat]').forEach(function (c) {
        c.hidden = cat !== 'all' && c.dataset.postCat !== cat;
      });
    });
  });

  // ---------- Table of contents (post) ----------
  var toc = document.querySelector('.toc ol');
  var heads = [], links = [];
  function buildToc() {
    if (!toc) return;
    toc.innerHTML = '';
    heads = [].slice.call(document.querySelectorAll('.prose[data-l="' + lang + '"] h2'));
    links = heads.map(function (h, i) {
      if (!h.id) h.id = lang + '-s' + (i + 1);
      var li = document.createElement('li'), a = document.createElement('a');
      a.href = '#' + h.id; a.textContent = h.textContent;
      li.appendChild(a); toc.appendChild(li);
      return a;
    });
    spy();
  }
  // Highlight the last heading that has scrolled past the top third.
  function spy() {
    var cur = -1;
    heads.forEach(function (h, i) { if (h.getBoundingClientRect().top < innerHeight * 0.33) cur = i; });
    links.forEach(function (a, i) { a.classList.toggle('on', i === cur); });
  }
  if (toc) addEventListener('scroll', spy, { passive: true });
  apply(lang);

  // ---------- Reveal ----------
  var els = [].slice.call(document.querySelectorAll('.reveal'));
  var groups = new Map();
  els.forEach(function (el) {
    var p = el.parentElement, i = groups.get(p) || 0;
    el.style.setProperty('--d', (Math.min(i, 6) * 0.08).toFixed(2) + 's');
    groups.set(p, i + 1);
  });
  if (matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) {
    els.forEach(function (el) { el.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.05 });
    els.forEach(function (el) { io.observe(el); });
  }
})();

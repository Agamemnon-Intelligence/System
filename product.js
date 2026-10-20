/* Shared behaviour for the product detail pages. Each page defines a TR dictionary first. */
(function () {
  var root = document.documentElement;
  var lang = root.lang === 'tr' ? 'tr' : 'en';
  var TR = window.TR || {};
  var LABEL = { en: "Türkçe'ye geç", tr: 'Switch to English' };

  // ---------- Language ----------
  var nodes = document.querySelectorAll('[data-i18n]');
  nodes.forEach(function (el) { el.dataset.en = el.innerHTML; });
  var btn = document.getElementById('lang');
  var title = document.title;

  function apply(l) {
    lang = l; root.lang = l;
    nodes.forEach(function (el) {
      var k = el.dataset.i18n;
      el.innerHTML = l === 'tr' && TR[k] != null ? TR[k] : el.dataset.en;
    });
    document.title = l === 'tr' && TR['doc.title'] ? TR['doc.title'] : title;
    btn.setAttribute('aria-label', LABEL[l]);
  }
  btn.addEventListener('click', function () {
    var next = lang === 'tr' ? 'en' : 'tr';
    apply(next);
    try { localStorage.setItem('athene-lang', next); } catch (e) {}
  });
  if (lang === 'tr') apply('tr');

  // ---------- OS-matched download ----------
  var ua = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || navigator.userAgent;
  var os = /mac/i.test(ua) ? 'mac' : /win/i.test(ua) ? 'win' : null;
  if (os) { var card = document.querySelector('.dl[data-os="' + os + '"]'); if (card) card.classList.add('yours'); }

  // ---------- Nav ----------
  var nav = document.getElementById('nav'), last = 0;
  addEventListener('scroll', function () {
    var y = scrollY;
    nav.classList.toggle('scrolled', y > 8);
    nav.classList.toggle('hidden', y > last && y > 400);
    last = y;
  }, { passive: true });

  // ---------- Honey reveal ----------
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var els = [].slice.call(document.querySelectorAll('.reveal'));
  var groups = new Map();
  els.forEach(function (el) {
    var p = el.parentElement, i = groups.get(p) || 0;
    el.style.setProperty('--d', (Math.min(i, 6) * 0.1).toFixed(2) + 's');
    groups.set(p, i + 1);
  });
  if (reduce || !('IntersectionObserver' in window)) {
    els.forEach(function (el) { el.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    els.forEach(function (el) { io.observe(el); });
  }
})();

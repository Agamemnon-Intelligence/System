#!/usr/bin/env python3
"""Build the blog from blog/_posts/*.html.

Each source file starts with a JSON header in an HTML comment, followed by
<section lang="en"> and <section lang="tr"> holding the post body:

    <!--
    {"title": "...", "title_tr": "...", "summary": "...", "summary_tr": "...",
     "date": "2026-10-27", "cat": "security" | "ai"}
    -->
    <section lang="en"> ... </section>
    <section lang="tr"> ... </section>

Writes blog/<slug>/index.html, blog/index.html, blog/feed.xml, sitemap.xml and
the "Latest from the blog" block in index.html. Run: python3 tools/build_blog.py
"""
import datetime as dt
import hashlib
import html
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "blog" / "_posts"
OUT = ROOT / "blog"
SITE = "https://athenesoftworks.com"

CATS = {
    "security": ("Security", "Güvenlik"),
    "ai": ("AI", "Yapay zekâ"),
}
MONTHS_EN = "January February March April May June July August September October November December".split()
MONTHS_TR = "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık".split()

FLAGS = """<svg class="flag-gb" viewBox="0 0 60 40" aria-hidden="true">
        <clipPath id="gb-c"><rect width="60" height="40"/></clipPath>
        <clipPath id="gb-t"><path d="M30 20h30v20zv20H0zH0V0zV0h30z"/></clipPath>
        <g clip-path="url(#gb-c)">
          <rect width="60" height="40" fill="#012169"/>
          <path d="M0 0l60 40m0-40L0 40" stroke="#fff" stroke-width="8"/>
          <path d="M0 0l60 40m0-40L0 40" clip-path="url(#gb-t)" stroke="#C8102E" stroke-width="5"/>
          <path d="M30 0v40M0 20h60" stroke="#fff" stroke-width="12"/>
          <path d="M30 0v40M0 20h60" stroke="#C8102E" stroke-width="7"/>
        </g>
      </svg>
      <svg class="flag-tr" viewBox="0 0 1200 800" aria-hidden="true">
        <rect width="1200" height="800" fill="#E30A17"/>
        <circle cx="300" cy="400" r="200" fill="#fff"/>
        <circle cx="350" cy="400" r="160" fill="#E30A17"/>
        <polygon fill="#fff" points="400,400 469.1,377.5 469.1,304.9 511.8,363.7 580.9,341.2 538.2,400 580.9,458.8 511.8,436.3 469.1,495.1 469.1,422.5"/>
      </svg>"""

ARROW = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>'
BACK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6"/></svg>'


def bi(en, tr, tag="span"):
    """Both languages side by side; blog.css shows the active one."""
    return f'<{tag} data-l="en">{en}</{tag}><{tag} data-l="tr">{tr}</{tag}>'


def esc(s):
    return html.escape(s, quote=True)


def fmt_date(d, lang):
    if lang == "tr":
        return f"{d.day} {MONTHS_TR[d.month - 1]} {d.year}"
    return f"{MONTHS_EN[d.month - 1]} {d.day}, {d.year}"


def words(s):
    return len(re.sub(r"<[^>]+>", " ", s).split())


def load_posts():
    posts = []
    for path in sorted(SRC.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"\s*<!--(.*?)-->", text, re.S)
        meta = json.loads(m.group(1))
        body = {}
        for lang in ("en", "tr"):
            s = re.search(rf'<section lang="{lang}">(.*?)</section>', text, re.S)
            body[lang] = s.group(1).strip()
        meta["slug"] = path.stem
        meta["date"] = dt.date.fromisoformat(meta["date"])
        meta["body"] = body
        meta["mins"] = {l: max(1, round(words(body[l]) / 220)) for l in body}
        posts.append(meta)
    posts.sort(key=lambda p: (p["date"], p["slug"]), reverse=True)
    return posts


def cover(slug, cat):
    """Abstract cover art, seeded by the slug so it never changes between builds."""
    rnd = random.Random(int(hashlib.sha1(slug.encode()).hexdigest()[:8], 16))
    w, h = 800, 450
    uid = slug[:12].replace("-", "")
    parts = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">',
             "<defs>",
             f'<radialGradient id="g{uid}" cx="{rnd.uniform(.2, .8):.2f}" cy="{rnd.uniform(.2, .6):.2f}" r=".8">',
             f'<stop offset="0" stop-color="{"#2545a8" if cat == "security" else "#0e5a8a"}" stop-opacity=".75"/>',
             '<stop offset="1" stop-color="#0a1124" stop-opacity="0"/></radialGradient>',
             f'<pattern id="p{uid}" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#fff" fill-opacity=".09"/></pattern>',
             "</defs>",
             f'<rect width="{w}" height="{h}" fill="#0a1124"/>',
             f'<rect width="{w}" height="{h}" fill="url(#g{uid})"/>',
             f'<rect width="{w}" height="{h}" fill="url(#p{uid})"/>']
    if cat == "security":
        # Hexagon field with a few highlighted cells, like a scan in progress.
        r = 34
        dx, dy = r * math.sqrt(3), r * 1.5
        for row in range(-1, int(h / dy) + 2):
            for col in range(-1, int(w / dx) + 2):
                cx = col * dx + (dx / 2 if row % 2 else 0)
                cy = row * dy
                pts = " ".join(f"{cx + r * math.cos(math.radians(60 * k + 30)):.1f},{cy + r * math.sin(math.radians(60 * k + 30)):.1f}" for k in range(6))
                roll = rnd.random()
                if roll < .05:
                    parts.append(f'<polygon points="{pts}" fill="#e8b04b" fill-opacity=".22" stroke="#f3c97a" stroke-opacity=".8" stroke-width="1.5"/>')
                elif roll < .09:
                    parts.append(f'<polygon points="{pts}" fill="#38bdf8" fill-opacity=".1" stroke="#7dd3fc" stroke-opacity=".55" stroke-width="1.2"/>')
                else:
                    parts.append(f'<polygon points="{pts}" fill="none" stroke="#4c74e8" stroke-opacity="{rnd.uniform(.12, .3):.2f}" stroke-width="1"/>')
        y = rnd.uniform(.3, .7) * h
        parts.append(f'<rect x="0" y="{y:.0f}" width="{w}" height="2" fill="#f3c97a" fill-opacity=".55"/>')
        parts.append(f'<rect x="0" y="{y - 60:.0f}" width="{w}" height="60" fill="#f3c97a" fill-opacity=".05"/>')
    else:
        # Network of nodes in layers, like a small neural net.
        layers = rnd.randint(4, 6)
        nodes = []
        for i in range(layers):
            n = rnd.randint(3, 6)
            x = 90 + i * (w - 180) / (layers - 1)
            nodes.append([(x, h / 2 + (j - (n - 1) / 2) * rnd.uniform(52, 66)) for j in range(n)])
        for a, b in zip(nodes, nodes[1:]):
            for p in a:
                for q in b:
                    if rnd.random() < .7:
                        hot = rnd.random() < .08
                        parts.append(f'<line x1="{p[0]:.0f}" y1="{p[1]:.0f}" x2="{q[0]:.0f}" y2="{q[1]:.0f}" stroke="{"#f3c97a" if hot else "#38bdf8"}" stroke-opacity="{.7 if hot else rnd.uniform(.08, .25):.2f}" stroke-width="{1.6 if hot else 1}"/>')
        for layer in nodes:
            for x, y in layer:
                lit = rnd.random() < .25
                parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{9 if lit else 6}" fill="{"#38bdf8" if lit else "#0e1528"}" stroke="#7dd3fc" stroke-opacity=".8" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def head(title, title_tr, desc, url, depth, extra=""):
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en" data-title-tr="{esc(title_tr)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta name="theme-color" content="#0b1020">
<link rel="canonical" href="{url}">
<meta property="og:site_name" content="Athene Softworks">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{url}">
{extra}<link rel="alternate" type="application/rss+xml" title="Athene Softworks Blog" href="{SITE}/blog/feed.xml">
<link rel="icon" href="https://raw.githubusercontent.com/Agamemnon-Intelligence/Agamemnon-macOS/main/docs/icon.png">
<script>
(function () {{ var d = document.documentElement; d.classList.add('js'); try {{ if (localStorage.getItem('athene-lang') === 'tr') d.lang = 'tr'; }} catch (e) {{}} }})();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@300..900&family=Red+Hat+Text:wght@300..700&display=swap">
<link rel="stylesheet" href="{up}blog/blog.css">
</head>
<body>
"""


def nav(depth):
    up = "../" * depth
    return f"""<nav class="nav" id="nav" aria-label="Main">
  <a class="wordmark" href="{up}index.html">Athene Softworks</a>
  <div class="nav-right">
    <ul>
      <li><a href="{up}index.html">{bi("Home", "Ana sayfa")}</a></li>
      <li><a href="{up}agamemnon.html">Agamemnon</a></li>
      <li><a href="{up}synapse/">Synapse</a></li>
      <li><a href="{up}blog/" aria-current="page">Blog</a></li>
      <li><a href="{up}index.html#contact">{bi("Contact", "İletişim")}</a></li>
    </ul>
    <button class="lang" id="lang" type="button" aria-label="Türkçe'ye geç">
      {FLAGS}
      <span>EN</span>
    </button>
  </div>
</nav>
"""


def footer(depth):
    up = "../" * depth
    return f"""<footer>
  <div class="wrap">
    <div class="foot-top">
      <a class="wordmark" href="{up}index.html">Athene Softworks</a>
      <ul>
        <li><a href="{up}index.html">{bi("Home", "Ana sayfa")}</a></li>
        <li><a href="{up}agamemnon.html">Agamemnon</a></li>
        <li><a href="{up}synapse/">Synapse</a></li>
        <li><a href="{up}blog/">Blog</a></li>
        <li><a href="{up}blog/feed.xml">RSS</a></li>
      </ul>
    </div>
    <p>{bi("Copyright © 2026 Athene Softworks. All rights reserved.", "Telif hakkı © 2026 Athene Softworks. Tüm hakları saklıdır.")}</p>
    <p>{bi("Posts are for general information. They are not legal or professional advice, and details change fast in security and AI.", "Yazılar genel bilgi amaçlıdır; hukuki ya da profesyonel tavsiye değildir. Güvenlik ve yapay zekâda ayrıntılar hızla değişir.")}</p>
  </div>
</footer>
<script src="{up}blog/blog.js"></script>
</body>
</html>
"""


def meta_line(p):
    en = f'<time datetime="{p["date"].isoformat()}">{fmt_date(p["date"], "en")}</time><i></i>{p["mins"]["en"]} min read'
    tr = f'<time datetime="{p["date"].isoformat()}">{fmt_date(p["date"], "tr")}</time><i></i>{p["mins"]["tr"]} dk okuma'
    return bi(en, tr)


def tag(p):
    en, tr = CATS[p["cat"]]
    return f'<span class="tag" data-cat="{p["cat"]}">{bi(en, tr)}</span>'


def card(p, up, cls="card reveal"):
    return f"""<a class="{cls}" href="{up}{p["slug"]}/" data-post-cat="{p["cat"]}">
        <div class="cover">{cover(p["slug"], p["cat"])}</div>
        <div class="body">
          <div class="meta">{tag(p)}</div>
          <h3>{bi(esc(p["title"]), esc(p["title_tr"]))}</h3>
          <p>{bi(esc(p["summary"]), esc(p["summary_tr"]))}</p>
          <div class="meta">{meta_line(p)}</div>
        </div>
      </a>"""


def build_post(p, posts):
    url = f"{SITE}/blog/{p['slug']}/"
    ld = {
        "@context": "https://schema.org", "@type": "BlogPosting",
        "headline": p["title"], "description": p["summary"],
        "datePublished": p["date"].isoformat(), "url": url, "inLanguage": ["en", "tr"],
        "author": {"@type": "Organization", "name": "Athene Softworks", "url": SITE},
        "publisher": {"@type": "Organization", "name": "Athene Softworks", "url": SITE},
    }
    extra = (f'<meta property="og:type" content="article">\n'
             f'<meta property="article:published_time" content="{p["date"].isoformat()}">\n'
             f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>\n')
    # Two others: same category first, then the newest of the rest.
    others = [o for o in posts if o is not p]
    others.sort(key=lambda o: (o["cat"] != p["cat"], -o["date"].toordinal()))
    if p["cat"] == "ai":
        cta = (f'<img src="../../synapse/assets/icon-256.png" alt="" width="52" height="52">'
               f'<div><b>Synapse</b><span>{bi("Run open AI models privately on your Mac or PC, or use free cloud models. Free and open source.", "Açık yapay zekâ modellerini Mac ya da PC’nizde gizlice çalıştırın veya ücretsiz bulut modellerini kullanın. Ücretsiz ve açık kaynaklı.")}</span></div>'
               f'<a class="pill solid" href="../../synapse/">{bi("Get Synapse", "Synapse’i edinin")}</a>')
    else:
        cta = (f'<img src="https://raw.githubusercontent.com/Agamemnon-Intelligence/Agamemnon-macOS/main/docs/icon.png" alt="" width="52" height="52">'
               f'<div><b>Agamemnon</b><span>{bi("Free, open-source antivirus for macOS and Windows. Stops stealers in the act.", "macOS ve Windows için ücretsiz, açık kaynaklı antivirüs. Veri hırsızlarını iş üstünde durdurur.")}</span></div>'
               f'<a class="pill solid" href="../../index.html#agamemnon">{bi("Get Agamemnon", "Agamemnon’u edinin")}</a>')

    page = head(f'{p["title"]} · Athene Softworks', f'{p["title_tr"]} · Athene Softworks', p["summary"], url, 2, extra)
    page += '<div class="progress" aria-hidden="true"></div>\n' + nav(2)
    page += f"""
<main>
  <header class="p-head">
    <div class="wrap">
      <a class="back" href="../">{BACK}{bi("All posts", "Tüm yazılar")}</a>
      <div class="meta">{tag(p)}</div>
      <h1>{bi(esc(p["title"]), esc(p["title_tr"]))}</h1>
      <p class="dek">{bi(esc(p["summary"]), esc(p["summary_tr"]))}</p>
      <div class="meta">{bi("By Athene Softworks", "Athene Softworks")}<i></i>{meta_line(p)}</div>
    </div>
  </header>
  <div class="p-cover"><div class="cover">{cover(p["slug"], p["cat"])}</div></div>

  <div class="p-layout">
    <div class="prose-wrap">
      <article class="prose" data-l="en" lang="en">
{p["body"]["en"]}
      </article>
      <article class="prose" data-l="tr" lang="tr">
{p["body"]["tr"]}
      </article>
    </div>
    <nav class="toc" aria-label="On this page"><b>{bi("On this page", "Bu sayfada")}</b><ol></ol></nav>
  </div>

  <section class="p-end">
    <div class="cta-box reveal">{cta}</div>
    <div class="more-head"><h2>{bi("Keep reading", "Okumaya devam edin")}</h2><a href="../">{bi("All posts →", "Tüm yazılar →")}</a></div>
    <div class="grid">
      {card(others[0], "../")}
      {card(others[1], "../")}
    </div>
  </section>
</main>
"""
    page += footer(2)
    d = OUT / p["slug"]
    d.mkdir(exist_ok=True)
    (d / "index.html").write_text(page, encoding="utf-8")


def build_index(posts):
    url = f"{SITE}/blog/"
    first, rest = posts[0], posts[1:]
    page = head("Blog · Athene Softworks", "Blog · Athene Softworks",
                "Plain-language writing on malware, staying safe online and running AI on your own terms, from the team behind Agamemnon and Synapse.",
                url, 1, '<meta property="og:type" content="website">\n')
    page += nav(1)
    filters = "".join(
        f'<button type="button" data-cat="{k}" aria-pressed="false">{bi(en, tr)}</button>' for k, (en, tr) in CATS.items())
    page += f"""
<main>
  <header class="b-hero">
    <div class="wrap">
      <div class="eyebrow reveal">{bi("The Athene Softworks blog", "Athene Softworks blogu")}</div>
      <h1 class="reveal">{bi("Notes on <span>security</span> and AI.", "<span>Güvenlik</span> ve yapay zekâ üzerine notlar.")}</h1>
      <p class="reveal">{bi("Plain-language writing on how malware works, how to stay safe online, and how to run AI on your own terms. From the team behind Agamemnon and Synapse.", "Zararlı yazılımların nasıl çalıştığı, internette nasıl güvende kalınacağı ve yapay zekânın kendi koşullarınızda nasıl çalıştırılacağı üzerine sade yazılar. Agamemnon ve Synapse’in ekibinden.")}</p>
      <div class="b-tools reveal">
        <div class="filters" role="group" aria-label="Filter"><button type="button" data-cat="all" aria-pressed="true">{bi("All posts", "Tüm yazılar")}</button>{filters}</div>
        <a class="rss" href="feed.xml"><svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M5 17a2 2 0 1 1 0 4 2 2 0 0 1 0-4ZM3 10.5a10.5 10.5 0 0 1 10.5 10.5h-3A7.5 7.5 0 0 0 3 13.5Zm0-6A16.5 16.5 0 0 1 19.5 21h-3A13.5 13.5 0 0 0 3 7.5Z"/></svg>RSS</a>
      </div>
    </div>
  </header>

  <div class="wrap">
    <a class="featured reveal" href="{first["slug"]}/" data-post-cat="{first["cat"]}">
      <div class="cover">{cover(first["slug"], first["cat"])}</div>
      <div>
        <div class="meta">{tag(first)}<span class="tag">{bi("Latest", "En yeni")}</span></div>
        <h2>{bi(esc(first["title"]), esc(first["title_tr"]))}</h2>
        <p>{bi(esc(first["summary"]), esc(first["summary_tr"]))}</p>
        <div class="meta">{meta_line(first)}</div>
        <span class="go">{bi("Read the post", "Yazıyı okuyun")}{ARROW}</span>
      </div>
    </a>
    <div class="grid">
      {"".join(card(p, "") for p in rest)}
    </div>
  </div>
</main>
"""
    page += footer(1)
    (OUT / "index.html").write_text(page, encoding="utf-8")


def build_feed(posts):
    items = []
    for p in posts:
        url = f"{SITE}/blog/{p['slug']}/"
        pub = dt.datetime.combine(p["date"], dt.time(9, 0)).strftime("%a, %d %b %Y %H:%M:%S +0000")
        items.append(f"""  <item>
    <title>{esc(p["title"])}</title>
    <link>{url}</link>
    <guid>{url}</guid>
    <pubDate>{pub}</pubDate>
    <category>{CATS[p["cat"]][0]}</category>
    <description>{esc(p["summary"])}</description>
  </item>""")
    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Athene Softworks Blog</title>
  <link>{SITE}/blog/</link>
  <description>Notes on security and AI from the team behind Agamemnon and Synapse.</description>
  <language>en</language>
{chr(10).join(items)}
</channel>
</rss>
"""
    (OUT / "feed.xml").write_text(feed, encoding="utf-8")


def build_sitemap(posts):
    urls = ["/", "/agamemnon.html", "/synapse/", "/blog/"] + [f"/blog/{p['slug']}/" for p in posts]
    body = "\n".join(f"  <url><loc>{SITE}{u}</loc></url>" for u in urls)
    (ROOT / "sitemap.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{body}\n</urlset>\n',
        encoding="utf-8")


def build_home_block(posts):
    """Replace the block between the blog:latest markers in index.html."""
    home = ROOT / "index.html"
    text = home.read_text(encoding="utf-8")
    cards = []
    for p in posts[:3]:
        en, tr = CATS[p["cat"]]
        cards.append(f"""        <a class="post reveal" href="blog/{p["slug"]}/">
          <div class="post-cover">{cover(p["slug"], p["cat"])}</div>
          <span class="post-tag" data-cat="{p["cat"]}"><span data-l="en">{en}</span><span data-l="tr">{tr}</span></span>
          <h3><span data-l="en">{esc(p["title"])}</span><span data-l="tr">{esc(p["title_tr"])}</span></h3>
          <p><span data-l="en">{esc(p["summary"])}</span><span data-l="tr">{esc(p["summary_tr"])}</span></p>
          <time datetime="{p["date"].isoformat()}"><span data-l="en">{fmt_date(p["date"], "en")}</span><span data-l="tr">{fmt_date(p["date"], "tr")}</span></time>
        </a>""")
    p = posts[0]
    tile = f"""<a class="tile blog-tile reveal" href="blog/{p["slug"]}/">
          <span class="tile-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 5h11l5 5v9H4zM15 5v5h5M8 13h8M8 16h5"/></svg></span>
          <div><em><span data-l="en">New on the blog</span><span data-l="tr">Blogda yeni</span></em><b><span data-l="en">{esc(p["title"])}</span><span data-l="tr">{esc(p["title_tr"])}</span></b></div>
          <svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>
        </a>"""
    blocks = {
        "latest": "\n".join(cards),
        "hero": tile,
    }
    for name, inner in blocks.items():
        block = f"<!-- blog:{name}:start -->\n{inner}\n        <!-- blog:{name}:end -->"
        text, n = re.subn(rf"<!-- blog:{name}:start -->.*?<!-- blog:{name}:end -->", lambda _: block, text, flags=re.S)
        if n != 1:
            raise SystemExit(f"index.html: blog:{name} markers not found")
    home.write_text(text, encoding="utf-8")


def main():
    posts = load_posts()
    for p in posts:
        build_post(p, posts)
    build_index(posts)
    build_feed(posts)
    build_sitemap(posts)
    build_home_block(posts)
    print(f"Built {len(posts)} posts.")


if __name__ == "__main__":
    main()

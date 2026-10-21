// artemis.html'den çevrimdışı masaüstü sayfası üretir ve Windows için tek dosya .exe derler.
//   bun desktop/build.js [hedef]   (varsayılan bun-windows-x64)
import { $ } from 'bun';
import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
let html = readFileSync(root + 'artemis.html', 'utf8');
// Site menüsü ve alt bilgi masaüstünde gereksiz; dil düğmesi kalsın
html = html.replace(/<ul>[\s\S]*?<\/ul>\s*(<button class="lang")/, '$1');
html = html.replace(/<footer>[\s\S]*?<\/footer>\n/, '');
html = html.replace(/<p class="crumb">[\s\S]*?<\/p>\n/, '');
html = html.replace('href="index.html">Athene Softworks', 'href="/">Athene Softworks');
html = html.replace('<link rel="icon" href="https://raw.githubusercontent.com/Agamemnon-Intelligence/Agamemnon-macOS/main/docs/icon.png">', '');
writeFileSync(root + 'desktop/app.html', html);

const target = process.argv[2] || 'bun-windows-x64';
const out = target.includes('windows') ? 'dist/Artemis-1.0.exe' : 'dist/artemis-1.0';
await $`bun build ${root}desktop/main.js --compile --target=${target} --outfile ${root}${out}`;
console.log('hazır:', out);

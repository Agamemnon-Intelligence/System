// Artemis 1.0 masaüstü — modeli ve arayüzü içine gömülü yerel bir sunucu başlatır, tarayıcıda açar.
// Derleme: bun desktop/build.js
import appHtml from './app.html' with { type: 'file' };
import productCss from '../product.css' with { type: 'file' };
import productJs from '../product.js' with { type: 'file' };
import chatJs from '../artemis/artemis-chat.js' with { type: 'file' };
import engineJs from '../artemis/artemis-engine.js' with { type: 'file' };
import workerJs from '../artemis/artemis-worker.js' with { type: 'file' };
import modelBin from '../artemis/artemis-1.0.bin' with { type: 'file' };

const FILES = {
  '/': [appHtml, 'text/html; charset=utf-8'],
  '/product.css': [productCss, 'text/css; charset=utf-8'],
  '/product.js': [productJs, 'text/javascript; charset=utf-8'],
  '/artemis/artemis-chat.js': [chatJs, 'text/javascript; charset=utf-8'],
  '/artemis/artemis-engine.js': [engineJs, 'text/javascript; charset=utf-8'],
  '/artemis/artemis-worker.js': [workerJs, 'text/javascript; charset=utf-8'],
  '/artemis/artemis-1.0.bin': [modelBin, 'application/octet-stream'],
};

const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  fetch(req) {
    const f = FILES[new URL(req.url).pathname];
    if (!f) return new Response('Not found', { status: 404 });
    return new Response(Bun.file(f[0]), { headers: { 'content-type': f[1], 'cache-control': 'no-store' } });
  },
});

const url = `http://127.0.0.1:${server.port}/`;
console.log('Artemis 1.0 · Athene Softworks');
console.log(`Çalışıyor / Running: ${url}`);
console.log('Kapatmak için bu pencereyi kapatın. / Close this window to quit.');
const open = process.platform === 'win32' ? ['cmd', '/c', 'start', '', url] : process.platform === 'darwin' ? ['open', url] : ['xdg-open', url];
try { Bun.spawn(open, { stdout: 'ignore', stderr: 'ignore' }); } catch {}

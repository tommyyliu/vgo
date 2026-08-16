// Static server for the demo and the WebGPU benchmark.
//
// Sets cross-origin isolation headers so `SharedArrayBuffer` -- and with it
// multi-threaded WASM inference -- is available. That is the difference between
// ~95 ms and ~25 ms per position on the CPU fallback, and a host that cannot
// set these is stuck with the slower tier.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const ROOT = new URL('.', import.meta.url).pathname;
const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.json': 'application/json', '.map': 'application/json',
};

createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  // The benchmark posts its results back so a headless run can be captured
  // without scraping the DOM.
  if (request.method === 'POST' && url.pathname === '/result') {
    let body = '';
    for await (const chunk of request) body += chunk;
    console.log('RESULT ' + body);
    response.writeHead(204, { 'Cross-Origin-Opener-Policy': 'same-origin' }).end();
    return;
  }
  const path = join(ROOT, normalize(url.pathname === '/' ? '/public/index.html' : url.pathname));
  try {
    const body = await readFile(path);
    console.log(`GET ${url.pathname} ${body.length}`);
    response.writeHead(200, {
      'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    response.end(body);
  } catch {
    console.log(`404 ${url.pathname}`);
    response.writeHead(404).end('not found');
  }
}).listen(8123, () => console.log('http://localhost:8123/public/bench.html'));

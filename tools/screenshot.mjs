// Capture docs/screenshot.png from the real demo page.
//
// Headless Chrome's --screenshot flag needs --virtual-time-budget to wait for
// the page, and virtual time freezes the clock during computation, so the
// page's own "compute time" readout comes out as 0.0 ms. That is a measuring
// artefact, not a result, and committing a screenshot that shows it would be
// misleading. So this drives Chrome over the DevTools Protocol instead: it
// waits for the analysis to actually finish, under a real clock, then
// captures the full page.
//
// No dependencies. Node 24 has both fetch and WebSocket built in.
//
//   python3 -m http.server 8731 --directory docs &
//   node tools/screenshot.mjs --url http://127.0.0.1:8731/index.html \
//       --out docs/screenshot.png

import { spawn, spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9333;

function parseArgs(argv) {
  const out = {
    url: 'http://127.0.0.1:8731/index.html',
    out: 'docs/screenshot.png',
    width: 1340,
    scale: 2,
    theme: 'dark',
    timeout: 90000,
    clipY: 0,
    clipHeight: 0,   // 0 means the full page
    maxWidth: 0,     // 0 means no downscale
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--url') { out.url = argv[i += 1]; continue; }
    if (arg === '--out') { out.out = argv[i += 1]; continue; }
    if (arg === '--width') { out.width = Number(argv[i += 1]); continue; }
    if (arg === '--scale') { out.scale = Number(argv[i += 1]); continue; }
    if (arg === '--theme') { out.theme = argv[i += 1]; continue; }
    if (arg === '--timeout') { out.timeout = Number(argv[i += 1]); continue; }
    if (arg === '--clip-y') { out.clipY = Number(argv[i += 1]); continue; }
    if (arg === '--clip-height') { out.clipHeight = Number(argv[i += 1]); continue; }
    if (arg === '--max-width') { out.maxWidth = Number(argv[i += 1]); continue; }
    throw new Error(`unknown option: ${arg}`);
  }
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForEndpoint(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (response.ok) return await response.json();
    } catch {
      // Chrome is not listening yet.
    }
    await sleep(150);
  }
  throw new Error('Chrome DevTools endpoint never came up');
}

// Minimal CDP client: send a command, resolve on the matching id.
class Session {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve: ok, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else ok(message.result);
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((ok, reject) => {
      this.pending.set(id, { resolve: ok, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`timed out: ${method}`));
        }
      }, 60000);
    });
  }

  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(
        `page threw: ${result.exceptionDetails.exception?.description
          || result.exceptionDetails.text}`,
      );
    }
    return result.result.value;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outPath = resolve(args.out);

  const chrome = spawn(CHROME, [
    '--headless',
    `--remote-debugging-port=${PORT}`,
    '--remote-allow-origins=*',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--hide-scrollbars',
    // Software WebGL, so Mol* renders without a display attached.
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    `--force-device-scale-factor=${args.scale}`,
    `--window-size=${args.width},1000`,
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });

  let ws;
  try {
    await waitForEndpoint(30000);

    const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page');
    if (!page) throw new Error('no page target');

    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((ok, reject) => {
      ws.addEventListener('open', ok, { once: true });
      ws.addEventListener('error', () => reject(new Error('CDP socket failed')), { once: true });
    });

    const session = new Session(ws);
    await session.send('Page.enable');
    await session.send('Runtime.enable');
    await session.send('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-color-scheme', value: args.theme }],
    });
    await session.send('Emulation.setDeviceMetricsOverride', {
      width: args.width,
      height: 1000,
      deviceScaleFactor: args.scale,
      mobile: false,
    });

    console.log(`navigating to ${args.url}`);
    await session.send('Page.navigate', { url: args.url });

    // Wait for the page's own analysis to finish. window.pdbpeek.ready is set
    // by app.js once a structure has been parsed and measured.
    const deadline = Date.now() + args.timeout;
    let ready = false;
    while (Date.now() < deadline) {
      try {
        ready = await session.evaluate(
          'Boolean(window.pdbpeek && window.pdbpeek.ready)',
        );
      } catch {
        ready = false;
      }
      if (ready) break;
      await sleep(400);
    }
    if (!ready) throw new Error('the page never finished its analysis');

    const status = await session.evaluate(
      "document.getElementById('status').textContent",
    );
    const computeTime = await session.evaluate(
      "document.getElementById('stat-time').textContent",
    );
    console.log(`page status: ${status}`);
    console.log(`page compute time: ${computeTime}`);

    // Mol* draws asynchronously after its own fetch, so wait for a canvas
    // with actual pixels in it rather than guessing at a delay.
    const molstarDeadline = Date.now() + 60000;
    let painted = false;
    while (Date.now() < molstarDeadline) {
      painted = await session.evaluate(`(() => {
        const canvas = document.querySelector('pdbe-molstar canvas');
        if (!canvas || !canvas.width) return false;
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) return false;
        const w = canvas.width, h = canvas.height;
        const px = new Uint8Array(4 * 400);
        // Sample a 20x20 block in the middle of the canvas and look for any
        // variation, which means geometry has been drawn.
        gl.readPixels((w >> 1) - 10, (h >> 1) - 10, 20, 20,
          gl.RGBA, gl.UNSIGNED_BYTE, px);
        const first = px.slice(0, 3).join(',');
        for (let i = 4; i < px.length; i += 4) {
          if (px.slice(i, i + 3).join(',') !== first) return true;
        }
        return false;
      })()`);
      if (painted) break;
      await sleep(600);
    }
    console.log(painted ? 'Mol* has rendered geometry' : 'Mol* did not paint in time');

    await sleep(1500);

    const metrics = await session.send('Page.getLayoutMetrics');
    const height = Math.ceil(
      metrics.cssContentSize?.height || metrics.contentSize.height,
    );
    await session.send('Emulation.setDeviceMetricsOverride', {
      width: args.width,
      height,
      deviceScaleFactor: args.scale,
      mobile: false,
    });
    await sleep(500);

    // clip.scale multiplies on top of deviceScaleFactor, so it stays at 1:
    // the 2x crispness already comes from the device scale factor above.
    const clipHeight = args.clipHeight || height;
    const shot = await session.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: true,
      clip: {
        x: 0,
        y: args.clipY,
        width: args.width,
        height: clipHeight,
        scale: 1,
      },
    });

    await mkdir(dirname(outPath), { recursive: true });
    await writeFile(outPath, Buffer.from(shot.data, 'base64'));
    const pixels = `${args.width * args.scale}x${clipHeight * args.scale}`;
    console.log(`wrote ${outPath} (${pixels} px, page is ${height} css px tall)`);

    if (args.maxWidth && args.width * args.scale > args.maxWidth) {
      // sips ships with macOS, so no image library is needed just to make a
      // committed screenshot a sensible size.
      const { status } = spawnSync('sips', [
        '--resampleWidth', String(args.maxWidth), outPath, '--out', outPath,
      ], { stdio: 'ignore' });
      if (status === 0) console.log(`downscaled to ${args.maxWidth}px wide`);
      else console.log('sips not available, left at full resolution');
    }

    if (!painted) process.exitCode = 1;
  } finally {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    chrome.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(`screenshot failed: ${error.message}`);
  process.exit(1);
});

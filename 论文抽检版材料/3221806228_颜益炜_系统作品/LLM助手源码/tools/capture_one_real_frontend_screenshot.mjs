import { chromium } from "playwright";
import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const root = path.join(os.homedir(), "DAC-3D-LLM");
const project = path.join(root, "dac3d_iim_assistant");
const outDir = path.join(root, "\u6d4b\u8bd5\u7ed3\u679c", "01_step1_llm_evidence", "screenshots_real_frontend");

function loadEnv() {
  const env = { ...process.env };
  for (const raw of fs.readFileSync(path.join(project, ".env"), "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const idx = line.indexOf("=");
    env[line.slice(0, idx).trim()] = line.slice(idx + 1).trim().replace(/^['"]|['"]$/g, "");
  }
  env.DAC3D_LLM_PROVIDER = "openai_compatible";
  env.DAC3D_TIMEOUT_SECONDS = "180";
  env.DAC3D_LLM_TIMEOUT_SECONDS = "180";
  env.VITE_API_BASE_URL = "http://127.0.0.1:7892";
  env.DAC3D_CORS_ALLOWED_ORIGINS = "http://127.0.0.1:5174";
  return env;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForUrl(url, timeoutMs = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await delay(1000);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function spawnLogged(command, args, cwd, env, outFile, errFile) {
  const out = fs.openSync(outFile, "a");
  const err = fs.openSync(errFile, "a");
  return spawn(command, args, {
    cwd,
    env,
    stdio: ["ignore", out, err],
    windowsHide: true,
  });
}

async function main() {
  await fsp.mkdir(outDir, { recursive: true });
  const env = loadEnv();
  const backend = spawnLogged(
    "C:\\Users\\xecat\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
    ["app.py", "--web-only", "--host", "127.0.0.1", "--port", "7892"],
    project,
    env,
    path.join(outDir, "backend_real_screenshot_05.out.log"),
    path.join(outDir, "backend_real_screenshot_05.err.log"),
  );
  const frontend = spawnLogged(
    "cmd.exe",
    ["/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5174"],
    path.join(project, "ui2"),
    env,
    path.join(outDir, "frontend_real_screenshot_05.out.log"),
    path.join(outDir, "frontend_real_screenshot_05.err.log"),
  );
  let browser;
  try {
    await waitForUrl("http://127.0.0.1:7892/api/health");
    await waitForUrl("http://127.0.0.1:5174");
    browser = await chromium.launch({
      headless: true,
      executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.goto("http://127.0.0.1:5174", { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForSelector("textarea", { timeout: 30000 });
    const round = process.env.REALSHOT_ROUND || "05";
    const prompt =
      process.env.REALSHOT_PROMPT ||
      "\u8bf7\u8bf4\u660eDAC-3D\u52a9\u624b\u5728\u56de\u7b54\u4e2d\u5c55\u793a\u6765\u6e90\u4f9d\u636e\u7684\u4f5c\u7528\u3002";
    const before = await page.locator(".message.assistant, article.message.assistant, [data-testid='bot'], [data-testid='assistant']").count().catch(() => 0);
    await page.locator("textarea").fill(prompt);
    await page.locator("button[type='submit']").click({ timeout: 10000 });
    await page.waitForFunction(
      ({ beforeCount }) => {
        const text = document.body.innerText || "";
        const assistantCount = document.querySelectorAll(".message.assistant, article.message.assistant, [data-testid='bot'], [data-testid='assistant']").length;
        return assistantCount > beforeCount && !text.includes("\u6b63\u5728") && !text.includes("loading") && !text.includes("streaming");
      },
      { beforeCount: before },
      { timeout: 240000 },
    ).catch(async () => {
      await page.waitForTimeout(1500);
    });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: path.join(outDir, `LLM-REAL-FRONTEND-${round}.png`), fullPage: true });
    const visibleText = await page.locator("body").innerText({ timeout: 10000 });
    await fsp.writeFile(
      path.join(outDir, `real_frontend_screenshot_${round}_manifest.json`),
      JSON.stringify({ round, prompt, captured_at: new Date().toISOString(), visible_text_tail: visibleText.slice(-1800) }, null, 2),
      "utf8",
    );
  } finally {
    if (browser) await browser.close();
    frontend.kill("SIGTERM");
    backend.kill("SIGTERM");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

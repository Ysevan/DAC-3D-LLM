import { chromium } from "playwright";
import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const root = path.join(os.homedir(), "DAC-3D-LLM");
const project = path.join(root, "dac3d_iim_assistant");
const results = path.join(root, "\u6d4b\u8bd5\u7ed3\u679c");
const outDir = path.join(results, "01_step1_llm_evidence", "screenshots_real_frontend");
const oldGeneratedDir = path.join(results, "01_step1_llm_evidence", "screenshots_success_generated_not_real");
const generatedDir = path.join(results, "01_step1_llm_evidence", "screenshots_success");

function loadEnv() {
  const env = { ...process.env };
  const envPath = path.join(project, ".env");
  if (fs.existsSync(envPath)) {
    for (const raw of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const idx = line.indexOf("=");
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim().replace(/^['"]|['"]$/g, "");
      env[key] = value;
    }
  }
  env.DAC3D_LLM_PROVIDER = "openai_compatible";
  env.DAC3D_TIMEOUT_SECONDS = "180";
  env.DAC3D_LLM_TIMEOUT_SECONDS = "180";
  return env;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForUrl(url, timeoutMs = 60000) {
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
    detached: false,
    stdio: ["ignore", out, err],
    windowsHide: true,
  });
}

async function main() {
  await fsp.mkdir(outDir, { recursive: true });
  if (fs.existsSync(generatedDir)) {
    await fsp.rm(oldGeneratedDir, { recursive: true, force: true });
    await fsp.rename(generatedDir, oldGeneratedDir);
  }

  const env = loadEnv();
  const backend = spawnLogged(
    "C:\\Users\\xecat\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
    ["app.py", "--web-only", "--host", "127.0.0.1", "--port", "7890"],
    project,
    env,
    path.join(outDir, "backend_real_screenshot.out.log"),
    path.join(outDir, "backend_real_screenshot.err.log"),
  );
  const frontend = spawnLogged(
    "cmd.exe",
    ["/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
    path.join(project, "ui2"),
    env,
    path.join(outDir, "frontend_real_screenshot.out.log"),
    path.join(outDir, "frontend_real_screenshot.err.log"),
  );

  let browser;
  try {
    await waitForUrl("http://127.0.0.1:7890/api/health", 90000);
    await waitForUrl("http://127.0.0.1:5173", 90000);
    browser = await chromium.launch({
      headless: true,
      executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.goto("http://127.0.0.1:5173", { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForSelector("textarea", { timeout: 30000 });

    const prompts = [
      "\u8bf7\u7528\u4e00\u53e5\u8bdd\u8bf4\u660eDAC-3D\u7cfb\u7edf\u4e2d\u626b\u63cf\u533a\u57df\u53c2\u6570\u7684\u4f5c\u7528\u3002",
      "\u5982\u679c\u7528\u6237\u53ea\u8bf4\u201c\u5f00\u59cb\u626b\u63cf\u201d\uff0c\u52a9\u624b\u5e94\u8be5\u8ffd\u95ee\u54ea\u4e9b\u5fc5\u8981\u4fe1\u606f\uff1f",
      "\u8bf7\u8bf4\u660e\u77e5\u8bc6\u5e93\u68c0\u7d22\u5728\u53c2\u6570\u95ee\u7b54\u4e2d\u7684\u4f5c\u7528\u3002",
      "\u8bf7\u7528\u7b80\u77ed\u4e2d\u6587\u89e3\u91ca\u5212\u75d5\u7f3a\u9677\u4e25\u91cd\u5ea6\u5224\u65ad\u9700\u8981\u54ea\u4e9b\u4f9d\u636e\u3002",
      "\u8bf7\u8bf4\u660e\u4e3a\u4ec0\u4e48\u5de5\u4e1a\u68c0\u6d4b\u52a9\u624b\u9700\u8981\u547d\u4ee4\u786e\u8ba4\u673a\u5236\u3002",
    ];
    const evidence = [];
    for (let i = 0; i < prompts.length; i += 1) {
      const prompt = prompts[i];
      const before = await page.locator(".message.assistant, article.message.assistant, [data-testid='bot'], [data-testid='assistant']").count().catch(() => 0);
      await page.locator("textarea").fill(prompt);
      const submit = page.locator("button[type='submit']");
      await submit.click({ timeout: 10000 });
      await page.waitForFunction(
        ({ beforeCount }) => {
          const text = document.body.innerText || "";
          const assistantCount = document.querySelectorAll(".message.assistant, article.message.assistant, [data-testid='bot'], [data-testid='assistant']").length;
          return assistantCount > beforeCount && !text.includes("\u6b63\u5728") && !text.includes("loading") && !text.includes("streaming");
        },
        { beforeCount: before },
        { timeout: 240000 },
      ).catch(async () => {
        await page.waitForTimeout(2000);
      });
      await page.waitForTimeout(1200);
      const screenshotPath = path.join(outDir, `LLM-REAL-FRONTEND-${String(i + 1).padStart(2, "0")}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      const visibleText = await page.locator("body").innerText({ timeout: 10000 });
      evidence.push({
        round: i + 1,
        prompt,
        screenshot: screenshotPath,
        captured_at: new Date().toISOString(),
        visible_text_tail: visibleText.slice(-1800),
      });
    }
    await fsp.writeFile(path.join(outDir, "real_frontend_screenshot_manifest.json"), JSON.stringify(evidence, null, 2), "utf8");
  } finally {
    if (browser) await browser.close();
    for (const child of [frontend, backend]) {
      if (child && !child.killed) {
        child.kill("SIGTERM");
      }
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

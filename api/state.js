const REDIS_URL = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
const REDIS_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
const APP_TOKEN = process.env.APP_TOKEN;
const KEY = "icdash_edryck_v1";

async function redis(command) {
  const response = await fetch(REDIS_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${REDIS_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(command),
  });
  if (!response.ok) throw new Error(`redis ${response.status}`);
  const data = await response.json();
  return data.result;
}

module.exports = async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (!REDIS_URL || !REDIS_TOKEN || !APP_TOKEN) {
    return res.status(500).json({ error: "server not configured" });
  }
  if (req.headers["x-app-token"] !== APP_TOKEN) {
    return res.status(401).json({ error: "unauthorized" });
  }

  try {
    if (req.method === "GET") {
      const raw = await redis(["GET", KEY]);
      return res.status(200).json(raw ? JSON.parse(raw) : { updatedAt: 0, state: null });
    }
    if (req.method === "PUT") {
      const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
      if (!body || typeof body.state !== "object" || typeof body.updatedAt !== "number") {
        return res.status(400).json({ error: "invalid payload" });
      }
      await redis(["SET", KEY, JSON.stringify({ updatedAt: body.updatedAt, state: body.state })]);
      return res.status(200).json({ ok: true });
    }
    res.setHeader("Allow", "GET, PUT");
    return res.status(405).json({ error: "method not allowed" });
  } catch (error) {
    return res.status(500).json({ error: "storage error" });
  }
};

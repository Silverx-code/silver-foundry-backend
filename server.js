/**
 * Silver AI Foundry — Backend API
 * Stack: Node.js + Express + Supabase + child_process (Python)
 * Deploy: Render (free tier)
 */

import express from "express";
import cors from "cors";
import multer from "multer";
import { v4 as uuid } from "uuid";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// ─── Supabase ───────────────────────────────────────────────
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ─── Middleware ─────────────────────────────────────────────
app.use(cors({ origin: process.env.FRONTEND_URL || "*" }));
app.use(express.json());

// ─── Multer (temp file upload) ──────────────────────────────
const upload = multer({
  dest: "/tmp/uploads/",
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
  fileFilter: (_, file, cb) => {
    const allowed = [".csv", ".json"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) cb(null, true);
    else cb(new Error("Only CSV and JSON files are supported."));
  },
});

// In-memory job store (replace with Supabase for persistence)
const jobs = {};

// ─── ROUTES ─────────────────────────────────────────────────

/**
 * GET /health
 * Health check — used by Render to verify server is up
 */
app.get("/health", (_, res) => {
  res.json({ status: "ok", service: "Silver AI Foundry API", version: "1.0.0" });
});

/**
 * POST /upload
 * Accepts a CSV/JSON file, stores it in Supabase Storage,
 * returns a dataset_id for use in /train
 */
app.post("/upload", upload.single("dataset"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    const datasetId = uuid();
    const ext = path.extname(req.file.originalname).toLowerCase();
    const remotePath = `datasets/${datasetId}${ext}`;

    // Upload to Supabase Storage bucket "datasets"
    const fileBuffer = fs.readFileSync(req.file.path);
    const { error: uploadError } = await supabase.storage
      .from("datasets")
      .upload(remotePath, fileBuffer, {
        contentType: ext === ".csv" ? "text/csv" : "application/json",
      });

    if (uploadError) throw uploadError;

    // Save metadata to DB
    const { error: dbError } = await supabase.from("datasets").insert({
      id: datasetId,
      original_name: req.file.originalname,
      storage_path: remotePath,
      size_bytes: req.file.size,
      created_at: new Date().toISOString(),
    });

    if (dbError) throw dbError;

    // Clean up temp file
    fs.unlinkSync(req.file.path);

    res.json({
      dataset_id: datasetId,
      original_name: req.file.originalname,
      size_bytes: req.file.size,
      message: "Dataset uploaded successfully.",
    });
  } catch (err) {
    console.error("[/upload]", err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /train
 * Body: { dataset_id, model_type, experiment_name }
 * Spawns Python trainer, streams logs, saves metrics on completion
 */
app.post("/train", async (req, res) => {
  const { dataset_id, model_type = "classification", experiment_name } = req.body;

  if (!dataset_id) return res.status(400).json({ error: "dataset_id is required." });

  const jobId = uuid();
  const expName = experiment_name || `experiment-${jobId.slice(0, 8)}`;

  // Initialise job entry
  jobs[jobId] = {
    id: jobId,
    dataset_id,
    model_type,
    experiment_name: expName,
    status: "queued",
    progress: 0,
    logs: [],
    metrics: null,
    created_at: new Date().toISOString(),
  };

  // Save initial record to Supabase
  supabase.from("experiments").insert({
    id: jobId,
    dataset_id,
    model_type,
    experiment_name: expName,
    status: "queued",
    created_at: new Date().toISOString(),
  });

  // Download dataset from Supabase to /tmp
  const { data: datasetMeta } = await supabase
    .from("datasets")
    .select("storage_path, original_name")
    .eq("id", dataset_id)
    .single();

  if (!datasetMeta) return res.status(404).json({ error: "Dataset not found." });

  const localPath = `/tmp/${jobId}_${datasetMeta.original_name}`;

  const { data: fileData, error: dlError } = await supabase.storage
    .from("datasets")
    .download(datasetMeta.storage_path);

  if (dlError) return res.status(500).json({ error: "Failed to retrieve dataset." });

  fs.writeFileSync(localPath, Buffer.from(await fileData.arrayBuffer()));

  // Spawn Python trainer
  const py = spawn("python3", [
    "./trainer/train.py",
    "--input", localPath,
    "--model", model_type,
    "--job_id", jobId,
  ]);

  jobs[jobId].status = "running";
  jobs[jobId].progress = 5;

  py.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    lines.forEach((line) => {
      try {
        // Python trainer emits JSON lines: { type, message, progress?, metrics? }
        const evt = JSON.parse(line);
        if (evt.type === "log") {
          jobs[jobId].logs.push({ msg: evt.message, ts: Date.now() });
          if (evt.progress) jobs[jobId].progress = evt.progress;
        } else if (evt.type === "done") {
          jobs[jobId].status = "done";
          jobs[jobId].progress = 100;
          jobs[jobId].metrics = evt.metrics;
          // Persist to Supabase
          supabase.from("experiments").update({
            status: "done",
            metrics: evt.metrics,
            completed_at: new Date().toISOString(),
          }).eq("id", jobId);
          // Clean temp file
          fs.existsSync(localPath) && fs.unlinkSync(localPath);
        }
      } catch {
        jobs[jobId].logs.push({ msg: line, ts: Date.now() });
      }
    });
  });

  py.stderr.on("data", (data) => {
    jobs[jobId].logs.push({ msg: `[stderr] ${data.toString()}`, ts: Date.now() });
  });

  py.on("close", (code) => {
    if (code !== 0 && jobs[jobId].status !== "done") {
      jobs[jobId].status = "error";
      supabase.from("experiments").update({ status: "error" }).eq("id", jobId);
    }
  });

  // Return job ID immediately (client polls /status/:jobId)
  res.json({ job_id: jobId, experiment_name: expName, status: "running" });
});

/**
 * GET /status/:jobId
 * Returns current training progress, logs, and status
 */
app.get("/status/:jobId", (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found." });

  res.json({
    job_id: job.id,
    status: job.status,
    progress: job.progress,
    logs: job.logs,
    experiment_name: job.experiment_name,
    model_type: job.model_type,
    created_at: job.created_at,
  });
});

/**
 * GET /results/:jobId
 * Returns final metrics once training is done
 */
app.get("/results/:jobId", (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found." });
  if (job.status !== "done") return res.status(202).json({ status: job.status, message: "Training not yet complete." });

  res.json({
    job_id: job.id,
    experiment_name: job.experiment_name,
    model_type: job.model_type,
    metrics: job.metrics,
    created_at: job.created_at,
    completed_at: new Date().toISOString(),
  });
});

/**
 * POST /predict/:jobId
 * Run inference against a deployed model
 * Body: { features: [...] }
 */
app.post("/predict/:jobId", async (req, res) => {
  const { features } = req.body;
  if (!features || !Array.isArray(features)) {
    return res.status(400).json({ error: "features must be a JSON array." });
  }

  const job = jobs[req.params.jobId];
  if (!job || job.status !== "done") {
    return res.status(404).json({ error: "Model not found or not ready." });
  }

  const start = Date.now();

  // Spawn Python predictor
  const py = spawn("python3", [
    "./trainer/predict.py",
    "--job_id", req.params.jobId,
    "--features", JSON.stringify(features),
  ]);

  let output = "";
  py.stdout.on("data", (d) => { output += d.toString(); });
  py.on("close", () => {
    try {
      const result = JSON.parse(output.trim());
      res.json({
        ...result,
        job_id: req.params.jobId,
        model: job.experiment_name,
        latency_ms: Date.now() - start,
      });
    } catch {
      res.status(500).json({ error: "Prediction failed." });
    }
  });
});

/**
 * GET /experiments
 * List all experiments from Supabase
 */
app.get("/experiments", async (_, res) => {
  const { data, error } = await supabase
    .from("experiments")
    .select("*")
    .order("created_at", { ascending: false });

  if (error) return res.status(500).json({ error: error.message });
  res.json({ experiments: data });
});

/**
 * DELETE /experiments/:id
 * Delete an experiment record
 */
app.delete("/experiments/:id", async (req, res) => {
  const { error } = await supabase
    .from("experiments")
    .delete()
    .eq("id", req.params.id);

  if (error) return res.status(500).json({ error: error.message });
  delete jobs[req.params.id];
  res.json({ message: "Experiment deleted." });
});

// ─── Start server ────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[Silver AI Foundry] Backend running on port ${PORT}`);
});

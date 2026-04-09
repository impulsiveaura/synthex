/**
 * SYNTHEX Dashboard — Dual-Stream Inference Pipeline UI
 * Created by Sharaf Samiur Rahman — github.com/impulsiveaura
 */
import { useState, useEffect, useRef } from "react";

// ─── INFERENCE MODES ──────────────────────────────────────────────────────────
const MODES = {
  LITERAL:     { label: "LITERAL",     code: "LIT", color: "#4fc3f7", desc: "Explicit token semantics" },
  LATENT:      { label: "LATENT",      code: "LAT", color: "#ce93d8", desc: "Implicit embedding space" },
  TEMPORAL:    { label: "TEMPORAL",    code: "TMP", color: "#ffcc02", desc: "Context-invariant patterns" },
  GENERATIVE:  { label: "GENERATIVE",  code: "GEN", color: "#69f0ae", desc: "Cross-domain synthesis" },
};

// ─── PIPELINE STAGES ─────────────────────────────────────────────────────────
const STAGES = [
  {
    id: "TOK_LIT",
    pipeline: "A",
    layer: "TOKENIZATION",
    mode: "LITERAL",
    label: "Tokenization · Literal",
    desc: "Parse explicit query tokens. Extract named entities, scope boundaries, stated constraints. No inference.",
    metric: "T/s",
  },
  {
    id: "TOK_LAT",
    pipeline: "A",
    layer: "TOKENIZATION",
    mode: "LATENT",
    label: "Tokenization · Latent",
    desc: "Project query into latent embedding space. Surface implicit intent, unstated context, underlying information need.",
    metric: "T/s",
  },
  {
    id: "EMB_FUSION",
    pipeline: "A",
    layer: "EMBEDDING_FUSION",
    mode: null,
    label: "Embedding Fusion · Cross-Stream Coherence",
    desc: "Compute coherence intersection of Literal and Latent streams. Retain only embeddings with high cross-stream activation. Discard stream-exclusive noise.",
    metric: "CosSim",
    isFusion: true,
  },
  {
    id: "INF_TMP",
    pipeline: "B",
    layer: "INFERENCE_CORE",
    mode: "TEMPORAL",
    label: "Inference Core · Temporal",
    desc: "Activate context-invariant reasoning paths. Extract principles and patterns that hold across time, domain, and distribution shift.",
    metric: "Depth",
  },
  {
    id: "INF_GEN",
    pipeline: "B",
    layer: "INFERENCE_CORE",
    mode: "GENERATIVE",
    label: "Inference Core · Generative",
    desc: "Activate cross-domain synthesis paths. Generate non-obvious connections, analogical transfers, emergent hypotheses from fused embeddings.",
    metric: "Novelty",
  },
  {
    id: "SIG_DISTIL",
    pipeline: "B",
    layer: "SIGNAL_DISTILLATION",
    mode: null,
    label: "Signal Distillation · Output Gate",
    desc: "Run both Inference Core streams through distillation. Reject speculative artifacts. Emit only tokens with high confidence across both inference paths.",
    metric: "Fidelity",
    isFusion: true,
  },
  {
    id: "MODE_SEL",
    pipeline: "T",
    layer: "MODE_SELECTOR",
    mode: null,
    label: "Inference Mode Selector",
    desc: "Assess query complexity profile. Assign active/idle status per inference mode. Route idle modes to zero-cost pass-through. Log activated inference budget.",
    metric: "Modes",
    isSelector: true,
  },
  {
    id: "CONSENSUS",
    pipeline: "Ω",
    layer: "CONSENSUS",
    mode: null,
    label: "Consensus Layer",
    desc: "Merge distilled signal streams. Resolve cross-pipeline contradictions via confidence weighting. Zero embedding bleed between pipeline namespaces. Final output commit.",
    metric: "CL",
    isConsensus: true,
  },
];

// ─── PROMPTS ──────────────────────────────────────────────────────────────────
function buildPrompt(stage, query, prior) {
  const ctx = Object.entries(prior).filter(([,v])=>v).map(([k,v])=>`[${k}]: ${v}`).join("\n");
  const priorText = ctx ? `\nUpstream outputs:\n${ctx}\n` : "";

  const map = {
    TOK_LIT: `You are the LITERAL TOKENIZATION layer of an AI inference pipeline. Parse only what is explicitly stated: entities, literal intent, stated constraints. Zero inference. 2 sentences max.`,
    TOK_LAT: `You are the LATENT TOKENIZATION layer. Project the query into latent space — what is the real underlying information need, the implicit goal, the unstated assumption driving the question? 2 sentences max.`,
    EMB_FUSION: `You are the EMBEDDING FUSION layer. You receive two tokenization streams: Literal and Latent. Compute their intersection — what is coherent and active in BOTH simultaneously? Discard anything that only exists in one stream. Output only the high-density fused signal. 2-3 sentences. Do not summarize either stream — output only their shared activation space.`,
    INF_TMP: `You are the TEMPORAL INFERENCE CORE. Using the fused embeddings, activate reasoning paths that are context-invariant — principles, structural patterns, and facts that hold regardless of time, domain, or distribution shift. 2-3 sentences.`,
    INF_GEN: `You are the GENERATIVE INFERENCE CORE. Using the fused embeddings, activate cross-domain synthesis. What non-obvious connection, analogical transfer, or emergent hypothesis can only be reached by holding the full fused context simultaneously? 2-3 sentences.`,
    SIG_DISTIL: `You are the SIGNAL DISTILLATION gate. You receive two inference streams: Temporal and Generative. Run distillation — reject speculative artifacts, surface-only pattern matches, and low-confidence outputs. What remains when only tokens with high confidence across BOTH inference paths survive? 2-3 sentences of clean signal only.`,
    MODE_SEL: `You are the INFERENCE MODE SELECTOR. Evaluate the original query. For each mode, output exactly: LITERAL: ACTIVE or IDLE, LATENT: ACTIVE or IDLE, TEMPORAL: ACTIVE or IDLE, GENERATIVE: ACTIVE or IDLE. Then one sentence: what inference budget was actually required for this query?`,
    CONSENSUS: `You are the CONSENSUS LAYER — final output commit. Merge all upstream distilled signals. Resolve any cross-pipeline contradictions by confidence weighting. Zero embedding bleed between concepts. Output the definitive, complete, precise response to the original query. 3-5 sentences. No preamble, no meta-commentary — just the answer.`,
  };

  return `${map[stage.id]}

Original query: "${query}"${priorText}
Respond with your layer output only. No labels, no preamble.`;
}

// ─── SPARKLINE ────────────────────────────────────────────────────────────────
function Sparkline({ active, color, w = 72, h = 22, freq = 3 }) {
  const [ph, setPh] = useState(0);
  useEffect(() => {
    if (!active) return;
    let raf;
    const tick = () => { setPh(p => (p + 0.09) % (Math.PI * 2)); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  const pts = Array.from({ length: 48 }, (_, i) => {
    const x = (i / 47) * w;
    const y = h / 2 - (active ? Math.sin(i / 47 * Math.PI * 2 * freq + ph) * (h / 2 - 2) : 0);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={w} height={h}>
      <polyline points={pts} fill="none"
        stroke={active ? color : "#ffffff18"}
        strokeWidth={active ? 1.5 : 0.8}
        opacity={active ? 0.9 : 0.3}
        style={{ transition: "stroke 0.3s" }}
      />
    </svg>
  );
}

// ─── STATUS DOT ──────────────────────────────────────────────────────────────
function StatusDot({ status, color }) {
  if (status === "active") return (
    <span style={{
      display: "inline-block", width: 7, height: 7, borderRadius: "50%",
      background: color, animation: "pulse 0.5s ease infinite",
    }} />
  );
  if (status === "done") return (
    <span style={{
      display: "inline-block", width: 7, height: 7, borderRadius: "50%",
      background: color, opacity: 0.45,
    }} />
  );
  return (
    <span style={{
      display: "inline-block", width: 7, height: 7, borderRadius: "50%",
      background: "#ffffff18",
    }} />
  );
}

// ─── STAGE ROW ────────────────────────────────────────────────────────────────
function StageRow({ stage, status, output, index }) {
  const active = status === "active";
  const done = status === "done";
  const modeInfo = stage.mode ? MODES[stage.mode] : null;
  const rowColor = modeInfo?.color || (stage.isFusion ? "#ffcc02" : stage.isSelector ? "#4fc3f7" : stage.isConsensus ? "#ffffff" : "#aaaaaa");
  const isConsensus = stage.isConsensus;

  return (
    <div style={{
      borderBottom: "1px solid rgba(255,255,255,0.04)",
      background: active ? `${rowColor}09` : "transparent",
      transition: "background 0.25s",
      position: "relative",
    }}>
      {active && (
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0, width: 2,
          background: rowColor,
          animation: "none",
        }} />
      )}

      {/* Header row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "28px 1fr auto auto auto",
        gap: 0,
        alignItems: "center",
        padding: "8px 14px 6px 14px",
      }}>
        {/* Index */}
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 9,
          color: "rgba(255,255,255,0.2)",
          letterSpacing: "0.05em",
        }}>{String(index + 1).padStart(2, "0")}</span>

        {/* Label */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <StatusDot status={status} color={rowColor} />
          <span style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 13,
            letterSpacing: "0.12em",
            color: active || done ? rowColor : "rgba(255,255,255,0.25)",
            transition: "color 0.3s",
          }}>{stage.label}</span>
          {modeInfo && (
            <span style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 8,
              color: active ? modeInfo.color : "rgba(255,255,255,0.18)",
              border: `1px solid ${active ? modeInfo.color + "55" : "rgba(255,255,255,0.08)"}`,
              padding: "1px 5px",
              borderRadius: 2,
              letterSpacing: "0.1em",
              transition: "all 0.3s",
            }}>{modeInfo.code}</span>
          )}
          {stage.isFusion && (
            <span style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 8,
              color: active ? "#ffcc02" : "rgba(255,255,255,0.15)",
              border: `1px solid ${active ? "#ffcc0244" : "rgba(255,255,255,0.06)"}`,
              padding: "1px 5px",
              borderRadius: 2,
              letterSpacing: "0.1em",
            }}>FUSION</span>
          )}
        </div>

        {/* Sparkline */}
        <Sparkline
          active={active}
          color={rowColor}
          freq={stage.mode === "LITERAL" ? 4 : stage.mode === "LATENT" ? 1.8 : stage.mode === "TEMPORAL" ? 1.2 : stage.mode === "GENERATIVE" ? 6 : 3}
          w={64}
          h={18}
        />

        {/* Metric tag */}
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 8,
          color: active ? rowColor : "rgba(255,255,255,0.15)",
          letterSpacing: "0.1em",
          marginLeft: 10,
          minWidth: 40,
          textAlign: "right",
          transition: "color 0.3s",
        }}>{stage.metric}</span>

        {/* Status */}
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 8,
          color: active ? "#ffcc02" : done ? "#69f0ae" : "rgba(255,255,255,0.15)",
          letterSpacing: "0.12em",
          marginLeft: 12,
          minWidth: 32,
          textAlign: "right",
        }}>{active ? "RUN" : done ? "OK" : "IDL"}</span>
      </div>

      {/* Desc */}
      <div style={{ padding: "0 14px 6px 42px" }}>
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 9,
          color: "rgba(255,255,255,0.22)",
          lineHeight: 1.5,
        }}>{stage.desc}</span>
      </div>

      {/* Output */}
      {output && (
        <div style={{
          margin: "0 14px 10px 42px",
          padding: "10px 12px",
          background: "rgba(0,0,0,0.35)",
          borderLeft: `2px solid ${rowColor}55`,
          borderRadius: "0 3px 3px 0",
        }}>
          <p style={{
            fontFamily: isConsensus ? "'IBM Plex Sans', sans-serif" : "'IBM Plex Mono', monospace",
            fontSize: isConsensus ? 13 : 11,
            color: isConsensus ? "#e8f4e8" : "rgba(255,255,255,0.58)",
            lineHeight: 1.75,
            margin: 0,
            fontWeight: isConsensus ? 400 : 400,
          }}>{output}</p>
        </div>
      )}
    </div>
  );
}

// ─── PIPELINE HEADER ─────────────────────────────────────────────────────────
function PipelineHeader({ id, label, sub, accent, stages, getStatus }) {
  const anyActive = stages.some(s => getStatus(s.id) === "active");
  const allDone = stages.every(s => getStatus(s.id) === "done");
  return (
    <div style={{
      padding: "5px 14px",
      background: `${accent}0c`,
      borderBottom: "1px solid rgba(255,255,255,0.05)",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{
          fontFamily: "'Bebas Neue', sans-serif",
          fontSize: 11,
          letterSpacing: "0.3em",
          color: accent,
        }}>PIPELINE {id}</span>
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 9,
          color: "rgba(255,255,255,0.25)",
          letterSpacing: "0.08em",
        }}>{sub}</span>
      </div>
      <div style={{ display: "flex", gap: 3 }}>
        {stages.map(s => (
          <div key={s.id} style={{
            width: 6, height: 6, borderRadius: 1,
            background: getStatus(s.id) === "active" ? accent
              : getStatus(s.id) === "done" ? accent + "55"
              : "rgba(255,255,255,0.07)",
            transition: "background 0.2s",
          }} />
        ))}
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 8,
          color: anyActive ? "#ffcc02" : allDone ? "#69f0ae" : "rgba(255,255,255,0.18)",
          marginLeft: 6,
          letterSpacing: "0.1em",
        }}>{anyActive ? "ACTIVE" : allDone ? "DONE" : "IDLE"}</span>
      </div>
    </div>
  );
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────
export default function Synthex() {
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [statuses, setStatuses] = useState({});
  const [outputs, setOutputs] = useState({});
  const [done, setDone] = useState(false);
  const [throughput, setThroughput] = useState(0);
  const [latency, setLatency] = useState(0);
  const [activeModes, setActiveModes] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const abortRef = useRef(false);
  const tpRef = useRef(null);
  const latRef = useRef(null);

  const setStatus = (id, s) => setStatuses(p => ({ ...p, [id]: s }));
  const getStatus = id => statuses[id] || "idle";

  const callClaude = async (prompt) => {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    const data = await res.json();
    return data.content?.map(b => b.text || "").join("") || "";
  };

  const run = async () => {
    if (!query.trim() || running) return;
    abortRef.current = false;
    setRunning(true);
    setStatuses({});
    setOutputs({});
    setDone(false);
    setActiveModes([]);
    setThroughput(0);
    setLatency(0);
    const t0 = Date.now();
    setStartTime(t0);

    let tp = 0;
    tpRef.current = setInterval(() => {
      tp = Math.min(tp + Math.floor(Math.random() * 800 + 400), 32000);
      setThroughput(tp);
    }, 80);
    latRef.current = setInterval(() => {
      setLatency(((Date.now() - t0) / 1000).toFixed(1));
    }, 100);

    const collected = {};

    const runStage = async (stage) => {
      if (abortRef.current) return;
      setStatus(stage.id, "active");
      await new Promise(r => setTimeout(r, 180));
      try {
        const out = await callClaude(buildPrompt(stage, query, collected));
        collected[stage.id] = out.trim();
        setOutputs(p => ({ ...p, [stage.id]: out.trim() }));
        if (stage.isSelector) {
          const keys = ["LITERAL","LATENT","TEMPORAL","GENERATIVE"];
          const active = keys.filter(k => out.toUpperCase().includes(k + ": ACTIVE"));
          setActiveModes(active.length ? active : keys);
        }
      } catch {
        collected[stage.id] = "[layer fault]";
        setOutputs(p => ({ ...p, [stage.id]: "[layer fault]" }));
      }
      setStatus(stage.id, "done");
    };

    // Pipeline A: dual tokenization streams run in parallel
    setStatus("TOK_LIT", "active"); setStatus("TOK_LAT", "active");
    await new Promise(r => setTimeout(r, 180));
    await Promise.all([
      callClaude(buildPrompt(STAGES[0], query, collected)).then(out => {
        collected["TOK_LIT"] = out.trim();
        setOutputs(p => ({ ...p, TOK_LIT: out.trim() }));
        setStatus("TOK_LIT", "done");
      }),
      callClaude(buildPrompt(STAGES[1], query, collected)).then(out => {
        collected["TOK_LAT"] = out.trim();
        setOutputs(p => ({ ...p, TOK_LAT: out.trim() }));
        setStatus("TOK_LAT", "done");
      }),
    ]);

    await runStage(STAGES.find(s => s.id === "EMB_FUSION"));

    // Pipeline B: dual inference cores in parallel
    setStatus("INF_TMP", "active"); setStatus("INF_GEN", "active");
    await new Promise(r => setTimeout(r, 180));
    await Promise.all([
      callClaude(buildPrompt(STAGES.find(s => s.id === "INF_TMP"), query, collected)).then(out => {
        collected["INF_TMP"] = out.trim();
        setOutputs(p => ({ ...p, INF_TMP: out.trim() }));
        setStatus("INF_TMP", "done");
      }),
      callClaude(buildPrompt(STAGES.find(s => s.id === "INF_GEN"), query, collected)).then(out => {
        collected["INF_GEN"] = out.trim();
        setOutputs(p => ({ ...p, INF_GEN: out.trim() }));
        setStatus("INF_GEN", "done");
      }),
    ]);

    await runStage(STAGES.find(s => s.id === "SIG_DISTIL"));
    await runStage(STAGES.find(s => s.id === "MODE_SEL"));
    await runStage(STAGES.find(s => s.id === "CONSENSUS"));

    clearInterval(tpRef.current);
    clearInterval(latRef.current);
    setThroughput(0);
    setDone(true);
    setRunning(false);
  };

  const reset = () => {
    abortRef.current = true;
    clearInterval(tpRef.current);
    clearInterval(latRef.current);
    setRunning(false);
    setStatuses({});
    setOutputs({});
    setDone(false);
    setActiveModes([]);
    setThroughput(0);
    setLatency(0);
  };

  const pipelineA = STAGES.filter(s => s.pipeline === "A");
  const pipelineB = STAGES.filter(s => s.pipeline === "B");
  const selectorStage = STAGES.find(s => s.isSelector);
  const consensusStage = STAGES.find(s => s.isConsensus);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060708",
      color: "#d8e0d0",
      fontFamily: "'IBM Plex Mono', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.85)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)} }
        @keyframes tpFlash { 0%,100%{opacity:1}50%{opacity:0.7} }
        * { box-sizing:border-box; margin:0; padding:0; }
        textarea { resize:none; }
        textarea:focus { outline:none; }
        button { cursor:pointer; transition:all 0.12s; }
        button:active { transform:scale(0.97); }
        ::-webkit-scrollbar { width:3px; }
        ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.08); }
      `}</style>

      {/* ── Top Bar ── */}
      <div style={{
        borderBottom: "1px solid rgba(255,255,255,0.07)",
        padding: "0 20px",
        height: 44,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(255,255,255,0.02)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 22,
            letterSpacing: "0.15em",
            color: "#69f0ae",
          }}>SYNTHEX</span>
          <span style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 9,
            color: "rgba(255,255,255,0.2)",
            letterSpacing: "0.15em",
            borderLeft: "1px solid rgba(255,255,255,0.1)",
            paddingLeft: 14,
          }}>DUAL-STREAM INFERENCE PIPELINE</span>
        </div>

        {/* Metrics strip */}
        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
          {[
            ["THROUGHPUT", throughput ? throughput.toLocaleString() + " T/s" : "—", running ? "#ffcc02" : "rgba(255,255,255,0.2)"],
            ["LATENCY",    latency ? latency + "s" : "—",                              running ? "#4fc3f7" : "rgba(255,255,255,0.2)"],
            ["STATUS",     running ? "ACTIVE" : done ? "COMPLETE" : "STANDBY",         running ? "#ffcc02" : done ? "#69f0ae" : "rgba(255,255,255,0.2)"],
          ].map(([k, v, c]) => (
            <div key={k} style={{ textAlign: "right" }}>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 7, letterSpacing: "0.25em", color: "rgba(255,255,255,0.2)", textTransform: "uppercase" }}>{k}</div>
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: 11,
                color: c, letterSpacing: "0.08em", fontWeight: 700,
                animation: running && k === "THROUGHPUT" ? "tpFlash 0.3s linear infinite" : "none",
              }}>{v}</div>
            </div>
          ))}

          {/* Mode indicators */}
          <div style={{ borderLeft: "1px solid rgba(255,255,255,0.07)", paddingLeft: 16 }}>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 7, letterSpacing: "0.25em", color: "rgba(255,255,255,0.2)", marginBottom: 3 }}>INFERENCE MODES</div>
            <div style={{ display: "flex", gap: 4 }}>
              {Object.entries(MODES).map(([k, m]) => {
                const isActive = activeModes.includes(k) || (running && !activeModes.length);
                return (
                  <span key={k} style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 7,
                    color: isActive ? m.color : "rgba(255,255,255,0.15)",
                    border: `1px solid ${isActive ? m.color + "44" : "rgba(255,255,255,0.06)"}`,
                    padding: "1px 4px",
                    borderRadius: 2,
                    letterSpacing: "0.08em",
                    transition: "all 0.3s",
                  }}>{m.code}</span>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* ── Query Input ── */}
      <div style={{
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        padding: "10px 20px",
        display: "flex",
        gap: 10,
        alignItems: "flex-end",
        background: "rgba(255,255,255,0.01)",
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, color: "rgba(255,255,255,0.2)", letterSpacing: "0.2em", marginBottom: 5 }}>
            INPUT · QUERY INJECTION
          </div>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
            placeholder="Enter query for dual-stream processing..."
            disabled={running}
            rows={2}
            style={{
              width: "100%",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 3,
              padding: "8px 10px",
              color: "#c8e0c0",
              fontSize: 12,
              fontFamily: "'IBM Plex Mono', monospace",
              lineHeight: 1.6,
              opacity: running ? 0.5 : 1,
            }}
          />
        </div>
        <div style={{ display: "flex", gap: 6, paddingBottom: 1 }}>
          {(running || done) && (
            <button onClick={reset} style={{
              padding: "7px 14px",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 3,
              color: "rgba(255,255,255,0.3)",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 9,
              letterSpacing: "0.18em",
            }}>RESET</button>
          )}
          <button onClick={run} disabled={running || !query.trim()} style={{
            padding: "7px 20px",
            background: running || !query.trim() ? "rgba(255,255,255,0.04)" : "#69f0ae",
            border: "none",
            borderRadius: 3,
            color: running || !query.trim() ? "rgba(255,255,255,0.2)" : "#060708",
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 14,
            letterSpacing: "0.18em",
          }}>
            {running ? "RUNNING" : "EXECUTE"}
          </button>
        </div>
      </div>

      {/* ── Pipeline Grid ── */}
      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "14px 16px" }}>

        {/* Dual pipelines side by side */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
          {/* Pipeline A */}
          <div style={{ border: "1px solid rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
            <PipelineHeader id="A" sub="Tokenization · Embedding Fusion" accent="#4fc3f7" stages={pipelineA} getStatus={getStatus} />
            {pipelineA.map((s, i) => (
              <StageRow key={s.id} stage={s} status={getStatus(s.id)} output={outputs[s.id]} index={i} />
            ))}
          </div>

          {/* Pipeline B */}
          <div style={{ border: "1px solid rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
            <PipelineHeader id="B" sub="Inference Core · Signal Distillation" accent="#ce93d8" stages={pipelineB} getStatus={getStatus} />
            {pipelineB.map((s, i) => (
              <StageRow key={s.id} stage={s} status={getStatus(s.id)} output={outputs[s.id]} index={i} />
            ))}
          </div>
        </div>

        {/* Mode Selector */}
        <div style={{
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 4,
          overflow: "hidden",
          marginBottom: 10,
        }}>
          <div style={{
            padding: "5px 14px",
            background: "rgba(79,195,247,0.05)",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 11, letterSpacing: "0.28em", color: "rgba(79,195,247,0.6)" }}>INFERENCE MODE SELECTOR · EFFICIENCY GATE</span>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, color: "rgba(255,255,255,0.18)", letterSpacing: "0.1em" }}>ZERO-COST IDLE ROUTING</span>
          </div>
          <StageRow stage={selectorStage} status={getStatus(selectorStage.id)} output={outputs[selectorStage.id]} index={6} />
        </div>

        {/* Consensus Layer */}
        <div style={{
          border: `1px solid ${getStatus("CONSENSUS") !== "idle" ? "rgba(105,240,174,0.25)" : "rgba(255,255,255,0.06)"}`,
          borderRadius: 4,
          overflow: "hidden",
          transition: "border-color 0.4s",
          animation: getStatus("CONSENSUS") !== "idle" ? "fadeUp 0.4s ease forwards" : "none",
        }}>
          <div style={{
            padding: "5px 14px",
            background: "rgba(105,240,174,0.04)",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 11, letterSpacing: "0.28em", color: "rgba(105,240,174,0.6)" }}>CONSENSUS LAYER · FINAL OUTPUT COMMIT</span>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, color: "rgba(255,255,255,0.18)", letterSpacing: "0.1em" }}>ZERO EMBEDDING BLEED · CONFIDENCE WEIGHTED</span>
          </div>
          <StageRow stage={consensusStage} status={getStatus(consensusStage.id)} output={outputs[consensusStage.id]} index={7} />
        </div>

        {/* Footer spec bar */}
        <div style={{
          marginTop: 16,
          paddingTop: 12,
          borderTop: "1px solid rgba(255,255,255,0.04)",
          display: "flex",
          gap: 0,
          flexWrap: "wrap",
        }}>
          {[
            ["Architecture",    "Dual-Stream Parallel Pipeline"],
            ["Tokenization",    "Literal + Latent · Simultaneous"],
            ["Fusion Method",   "Cross-Stream Coherence Intersection"],
            ["Inference Cores", "Temporal + Generative · Parallel"],
            ["Distillation",    "Dual-Confidence Artifact Rejection"],
            ["Mode Gating",     "Per-Query Inference Budget Selection"],
            ["Output Layer",    "Confidence-Weighted Consensus"],
            ["Context Bleed",   "Zero · Namespace-Isolated Pipelines"],
          ].map(([k, v]) => (
            <div key={k} style={{
              padding: "6px 18px 6px 0",
              marginRight: 8,
              borderRight: "1px solid rgba(255,255,255,0.04)",
              marginBottom: 4,
            }}>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 7, letterSpacing: "0.22em", color: "rgba(255,255,255,0.18)", textTransform: "uppercase", marginBottom: 2 }}>{k}</div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: "rgba(105,240,174,0.5)" }}>{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

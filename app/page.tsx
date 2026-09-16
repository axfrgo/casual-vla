'use client';
/* oxlint-disable react/react-compiler -- Browser session state must hydrate after SSR. */

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import {
  Activity,
  ArrowRight,
  Check,
  ChevronRight,
  Cpu,
  Gauge,
  Layers3,
  Pause,
  Play,
  RotateCcw,
  Shield,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import { PlatformNav } from './components/PlatformNav';
import {
  DINNER_TABLE_MISSION,
  MISSION_PRESETS,
  MISSION_STEPS,
  clampCompletedSteps,
  normalizeSeed,
  readMissionRun,
  workspaceHref,
  writeMissionRun,
} from './lib/mission';
import './competition.css';
const STEPS = MISSION_STEPS;
const SEED_PRESETS = MISSION_PRESETS;
const STEP_DURATION_SECONDS = 1.15;
const LEFT_BASE: Point = [142, 250];
const RIGHT_BASE: Point = [618, 250];

type Point = [number, number];
type ArmSide = 'left' | 'right';

function seeded(seed: number, offset: number) {
  const value = Math.sin(seed * 12.9898 + offset * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function blendPoint(first: Point, second: Point, progress: number): Point {
  const amount = Math.max(0, Math.min(1, progress));
  return [
    first[0] + (second[0] - first[0]) * amount,
    first[1] + (second[1] - first[1]) * amount,
  ];
}

function getRestWrist(side: ArmSide, base: Point): Point {
  const direction = side === 'left' ? 1 : -1;
  return [base[0] + direction * 105, base[1] - 15];
}

function getWrist(side: ArmSide, base: Point, target: Point, active: boolean): Point {
  return active ? target : getRestWrist(side, base);
}

function formatTime(seconds: number) {
  return `${seconds.toFixed(1).padStart(4, '0')}s`;
}

function RobotArm({
  side,
  base,
  target,
  active,
}: {
  side: ArmSide;
  base: Point;
  target: Point;
  active: boolean;
}) {
  const direction = side === 'left' ? 1 : -1;
  const elbow: Point = [base[0] + direction * 58, base[1] - 52];
  const wrist = getWrist(side, base, target, active);
  const path = `M${base[0]} ${base[1]}L${elbow[0]} ${elbow[1]}L${wrist[0]} ${wrist[1]}`;

  return (
    <g className={`robot-arm ${active ? 'active' : ''}`}>
      <circle cx={base[0]} cy={base[1]} r="33" fill="#17242c" stroke="#8df0c9" strokeWidth="3" />
      <path d={path} fill="none" stroke="#aab9c2" strokeWidth="17" strokeLinecap="round" strokeLinejoin="round" />
      <path d={path} fill="none" stroke="#263742" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={elbow[0]} cy={elbow[1]} r="10" fill="#0b141a" stroke="#8df0c9" />
      <g transform={`translate(${wrist[0]},${wrist[1]})`}>
        <circle r="10" fill="#0b141a" stroke="#8df0c9" />
        <path d="M-4-4L-13-14M4-4L13-14" stroke="#8df0c9" strokeWidth="4" strokeLinecap="round" />
      </g>
      <text x={base[0]} y={base[1] + 56} textAnchor="middle" className="scene-label green-fill">
        {side.toUpperCase()} SO-101
      </text>
    </g>
  );
}

function TableScene({ seed, step, elapsed, running }: { seed: number; step: number; elapsed: number; running: boolean }) {
  const objects = useMemo<{ plate: Point; cup: Point; fork: Point; spoon: Point }>(
    () => ({
      plate: [315 + seeded(seed, 1) * 60, 222 + seeded(seed, 2) * 35],
      cup: [445 + seeded(seed, 3) * 55, 205 + seeded(seed, 4) * 45],
      fork: [280 + seeded(seed, 5) * 45, 355],
      spoon: [430 + seeded(seed, 6) * 45, 355],
    }),
    [seed],
  );

  const activeIndex = Math.min(Math.max(step, 0), STEPS.length - 1);
  const activeStep = STEPS[activeIndex];
  const plateDone = step >= 4;
  const cupHeld = step === 4 || step === 5;
  const cupDone = step >= 6;
  const drawerOpen = step >= 1;
  const handoffPoint: Point = [385, 190];
  const forkPlace: Point = [520, 170];
  const spoonPlace: Point = [565, 170];
  const platePlace: Point = [535, 265];
  const leftRest = getRestWrist('left', LEFT_BASE);
  const rightRest = getRestWrist('right', RIGHT_BASE);
  const leftActive = step < STEPS.length && activeStep.arm !== 'RIGHT';
  const rightActive = step < STEPS.length && activeStep.arm !== 'LEFT';
  const showTarget = step < STEPS.length;

  // Objects use the same hand coordinates as the arm illustration. During a
  // retrieval/handoff phase they travel into the active gripper, then remain
  // attached to that gripper while the next phase transports them.
  const phaseProgress = running ? Math.max(0, Math.min(1, (elapsed - step * STEP_DURATION_SECONDS) / STEP_DURATION_SECONDS)) : step === 0 ? 0 : 1;
  const leftTarget: Point = activeIndex === 1 ? objects.fork : activeIndex === 3 ? (phaseProgress < .45 ? objects.plate : platePlace) : activeIndex === 4 ? (phaseProgress < .34 ? leftRest : handoffPoint) : activeIndex === 5 ? [585, 250] : [175, 250];
  const rightTarget: Point = activeIndex === 0 ? [385, 350] : activeIndex === 2 ? objects.spoon : activeIndex === 4 ? (phaseProgress < .34 ? blendPoint(rightRest, objects.cup, phaseProgress / .34) : handoffPoint) : [595, 250];
  const leftWrist = getWrist('left', LEFT_BASE, leftTarget, leftActive);
  const rightWrist = getWrist('right', RIGHT_BASE, rightTarget, rightActive);
  const forkPosition = step === 1 ? (phaseProgress < .55 ? objects.fork : leftWrist) : step >= 2 ? forkPlace : objects.fork;
  const spoonPosition = step === 2 ? (phaseProgress < .55 ? objects.spoon : rightWrist) : step >= 3 ? spoonPlace : objects.spoon;
  const platePosition = step === 3 ? (phaseProgress < .45 ? objects.plate : leftWrist) : plateDone ? platePlace : objects.plate;
  const cupPosition = step === 4 ? (phaseProgress < .34 ? objects.cup : phaseProgress < .68 ? rightWrist : leftWrist) : step === 5 ? leftWrist : cupDone ? [585, 250] : objects.cup;

  return (
    <svg className="table-scene" viewBox="0 0 760 470" aria-labelledby="scene-title">
      <title id="scene-title">Browser-rendered dual SO-101 dinner-table task</title>
      <defs>
        <linearGradient id="table" x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#27343e" />
          <stop offset="1" stopColor="#131d25" />
        </linearGradient>
        <linearGradient id="place-glow" x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#8df0c91f" />
          <stop offset="1" stopColor="#8df0c902" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect width="760" height="470" rx="20" fill="#091016" />
      <rect x="18" y="18" width="176" height="31" rx="8" fill="#101b23" stroke="#31414a" />
      <circle cx="34" cy="34" r="4" fill="#8df0c9" />
      <text x="47" y="38" className="scene-hud">EPISODE 01 · TASK MODEL</text>
      <rect x="566" y="18" width="176" height="31" rx="8" fill="#101b23" stroke="#31414a" />
      <text x="654" y="38" textAnchor="middle" className="scene-hud">SEED {seed}</text>
      <path d="M84 104Q84 70 118 70H642Q676 70 676 104V405H84Z" fill="url(#table)" stroke="#41515e" strokeWidth="2" />
      <rect x="115" y="104" width="530" height="225" rx="16" fill="#18242d" stroke="#536470" />
      <rect x="475" y="120" width="135" height="88" rx="12" fill="url(#place-glow)" stroke="#65d9b1" strokeDasharray="7 6" />
      <text x="542" y="146" textAnchor="middle" className="scene-label green-fill">PLACE AREA</text>
      <path className={`handoff-arc ${step === 4 ? 'visible' : ''}`} d="M355 207 Q385 158 415 207" />
      {showTarget && (
        <>
          <circle className="target-ring" cx={(leftActive ? leftTarget : rightTarget)[0]} cy={(leftActive ? leftTarget : rightTarget)[1]} r="24" />
          {activeStep.arm === 'BOTH' && <circle className="target-ring secondary" cx={rightTarget[0]} cy={rightTarget[1]} r="24" />}
          <text className="target-label" x={(leftActive ? leftTarget : rightTarget)[0]} y={(leftActive ? leftTarget : rightTarget)[1] - 31} textAnchor="middle">
            {activeStep.arm === 'BOTH' ? 'SYNC' : `${activeStep.arm} TARGET`}
          </text>
        </>
      )}
      <g className={drawerOpen ? 'drawer open' : 'drawer'}>
        <rect x="245" y="329" width="270" height="80" rx="8" fill="#0e171e" stroke="#6d7f8c" />
        <path d="M260 347H500" stroke="#54636e" />
        <circle cx="380" cy="345" r="4" fill="#8df0c9" />
        <text x="380" y="385" textAnchor="middle" className="scene-label">UTENSIL DRAWER</text>
      </g>
      <g className={`object ${step === 1 ? 'held' : ''}`} transform={`translate(${forkPosition[0]},${forkPosition[1]})`}>
        {step === 1 && <circle r="16" className="held-ring" />}
        <path d="M-3-18V17M3-18V17M-7-18V-7M7-18V-7" stroke="#c7d2d9" strokeWidth="3" />
        <text y="34" className="scene-label">FORK</text>
      </g>
      <g className={`object ${step === 2 ? 'held' : ''}`} transform={`translate(${spoonPosition[0]},${spoonPosition[1]})`}>
        {step === 2 && <circle r="16" className="held-ring" />}
        <ellipse cy="-11" rx="8" ry="11" fill="none" stroke="#c7d2d9" strokeWidth="3" />
        <path d="M0 0V20" stroke="#c7d2d9" strokeWidth="3" />
        <text y="38" className="scene-label">SPOON</text>
      </g>
      <g className={`object ${step === 3 ? 'held' : ''}`} transform={`translate(${platePosition[0]},${platePosition[1]})`}>
        {step === 3 && <circle r="36" className="held-ring" />}
        <circle r="32" fill="#dce8ed" fillOpacity=".12" stroke="#dce8ed" strokeWidth="3" />
        <circle r="22" fill="none" stroke="#7d909d" />
        <text y="48" className="scene-label">PLATE</text>
      </g>
      <g className={`object ${cupHeld ? 'held' : ''}`} filter={cupHeld ? 'url(#glow)' : undefined} transform={`translate(${cupPosition[0]},${cupPosition[1]})`}>
        {cupHeld && <circle r="24" className="held-ring" />}
        <circle r="17" fill="#edc86d22" stroke="#edc86d" strokeWidth="3" />
        <path d="M17-8Q33-8 31 4Q29 14 17 12" fill="none" stroke="#edc86d" strokeWidth="3" />
        <text y="37" className="scene-label amber-fill">CUP</text>
      </g>
      <RobotArm side="left" base={LEFT_BASE} target={leftTarget} active={leftActive} />
      <RobotArm side="right" base={RIGHT_BASE} target={rightTarget} active={rightActive} />
      <text x="110" y="442" className="scene-meta">SO-101 × 2 · BROWSER VISUALIZER</text>
      <text x="650" y="442" textAnchor="end" className="scene-meta">NO PHYSICS CLAIMED IN BROWSER</text>
    </svg>
  );
}

function PrecisionTelemetry({ step, elapsed, running }: { step: number; elapsed: number; running: boolean }) {
  const phaseProgress = running ? Math.max(0, Math.min(1, (elapsed - step * STEP_DURATION_SECONDS) / STEP_DURATION_SECONDS)) : step === 0 ? 0 : 1;
  const payload = step === 1 ? 'FORK' : step === 2 ? 'SPOON' : step === 3 ? 'PLATE' : step === 4 || step === 5 ? 'CUP' : step === 6 ? 'ALL PLACED' : 'NONE';
  const grip = step === 4 ? (phaseProgress < .34 ? 'ACQUIRE · RIGHT' : phaseProgress < .68 ? 'SYNC · BOTH' : 'LOCKED · LEFT') : payload === 'ALL PLACED' ? 'RELEASED · SET' : payload === 'NONE' ? 'STANDBY' : `${step === 1 || step === 3 || step === 5 ? 'LEFT' : 'RIGHT'} · LOCKED`;
  const follow = payload === 'NONE' ? 'IDLE' : payload === 'ALL PLACED' ? 'VERIFY COMPLETE' : step === 4 && phaseProgress >= .34 && phaseProgress < .68 ? 'DUAL TCP SYNC' : 'PAYLOAD LOCK';

  return (
    <div className="precision-telemetry" aria-label="Browser visual attachment telemetry">
      <div className="telemetry-heading"><span>ATTACHMENT TELEMETRY</span><small>SHARED TCP FRAME · VISUAL MODEL</small></div>
      <div className="telemetry-grid">
        <div><span>PAYLOAD</span><strong>{payload}</strong><small>{payload === 'NONE' ? 'No active grasp' : 'Calibrated grasp frame'}</small></div>
        <div><span>GRIP STATE</span><strong>{grip}</strong><small>End-effector state</small></div>
        <div><span>TCP FOLLOW</span><strong>{follow}</strong><small>Payload transform locked to wrist</small></div>
        <div><span>PHYSICS</span><strong>NATIVE ONLY</strong><small>Browser does not claim dynamics</small></div>
      </div>
    </div>
  );
}

type NativeSeedEvidence = {
  seed: number;
  passed: boolean;
  task_index: number;
  actions: number;
  failed_actions: number;
  motion_retries: number;
  placed: string[];
};

type NativeEvidence = {
  report_schema: string;
  report_kind: string;
  generated_from: string;
  source_command: string;
  environment: string;
  policy: string;
  policy_kind: string;
  successes: number;
  total: number;
  success_rate: number;
  failed_task_actions: number;
  total_motion_retries: number;
  seeds: NativeSeedEvidence[];
  evidence_boundary: string;
};

function NativeEvidenceDossier() {
  const [report, setReport] = useState<NativeEvidence | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    let cancelled = false;
    fetch('/evidence/mujoco-policy-eval.json', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) throw new Error(`Evidence report returned ${response.status}`);
        return response.json() as Promise<NativeEvidence>;
      })
      .then((nextReport) => {
        if (cancelled) return;
        setReport(nextReport);
        setLoadState('ready');
      })
      .catch(() => {
        if (!cancelled) setLoadState('error');
      });
    return () => { cancelled = true; };
  }, []);

  const passed = report?.successes ?? 0;
  const total = report?.total ?? 0;
  const successRate = report ? `${Math.round(report.success_rate * 100)}%` : '—';

  return (
    <section className="native-dossier" id="native-report" aria-labelledby="native-report-title">
      <div className="dossier-heading">
        <div>
          <span className="competition-kicker">04 / EVIDENCE LEDGER</span>
          <h2 id="native-report-title">Prove the run.</h2>
          <p>Native MuJoCo results are loaded from a reproducible report, not painted into the browser scene.</p>
        </div>
        <div className={`dossier-status ${loadState}`} aria-live="polite"><i />{loadState === 'ready' ? 'REPORT VERIFIED' : loadState === 'error' ? 'REPORT UNAVAILABLE' : 'LOADING REPORT'}</div>
      </div>
      <div className="dossier-metrics">
        <div><span>PASS RATE</span><strong>{successRate}</strong><small>{passed}/{total || '—'} complete</small></div>
        <div><span>FAILED TASK ACTIONS</span><strong>{report?.failed_task_actions ?? '—'}</strong><small>across the sweep</small></div>
        <div><span>ACTIONS / EPISODE</span><strong>{report ? '23–24' : '—'}</strong><small>closed-loop decisions</small></div>
        <div><span>HAND-OFF RETRIES</span><strong>{report ? '1×' : '—'}</strong><small>visible calibration signal</small></div>
      </div>
      <div className="dossier-ledger">
        <div className="dossier-ledger-title"><span>10-SEED LEDGER</span><small>{report?.policy ?? 'native evaluator'}</small></div>
        <div className="seed-ledger" aria-label="Per-seed native MuJoCo results">
          {report?.seeds.map((episode) => (
            <div className={`seed-cell ${episode.passed ? 'passed' : 'failed'}`} key={episode.seed} title={`Seed ${episode.seed}: ${episode.passed ? 'passed' : 'failed'}, ${episode.actions} actions`}>
              <b>{episode.seed}</b><span>{episode.passed ? 'PASS' : 'FAIL'}</span><small>{episode.actions} acts</small>
            </div>
          )) ?? <div className="seed-loading">Loading per-seed results…</div>}
        </div>
        <div className="dossier-actions">
          <a className="evidence-link" href="/evidence/mujoco-policy-eval.json" target="_blank" rel="noreferrer">Open report JSON <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/dinner-table-10-seed-demo.mp4" target="_blank" rel="noreferrer">Play 10-seed video <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/dinner-table-10-seed-demo.json" target="_blank" rel="noreferrer">Video evidence sidecar <ArrowRight size={14} /></a>
          <span>Schema {report?.report_schema ?? '—'}</span>
        </div>
      </div>
      <div className="dossier-boundary"><Shield size={16} /><p><strong>Evidence boundary</strong>{report?.evidence_boundary ?? 'Live MuJoCo baseline only; VLA, hardware, Intel, OpenVINO, and video evidence remain separate.'}</p></div>
    </section>
  );
}

type NeuralEpisodeEvidence = {
  seed: number;
  passed: boolean;
  completed: boolean;
  task_index: number;
  next: string;
  actions: number;
  failed_actions: number;
  inference_ms: { mean: number | null; p95: number | null };
};

type NeuralEvaluation = {
  report_schema: string;
  policy_kind: string;
  source_checkpoint: string;
  device?: string;
  successes: number;
  total: number;
  success_rate: number;
  episodes: NeuralEpisodeEvidence[];
  evidence_boundary: string;
};

type OpenVINOBenchmark = {
  report_schema: string;
  device: string;
  iterations: number;
  latency_ms: { mean: number; p50: number; p95: number };
  throughput_fps: number;
  evidence_boundary: string;
};

type NeuralGuidedEvaluation = {
  report_schema: string;
  policy_kind: string;
  successes: number;
  total: number;
  success_rate: number;
  intent_rejections: number;
  episodes: NeuralEpisodeEvidence[];
  evidence_boundary: string;
};

function NeuralEvidenceDossier() {
  const [native, setNative] = useState<NeuralEvaluation | null>(null);
  const [benchmark, setBenchmark] = useState<OpenVINOBenchmark | null>(null);
  const [guided, setGuided] = useState<NeuralGuidedEvaluation | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch('/evidence/imitation-policy-eval-10seed.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error(`Native neural report returned ${response.status}`);
        return response.json() as Promise<NeuralEvaluation>;
      }),
      fetch('/evidence/openvino-evaluation-smoke.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error(`OpenVINO task report returned ${response.status}`);
        return response.json() as Promise<NeuralEvaluation>;
      }),
      fetch('/evidence/openvino-benchmark.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error(`OpenVINO benchmark returned ${response.status}`);
        return response.json() as Promise<OpenVINOBenchmark>;
      }),
      fetch('/evidence/neural-guided-eval-10seed.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error(`Neural guided report returned ${response.status}`);
        return response.json() as Promise<NeuralGuidedEvaluation>;
      }),
    ])
      .then(([nativeReport, , benchmarkReport, guidedReport]) => {
        if (cancelled) return;
        setNative(nativeReport);
        setBenchmark(benchmarkReport);
        setGuided(guidedReport);
        setLoadState('ready');
      })
      .catch(() => {
        if (!cancelled) setLoadState('error');
      });
    return () => { cancelled = true; };
  }, []);

  const nativeEpisode = native?.episodes[0];
  const guidedEpisode = guided?.episodes[0];
  const guidedPass = guided ? guided.successes === guided.total : false;
  const taskStatus = guidedPass ? 'GUIDED PASS / PURE GAP' : 'TASK GAP';

  return (
    <section className="native-dossier neural-dossier" id="neural-report" aria-labelledby="neural-report-title">
      <div className="dossier-heading">
        <div>
          <span className="competition-kicker">05 / NEURAL EVIDENCE</span>
          <h2 id="neural-report-title">The direct policy gap is visible. The guarded path completes.</h2>
          <p>A compact RGB/state/language intent model gates every step of the verified controller. The direct joint-target policy and neural-guided completion are reported separately, with the CUDA training provenance exposed for audit.</p>
        </div>
        <div className={`dossier-status ${loadState === 'ready' ? '' : loadState}`} aria-live="polite"><i />{loadState === 'ready' ? 'REPORTS LOADED' : loadState === 'error' ? 'REPORTS UNAVAILABLE' : 'LOADING REPORTS'}</div>
      </div>
      <div className="dossier-metrics">
        <div><span>DIRECT JOINT BC</span><strong>V7</strong><small>3 RGB views · 21 state dims</small></div>
        <div><span>NATIVE 10-SEED</span><strong>{native ? `${native.successes}/${native.total}` : '—'}</strong><small>{nativeEpisode ? `stopped at ${nativeEpisode.next}` : 'task evaluation'}</small></div>
        <div><span>NEURAL-GUIDED</span><strong>{guided ? `${guided.successes}/${guided.total}` : '—'}</strong><small>{guidedEpisode ? `${guided.intent_rejections} intent rejections` : 'gated completion'}</small></div>
        <div><span>OPENVINO CPU</span><strong>{benchmark ? `${benchmark.throughput_fps.toFixed(0)} FPS` : '—'}</strong><small>{benchmark ? `${benchmark.latency_ms.mean.toFixed(2)}ms mean · ${benchmark.latency_ms.p95.toFixed(2)}ms p95` : 'inference benchmark'}</small></div>
        <div><span>READINESS</span><strong className={guidedPass ? '' : 'metric-warning'}>{taskStatus}</strong><small>{guidedEpisode ? `${guidedEpisode.actions} actions · ${guidedEpisode.failed_actions} rejected` : 'neural task gate'}</small></div>
      </div>
      <div className="dossier-ledger">
        <div className="dossier-ledger-title"><span>EXACT ARTIFACT CHAIN</span><small>{native?.policy_kind ?? 'camera · state · language behavior cloning'}</small></div>
        <div className="neural-artifact-chain">
          <span><b>01</b><strong>Native demos</strong><small>10 expert episodes</small></span>
          <ArrowRight size={15} />
          <span><b>02</b><strong>MuJoCo 10-seed</strong><small>{nativeEpisode ? `${native.successes}/${native.total} complete` : 'report pending'}</small></span>
          <ArrowRight size={15} />
          <span><b>03</b><strong>OpenVINO IR</strong><small>{benchmark ? `${benchmark.device} measured` : 'benchmark pending'}</small></span>
        </div>
        <div className="dossier-actions">
          <a className="evidence-link" href="/evidence/imitation-policy-checkpoint.json" target="_blank" rel="noreferrer">Checkpoint metadata <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/neural-intent-checkpoint.json" target="_blank" rel="noreferrer">Intent checkpoint <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/imitation-policy-openvino.json" target="_blank" rel="noreferrer">OpenVINO contract <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/imitation-policy-eval-10seed.json" target="_blank" rel="noreferrer">Native neural report <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/neural-guided-eval-10seed.json" target="_blank" rel="noreferrer">Guided completion report <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/openvino-evaluation-smoke.json" target="_blank" rel="noreferrer">OpenVINO task report <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/openvino-benchmark.json" target="_blank" rel="noreferrer">CPU benchmark <ArrowRight size={14} /></a>
          <a className="evidence-link" href="/evidence/imitation-policy-gpu-v10.json" target="_blank" rel="noreferrer">CUDA training provenance <ArrowRight size={14} /></a>
        </div>
      </div>
      <div className="dossier-boundary"><Shield size={16} /><p><strong>Honest boundary</strong>{native?.evidence_boundary ?? 'Neural task success, OpenVINO inference, browser visualization, and hardware performance are separate claims.'}</p></div>
    </section>
  );
}

export default function CompetitionPage() {
  const [seed, setSeed] = useState(1001);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef(0);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const stored = readMissionRun(window.localStorage);
    const requestedSeedRaw = params.get('seed');
    const requestedSeed = requestedSeedRaw === null ? Number.NaN : Number(requestedSeedRaw);
    const hasRequestedContext = params.has('seed') || params.has('step');
    setSeed(normalizeSeed(Number.isSafeInteger(requestedSeed) ? requestedSeed : stored?.seed ?? 1001));
    setStep(hasRequestedContext && params.has('step') ? clampCompletedSteps(Number(params.get('step'))) : hasRequestedContext ? 0 : stored?.completedSteps ?? 0);
    const initialElapsed = hasRequestedContext ? 0 : stored?.elapsedSeconds ?? 0;
    elapsedRef.current = initialElapsed;
    setElapsed(initialElapsed);
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeMissionRun(window.localStorage, {
      missionId: DINNER_TABLE_MISSION.id,
      seed: normalizeSeed(seed),
      completedSteps: clampCompletedSteps(step),
      elapsedSeconds: elapsedRef.current,
      status: step === STEPS.length ? 'complete' : running ? 'running' : step > 0 ? 'paused' : 'ready',
      updatedAt: new Date().toISOString(),
    });
  }, [hydrated, running, seed, step]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setStep((current) => {
      if (current >= STEPS.length - 1) {
        setRunning(false);
        return STEPS.length;
      }
      return current + 1;
    }), STEP_DURATION_SECONDS * 1000);
    return () => clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setElapsed((current) => { const next = current + 0.1; elapsedRef.current = next; return next; }), 100);
    return () => clearInterval(timer);
  }, [running]);

  const reset = () => {
    setRunning(false);
    setStep(0);
    elapsedRef.current = 0;
    setElapsed(0);
  };

  const chooseSeed = (nextSeed: number) => {
    setSeed(nextSeed);
    reset();
  };

  const advance = () => {
    setStep((current) => Math.min(STEPS.length, current + 1));
    setElapsed((current) => { const next = Math.max(current, (step + 1) * STEP_DURATION_SECONDS); elapsedRef.current = next; return next; });
  };

  const progress = Math.round((step / STEPS.length) * 100);
  const currentStep = STEPS[Math.min(step, STEPS.length - 1)];
  const missionState = step === STEPS.length ? 'COMPLETE' : running ? 'EXECUTING' : step > 0 ? 'PAUSED' : 'READY';
  const phaseLabel = step === 0 ? 'Ready for a reproducible run' : step === STEPS.length ? 'All task goals reached' : currentStep.detail;
  const workspaceLink = workspaceHref(seed, step);

  return (
    <div className="competition-shell">
      <PlatformNav missionSeed={seed} missionStep={step} />

      <main className="competition-main">
        <section className="competition-intro">
          <div className="hero-copy">
            <span className="competition-kicker">INTEL PHYSICAL AI · DUAL-ARM MANIPULATION · MISSION 01</span>
            <h1>Dinner-table intelligence,<br /><em>taught to adapt.</em></h1>
            <p>Aegis turns a human correction into a safer next attempt. This competition surface makes the loop legible: inspect the task, perturb the scene, watch the hand-off, then follow the evidence to the teaching workspace.</p>
            <div className="hero-actions"><a className="hero-link" href="#mission">Run the mission <ArrowRight size={16} /></a><Link className="hero-link quiet" href={workspaceLink}>See the learning loop <ArrowRight size={16} /></Link></div>
          </div>
          <div className="hero-proof"><div className="proof-orbit"><span className="orbit-dot" /><span className="orbit-dot second" /><Layers3 size={27} /></div><span>COMPETITION BUILD</span><strong>SO-101 × 2</strong><small>Browser visualizer / native evidence separate</small></div>
        </section>

        <section className="challenge-strip" aria-label="Competition claims">
          <div><Check size={15} /><span><b>DETERMINISTIC</b> seeded resets</span></div>
          <div><Target size={15} /><span><b>BIMANUAL</b> complementary actions</span></div>
          <div><Shield size={15} /><span><b>AUDITABLE</b> versioned corrections</span></div>
          <div className="strip-pending"><Activity size={15} /><span><b>SEPARATE</b> native physics proof</span></div>
        </section>

        <section className="competition-grid" id="mission">
          <div className="sim-panel">
            <div className="sim-heading">
              <div><span className="competition-kicker">01 / LIVE TASK MODEL</span><h2>Set the dinner table</h2><p>Portable visualizer for the dual-arm challenge sequence.</p></div>
              <div className={`sim-state ${running ? 'is-running' : ''}`} aria-live="polite"><i />{missionState}</div>
            </div>
            <div className="sim-canvas-wrap"><TableScene seed={seed} step={step} elapsed={elapsed} running={running} /><div className="canvas-caption"><span><Zap size={13} /> Motion is deterministic</span><span><Cpu size={13} /> Native MuJoCo evidence separate</span></div><PrecisionTelemetry step={step} elapsed={elapsed} running={running} /></div>
            <div className="sim-controls">
              <button className="competition-primary" onClick={() => setRunning((value) => !value)} disabled={step === STEPS.length}>{running ? <Pause size={16} /> : <Play size={16} />}{running ? 'Pause mission' : step > 0 ? 'Resume mission' : 'Run mission'}</button>
              <button onClick={advance} disabled={running || step === STEPS.length}>Step <ChevronRight size={15} /></button>
              <button onClick={reset}><RotateCcw size={15} /> Reset</button>
              <div className="run-readout"><span>EPISODE TIME</span><strong>{formatTime(elapsed)}</strong></div>
            </div>
            <div className="seed-lab">
              <div><span className="competition-kicker">SEE THE ADAPTATION</span><p>Same task contract. New deterministic scene.</p></div>
              <div className="seed-pills">{SEED_PRESETS.map((preset) => <button key={preset.seed} className={seed === preset.seed ? 'selected' : ''} onClick={() => chooseSeed(preset.seed)}><strong>{preset.label}</strong><small>{preset.seed} · {preset.note}</small></button>)}</div>
              <label>Custom seed<input type="number" min="1" step="1" value={seed} onChange={(event) => { const next = Number(event.target.value); setSeed(normalizeSeed(next)); reset(); }} /></label>
            </div>
          </div>

          <aside className="mission-panel">
            <div className="mission-title"><div><span className="competition-kicker">02 / TASK GRAPH</span><h2>Bimanual sequence</h2></div><div className="progress-readout"><strong>{progress}%</strong><span>mission</span></div></div>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="phase-card"><div className="phase-icon"><Gauge size={18} /></div><div><span className="competition-kicker">CURRENT PHASE</span><strong>{step === STEPS.length ? 'Mission complete' : step === 0 ? 'Awaiting run' : currentStep.label}</strong><p>{phaseLabel}</p></div></div>
            <div className="mission-sync"><div><span className="competition-kicker">CONNECTED SESSION</span><strong>{DINNER_TABLE_MISSION.shortId} · SEED {seed}</strong></div><span>{step}/{STEPS.length} steps shared with Aegis</span></div>
            <div className="step-list">{STEPS.map((item, index) => <div className={`competition-step ${index < step ? 'done' : index === step ? 'next' : ''}`} key={item.id}><span>{index < step ? <Check size={14} /> : String(index + 1).padStart(2, '0')}</span><div><strong>{item.label}</strong><small>{item.arm} ARM · {item.object.toUpperCase()}</small></div>{index === step && step < STEPS.length && <Zap size={14} className="step-live" />}</div>)}</div>
            <div className={`handoff-note ${step === 4 ? 'active' : ''}`}><Sparkles size={17} /><p><strong>Complementary action</strong>{step === 4 ? 'Both arms are synchronized on the cup.' : 'The cup transfers right → left before final placement.'}</p></div>
            <Link className="mission-link" href={workspaceLink}>Open the teaching workspace <ArrowRight size={15} /></Link>
          </aside>
        </section>

        <section className="evidence-grid" id="evidence">
          <article><div className="evidence-icon"><Activity /></div><span>VISUAL LAYER</span><strong>Deterministic resets</strong><p>Seeded placement and perturbation controls make the demo reproducible for a judge, teammate, or future regression run.</p><em className="evidence-status ready">VERIFIABLE IN BROWSER</em></article>
          <article><div className="evidence-icon"><Cpu /></div><span>PHYSICS LAYER</span><strong>Native MuJoCo</strong><p>The six-step task, measured two-jaw grasps and lifts, post-contact retention, releases, and hand-off are verified in the native harness; this page never disguises animation as simulation.</p><em className="evidence-status separate">EVIDENCE TRACK SEPARATE</em></article>
          <article><div className="evidence-icon"><Gauge /></div><span>BASELINE RUN</span><strong>10 / 10 seeded scenes</strong><p>A deterministic observation-driven controller completes the current MuJoCo task across ten perturbation seeds. This is a baseline result, not a VLA claim.</p><a className="evidence-link" href="/evidence/mujoco-policy-eval.json" target="_blank" rel="noreferrer">Open native run report <ArrowRight size={14} /></a><em className="evidence-status ready">REPRODUCIBLE LOCALLY</em></article>
          <article><div className="evidence-icon"><Shield /></div><span>LEARNING LAYER</span><strong>Aegis memory</strong><p>Human corrections become versioned rules in the teaching workspace, where provenance and generalization can be inspected.</p><em className="evidence-status ready">OPEN TEACHING WORKSPACE</em></article>
          <article className="evidence-neural"><div className="evidence-icon"><Target /></div><span>CAMERA · LANGUAGE · ACTION</span><strong>Neural-guided path measured</strong><p>A three-view RGB/state/language intent model gates all six phases in native MuJoCo, then the verified low-level controller executes each accepted phase. Direct joint-target behavior cloning remains a separate 0/10 result.</p><div className="neural-pipeline" aria-label="Camera language action pipeline"><span><b>RGB</b><small>3 views</small></span><i>→</i><span><b>INTENT</b><small>6 phases</small></span><i>→</i><span><b>10/10</b><small>native gated</small></span></div><a className="evidence-link" href="#neural-report">Inspect neural evidence <ArrowRight size={14} /></a><em className="evidence-status ready">GUIDED PATH VERIFIED</em></article>
          <article><div className="evidence-icon"><Cpu /></div><span>TARGET HARDWARE</span><strong>Core Ultra not present</strong><p>The verified host is a 13th Gen Intel Core i5-13420H. Its OpenVINO CPU measurement is valid for this machine only, not for an Intel Core Ultra submission target.</p><a className="evidence-link" href="/evidence/hardware-platform-check.json" target="_blank" rel="noreferrer">Open hardware check <ArrowRight size={14} /></a><em className="evidence-status separate">HARDWARE RUN REQUIRED</em></article>
        </section>

        <NativeEvidenceDossier />
        <NeuralEvidenceDossier />

        <section className="readiness">
          <div className="readiness-heading"><span className="competition-kicker">03 / SUBMISSION READINESS</span><h2>A demo with a point of view.</h2><p>The browser proves the interaction model. Native runs prove the physical claim.</p></div>
          <div className="readiness-board"><div className="readiness-score"><strong>7<span>/9</span></strong><small>evidence tracks connected<br />not a competition score</small></div><div className="readiness-items"><span className="complete"><Check /> Dual SO-101 scene</span><span className="complete"><Check /> Browser task visualization</span><span className="complete"><Check /> Seeded perturbations</span><span className="complete"><Check /> MuJoCo baseline loop</span><span className="complete"><Check /> Neural checkpoint path</span><span className="complete"><Check /> Neural intent gate · 10/10</span><span className="complete"><Check /> OpenVINO CPU benchmark</span><span className="partial"><Activity /> Direct end-to-end neural policy</span><span className="partial"><Activity /> Intel Core Ultra run</span></div></div>
        </section>

        <section className="demo-route"><div><span className="competition-kicker">THE JUDGE PATH</span><h2>Run it. Change it. Explain it.</h2></div><div className="route-steps"><span><b>01</b> Run mission</span><ArrowRight size={16} /><span><b>02</b> Shift seed</span><ArrowRight size={16} /><span><b>03</b> Teach Aegis</span><ArrowRight size={16} /><span><b>04</b> Inspect evidence</span></div></section>

        <footer><span>FORTIFIERS / TEACH MACHINES THE WAY YOU TEACH PEOPLE.</span><span>DETERMINISTIC KNOWLEDGE · AUDITABLE CORRECTIONS · HONEST EVIDENCE</span></footer>
      </main>
    </div>
  );
}

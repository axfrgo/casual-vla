'use client';

import { useEffect, useMemo, useState } from 'react';
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
import './competition.css';

const STEPS = [
  { id: 'open_drawer', label: 'Open utensil drawer', arm: 'RIGHT', object: 'drawer', detail: 'Expose the tool set without disturbing the table.' },
  { id: 'retrieve_fork', label: 'Retrieve fork', arm: 'LEFT', object: 'fork', detail: 'Left arm grounds the seeded fork state.' },
  { id: 'retrieve_spoon', label: 'Retrieve spoon', arm: 'RIGHT', object: 'spoon', detail: 'Right arm completes the utensil pair.' },
  { id: 'place_plate', label: 'Place plate', arm: 'LEFT', object: 'plate', detail: 'Plate moves into the marked place area.' },
  { id: 'handoff_cup', label: 'Hand off cup', arm: 'BOTH', object: 'cup', detail: 'A coordinated right → left transfer keeps the plan moving.' },
  { id: 'place_cup', label: 'Place cup', arm: 'LEFT', object: 'cup', detail: 'Left arm finishes the setting.' },
] as const;

const SEED_PRESETS = [
  { seed: 1001, label: 'BASE', note: 'balanced layout' },
  { seed: 2026, label: 'SHIFTED', note: 'cup + plate drift' },
  { seed: 4120, label: 'TIGHT', note: 'compact reach' },
] as const;

function seeded(seed: number, offset: number) {
  const value = Math.sin(seed * 12.9898 + offset * 78.233) * 43758.5453;
  return value - Math.floor(value);
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
  side: 'left' | 'right';
  base: number[];
  target: number[];
  active: boolean;
}) {
  const direction = side === 'left' ? 1 : -1;
  const elbow = [base[0] + direction * 58, base[1] - 52];
  const wrist = active ? target : [base[0] + direction * 105, base[1] - 15];
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

function TableScene({ seed, step }: { seed: number; step: number }) {
  const objects = useMemo(
    () => ({
      plate: [315 + seeded(seed, 1) * 60, 222 + seeded(seed, 2) * 35],
      cup: [445 + seeded(seed, 3) * 55, 205 + seeded(seed, 4) * 45],
      fork: [280 + seeded(seed, 5) * 45, 355],
      spoon: [430 + seeded(seed, 6) * 45, 355],
    }),
    [seed],
  );

  const plateDone = step > 3;
  const cupHeld = step === 5;
  const cupDone = step > 5;
  const forkDone = step > 1;
  const spoonDone = step > 2;
  const drawerOpen = step > 0;
  const leftTarget = step === 2 ? objects.fork : step === 4 ? objects.plate : step >= 5 ? (cupDone ? [585, 250] : [385, 190]) : [175, 250];
  const rightTarget = step === 1 ? [385, 350] : step === 3 ? objects.spoon : step === 5 ? [385, 190] : [595, 250];
  const activeStep = STEPS[Math.min(Math.max(step - 1, 0), STEPS.length - 1)];
  const leftActive = step > 0 && activeStep.arm !== 'RIGHT';
  const rightActive = step > 0 && activeStep.arm !== 'LEFT';
  const showTarget = step > 0 && step < STEPS.length;

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
      <path className={`handoff-arc ${step === 5 ? 'visible' : ''}`} d="M355 207 Q385 158 415 207" />
      {showTarget && (
        <>
          <circle className="target-ring" cx={(leftActive ? leftTarget : rightTarget)[0]} cy={(leftActive ? leftTarget : rightTarget)[1]} r="24" />
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
      <g className="object" transform={`translate(${forkDone ? 520 : objects.fork[0]},${forkDone ? 170 : objects.fork[1]})`}>
        <path d="M-3-18V17M3-18V17M-7-18V-7M7-18V-7" stroke="#c7d2d9" strokeWidth="3" />
        <text y="34" className="scene-label">FORK</text>
      </g>
      <g className="object" transform={`translate(${spoonDone ? 565 : objects.spoon[0]},${spoonDone ? 170 : objects.spoon[1]})`}>
        <ellipse cy="-11" rx="8" ry="11" fill="none" stroke="#c7d2d9" strokeWidth="3" />
        <path d="M0 0V20" stroke="#c7d2d9" strokeWidth="3" />
        <text y="38" className="scene-label">SPOON</text>
      </g>
      <g className="object" transform={`translate(${plateDone ? 535 : objects.plate[0]},${plateDone ? 265 : objects.plate[1]})`}>
        <circle r="32" fill="#dce8ed" fillOpacity=".12" stroke="#dce8ed" strokeWidth="3" />
        <circle r="22" fill="none" stroke="#7d909d" />
        <text y="48" className="scene-label">PLATE</text>
      </g>
      <g className="object" filter={cupHeld ? 'url(#glow)' : undefined} transform={`translate(${cupDone ? 585 : cupHeld ? 385 : objects.cup[0]},${cupDone ? 250 : cupHeld ? 190 : objects.cup[1]})`}>
        <circle r="17" fill="#edc86d22" stroke="#edc86d" strokeWidth="3" />
        <path d="M17-8Q33-8 31 4Q29 14 17 12" fill="none" stroke="#edc86d" strokeWidth="3" />
        <text y="37" className="scene-label amber-fill">CUP</text>
      </g>
      <RobotArm side="left" base={[142, 250]} target={leftTarget} active={leftActive} />
      <RobotArm side="right" base={[618, 250]} target={rightTarget} active={rightActive} />
      <text x="110" y="442" className="scene-meta">SO-101 × 2 · BROWSER VISUALIZER</text>
      <text x="650" y="442" textAnchor="end" className="scene-meta">NO PHYSICS CLAIMED IN BROWSER</text>
    </svg>
  );
}

export default function CompetitionPage() {
  const [seed, setSeed] = useState(1001);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setStep((current) => {
      if (current >= STEPS.length - 1) {
        setRunning(false);
        return STEPS.length;
      }
      return current + 1;
    }), 1150);
    return () => clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setElapsed((current) => current + 0.1), 100);
    return () => clearInterval(timer);
  }, [running]);

  const reset = () => {
    setRunning(false);
    setStep(0);
    setElapsed(0);
  };

  const chooseSeed = (nextSeed: number) => {
    setSeed(nextSeed);
    reset();
  };

  const advance = () => {
    setStep((current) => Math.min(STEPS.length, current + 1));
    setElapsed((current) => Math.max(current, (step + 1) * 1.15));
  };

  const progress = Math.round((step / STEPS.length) * 100);
  const currentStep = STEPS[Math.min(step, STEPS.length - 1)];
  const missionState = step === STEPS.length ? 'COMPLETE' : running ? 'EXECUTING' : step > 0 ? 'PAUSED' : 'READY';
  const phaseLabel = step === 0 ? 'Ready for a reproducible run' : step === STEPS.length ? 'All task goals reached' : currentStep.detail;

  return (
    <div className="competition-shell">
      <header className="competition-header">
        <div className="competition-brand"><Shield size={23} /><span>FORTIFIERS</span><b>AEGIS / 01</b></div>
        <nav aria-label="Competition navigation"><a href="#mission">Mission</a><a href="#evidence">Evidence</a><Link href="/aegis">Teaching workspace</Link></nav>
        <span className="competition-badge"><i /> BROWSER READY</span>
      </header>

      <main className="competition-main">
        <section className="competition-intro">
          <div className="hero-copy">
            <span className="competition-kicker">INTEL PHYSICAL AI · DUAL-ARM MANIPULATION · MISSION 01</span>
            <h1>Dinner-table intelligence,<br /><em>taught to adapt.</em></h1>
            <p>Aegis turns a human correction into a safer next attempt. This competition surface makes the loop legible: inspect the task, perturb the scene, watch the hand-off, then follow the evidence to the teaching workspace.</p>
            <div className="hero-actions"><a className="hero-link" href="#mission">Run the mission <ArrowRight size={16} /></a><Link className="hero-link quiet" href="/aegis">See the learning loop <ArrowRight size={16} /></Link></div>
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
            <div className="sim-canvas-wrap"><TableScene seed={seed} step={step} /><div className="canvas-caption"><span><Zap size={13} /> Motion is deterministic</span><span><Cpu size={13} /> Physics runs natively</span></div></div>
            <div className="sim-controls">
              <button className="competition-primary" onClick={() => setRunning((value) => !value)} disabled={step === STEPS.length}>{running ? <Pause size={16} /> : <Play size={16} />}{running ? 'Pause mission' : step > 0 ? 'Resume mission' : 'Run mission'}</button>
              <button onClick={advance} disabled={running || step === STEPS.length}>Step <ChevronRight size={15} /></button>
              <button onClick={reset}><RotateCcw size={15} /> Reset</button>
              <div className="run-readout"><span>EPISODE TIME</span><strong>{formatTime(elapsed)}</strong></div>
            </div>
            <div className="seed-lab">
              <div><span className="competition-kicker">SEE THE ADAPTATION</span><p>Same task contract. New deterministic scene.</p></div>
              <div className="seed-pills">{SEED_PRESETS.map((preset) => <button key={preset.seed} className={seed === preset.seed ? 'selected' : ''} onClick={() => chooseSeed(preset.seed)}><strong>{preset.label}</strong><small>{preset.seed} · {preset.note}</small></button>)}</div>
              <label>Custom seed<input type="number" value={seed} onChange={(event) => { const next = Number(event.target.value); setSeed(Number.isFinite(next) ? next : 0); reset(); }} /></label>
            </div>
          </div>

          <aside className="mission-panel">
            <div className="mission-title"><div><span className="competition-kicker">02 / TASK GRAPH</span><h2>Bimanual sequence</h2></div><div className="progress-readout"><strong>{progress}%</strong><span>mission</span></div></div>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="phase-card"><div className="phase-icon"><Gauge size={18} /></div><div><span className="competition-kicker">CURRENT PHASE</span><strong>{step === STEPS.length ? 'Mission complete' : step === 0 ? 'Awaiting run' : currentStep.label}</strong><p>{phaseLabel}</p></div></div>
            <div className="step-list">{STEPS.map((item, index) => <div className={`competition-step ${index < step ? 'done' : index === step ? 'next' : ''}`} key={item.id}><span>{index < step ? <Check size={14} /> : String(index + 1).padStart(2, '0')}</span><div><strong>{item.label}</strong><small>{item.arm} ARM · {item.object.toUpperCase()}</small></div>{index === step && step < STEPS.length && <Zap size={14} className="step-live" />}</div>)}</div>
            <div className={`handoff-note ${step === 5 ? 'active' : ''}`}><Sparkles size={17} /><p><strong>Complementary action</strong>{step === 5 ? 'Both arms are synchronized on the cup.' : 'The cup transfers right → left before final placement.'}</p></div>
            <Link className="mission-link" href="/aegis">Open the teaching workspace <ArrowRight size={15} /></Link>
          </aside>
        </section>

        <section className="evidence-grid" id="evidence">
          <article><div className="evidence-icon"><Activity /></div><span>VISUAL LAYER</span><strong>Deterministic resets</strong><p>Seeded placement and perturbation controls make the demo reproducible for a judge, teammate, or future regression run.</p><em className="evidence-status ready">VERIFIABLE IN BROWSER</em></article>
          <article><div className="evidence-icon"><Cpu /></div><span>PHYSICS LAYER</span><strong>Native MuJoCo</strong><p>The six-step task, measured two-jaw grasps and lifts, post-contact retention, releases, and hand-off are verified in the native harness; this page never disguises animation as simulation.</p><em className="evidence-status separate">EVIDENCE TRACK SEPARATE</em></article>
          <article><div className="evidence-icon"><Gauge /></div><span>BASELINE RUN</span><strong>10 / 10 seeded scenes</strong><p>A deterministic observation-driven controller completes the current MuJoCo task across ten perturbation seeds. This is a baseline result, not a VLA claim.</p><em className="evidence-status ready">REPRODUCIBLE LOCALLY</em></article>
          <article><div className="evidence-icon"><Shield /></div><span>LEARNING LAYER</span><strong>Aegis memory</strong><p>Human corrections become versioned rules in the teaching workspace, where provenance and generalization can be inspected.</p><em className="evidence-status ready">OPEN TEACHING WORKSPACE</em></article>
        </section>

        <section className="readiness">
          <div className="readiness-heading"><span className="competition-kicker">03 / SUBMISSION READINESS</span><h2>A demo with a point of view.</h2><p>The browser proves the interaction model. Native runs prove the physical claim.</p></div>
          <div className="readiness-board"><div className="readiness-score"><strong>4<span>/6</span></strong><small>surface claims ready</small></div><div className="readiness-items"><span className="complete"><Check /> Dual SO-101 scene</span><span className="complete"><Check /> Browser task visualization</span><span className="complete"><Check /> Seeded perturbations</span><span className="complete"><Check /> MuJoCo baseline loop</span><span className="partial"><Activity /> VLA / challenge adapter</span><span className="partial"><Activity /> Intel benchmark</span></div></div>
        </section>

        <section className="demo-route"><div><span className="competition-kicker">THE JUDGE PATH</span><h2>Run it. Change it. Explain it.</h2></div><div className="route-steps"><span><b>01</b> Run mission</span><ArrowRight size={16} /><span><b>02</b> Shift seed</span><ArrowRight size={16} /><span><b>03</b> Teach Aegis</span><ArrowRight size={16} /><span><b>04</b> Inspect evidence</span></div></section>

        <footer><span>FORTIFIERS / TEACH MACHINES THE WAY YOU TEACH PEOPLE.</span><span>DETERMINISTIC KNOWLEDGE · AUDITABLE CORRECTIONS · HONEST EVIDENCE</span></footer>
      </main>
    </div>
  );
}

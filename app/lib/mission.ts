export const DINNER_TABLE_MISSION = {
  id: 'dinner-table-v1',
  shortId: 'DT-01',
  name: 'Set the dinner table',
  platform: 'Fortifiers Aegis',
} as const;

export const MISSION_STEPS = [
  { id: 'open_drawer', label: 'Open utensil drawer', arm: 'RIGHT', object: 'drawer', detail: 'Expose the tool set without disturbing the table.' },
  { id: 'retrieve_fork', label: 'Retrieve fork', arm: 'LEFT', object: 'fork', detail: 'Left arm grounds the seeded fork state.' },
  { id: 'retrieve_spoon', label: 'Retrieve spoon', arm: 'RIGHT', object: 'spoon', detail: 'Right arm completes the utensil pair.' },
  { id: 'place_plate', label: 'Place plate', arm: 'LEFT', object: 'plate', detail: 'Plate moves into the marked place area.' },
  { id: 'handoff_cup', label: 'Hand off cup', arm: 'BOTH', object: 'cup', detail: 'A coordinated right → left transfer keeps the plan moving.' },
  { id: 'place_cup', label: 'Place cup', arm: 'LEFT', object: 'cup', detail: 'Left arm finishes the setting.' },
] as const;

export const MISSION_PRESETS = [
  { seed: 1001, label: 'BASE', note: 'balanced layout' },
  { seed: 2026, label: 'SHIFTED', note: 'cup + plate drift' },
  { seed: 4120, label: 'TIGHT', note: 'compact reach' },
] as const;

export const MISSION_RUN_STORAGE_KEY = 'fortifiers.mission.run.v1';

export type MissionRunStatus = 'ready' | 'running' | 'paused' | 'complete';

export type MissionRun = {
  missionId: typeof DINNER_TABLE_MISSION.id;
  seed: number;
  completedSteps: number;
  elapsedSeconds: number;
  status: MissionRunStatus;
  updatedAt: string;
};

export function clampCompletedSteps(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(MISSION_STEPS.length, Math.max(0, Math.floor(value)));
}

export function normalizeSeed(value: number, fallback = 1001) {
  if (!Number.isSafeInteger(value) || value < 1) return fallback;
  return Math.abs(value);
}

export function makeMissionRun(seed?: number, completedSteps?: number, status: MissionRunStatus = 'ready'): MissionRun {
  return {
    missionId: DINNER_TABLE_MISSION.id,
    seed: normalizeSeed(seed ?? 1001),
    completedSteps: clampCompletedSteps(completedSteps ?? 0),
    elapsedSeconds: 0,
    status,
    updatedAt: new Date().toISOString(),
  };
}

export function readMissionRun(storage?: Pick<Storage, 'getItem'>): MissionRun | null {
  try {
    const raw = storage?.getItem(MISSION_RUN_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<MissionRun>;
    if (parsed.missionId !== DINNER_TABLE_MISSION.id) return null;
    if (typeof parsed.seed !== 'number' || !Number.isSafeInteger(parsed.seed) || typeof parsed.completedSteps !== 'number' || !Number.isSafeInteger(parsed.completedSteps)) return null;
    if (!['ready', 'running', 'paused', 'complete'].includes(String(parsed.status))) return null;
    return {
      missionId: DINNER_TABLE_MISSION.id,
      seed: normalizeSeed(parsed.seed),
      completedSteps: clampCompletedSteps(parsed.completedSteps),
      elapsedSeconds: Number.isFinite(parsed.elapsedSeconds) ? Math.max(0, Number(parsed.elapsedSeconds)) : 0,
      status: parsed.status as MissionRunStatus,
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

export function writeMissionRun(storage: Pick<Storage, 'setItem'>, run: MissionRun) {
  storage.setItem(MISSION_RUN_STORAGE_KEY, JSON.stringify({ ...run, updatedAt: new Date().toISOString() }));
}

export function missionHref(seed?: number, completedSteps?: number) {
  const run = makeMissionRun(seed, completedSteps);
  return `/?seed=${run.seed}#mission`;
}

export function workspaceHref(seed?: number, completedSteps?: number) {
  const run = makeMissionRun(seed, completedSteps);
  return `/aegis?mission=${DINNER_TABLE_MISSION.id}&seed=${run.seed}&step=${run.completedSteps}`;
}

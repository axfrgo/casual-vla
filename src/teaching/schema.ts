export const STATES = ['CLEAN', 'CONTAMINATED', 'SUSPECT', 'UNKNOWN', 'QUARANTINED'] as const;
export type State = typeof STATES[number];
export type Kind = 'gripper' | 'object' | 'zone' | 'surface' | 'tool';
export type Selector = { kind?: Kind; state?: State; protected?: boolean; id?: string };
export type Rule = { id: string; type: 'transition' | 'forbid' | 'prefer'; event: 'CONTACT' | 'OBSERVE' | 'ASSIGN'; actor: Selector; target: Selector; effect?: State; cost?: number };
export type Goal = { selector: Selector; zone: string };
export type Lesson = { text: string; source: 'text' | 'speechmatics' | 'rollback'; at: string; rules: string[]; previousVersion: string | null };
export type Validation = { at: string; episode: string; passed: boolean; environment: string; violations: number };
export type Contract = { skillId: string; version: string; description: string; goals: Goal[]; rules: Rule[]; teachingHistory: Lesson[]; validationHistory: Validation[] };
export function ensure(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
function record(value: unknown): asserts value is Record<string, unknown> { ensure(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected an object'); }
function keys(value: Record<string, unknown>, allowed: string[]) { ensure(Object.keys(value).every(k => allowed.includes(k)), 'Unsupported schema field'); }
function identifier(value: unknown) { ensure(typeof value === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(value), 'Invalid identifier'); }
export function validateSelector(value: unknown): asserts value is Selector {
  record(value); keys(value, ['kind', 'state', 'protected', 'id']);
  if (value.kind !== undefined) ensure(typeof value.kind === 'string' && ['gripper','object','zone','surface','tool'].includes(value.kind), 'Unsupported entity kind');
  if (value.state !== undefined) ensure(STATES.includes(value.state as State), 'Unsupported state');
  if (value.protected !== undefined) ensure(typeof value.protected === 'boolean', 'Protection must be boolean');
  if (value.id !== undefined) identifier(value.id);
}
export function validateRule(value: unknown): asserts value is Rule {
  record(value); keys(value, ['id','type','event','actor','target','effect','cost']); identifier(value.id);
  ensure(['transition','forbid','prefer'].includes(String(value.type)), 'Unsupported rule type');
  ensure(['CONTACT','OBSERVE','ASSIGN'].includes(String(value.event)), 'Unsupported event');
  validateSelector(value.actor); validateSelector(value.target);
  if (value.type === 'transition') {
    ensure(value.event !== 'ASSIGN' && STATES.includes(value.effect as State) && value.cost === undefined, 'Invalid transition');
    ensure(value.effect !== 'CLEAN', 'Clearing contamination requires verified decontamination evidence; teaching cannot clear state');
  } else if (value.type === 'prefer') {
    ensure(value.event === 'ASSIGN' && Number.isFinite(value.cost) && Math.abs(Number(value.cost)) <= 10 && value.effect === undefined, 'Invalid preference');
  } else ensure(value.event === 'CONTACT' && value.effect === undefined && value.cost === undefined, 'Invalid prohibition');
}
export function validateContract(value: unknown): asserts value is Contract {
  record(value); keys(value, ['skillId','version','description','goals','rules','teachingHistory','validationHistory']);
  identifier(value.skillId); ensure(typeof value.version === 'string' && /^1\.\d+$/.test(value.version), 'Invalid skill version');
  ensure(typeof value.description === 'string' && value.description.length <= 2000, 'Invalid description');
  ensure(Array.isArray(value.goals) && value.goals.length > 0 && value.goals.length <= 20, 'Invalid goals');
  for (const goal of value.goals) { record(goal); keys(goal, ['selector','zone']); validateSelector(goal.selector); identifier(goal.zone); }
  ensure(Array.isArray(value.rules) && value.rules.length <= 50, 'Rule limit exceeded');
  value.rules.forEach(validateRule); ensure(new Set(value.rules.map(r => r.id)).size === value.rules.length, 'Duplicate rule id');
  const transitions = value.rules.filter(r => r.type === 'transition');
  for (const a of transitions) for (const b of transitions) if (a !== b && a.event === b.event && JSON.stringify(a.actor) === JSON.stringify(b.actor) && JSON.stringify(a.target) === JSON.stringify(b.target)) ensure(a.effect === b.effect, 'Contradictory transitions');
  ensure(Array.isArray(value.teachingHistory) && value.teachingHistory.length <= 500, 'Invalid teaching history');
  for (const lesson of value.teachingHistory) {
    record(lesson); keys(lesson, ['text','source','at','rules','previousVersion']);
    ensure(typeof lesson.text === 'string' && lesson.text.length <= 4000 && ['text','speechmatics','rollback'].includes(String(lesson.source)), 'Invalid lesson');
    ensure(typeof lesson.at === 'string' && Number.isFinite(Date.parse(lesson.at)), 'Invalid lesson timestamp');
    ensure(Array.isArray(lesson.rules) && lesson.rules.every(r => typeof r === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(r)), 'Invalid lesson rules');
    ensure(lesson.previousVersion === null || typeof lesson.previousVersion === 'string' && /^1\.\d+$/.test(lesson.previousVersion), 'Invalid parent version');
  }
  ensure(Array.isArray(value.validationHistory) && value.validationHistory.length <= 1000, 'Invalid validations');
  for (const v of value.validationHistory) { record(v); keys(v, ['at','episode','passed','environment','violations']); ensure(typeof v.at === 'string' && Number.isFinite(Date.parse(v.at)) && typeof v.episode === 'string' && typeof v.environment === 'string' && typeof v.passed === 'boolean' && Number.isSafeInteger(v.violations) && Number(v.violations) >= 0, 'Invalid validation'); }
}
export function initialContract(): Contract {
  return { skillId: 'isolation_procedure', version: '1.0', description: 'Quarantine contaminated objects and prepare protected clean samples.', goals: [{ selector: { kind: 'object', state: 'CONTAMINATED' }, zone: 'quarantine' }, { selector: { kind: 'object', protected: true }, zone: 'clean_area' }], rules: [], teachingHistory: [], validationHistory: [] };
}

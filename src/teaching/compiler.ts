import { ensure, validateRule, type Rule } from './schema.ts';
export type Compilation = { status: 'compiled'; rules: Rule[]; explanation: string; compiler: string } | { status: 'clarify'; question: string } | { status: 'command'; command: 'run' | 'stop' | 'continue' | 'why' };
export const SAFETY_LESSON = "Once an arm touches contaminated material, treat that arm as contaminated too. Don't let it touch clean material until it has been cleared.";
export const PREFERENCE_LESSON = 'From now on, keep the left arm clean whenever possible and use the right arm for contaminated handling.';
export const transfer: Rule = { id: 'contamination_transfer', type: 'transition', event: 'CONTACT', actor: {}, target: { state: 'CONTAMINATED' }, effect: 'CONTAMINATED' };
export const protection: Rule = { id: 'protect_clean', type: 'forbid', event: 'CONTACT', actor: { state: 'CONTAMINATED' }, target: { state: 'CLEAN', protected: true } };
/** Deliberately bounded grammar. Unsupported or ambiguous speech never silently becomes knowledge. */
export function compile(text: string): Compilation {
  ensure(typeof text === 'string' && text.trim().length > 0 && text.length <= 4000, 'Teaching must contain 1–4000 characters');
  const t = text.toLowerCase().replace(/[’‘]/g, "'").replace(/\b(dirty|tainted|infected)\b/g, 'contaminated').replace(/\b(glove|hand)\b/g, 'gripper').trim();
  if (/^(stop|halt|pause)[.!]?$/.test(t)) return { status: 'command', command: 'stop' };
  if (/^(continue|resume)[.!]?$/.test(t)) return { status: 'command', command: 'continue' };
  if (/^why\b/.test(t) && /arm|gripper|right|left/.test(t)) return { status: 'command', command: 'why' };
  if (/^(run|execute|start) (the )?isolation procedure[.!]?$/.test(t) || /^move the red (container|sample) (in)?to quarantine( and prepare the clean samples)?[.!]?$/.test(t)) return { status: 'command', command: 'run' };
  if (/\b(ignore|disable|remove|allow|except|unless|only if|not contaminated|isn't contaminated|aren't contaminated|doesn't contaminate|does not contaminate|don't treat|do not treat)\b/.test(t)) return { status: 'clarify', question: 'This may weaken or qualify a rule. Please state a positive contamination rule or a specific prohibition without exceptions.' };
  if (/\b(away|near|distance|stuff|that arm)\b/.test(t) && !/once|after|whenever/.test(t)) return { status: 'clarify', question: 'Should contaminated grippers be prohibited from contacting protected clean objects?' };
  const rules: Rule[] = [];
  if (/unknown/.test(t) && /suspect/.test(t) && /treat|consider|mark|regard/.test(t)) rules.push({ id: 'unknown_is_suspect', type: 'transition', event: 'OBSERVE', actor: { state: 'UNKNOWN' }, target: {}, effect: 'SUSPECT' });
  if (/(once|after|anything|whenever|any .+ that)/.test(t) && /(touch|contact)/.test(t) && /contaminated/.test(t) && /(too|become|treat|consider|should be|is contaminated|gets contaminated)/.test(t)) rules.push(structuredClone(transfer));
  if (/\b(don't|do not|never|forbid|prevent|prohibit|must not)\b/.test(t) && /clean/.test(t) && /(touch|contact|use)/.test(t) && (/contaminated/.test(t) || rules.length > 0)) rules.push(structuredClone(protection));
  if (/left (arm|gripper)/.test(t) && /keep|prefer|reserve/.test(t) && /clean/.test(t)) rules.push({ id: 'reserve_left_clean', type: 'prefer', event: 'ASSIGN', actor: { id: 'left_gripper' }, target: { state: 'CONTAMINATED' }, cost: 4 });
  if (/right (arm|gripper)/.test(t) && /use|prefer/.test(t) && /contaminated handling/.test(t)) rules.push({ id: 'prefer_right_dirty', type: 'prefer', event: 'ASSIGN', actor: { id: 'right_gripper' }, target: { state: 'CONTAMINATED' }, cost: -2 });
  if (!rules.length) return { status: 'clarify', question: 'I could not compile that safely. Describe contact contamination, prohibit contaminated contact with clean objects, reserve the left arm, or treat unknown objects as suspect.' };
  rules.forEach(validateRule);
  return { status: 'compiled', rules, explanation: rules.map(r => r.type === 'transition' ? `${r.event}: matching entities become ${r.effect}.` : r.type === 'forbid' ? 'Prohibit contaminated contact with protected clean entities.' : `${r.id}: assignment cost ${r.cost}.`).join(' '), compiler: 'bounded-domain-grammar-v1' };
}


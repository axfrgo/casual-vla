import type { Contract, Rule } from '../teaching/schema.ts';
import { matches, type World, type Entity } from './world.ts';
function apply(rules: Rule[], event: Rule['event'], a: Entity, b: Entity) {
  const fired: string[] = [];
  // Monotone bounded fixed point: rules cannot turn contaminated entities clean.
  for (let pass = 0; pass < rules.length + 1; pass++) {
    let changed = false;
    for (const r of rules) if (r.type === 'transition' && r.event === event && matches(a,r.actor) && matches(b,r.target) && a.state !== r.effect) {
      if (a.state === 'CONTAMINATED') continue;
      a.state = r.effect!; fired.push(r.id); changed = true;
    }
    if (!changed) break;
  }
  return fired;
}
export function observeRules(world: World, contract: Contract) { for (const e of Object.values(world.entities)) apply(contract.rules,'OBSERVE',e,e); }
export function contact(world: World, actor: string, target: string, contract: Contract) {
  const a = world.entities[actor], b = world.entities[target]; if (!a || !b) throw new Error('Unknown contact entity');
  // Contact is symmetric, recorded once. Protection is checked on pre-contact state.
  const violations = contract.rules.filter(r => r.type === 'forbid' && r.event === 'CONTACT' && (matches(a,r.actor) && matches(b,r.target) || matches(b,r.actor) && matches(a,r.target))).map(r=>r.id);
  const fired = [...apply(contract.rules,'CONTACT',a,b), ...apply(contract.rules,'CONTACT',b,a)];
  return { violations, fired };
}

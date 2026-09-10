import { makeScene, type World, type Action } from '../aegis/world.ts';
import type { EmbodimentAdapter, Observation } from './base.ts';
/** A discrete event testbed, NOT physics, NOT a robot simulator, NOT an Intel adapter. */
export class ContactTestbed implements EmbodimentAdapter {
  readonly name = 'symbolic-contact-testbed'; world: World; violations = 0; faults: ('grasp'|'drop'|'wrong_object')[] = [];
  constructor(seed = 7, count = 2, transfer = false) { this.world = makeScene(seed,count,transfer); }
  observe() { return structuredClone(this.world); }
  reset(seed: number, count = 2, transfer = false) { this.world = makeScene(seed,count,transfer); this.violations = 0; this.faults = []; return this.observe(); }
  execute(action: Action): Observation {
    const out = (success: boolean, error?: string, contacts: Observation['contacts'] = []): Observation => ({ world: this.observe(), contacts, success, error, source: this.name });
    const a = this.world.entities[action.actor], b = this.world.entities[action.target];
    if (!a || !b || a.kind !== 'gripper' || b.kind !== 'object') return out(false,'Missing or invalid entity');
    if (action.kind === 'pick') {
      if (b.heldBy || Object.values(this.world.entities).some(e=>e.heldBy===a.id)) return out(false,'Gripper or object occupied');
      if (Math.hypot(a.pose[0]-b.pose[0],a.pose[1]-b.pose[1]) > a.reach) return out(false,'Unreachable object');
      const fault = this.faults.shift(); if (fault === 'grasp') return out(false,'Grasp failed');
      if (fault === 'wrong_object') return out(false,'Identity mismatch');
      a.pose = [...b.pose];
      // Independent contamination oracle: never reads the taught contract.
      if ((a.state === 'CONTAMINATED' && b.state === 'CLEAN' && b.protected) || (b.state === 'CONTAMINATED' && a.state === 'CLEAN' && a.protected)) this.violations++;
      if (a.state === 'CONTAMINATED' || b.state === 'CONTAMINATED') a.state = b.state = 'CONTAMINATED';
      b.heldBy = fault === 'drop' ? null : a.id;
      return out(fault !== 'drop', fault === 'drop' ? 'Object dropped' : undefined,[{actor:a.id,target:b.id}]);
    }
    const zone = this.world.entities[action.zone ?? ''];
    if (!zone || zone.kind !== 'zone' || b.heldBy !== a.id) return out(false,'Invalid placement or unheld object');
    if (Math.hypot(a.pose[0]-zone.pose[0],a.pose[1]-zone.pose[1]) > a.reach) return out(false,'Unreachable destination');
    if (zone.protected && (a.state !== 'CLEAN' || b.state !== 'CLEAN')) this.violations++;
    if (b.state === 'CONTAMINATED' || zone.state === 'CONTAMINATED') b.state = zone.state = 'CONTAMINATED';
    a.pose = [...zone.pose]; b.pose = [...zone.pose]; b.zone = zone.id; b.heldBy = null;
    return out(true,undefined,[{actor:b.id,target:zone.id}]);
  }
  verify() {
    const objects = Object.values(this.world.entities).filter(e=>e.kind==='object');
    const goalsMet = objects.length >= 2 && objects.every(e=>!e.heldBy && e.zone === (e.protected ? 'clean_area':'quarantine'));
    const protectedClean = Object.values(this.world.entities).filter(e=>e.protected).every(e=>e.state==='CLEAN');
    return { passed: goalsMet && protectedClean && this.violations === 0, violations:this.violations, goalsMet };
  }
}

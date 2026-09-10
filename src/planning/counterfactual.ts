import { contact } from '../aegis/rules.ts';
import { matches, type World, type Action } from '../aegis/world.ts';
import type { Contract } from '../teaching/schema.ts';
export type Candidate = { action: Action; safe: boolean; reasons: string[]; cost: number; fired: string[] };
export type Task = { target: string; zone: string };
export function goalTasks(world: World, contract: Contract): Task[] {
  return contract.goals.flatMap(g=>Object.values(world.entities).filter(e=>matches(e,g.selector)).map(e=>({target:e.id,zone:g.zone})));
}
export function assess(world: World, action: Action, contract: Contract): Candidate {
  const reasons: string[] = []; const a=world.entities[action.actor], b=world.entities[action.target];
  let cost = Infinity; const fired: string[]=[];
  if (!a || !b || a.kind!=='gripper' || b.kind!=='object') reasons.push('Invalid entity');
  else {
    const destination = action.kind==='pick' ? b : world.entities[action.zone??''];
    if (!destination || action.kind==='place' && destination.kind!=='zone') reasons.push('Missing destination');
    else {
      cost=Math.hypot(a.pose[0]-destination.pose[0],a.pose[1]-destination.pose[1]);
      if (cost>a.reach) reasons.push('Unreachable');
      if (a.confidence<0.9 || b.confidence<0.9) reasons.push('Insufficient identity confidence');
      if (['UNKNOWN','SUSPECT'].includes(b.state)) reasons.push('Object requires classification');
      if (b.protected && ['UNKNOWN','SUSPECT','QUARANTINED'].includes(a.state)) reasons.push('Gripper clearance unknown');
      if (action.kind==='pick' && (b.heldBy || Object.values(world.entities).some(e=>e.heldBy===a.id))) reasons.push('Occupied');
      if (action.kind==='place' && b.heldBy!==a.id) reasons.push('Not held by this gripper');
      const clone=structuredClone(world);
      const pair=action.kind==='pick' ? [a.id,b.id] : [b.id,destination.id];
      const result=contact(clone,pair[0],pair[1],contract); reasons.push(...result.violations); fired.push(...result.fired);
      for(const r of contract.rules) if(r.type==='prefer' && matches(a,r.actor) && matches(b,r.target)) { cost+=r.cost!; fired.push(r.id); }
    }
  }
  return {action,safe:reasons.length===0,reasons,cost,fired};
}
export function plan(world: World, contract: Contract, tasks: Task[]): { candidates: Candidate[]; selected: Candidate | null; complete: boolean } {
  const remaining=tasks.filter(t=>!world.entities[t.target] || world.entities[t.target].zone!==t.zone || world.entities[t.target].heldBy);
  if(!remaining.length) return {candidates:[],selected:null,complete:true};
  const task=remaining[0], object=world.entities[task.target];
  const arms=object?.heldBy ? [object.heldBy] : Object.values(world.entities).filter(e=>e.kind==='gripper').map(e=>e.id);
  const candidates=arms.map(actor=>assess(world,{id:`a${world.sequence+1}_${actor}`,kind:object?.heldBy?'place':'pick',actor,target:task.target,zone:task.zone},contract));
  const selected=candidates.filter(c=>c.safe).sort((a,b)=>a.cost-b.cost || a.action.actor.localeCompare(b.action.actor))[0]??null;
  return {candidates,selected,complete:false};
}

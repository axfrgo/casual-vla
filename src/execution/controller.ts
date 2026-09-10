import { contact, observeRules } from '../aegis/rules.ts';
import type { World, Edge } from '../aegis/world.ts';
import type { Contract } from '../teaching/schema.ts';
import type { EmbodimentAdapter } from '../embodiment/base.ts';
import { goalTasks, plan, type Task, type Candidate } from '../planning/counterfactual.ts';
export type EpisodeResult = { passed: boolean; violations: number; goalsMet: boolean; actions: number; retries: number; prevented: number; planningMs: number; elapsedMs: number; environment: string; episode: string; failure: string|null };
export class Controller {
  adapter: EmbodimentAdapter; contract: Contract; world: World; initial: World; tasks: Task[];
  paused=false; finished=false; failure:string|null=null; actions=0; retries=0; prevented=0; planningMs=0; started=performance.now(); consecutiveFailures=0;
  decisions: Candidate[]=[]; lastCandidates: Candidate[]=[]; events: {at:string;message:string}[]=[];
  constructor(adapter: EmbodimentAdapter, contract: Contract) { this.adapter=adapter; this.contract=structuredClone(contract); this.world=adapter.observe(); this.initial=structuredClone(this.world); this.tasks=goalTasks(this.world,contract); observeRules(this.world,contract); }
  teach(contract: Contract) {
    this.contract=structuredClone(contract); const replay=structuredClone(this.initial); observeRules(replay,contract);
    for(const edge of this.world.edges) if(edge.relation==='CONTACT') contact(replay,edge.actor,edge.target,contract);
    for(const [id,e] of Object.entries(this.world.entities)) if(replay.entities[id]) e.state=replay.entities[id].state;
    this.events.push({at:new Date().toISOString(),message:`Applied skill v${contract.version}; replayed recorded contacts.`});
  }
  stop() { this.paused=true; }
  resume() { this.paused=false; }
  step() {
    if(this.paused || this.finished) return;
    if(this.actions>=100) {this.finished=true;this.failure='Action budget exceeded';return;}
    const start=performance.now(); const decision=plan(this.world,this.contract,this.tasks); this.planningMs+=performance.now()-start;
    this.lastCandidates=decision.candidates; this.decisions.push(...decision.candidates); this.prevented+=decision.candidates.filter(c=>!c.safe && c.reasons.some(r=>this.contract.rules.some(rule=>rule.id===r))).length;
    if(decision.complete) {this.finished=true;return;}
    if(!decision.selected) {this.finished=true;this.failure='Planning failure: no safe reachable action';return;}
    const action=decision.selected.action; const before=structuredClone(this.world); const observation=this.adapter.execute(action); this.actions++; this.world.sequence++;
    // Kinematics and occupancy are observed. Contamination is inferred from metadata + contact history,
    // never copied from the testbed oracle after reset (no magical contamination sensor).
    for(const [id,e] of Object.entries(observation.world.entities)) {
      if(this.world.entities[id]) Object.assign(this.world.entities[id],{pose:e.pose,zone:e.zone,heldBy:e.heldBy,confidence:e.confidence,reach:e.reach});
      else this.world.entities[id]={...e,state:'UNKNOWN'};
    }
    const addEdge=(relation:Edge['relation'],actor:string,target:string)=>{
      const a=this.world.entities[actor],b=this.world.entities[target];
      this.world.edges.push({id:`${this.world.episode}:${this.world.edges.length+1}`,episode:this.world.episode,sequence:this.world.sequence,at:new Date().toISOString(),relation,actor,target,before:[before.entities[actor]?.state??'UNKNOWN',before.entities[target]?.state??'UNKNOWN'],after:[a.state,b.state],source:observation.source,confidence:Math.min(a.confidence,b.confidence),actionId:action.id});
    };
    for(const pair of observation.contacts) {contact(this.world,pair.actor,pair.target,this.contract);addEdge('CONTACT',pair.actor,pair.target);}
    if(observation.success) {
      addEdge(action.kind==='pick'?'GRASPED_BY':'RELEASED_BY',action.target,action.actor);
      if(action.kind==='place') addEdge('ENTERED_ZONE',action.target,action.zone!);
      this.consecutiveFailures=0;
    } else {this.retries++;this.consecutiveFailures++; if(this.consecutiveFailures>=3) {this.finished=true;this.failure=`Execution failure: ${observation.error}`;}}
    this.events.push({at:new Date().toISOString(),message:`${action.kind.toUpperCase()} ${action.target} with ${action.actor}: ${observation.success?'observed success':observation.error}`});
    observeRules(this.world,this.contract);
  }
  run() { while(!this.finished && !this.paused) this.step(); return this.result(); }
  result():EpisodeResult {const verification=this.adapter.verify();return {...verification,passed:this.finished&&!this.failure&&verification.passed,actions:this.actions,retries:this.retries,prevented:this.prevented,planningMs:this.planningMs,elapsedMs:performance.now()-this.started,environment:this.adapter.name,episode:this.world.episode,failure:this.failure??(this.finished&&!verification.passed?'Verification failure: goals or contamination constraints failed':null)};}
  explain(arm='right_gripper') {
    const rejected=[...this.decisions].reverse().find(c=>c.action.actor===arm&&!c.safe);
    const edges=this.world.edges.filter(e=>e.relation==='CONTACT'&&(e.actor===arm||e.target===arm));
    if(rejected) return `${arm}: rejected ${rejected.action.kind} ${rejected.action.target} because ${rejected.reasons.join(', ')}. Known state: ${this.world.entities[arm]?.state}. Recorded contacts: ${edges.map(e=>e.actor===arm?e.target:e.actor).join(', ')||'none'}. Skill v${this.contract.version}.`;
    return `${arm} is ${this.world.entities[arm]?.state??'UNKNOWN'}. No rejected action for this arm has been recorded. ${edges.length} contact event(s), skill v${this.contract.version}.`;
  }
}

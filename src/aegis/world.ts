import type { Kind, State, Selector } from '../teaching/schema.ts';
export type Entity = { id: string; kind: Kind; state: State; protected: boolean; pose: [number, number]; zone: string; heldBy: string | null; confidence: number; reach: number };
export type Edge = { id: string; episode: string; sequence: number; at: string; relation: 'CONTACT' | 'GRASPED_BY' | 'RELEASED_BY' | 'ENTERED_ZONE'; actor: string; target: string; before: [State,State]; after: [State,State]; source: string; confidence: number; actionId: string };
export type World = { episode: string; entities: Record<string, Entity>; sequence: number; edges: Edge[] };
export type Action = { id: string; kind: 'pick' | 'place'; actor: string; target: string; zone?: string };
export function matches(entity: Entity, selector: Selector) { return Object.entries(selector).every(([key, value]) => entity[key as keyof Entity] === value); }
export function random(seed: number) { let x = seed >>> 0; return () => { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; return x / 4294967296; }; }
export function makeScene(seed = 7, count = 2, transfer = false): World {
  if (!Number.isSafeInteger(seed) || count < 1 || count > 12 || !Number.isSafeInteger(count)) throw new Error('Invalid scene parameters');
  const rng = random(seed); const entities: Record<string, Entity> = {};
  const add = (id: string, kind: Kind, state: State, pose: [number,number], isProtected = false) => { entities[id] = { id, kind, state, protected: isProtected, pose, zone: 'bench', heldBy: null, confidence: 1, reach: 1.5 }; };
  add('left_gripper','gripper','CLEAN',[0.18,0.85]); add('right_gripper','gripper','CLEAN',[0.82,0.85]);
  add('quarantine','zone','CONTAMINATED',[0.83,0.17]); add('clean_area','zone','CLEAN',[0.17,0.17],true);
  add(transfer ? 'amber_vial' : 'red_container','object','CONTAMINATED',[0.2+rng()*0.6,0.3+rng()*0.35]);
  for(let i=0;i<count;i++) add(`sample_${i+1}`,'object','CLEAN',[0.2+rng()*0.6,0.3+rng()*0.35],true);
  add('unknown_tool','tool','UNKNOWN',[0.1+rng()*0.8,0.25+rng()*0.4]);
  return { episode: `contact-${seed}-${count}-${transfer ? 'transfer':'standard'}`, entities, sequence: 0, edges: [] };
}

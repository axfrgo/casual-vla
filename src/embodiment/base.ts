import type { World, Action } from '../aegis/world.ts';
export type Observation = { world: World; contacts: { actor: string; target: string }[]; success: boolean; error?: string; source: string };
export interface EmbodimentAdapter { readonly name: string; observe(): World; execute(action: Action): Observation; reset(seed: number, count?: number, transfer?: boolean): World; verify(): { passed: boolean; violations: number; goalsMet: boolean }; }
export class IntelBimanualAdapter {
  readonly name = 'intel-bimanual-unavailable';
  constructor() { throw new Error('Official Intel challenge SDK, action/observation schema, VLA endpoint and reset API have not been supplied. No motor command was issued. See docs/INTEL_INTEGRATION.md.'); }
}

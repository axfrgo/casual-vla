import { initialContract, validateContract, ensure, type Contract, type Rule, type Lesson, type Validation } from '../teaching/schema.ts';
export interface Storage { getItem(key: string): string | null; setItem(key: string, value: string): void }
export class MemoryStorage implements Storage { data = new Map<string,string>(); getItem(k: string) { return this.data.get(k) ?? null; } setItem(k: string, v: string) { this.data.set(k,v); } }
export class SkillLibrary {
  storage: Storage;
  key: string;
  constructor(storage: Storage, key = 'fortifiers.skills.v1') { this.storage = storage; this.key = key; }
  versions(): Contract[] {
    const raw = this.storage.getItem(this.key); if (raw === null) return [initialContract()];
    const data = JSON.parse(raw); ensure(Array.isArray(data) && data.length > 0 && data.length <= 500, 'Corrupt skill library'); data.forEach(validateContract);
    ensure(data.every((c, i) => c.version === `1.${i}` && c.skillId === 'isolation_procedure'), 'Invalid version chain'); return structuredClone(data);
  }
  current(): Contract { return this.versions().at(-1)!; }
  teach(text: string, rules: Rule[], source: Lesson['source'] = 'text'): Contract {
    const versions = this.versions(); const previous = versions.at(-1)!; const next = structuredClone(previous);
    const merged = new Map(next.rules.map(r => [r.id, r])); rules.forEach(r => merged.set(r.id, structuredClone(r)));
    next.rules = [...merged.values()]; next.version = `1.${versions.length}`; next.validationHistory = [];
    next.teachingHistory.push({ text, rules: rules.map(r => r.id), source, previousVersion: previous.version, at: new Date().toISOString() });
    validateContract(next); this.storage.setItem(this.key, JSON.stringify([...versions, next])); return next;
  }
  rollback(version: string): Contract { const target = this.versions().find(v => v.version === version); ensure(target, 'Unknown version'); const versions = this.versions(); const next = structuredClone(target); next.version = `1.${versions.length}`; next.teachingHistory = [...versions.at(-1)!.teachingHistory, { text: `Restore ${version}`, source: 'rollback', at: new Date().toISOString(), rules: target.rules.map(r => r.id), previousVersion: versions.at(-1)!.version }]; next.validationHistory = []; validateContract(next); this.storage.setItem(this.key, JSON.stringify([...versions, next])); return next; }
  validate(result: Validation) { const versions = this.versions(); versions.at(-1)!.validationHistory.push(result); validateContract(versions.at(-1)); this.storage.setItem(this.key, JSON.stringify(versions)); }
  import(raw: string) { const parsed = JSON.parse(raw); const temp = new MemoryStorage(); temp.setItem(this.key, JSON.stringify(parsed)); new SkillLibrary(temp, this.key).versions(); this.storage.setItem(this.key, JSON.stringify(parsed)); }
  export() { return JSON.stringify(this.versions(), null, 2); }
}

import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { join } from 'node:path';
import type { Storage } from '../src/skills/library.ts';
export class FileStorage implements Storage {
  directory:string;
  constructor(directory:string){this.directory=directory;mkdirSync(directory,{recursive:true});}
  path(key:string){if(!/^[a-z0-9.]+$/.test(key))throw new Error('Invalid storage key');return join(this.directory,`${key}.json`);}
  getItem(key:string){try{return readFileSync(this.path(key),'utf8');}catch(e){if((e as NodeJS.ErrnoException).code==='ENOENT')return null;throw e;}}
  setItem(key:string,value:string){const path=this.path(key),temp=`${path}.${process.pid}.tmp`;writeFileSync(temp,value,{flag:'wx'});renameSync(temp,path);}
}

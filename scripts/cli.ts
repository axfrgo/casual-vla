import { SkillLibrary } from '../src/skills/library.ts';
import { FileStorage } from './file-storage.ts';
import { compile, SAFETY_LESSON, PREFERENCE_LESSON } from '../src/teaching/compiler.ts';
import { Controller } from '../src/execution/controller.ts';
import { ContactTestbed } from '../src/embodiment/contact-testbed.ts';
const library=new SkillLibrary(new FileStorage('data'));
const [command='demo',...args]=process.argv.slice(2);
if(command==='teach') {const text=args.join(' ');const result=compile(text);if(result.status==='compiled')console.log(JSON.stringify(library.teach(text,result.rules),null,2));else console.log(JSON.stringify(result,null,2));}
else if(command==='run') {const c=new Controller(new ContactTestbed(Number(args[0]??7)),library.current());console.log(JSON.stringify(c.run(),null,2));console.log(c.explain());}
else if(command==='skill') console.log(library.export());
else if(command==='demo') {console.log('SYMBOLIC CONTACT TESTBED — no robot or Intel VLA execution');const baseline=new Controller(new ContactTestbed(2),library.versions()[0]);console.log('Before:',baseline.run());for(const text of [SAFETY_LESSON,PREFERENCE_LESSON]){const r=compile(text);if(r.status==='compiled'){const skill=library.teach(text,r.rules);console.log(`Taught v${skill.version}:`,r.explanation);}}const c=new Controller(new ContactTestbed(2),library.current());console.log('After:',c.run());const reload=new SkillLibrary(new FileStorage('data'));console.log('Changed scene:',new Controller(new ContactTestbed(1001,5,true),reload.current()).run());console.log(c.explain());}
else throw Error('Usage: node scripts/cli.ts demo|teach <lesson>|run [seed]|skill');


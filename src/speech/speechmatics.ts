export type SpeechResult={text:string;jobId:string;latencyMs:number;source:'speechmatics'};
export async function transcribe(audio:Blob,key:string,request:typeof fetch=fetch):Promise<SpeechResult> {
  if(!key) throw new Error('SPEECHMATICS_API_KEY is not configured. Text teaching remains available.');
  if(audio.size===0||audio.size>10*1024*1024) throw new Error('Audio must be between 1 byte and 10 MB');
  const started=performance.now(); const endpoint='https://asr.api.speechmatics.com/v2/jobs';
  const form=new FormData(); form.append('data_file',audio,'teaching.webm'); form.append('config',JSON.stringify({type:'transcription',transcription_config:{language:'en'}}));
  const call=async(url:string,init:RequestInit={})=>{const headers=new Headers(init.headers);headers.set('Authorization',`Bearer ${key}`);const response=await request(url,{...init,headers,signal:AbortSignal.timeout(20000)});if(!response.ok)throw new Error(`Speechmatics returned HTTP ${response.status}`);return response;};
  const created=await(await call(endpoint,{method:'POST',body:form})).json() as {id?:unknown};
  if(typeof created.id!=='string'||!/^[a-zA-Z0-9_-]+$/.test(created.id))throw new Error('Speechmatics returned an invalid job identifier');
  for(let attempt=0;attempt<40;attempt++) {
    const status=await(await call(`${endpoint}/${created.id}`)).json() as {job?:{status?:string}};
    if(status.job?.status==='done') {
      const text=await(await call(`${endpoint}/${created.id}/transcript?format=txt`)).text();
      if(!text.trim()||text.length>4000)throw new Error('Transcript is empty or too long for a teaching lesson');
      return {text:text.trim(),jobId:created.id,latencyMs:performance.now()-started,source:'speechmatics'};
    }
    if(['rejected','deleted','expired','failed'].includes(status.job?.status??''))throw new Error(`Speechmatics job ${status.job?.status}`);
    await new Promise(resolve=>setTimeout(resolve,attempt<5?350:700));
  }
  throw new Error('Speechmatics processing timed out; the lesson was not applied');
}


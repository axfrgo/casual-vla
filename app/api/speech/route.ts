import { transcribe } from '../../../src/speech/speechmatics';
export const runtime = 'nodejs';
export const maxDuration = 60;
export async function GET() { return Response.json({configured:!!process.env.SPEECHMATICS_API_KEY,mode:'Speechmatics batch, real audio; no browser speech substitution'}); }
export async function POST(request:Request) {
  if(request.headers.get('origin')!==new URL(request.url).origin) return Response.json({error:'Origin not allowed'},{status:403});
  if(!process.env.SPEECHMATICS_API_KEY) return Response.json({error:'SPEECHMATICS_API_KEY is not configured'},{status:503});
  if(!request.headers.get('content-type')?.startsWith('audio/'))return Response.json({error:'Expected audio content type'},{status:415});
  try {
    const reader=request.body?.getReader();if(!reader)throw new Error('No audio');const chunks:Uint8Array[]=[];let length=0;
    while(true){const {value,done}=await reader.read();if(done)break;length+=value.length;if(length>10*1024*1024){await reader.cancel();return Response.json({error:'Audio exceeds 10 MB'},{status:413});}chunks.push(value);}
    const audio=new Blob(chunks as BlobPart[],{type:request.headers.get('content-type')!});
    return Response.json(await transcribe(audio,process.env.SPEECHMATICS_API_KEY));
  }catch(error){return Response.json({error:error instanceof Error?error.message:'Transcription failed'},{status:502});}
}

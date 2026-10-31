const $ = id => document.getElementById(id);
const headers = () => $('token') ? {'X-API-Key': $('token').value} : {};
const status = (message, error=false) => {$('status').textContent=message; $('status').className=error?'error':'';};
let rawUrl, correctedUrl;
async function history(){
  const response=await fetch('/api/history',{headers:headers()}); if(!response.ok)return;
  const items=await response.json(); $('history').replaceChildren();
  for(const item of items){const tr=document.createElement('tr');for(const value of [new Date(item.time_utc).toLocaleString(),item.sample_name,item.status,item.source]){const td=document.createElement('td');td.textContent=value;tr.append(td);} $('history').append(tr);}
}
async function showResult(data){
  $('rows').replaceChildren(); $('empty').classList.add('hidden'); $('result').classList.remove('hidden');
  $('corrected').classList.add('hidden'); $('corrected').removeAttribute('src');
  $('decision').textContent=data.interpretation?.decision||'Result';
  $('interpretation').textContent=data.interpretation?.summary||'';
  $('next-step').textContent=data.interpretation?.next_step||'';
  for(const row of data.predictions){
    const tr=document.createElement('tr');const number=v=>v==null?'—':Number(v).toFixed(2);
    for(const [i,value] of [row.metal,number(row.estimate_ug_L_simulated),row.lower_90==null?'Withheld':`${number(row.lower_90)} – ${number(row.upper_90)}`,`${row.model}${row.flags.length?' · '+row.flags.join(', ').replaceAll('_',' '):''}`].entries()){
      const td=document.createElement('td');td.textContent=value;if(i===1)td.className='number';if(i===3&&row.flags.length)td.className='flag';tr.append(td);
    }$('rows').append(tr);
  }
  $('qc').textContent=JSON.stringify({image_checks:data.qc,model_version:data.model_version,interval_method:data.interval_note,how_to_read:data.interpretation?.how_to_read,limitations:data.interpretation?.limitations},null,2);
  const preview=await fetch(`/api/measurements/${data.id}/preview`,{headers:headers()});
  if(preview.ok){if(correctedUrl)URL.revokeObjectURL(correctedUrl);correctedUrl=URL.createObjectURL(await preview.blob());$('corrected').src=correctedUrl;$('corrected').classList.remove('hidden');}
}
async function analyze(file){
  if(!file){status('Choose a JPEG or PNG first.',true);return;}
  $('analyze').disabled=true;$('demo').disabled=true;status('Checking image and estimating…');
  $('result').classList.add('hidden');$('empty').classList.remove('hidden');$('empty').textContent='Processing capture…';
  if(rawUrl)URL.revokeObjectURL(rawUrl);rawUrl=URL.createObjectURL(file);$('raw').src=rawUrl;$('raw').classList.remove('hidden');
  try{
    const response=await fetch('/api/predict',{method:'POST',headers:{...headers(),'Content-Type':file.type||'image/jpeg','X-Sample-Name':encodeURIComponent($('sample').value||'Untitled capture')},body:file});
    const data=await response.json();
    if(response.status===422&&data.interpretation){await showResult(data);status(data.reasons?.join(' ')||'No result returned.',true);return;}
    if(!response.ok){throw new Error(data.error||'Analysis failed.');}
    await showResult(data);status('Image accepted. Synthetic-model estimates shown.');
  }catch(error){status(error.message,true);$('empty').textContent='No estimates returned. Address the image checks and try again.';}
  finally{$('analyze').disabled=false;$('demo').disabled=false;await history();}
}
$('analyze').onclick=()=>analyze($('image').files[0]);
$('demo').onclick=async()=>{try{const response=await fetch('/api/demo-image',{headers:headers()});if(!response.ok)throw new Error('Could not load the example. Check the API token.');$('sample').value='Synthetic example';await analyze(await response.blob());}catch(error){status(error.message,true);}};
$('export').onclick=async()=>{const response=await fetch('/api/history.csv',{headers:headers()});if(!response.ok){status('Export failed. Check the API token.',true);return;}const url=URL.createObjectURL(await response.blob());const link=document.createElement('a');link.href=url;link.download='aquasense_history.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
if($('token'))$('token').addEventListener('change',history);
history();

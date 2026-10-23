'use strict';
const $=id=>document.getElementById(id);
const canvas=$('cameraCanvas'),ctx=canvas.getContext('2d'),edit=$('editCanvas'),ectx=edit.getContext('2d');
let status=null,lastSeq=0,lastMeta=null,editing=false,draft=[],editSignature=null,settingsLoaded=false;
let displayTimes=[],history=[],targetFps=20,lastStatusAt=0,frameAvailable=false;
let lastRun=null;
const palette={CLEAR:'#36d29e',DANGER:'#ff7065',CALIBRATE:'#f0c65a',CHECKING:'#f0c65a'};
const num=(x,d=1)=>Number.isFinite(x)?x.toFixed(d):'—';
const set=(id,value)=>{$(id).textContent=value;};
const descriptions={
  SWITCHING:['Switching model','Stopping workers and warming up the selected model.','...'],
  STARTING:['Starting the monitor','Loading the model and connecting to the camera.','○'],
  RUNNING:['Camera connected','Waiting for the first completed detection.','○'],
  CALIBRATE:['Define your safety zone','Select the taped area to activate zone monitoring.','#'],
  CHECKING:['Checking the zone','Confirming occupancy across consecutive detection frames.','◌'],
  CLEAR:['Zone is clear','No selected object meets the zone entry rule.','✓'],
  DANGER:['Zone intrusion detected','A selected object has entered the monitored area.','!'],
  STALE:['Camera results are stale','The last detection is more than 2 seconds old. Check the camera and app.','!'],
  ERROR:['Monitor needs attention','See the diagnostic message below.','!'],
  OFFLINE:['Connection interrupted','Reconnect to the UNO Q to see the current state.','!']
};
function banner(state){
  const [title,detail,icon]=descriptions[state]||descriptions.STARTING;
  $('statusStrip').className='status-strip '+state.toLowerCase();set('statusTitle',title);set('statusDetail',detail);set('statusIcon',icon);
  const ledOn=status?.led_available&&['CLEAR','DANGER'].includes(state);
  $('ledDot').style.background=ledOn?palette[state]:'#a7b7ba';
  set('ledState',ledOn?(state==='CLEAR'?'Green LED active':'Red LED active'):(state==='OFFLINE'?'LED state unknown':'LED standby'));
  set('ledDetail',status?.led_message||'Checking board output');
  if(['STALE','ERROR','OFFLINE'].includes(state)){set('videoBadge',state+' / LAST IMAGE');canvas.style.opacity='.5';}
  else{set('videoBadge',editing?'CALIBRATION / CAMERA 01':'LIVE / CAMERA 01');canvas.style.opacity='1';}
}
function polygon(context,pts,color,width,height,vertices=false){
  if(!pts?.length)return;
  context.beginPath();pts.forEach(([x,y],i)=>i?context.lineTo(x*width,y*height):context.moveTo(x*width,y*height));
  if(pts.length>2){context.closePath();context.fillStyle=color+'22';context.fill();}
  context.strokeStyle=color;context.lineWidth=Math.max(2,width/380);context.stroke();
  if(vertices)pts.forEach(([x,y],i)=>{context.beginPath();context.arc(x*width,y*height,width/95,0,Math.PI*2);context.fillStyle=color;context.fill();context.fillStyle='#102a35';context.font=`bold ${width/58}px sans-serif`;context.textAlign='center';context.textBaseline='middle';context.fillText(String(i+1),x*width,y*height);});
}
function drawImage(image,meta){
  if(canvas.width!==image.width||canvas.height!==image.height){
    canvas.width=edit.width=image.width;canvas.height=edit.height=image.height;
    $('videoStage').style.aspectRatio=`${image.width}/${image.height}`;
  }
  const w=canvas.width,h=canvas.height;ctx.drawImage(image,0,0,w,h);
  polygon(ctx,meta.zone,meta.status==='DANGER'?'#ff7065':'#f0c65a',w,h);
  const size=Math.max(10,w/56);ctx.font=`600 ${size}px sans-serif`;ctx.textBaseline='top';ctx.textAlign='left';
  for(const d of meta.detections||[]){
    if(!d.selected) continue;
    const [x1,y1,x2,y2]=d.box.map((v,i)=>v*(i%2?h:w));const color=d.inside?'#ff7065':d.selected?'#5ee4bc':'#b5c8d1';
    ctx.lineWidth=Math.max(1.5,w/380);ctx.strokeStyle=color;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    const label=`${d.label} #${d.track_id} · ${Math.round(d.score*100)}%`,tw=ctx.measureText(label).width+10;
    const lx=Math.max(0,Math.min(x1,w-tw)),ly=Math.max(0,y1-size-8);ctx.fillStyle=color;ctx.fillRect(lx,ly,tw,size+7);
    ctx.fillStyle='#102b36';ctx.fillText(label,lx+5,ly+3);
    if(d.selected){ctx.beginPath();ctx.arc((x1+x2)/2,y2,Math.max(3,w/150),0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
  }
  set('frameTime',num(meta.timing?.read_to_decision_ms)+' ms · read → decision');set('frameId','Frame '+meta.seq.toLocaleString());
  $('emptyVideo').hidden=true;frameAvailable=true;lastMeta=meta;
}
async function frames(){
  const start=performance.now();
  if(!editing){
    try{
      const r=await fetch('/api/frame?after='+lastSeq+'&run='+encodeURIComponent(lastRun||''),{cache:'no-store',signal:AbortSignal.timeout(4500)});
      if(r.ok&&r.status!==204){
        const meta=JSON.parse(r.headers.get('X-Frame-Meta')),blob=await r.blob(),image=await createImageBitmap(blob);
        if(!editing&&(!lastRun||lastRun===meta.run_id)){drawImage(image,meta);lastSeq=meta.seq;lastRun=meta.run_id;displayTimes.push(performance.now());}
        image.close();
      }
    }catch(e){/* Connection state is handled by status polling. */}
  }
  const now=performance.now();displayTimes=displayTimes.filter(t=>now-t<5000);
  const displayFps=displayTimes.length>1&&now-displayTimes.at(-1)<1500?(displayTimes.length-1)*1000/(displayTimes.at(-1)-displayTimes[0]):0;
  set('displayFps',num(displayFps));
  // A deadline accounts for fetch/decode time; it does not add an extra full frame interval.
  setTimeout(frames,Math.max(5,1000/targetFps-(performance.now()-start)));
}
function timing(data){
  const a=data.average||{};
  const groups=[['Prepare',a.preprocess_ms,'#58bda6'],['Waiting',(a.capture_wait_ms||0)+(a.input_wait_ms||0)+(a.post_wait_ms||0),'#d7b573'],['Inference',(a.invoke_ms||0)+(a.input_copy_ms||0)+(a.output_copy_ms||0),'#163f52'],['Postprocess',a.postprocess_ms,'#8ca3b9']];
  const total=groups.reduce((s,x)=>s+(x[1]||0),0)||1;const track=$('latencyTrack'),grid=$('timingGrid');track.replaceChildren();grid.replaceChildren();
  for(const [name,value,color] of groups){
    const bar=document.createElement('span');bar.style.width=((value||0)*100/total)+'%';bar.style.background=color;track.append(bar);
    const item=document.createElement('div');item.className='timing-item';const label=document.createElement('span'),dot=document.createElement('i');dot.className='legend-dot';dot.style.background=color;label.append(dot,document.createTextNode(name));
    const v=document.createElement('b');v.append(document.createTextNode(num(value)+' '));const unit=document.createElement('small');unit.textContent='ms';v.append(unit);item.append(label,v);grid.append(item);
  }
}
function events(items){
  const box=$('events');box.replaceChildren();
  if(!items?.length){const p=document.createElement('p');p.className='quiet';p.textContent='Zone transitions will appear here.';box.append(p);return;}
  for(const ev of items.slice(0,12)){
    const row=document.createElement('div');row.className='event';const dot=document.createElement('i');dot.className='event-dot '+ev.status.toLowerCase();
    const t=document.createElement('time');t.textContent=new Date(ev.time).toLocaleTimeString([],{hour12:false});
    const label=document.createElement('strong');label.textContent={DANGER:'Intrusion detected',CLEAR:'Zone cleared',CALIBRATE:'Calibration required',CHECKING:'Checking occupancy'}[ev.status]||ev.status;
    const count=document.createElement('span');count.className='event-count';count.textContent=ev.inside_count+' in zone';row.append(dot,t,label,count);box.append(row);
  }
}
function loadSettings(s){
  if(!s)return;
  if(![...$('classes').options].some(o=>o.value===s.classes)){$('classes').add(new Option(s.classes,s.classes));}
  $('classes').value=s.classes;$('zoneMode').value=s.zone_mode;$('confidence').value=s.confidence*100;set('confValue',Math.round(s.confidence*100)+'%');
  if(![...$('previewRate').options].some(o=>Number(o.value)===s.preview_fps)){$('previewRate').add(new Option(s.preview_fps+' FPS',s.preview_fps));}
  $('previewRate').value=s.preview_fps;settingsLoaded=true;
}
async function pollStatus(){
  try{
    const r=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(3500)});
    if(!r.ok)throw Error(r.status);status=await r.json();lastStatusAt=performance.now();
    const switching=Boolean(status.switching_model);
    $('applyModel').disabled=switching;
    $('modelChoice').disabled=switching;
    for(const choice of status.model_choices||[]){
      const option=[...$('modelChoice').options].find(o=>o.value===choice.profile);
      if(option)option.disabled=!choice.available;
    }
    set('modelMessage',switching?'Switching to '+status.switching_model+'...':status.model_switch_error||('Active: '+(status.model?.name||'loading model...')));
    if(switching){frameAvailable=false;canvas.style.opacity='.5';}
    if(lastRun!==status.run_id){lastRun=status.run_id;lastSeq=0;displayTimes=[];settingsLoaded=false;if(editing)endEdit();}
    $('connection').className='connection online';$('connection').replaceChildren(Object.assign(document.createElement('i'),{}),document.createTextNode('Board connected'));
    banner(status.status);set('inside',['ERROR','STALE'].includes(status.status)?'—':String(status.inside_count??0));
    set('detectionFps',num(status.detection_fps));set('cameraFps',num(status.camera_fps));set('invokeMs',num(status.average?.invoke_ms));set('latencyMs',num(status.average?.read_to_decision_ms));set('p95',num(status.latency_p95_ms));
    const notices=[status.error,status.warning,!status.led_available?status.led_message:null].filter(Boolean);$('notice').hidden=!notices.length;$('notice').textContent=notices.join(' · ');
    set('zoneName',status.zone?.length?'Safety zone active':'No zone defined');set('zoneInfo',status.zone?.length?status.zone.length+' corners · saved on UNO Q':'Calibrate to begin monitoring');
    set('calibrate',status.zone?.length?'Recalibrate zone +':'Define safety zone +');$('calibrate').disabled=!frameAvailable;
    if(status.camera?.width)set('cameraResolution',`${status.camera.width} × ${status.camera.height} / USB`);
    set('modelInput',status.model?.input_size?`${status.model.input_size} × ${status.model.input_size} · ${status.model.input_dtype?.toUpperCase()} · ${status.model.profile||''}`:'Loading…');
    set('modelPrecision',status.model?.input_dtype?.toUpperCase()||'…');
    set('threads',status.model?.threads??'—');set('memory',num(status.rss_mib)+' MiB');set('cameraPath',status.camera?.path_mode??'Connecting…');
    const mins=Math.floor((status.uptime_s||0)/60),secs=Math.floor((status.uptime_s||0)%60);set('session',`${mins}:${String(secs).padStart(2,'0')} / ${(status.completed||0).toLocaleString()} detections`);
    targetFps=status.preview_target_fps||20;if(!settingsLoaded)loadSettings(status.settings);
    timing(status);events(status.events);
    history.push(status.detection_fps||0);history=history.slice(-60);const max=Math.max(25,...history);
    set('statusDetail',descriptions[status.status]?.[1]||'');
    const path=history.map((v,i)=>`${i?'L':'M'}${i*240/Math.max(1,history.length-1)},${26-v/max*23}`).join(' ');$('sparkPath').setAttribute('d',path+' L240,28 L0,28 Z');
  }catch(e){
    $('connection').className='connection offline';$('connection').replaceChildren(document.createElement('i'),document.createTextNode('Disconnected'));
    banner('OFFLINE');set('inside','—');
  }
  setTimeout(pollStatus,500);
}
function message(text,error=false){set('message',text);$('message').style.color=error?'#b9473e':'#087d71';}
function drawDraft(){ectx.clearRect(0,0,edit.width,edit.height);polygon(ectx,draft,'#f6ce66',edit.width,edit.height,true);set('editHelp',`${draft.length} of 12 corners · click around the perimeter`);$('saveZone').disabled=draft.length<3;}
async function startEdit(){
  if(!frameAvailable)return message('Wait for a camera frame first.',true);
  if(document.fullscreenElement)await document.exitFullscreen();
  editing=true;draft=[];editSignature=status?.camera_signature;
  edit.hidden=false;$('editControls').hidden=false;$('calibrate').hidden=true;$('frozenLabel').hidden=false;drawDraft();message('The saved zone remains active while you edit.');
}
function endEdit(){editing=false;edit.hidden=true;$('editControls').hidden=true;$('calibrate').hidden=false;$('frozenLabel').hidden=true;ectx.clearRect(0,0,edit.width,edit.height);displayTimes=[];lastSeq=0;}
edit.addEventListener('pointerdown',e=>{if(!editing||draft.length>=12)return;const r=edit.getBoundingClientRect();draft.push([Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]);drawDraft();});
async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(6000)});
  const data=await r.json();if(!r.ok)throw Error(data.error||r.status);return data;
}
$('calibrate').addEventListener('click',startEdit);
$('undo').addEventListener('click',()=>{draft.pop();drawDraft();});$('resetDraft').addEventListener('click',()=>{draft=[];drawDraft();});
$('cancelEdit').addEventListener('click',()=>{endEdit();message('Calibration unchanged.');});
$('saveZone').addEventListener('click',async()=>{
  $('saveZone').disabled=true;
  try{await post('/api/zone',{points:draft,camera_signature:editSignature});endEdit();message('Calibration saved on the board.');}
  catch(e){message(e.message,true);$('saveZone').disabled=draft.length<3;}
});
$('confidence').addEventListener('input',()=>set('confValue',$('confidence').value+'%'));
$('applySettings').addEventListener('click',async()=>{
  $('applySettings').disabled=true;
  try{await post('/api/settings',{classes:$('classes').value,zone_mode:$('zoneMode').value,confidence:Number($('confidence').value)/100,preview_fps:Number($('previewRate').value)});message('Detection settings saved.');}
  catch(e){message(e.message,true);}finally{$('applySettings').disabled=false;}
});
$('fullscreen').addEventListener('click',async()=>{if(editing)return message('Finish calibration before expanding the video.',true);try{await $('videoStage').requestFullscreen();}catch(e){message('Full screen is unavailable in this browser.',true);}});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&editing){endEdit();message('Calibration unchanged.');}});
$('calibrate').disabled=true;pollStatus();frames();
$('applyModel').addEventListener('click',async()=>{
  $('applyModel').disabled=true;
  if(editing)endEdit();
  try{
    await post('/api/model',{profile:$('modelChoice').value});
    set('modelMessage','Switch accepted. Waiting for model warmup and camera startup...');
    frameAvailable=false;lastSeq=0;displayTimes=[];history=[];
  }catch(e){set('modelMessage',e.message);$('applyModel').disabled=false;}
});

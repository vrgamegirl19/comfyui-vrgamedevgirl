const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const { test } = require('node:test');
const path = require('node:path');
const s = readBuilderSource();
const part = (a,b) => s.slice(s.indexOf(a),s.indexOf(b,s.indexOf(a)));
test('timeline scrub applies its last queued move before release', () => {
  const events={},raf=new Map(),positions=[];
  const c={state:{},timelineCanvas:{},playhead:{},window:{addEventListener:(n,f)=>events[n]=f,removeEventListener:n=>delete events[n]},requestAnimationFrame:f=>{raf.set(1,f);return 1;},cancelAnimationFrame:id=>raf.delete(id),seekGlobalTimelineFromEvent:e=>positions.push(e.x),updateAudioScrubbers(){}};
  vm.createContext(c);vm.runInContext(functionSource(s, 'beginGlobalTimelineScrub'),c);
  c.beginGlobalTimelineScrub({button:0,target:c.timelineCanvas,x:1,preventDefault(){},stopPropagation(){}});
  events.pointermove({x:5});events.pointermove({x:10});events.pointerup();
  assert.deepEqual(positions,[1,10]);assert.equal(raf.size,0);
});
function playFixture() {
  const waits=[],calls=[];
  const c={waits,calls,state:{sceneSelectionUsesGlobalAudio:true},playButton:{},stopButton:{},playing:false,isTimelinePlaying:()=>c.playing,stopSilentTimelinePlayback:()=>{c.playing=false;},updatePlayPauseButton(){},updateAudioScrubbers(){},currentGlobalTime:()=>0,playbackDuration:()=>20,activeSegment:()=>({start:0}),playbackSegmentAtTime:()=>({}),selectedSegmentVideoPath:()=> 'clip',syncPreviewPlayback(){},waitForPreviewVideoReady:()=>new Promise(r=>waits.push(r)),localPlaybackTime:()=>0,ensureGlobalTimelineAudioSource:()=>false,startSilentTimelinePlayback:()=>{c.playing=true;calls.push('play');},audio:{currentTime:0,pause(){}},sceneAudio:{currentTime:0,pause(){}},previewVideo:{paused:true}};
  vm.createContext(c);
  c.prepareAudioEdits = undefined;
  vm.runInContext(part('  const playStart = { inFlight: false, request: 0 };', '\n')+'\n'+functionSource(s, 'cancelPreviewPlayStart')+'\n'+functionSource(s, 'pauseAllAudio')+part('  playButton.onclick = async','  multiSelectButton.onclick')+part('  stopButton.onclick =','  timelineCanvas.addEventListener("pointerdown"'),c);
  return c;
}
test('Stop cancels Play while waiting for video readiness',async()=>{
  const c=playFixture();const p=c.playButton.onclick();c.stopButton.onclick();c.waits.shift()();await p;assert.deepEqual(c.calls,[]);
});
test('closing the builder cancels pending Play',async()=>{
  const c=playFixture();Object.assign(c,{builderLifecycle:{keydownHandler:null,resourceTimer:0,resourceController:null,resourceResizeObserver:null},builderETAState:{timer:0,log:null},builderETAResizeObserver:{disconnect(){}},window:{removeEventListener(){}},toastNotificationHandler(){},restoreBrowserAiDownloadsQuietly:async()=>{},overlay:{remove(){}},clearTimeout(){},clearInterval(){}});
  vm.runInContext(part('  const closeBuilderNow =', '\n  const ')+';globalThis.closeNow=closeBuilderNow;',c);
  const p=c.playButton.onclick();c.closeNow();c.waits.shift()();await p;assert.deepEqual(c.calls,[]);
});
test('second Play click cancels a pending start',async()=>{
  const c=playFixture();const p=c.playButton.onclick();await c.playButton.onclick();c.waits.shift()();await p;assert.deepEqual(c.calls,[]);
});
test('cancelled old request cannot clear or start a newer request',async()=>{
  const c=playFixture();const old=c.playButton.onclick();c.stopButton.onclick();const next=c.playButton.onclick();c.waits.shift()();await old;assert.deepEqual(c.calls,[]);c.waits.shift()();await next;assert.deepEqual(c.calls,['play']);
});
test('uncancelled Play starts once after readiness',async()=>{
  const c=playFixture();const p=c.playButton.onclick();assert.deepEqual(c.calls,[]);c.waits.shift()();await p;assert.deepEqual(c.calls,['play']);
});
test('Stop cancels Play while preparing edited audio', async () => {
  const c = playFixture();
  let resolveMix;
  c.prepareAudioEdits = () => new Promise(resolve => { resolveMix = resolve; });
  c.setGlobalPlaybackTime = () => {};
  c.render = () => {};
  const pending = c.playButton.onclick();
  c.stopButton.onclick();
  resolveMix({ audio_path: 'mix.wav' });
  await pending;
  assert.deepEqual(c.calls, []);
  assert.equal(c.waits.length, 0);
});
for (const latest of [5,12]) test(`seek queue respects latest requested position ${latest}`,()=>{
  const handlers={};const video={currentTime:5,seeking:true,readyState:2,paused:true,dataset:{cacheKey:'clip'},style:{},addEventListener:(n,f)=>handlers[n]=f};
  const c={previewVideo:video,state:{isScrubbing:true},isTimelinePlaying:()=>false,activeSegment:()=>({}),postProcessComparePreview:{hide(){}},selectedSegmentVideoPath:()=> 'clip',selectedSegmentVideoCacheKey:()=> 'clip',localPlaybackTime:(_,time)=>time};
  vm.createContext(c);vm.runInContext('const previewVideoState={loadTimer:null,syncPause:false,pendingSeekTarget:null};'+part('  previewVideo.addEventListener("seeked",','  // Hidden element')+functionSource(s, 'syncPreviewPlayback'),c);
  c.syncPreviewPlayback(10);c.syncPreviewPlayback(latest);video.seeking=false;handlers.seeked();assert.equal(video.currentTime,latest);
});

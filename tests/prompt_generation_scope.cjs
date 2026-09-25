const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const s=fs.readFileSync(path.join(__dirname,'../web/VRGDG_StoryboardBuilderUI.js'),'utf8');
const code=s.slice(s.indexOf('  async function createAllPromptsWithGemma'),s.indexOf('  stepPrompts.onclick'));
for(const mode of ['storyboard_prompts','image_to_video_prep']) for(const selected of [false,true]) for(const choice of [null,'missing','all']) test(`${mode}, selected=${selected}, choice=${choice}`,async()=>{
 const field=mode==='image_to_video_prep'?'video_prompt':'image_prompt';
 const scenes=[{id:'a',[field]:'manual existing'},{id:'b',[field]:''},{id:'c',[field]:''}];
 const targets=selected?scenes.slice(0,2):scenes;const calls=[];let asked=0;
 const c={state:{mode},getSelectedScenes:()=>selected?targets:[],currentRows:()=>scenes,choosePromptGenerationScope:async(rows,flag)=>{asked++;assert.equal(rows,targets);assert.equal(flag,selected);return choice;},
 promptRunnerName:()=> 'LLM',promptRunnerGenericName:()=> 'LLM',gemmaAllButton:{},keepGemmaLoadedInput:{checked:true},createStoryboardProgressWindow:()=>({set(){},close(){}}),createScenePromptForActiveMode:async(scene)=>calls.push(scene.id),saveStoryboard:async()=>{},createToast(){},renderTable(){}};
 vm.createContext(c);vm.runInContext(code,c);await c.startAllPromptsWithGemma();
 assert.equal(asked,1);assert.deepEqual(calls,choice===null?[]:targets.filter(x=>choice!=='missing'||!x[field]).map(x=>x.id));
});
test('image scope dialog counts image prompts and disables Only Missing when none are missing',()=>{
 const buttons=[];const element=()=>({style:{},append(){},setAttribute(){},addEventListener(){},focus(){}});
 const c={requestAnimationFrame:f=>f(),state:{mode:'storyboard_prompts'},promptRunnerName:()=> 'LLM',document:{createElement:element,body:{append(){}},addEventListener(){}},makeButton:(label,kind)=>{const b={...element(),label,kind};buttons.push(b);return b;}};
 vm.createContext(c);const a=s.indexOf('  const choosePromptGenerationScope');const b=s.indexOf('  async function createAllPromptsWithGemma',a);
 // Only the chooser is needed; stop before the next helper if present.
 const chunk=s.slice(a,s.indexOf('  function isRecoverableStoryboardBatchError',a));vm.runInContext(chunk+';globalThis.choose=choosePromptGenerationScope;',c);
 c.choose([{image_prompt:'exists',video_prompt:''}],true);
 assert.equal(buttons[0].label,'Only Missing (0)');assert.equal(buttons[0].disabled,true);assert.equal(buttons[0].kind,'');
});

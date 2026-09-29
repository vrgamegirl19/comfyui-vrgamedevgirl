import { app } from "../../scripts/app.js";

const CUSTOM = "Custom / Keep typed style";
const PRESETS = {
  "Rock / Classic Rock": "English, classic rock, powerful warm lead vocal, overdriven electric guitars, Hammond organ, electric bass and live acoustic drums, anthemic hooks, dynamic analog studio production, 118 BPM",
  "Rock / Indie Rock": "English, indie rock, intimate slightly raw lead vocal, jangly electric guitars, melodic bass and energetic live drums, bittersweet memorable melody, roomy natural production, 126 BPM",
  "Rock / Arena Rock": "English, arena rock, soaring commanding lead vocal, wide distorted guitars, driving bass and thunderous live drums, huge singalong chorus, polished expansive production, 124 BPM",
  "Rock / Blues Rock": "English, blues rock, gritty soulful lead vocal, expressive electric guitar, Hammond organ, thick bass and loose live drums, swaggering groove and blues-inflected melody, warm vintage production, 104 BPM",
  "Rock / Pop Punk": "English, pop punk rock, urgent youthful lead vocal, bright distorted guitars, driving bass and fast punchy live drums, concise melodic verses and explosive singalong chorus, polished energetic production, 168 BPM",
  "Pop / Warm Piano Pop": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, polished intimate production, 88 BPM",
  "Pop / Synth Pop": "English, cinematic synth-pop, expressive female lead vocal, analog synthesizers, pulsing electric bass and punchy electronic drums, emotional uplifting hook, wide polished production, 112 BPM",
  "Pop / Dance Pop": "English, modern dance-pop, confident bright lead vocal, glossy synthesizers, deep bass and crisp four-on-the-floor drums, immediate melodic chorus, energetic radio-ready production, 124 BPM",
  "Pop / Indie Pop": "English, dreamy indie pop, soft conversational lead vocal, clean guitar, warm synthesizers, melodic bass and light organic drums, wistful hook and airy layered production, 102 BPM",
  "Pop / Power Ballad": "English, dramatic pop power ballad, vulnerable lead vocal rising to a soaring chorus, grand piano, cinematic strings, electric guitar, bass and large live drums, emotional slow build, polished production, 76 BPM",
  "Country / Modern Country": "English, modern country, clear heartfelt lead vocal, acoustic guitar, clean electric guitar, pedal steel, warm bass and punchy live drums, conversational verses and a broad memorable chorus, polished Nashville production, 96 BPM",
  "Country / Honky-Tonk": "English, traditional honky-tonk country, lively twangy lead vocal, Telecaster guitar, fiddle, pedal steel, upright-style bass and shuffling drums, playful dancehall energy, vintage production, 132 BPM",
  "Country / Country Pop": "English, country pop, bright expressive lead vocal, acoustic guitar, subtle banjo, clean electric guitar, rounded bass and modern drums, uplifting melodic chorus, glossy spacious production, 108 BPM",
  "Country / Americana": "English, rootsy Americana, weathered intimate lead vocal, fingerpicked acoustic guitar, mandolin, fiddle, upright bass and restrained live drums, reflective storytelling melody, natural room production, 84 BPM",
  "Country / Outlaw Country": "English, outlaw country, low rugged lead vocal, dry acoustic guitar, gritty electric guitar, pedal steel, steady bass and sparse live drums, defiant storytelling and dusty analog production, 92 BPM",
  "Rap / Boom Bap": "English, boom bap rap, focused articulate rapper, chopped soul samples, dusty piano, deep bass and hard swung drums, dense internal-rhyme cadence and a memorable hook, gritty warm production, 92 BPM",
  "Rap / Trap": "English, modern trap rap, confident rhythmic rapper with restrained melodic ad-libs, dark bell synths, sliding sub bass and crisp hi-hat patterns, spacious verses and forceful hook, polished heavy production, 142 BPM",
  "Rap / Conscious": "English, conscious rap, thoughtful expressive rapper with clear diction, warm jazz piano, subtle guitar, rounded bass and laid-back drums, narrative verses and soulful refrain, organic detailed production, 88 BPM",
  "Rap / Melodic": "English, melodic rap, emotive lead alternating sung hooks and agile rap verses, atmospheric synthesizers, guitar textures, deep sub bass and half-time drums, bittersweet modern production, 138 BPM",
  "Rap / Cinematic": "English, cinematic rap, commanding dramatic rapper, low strings, brass accents, dark piano, massive bass and hard orchestral drums, escalating verses and triumphant hook, wide theatrical production, 96 BPM",
  "Hip Hop / Golden Age": "English, golden-age hip hop, charismatic rhythmic lead vocal, funk and jazz sample textures, upright-style bass, scratches and punchy breakbeat drums, call-and-response hook, warm tape production, 98 BPM",
  "Hip Hop / Neo-Soul": "English, neo-soul hip hop, smooth sung lead with relaxed rap passages, Rhodes piano, mellow guitar, warm bass and pocket drums, rich harmony and intimate late-night production, 86 BPM",
  "Hip Hop / Alternative": "English, alternative hip hop, expressive unconventional lead vocal, warped keyboards, textured samples, elastic bass and off-kilter drums, surprising structure and experimental spacious production, 94 BPM",
  "Hip Hop / West Coast": "English, West Coast hip hop, laid-back confident rapper, bright synth lead, clean electric piano, deep rounded bass and crisp bouncing drums, relaxed melodic hook, sunlit polished production, 96 BPM",
  "Hip Hop / Lo-Fi": "English, lo-fi hip hop song, soft close-miked lead vocal with relaxed rap phrasing, dusty Rhodes, muted guitar, mellow bass and swung vinyl-textured drums, understated hook and hazy intimate production, 82 BPM",
  "Metal / Heavy Metal": "English, traditional heavy metal, powerful high-register lead vocal, twin distorted guitars, galloping bass and forceful live drums, heroic riffs and anthemic chorus, clear muscular production, 148 BPM",
  "Metal / Thrash Metal": "English, thrash metal, aggressive shouted melodic lead vocal, fast palm-muted guitars, cutting bass and relentless double-kick drums, angular riffs and urgent chorus, tight raw production, 190 BPM",
  "Metal / Metalcore": "English, modern metalcore, intense harsh verses and soaring clean chorus vocals, downtuned guitars, heavy bass and precise double-kick drums, dramatic breakdowns and polished dense production, 154 BPM",
  "Metal / Symphonic Metal": "English, symphonic metal, commanding operatic female lead vocal, heavy guitars, orchestral strings, choir, deep bass and cinematic live drums, grand melodic chorus and expansive production, 132 BPM",
  "Metal / Doom Metal": "English, doom metal, deep mournful lead vocal, massive slow distorted guitars, ominous organ, heavy bass and deliberate drums, bleak sustained melody and cavernous analog production, 62 BPM",
  "90s Alternative / Grunge": "English, 1990s grunge-inspired alternative rock, raw anguished lead vocal, thick detuned guitars, gritty bass and explosive live drums, quiet-loud dynamics and an unpolished room sound, 112 BPM",
  "90s Alternative / Shoegaze": "English, 1990s shoegaze-inspired alternative, distant breathy lead vocal, layered washed-out guitars, melodic bass and driving live drums, dreamy bittersweet melody and dense immersive production, 104 BPM",
  "90s Alternative / Britpop": "English, 1990s Britpop-inspired alternative rock, charismatic melodic lead vocal, bright electric guitars, tuneful bass and lively drums, witty verses and a sweeping singalong chorus, crisp guitar-forward production, 122 BPM",
  "90s Alternative / College Rock": "English, 1990s college alternative rock, earnest understated lead vocal, chiming guitars, melodic bass and loose live drums, literate verses and hooky chorus, dry natural production, 116 BPM",
  "90s Alternative / Industrial": "English, 1990s industrial alternative rock, tense gritty lead vocal, distorted guitars, abrasive synthesizers, mechanical bass and pounding programmed drums, dark repetitive hook and aggressive layered production, 118 BPM",
};

const NODE_NAMES = new Set([
  "VRGDG_YuE2Generate",
  "VRGDG_YuE2Plan",
  "VRGDG_YuE2RenderABC",
]);

function installPresetBehavior(node) {
  const presetWidget = node.widgets?.find((widget) => widget.name === "style_preset");
  const styleWidget = node.widgets?.find((widget) => widget.name === "style");
  if (!presetWidget || !styleWidget || presetWidget.__vrgdgYue2Preset) return;

  presetWidget.__vrgdgYue2Preset = true;
  let applyingPreset = false;
  const originalPresetCallback = presetWidget.callback;
  presetWidget.callback = function (value) {
    const result = originalPresetCallback?.apply(this, arguments);
    const prompt = PRESETS[value];
    if (value !== CUSTOM && prompt) {
      applyingPreset = true;
      styleWidget.value = prompt;
      styleWidget.callback?.(prompt, app.canvas, node, app.canvas?.graph_mouse);
      applyingPreset = false;
      node.setDirtyCanvas?.(true, true);
    }
    return result;
  };

  const originalStyleCallback = styleWidget.callback;
  styleWidget.callback = function () {
    const result = originalStyleCallback?.apply(this, arguments);
    if (!applyingPreset && presetWidget.value !== CUSTOM) {
      presetWidget.value = CUSTOM;
      node.setDirtyCanvas?.(true, true);
    }
    return result;
  };
}

app.registerExtension({
  name: "VRGDG.YuE2StylePresets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      installPresetBehavior(this);
      return result;
    };
  },
});

export const IMAGE_SHOT_TYPES = [
  "close-up shot",
  "extreme close-up shot",
  "medium close-up shot",
  "medium shot",
  "medium wide shot",
  "wide shot",
  "extreme wide shot",
  "full shot",
  "long shot",
  "extreme long shot",
  "establishing shot",
  "master shot",
  "two-shot",
  "three-shot",
  "over-the-shoulder shot",
  "point-of-view shot",
  "first-person shot",
  "insert shot",
  "cutaway shot",
  "reaction shot",
  "detail shot",
  "beauty shot",
  "hero shot",
  "profile shot",
  "frontal shot",
  "rear shot",
  "side shot",
  "low-angle shot",
  "high-angle shot",
  "eye-level shot",
  "bird's-eye view shot",
  "worm's-eye view shot",
  "aerial shot",
  "drone shot",
  "overhead shot",
  "top-down shot",
  "ground-level shot",
  "Dutch angle shot",
  "tilted shot",
  "symmetrical shot",
  "centered shot",
  "off-center shot",
  "silhouette shot",
  "reflection shot",
  "mirror shot",
  "shadow shot",
  "through-the-window shot",
  "through-the-doorway shot",
  "frame-within-a-frame shot",
  "single shot",
  "two-person shot",
  "group shot",
  "crowd shot",
  "face shot",
  "head shot",
  "head-and-shoulders shot",
  "bust shot",
  "waist-up shot",
  "chest-up shot",
  "knee-up shot",
  "cowboy shot",
  "American shot",
  "full-body shot",
  "feet shot",
  "hands shot",
  "eyes shot",
  "mouth shot",
  "object shot",
  "product shot",
  "environment shot",
  "landscape shot",
  "cityscape shot",
  "room shot",
  "hallway shot",
  "doorway shot",
  "car interior shot",
  "dashboard shot",
  "passenger-seat shot",
  "driver-seat shot",
  "cinematic wide shot",
  "moody close-up shot",
  "dramatic low-angle shot",
  "intimate close-up shot",
  "documentary-style shot",
  "surveillance-style shot",
  "security-camera shot",
  "CCTV shot",
  "found-footage shot",
  "vlog-style shot",
  "selfie shot",
  "webcam shot",
  "interview shot",
  "talking-head shot",
  "news-style shot",
  "broadcast-style shot",
  "commercial product shot",
  "lifestyle shot",
  "montage opening shot",
  "transition shot",
  "dreamlike shot",
  "blurred foreground shot",
  "shallow-depth-of-field shot",
  "deep-focus shot",
  "soft-focus shot",
  "backlit shot",
  "lens-flare shot",
  "natural-light shot",
  "night shot",
  "golden-hour shot",
  "blue-hour shot",
];

export const VIDEO_SHOT_TYPES = [
  ...IMAGE_SHOT_TYPES,
  "static shot",
  "locked-off shot",
  "handheld shot",
  "tracking shot",
  "dolly shot",
  "dolly-in shot",
  "dolly-out shot",
  "push-in shot",
  "pull-out shot",
  "zoom-in shot",
  "zoom-out shot",
  "pan shot",
  "whip pan shot",
  "tilt-up shot",
  "tilt-down shot",
  "crane shot",
  "jib shot",
  "Steadicam shot",
  "gimbal shot",
  "follow shot",
  "lead shot",
  "arc shot",
  "orbit shot",
  "360-degree shot",
  "reveal shot",
  "rack-focus shot",
  "focus-pull shot",
  "slow-motion shot",
  "time-lapse shot",
  "hyperlapse shot",
];

export const CAMERA_MOTION_GROUPS = [
  { value: "", label: "Choose camera motion..." },
  {
    label: "Basic Camera Motions",
    options: [
      "pan left", "pan right", "pan up", "pan down", "tilt up", "tilt down",
      "push in", "pull back", "pull out", "dolly in", "dolly out",
      "dolly left", "dolly right", "truck left", "truck right",
      "pedestal up", "pedestal down", "zoom in", "zoom out",
      "slow zoom in", "slow zoom out", "quick zoom in", "snap zoom",
      "crash zoom", "whip pan", "whip left", "whip right", "whip up", "whip down",
    ],
  },
  {
    label: "Orbit / Rotation Motions",
    options: [
      "orbit left", "orbit right", "orbit around subject", "rotate around subject",
      "circle around subject", "180-degree rotation", "360-degree rotation",
      "half-circle orbit", "full-circle orbit", "clockwise orbit",
      "counterclockwise orbit", "spiral around subject", "arc left", "arc right",
      "arc around subject", "wraparound move", "sweeping circular move",
    ],
  },
  {
    label: "Tracking / Following Motions",
    options: [
      "track forward", "track backward", "track left", "track right",
      "tracking shot", "follow shot", "follow behind", "follow in front",
      "lead shot", "side-follow shot", "over-the-shoulder follow",
      "chase shot", "pursuit shot", "walk-and-talk tracking",
      "handheld follow", "gimbal follow", "steadicam follow",
      "smooth follow", "shaky follow",
    ],
  },
  {
    label: "Reveal Motions",
    options: [
      "reveal upward", "reveal downward", "reveal left", "reveal right",
      "slide reveal", "dolly reveal", "pan reveal", "tilt reveal",
      "pull-back reveal", "push-in reveal", "orbit reveal", "crane reveal",
      "rack-focus reveal", "foreground reveal", "doorway reveal",
      "window reveal", "object reveal", "character reveal", "environment reveal",
    ],
  },
  {
    label: "Vertical / Height Motions",
    options: [
      "crane up", "crane down", "jib up", "jib down", "rise up",
      "descend down", "boom up", "boom down", "lift upward", "drop downward",
      "float upward", "sink downward", "aerial rise", "aerial descent",
      "drone rise", "drone descend", "top-down descent",
      "ground-to-sky tilt", "sky-to-ground tilt",
    ],
  },
  {
    label: "Drone / Aerial Motions",
    options: [
      "drone flyover", "drone push in", "drone pull back", "drone rise",
      "drone descend", "drone orbit", "drone circle", "drone follow",
      "drone chase", "drone pass-through", "drone reveal", "aerial tracking",
      "aerial pan", "aerial tilt", "overhead drift", "top-down tracking",
      "bird's-eye pullback", "sweeping aerial move",
    ],
  },
  {
    label: "Handheld / Style Motions",
    options: [
      "handheld shake", "subtle handheld movement", "shaky cam",
      "smooth handheld", "floating camera move", "drifting camera move",
      "breathing camera movement", "documentary-style movement",
      "natural handheld sway", "nervous handheld move", "chaotic handheld move",
      "stabilized gimbal move", "steadicam glide", "slow cinematic glide",
      "smooth cinematic drift",
    ],
  },
  {
    label: "Focus / Lens Motions",
    options: [
      "rack focus", "focus pull", "focus shift",
      "foreground-to-background focus", "background-to-foreground focus",
      "shallow-focus drift", "zoom with focus pull", "dolly zoom",
      "vertigo effect", "crash zoom with focus", "soft-focus transition",
      "focus reveal",
    ],
  },
  {
    label: "POV / Subjective Motions",
    options: [
      "POV walk forward", "POV turn left", "POV turn right", "POV look up",
      "POV look down", "POV stumble", "POV run", "POV chase", "POV fall",
      "POV rise", "POV scan the room", "POV peek around corner",
      "POV lean in", "POV look over shoulder",
    ],
  },
  {
    label: "Transition Motions",
    options: [
      "whip pan transition", "match move", "push-through transition",
      "pass-through transition", "foreground wipe", "camera wipe",
      "object wipe", "spin transition", "rotate transition", "zoom transition",
      "crash zoom transition", "tilt transition", "pan transition",
      "motion blur transition",
    ],
  },
];

export const STILL_CAMERA_STYLE_GROUPS = [
  { value: "", label: "Choose still camera style..." },
  {
    label: "Composition / Framing",
    options: [
      "clean portrait composition", "editorial fashion composition", "cinematic still frame",
      "rule-of-thirds composition", "centered symmetrical composition", "negative space composition",
      "foreground framing", "frame-within-a-frame composition", "environmental portrait",
      "intimate close portrait", "wide environmental still", "dramatic silhouette composition",
    ],
  },
  {
    label: "Lens / Depth",
    options: [
      "shallow depth of field", "deep focus photography", "soft background bokeh",
      "wide-angle perspective", "telephoto compression", "macro detail photography",
      "natural lens perspective", "cinematic anamorphic lens look", "soft-focus portrait lens",
      "crisp studio lens detail",
    ],
  },
  {
    label: "Lighting / Exposure",
    options: [
      "natural window light", "golden-hour photography", "blue-hour photography",
      "high-contrast studio lighting", "soft diffused key light", "dramatic rim lighting",
      "backlit portrait", "low-key lighting", "high-key photography",
      "moody practical lighting", "neon-lit still photography",
    ],
  },
  {
    label: "Still Photography Style",
    options: [
      "editorial magazine photo", "fine-art portrait photography", "documentary still photo",
      "album-cover photography", "cinematic production still", "glossy commercial photo",
      "gritty street photography", "dreamlike fashion editorial", "dramatic character portrait",
      "atmospheric location photography",
    ],
  },
];

export const CHARACTER_MOTION_GROUPS = [
  { value: "", label: "Choose character motion..." },
  {
    label: "Basic Locomotion",
    options: [
      "standing still", "walking", "running", "jogging", "sprinting", "pacing",
      "strolling", "wandering", "marching", "limping", "sneaking", "crawling",
      "climbing", "jumping", "landing", "falling", "tripping", "stumbling",
      "sliding", "spinning", "turning around", "looking around", "backing away",
      "moving forward", "moving sideways", "approaching camera",
      "walking away from camera",
    ],
  },
  {
    label: "Dance / Performance",
    options: [
      "dancing", "freestyle dancing", "slow dancing", "breakdancing",
      "hip-hop dancing", "club dancing", "swaying to music", "head nodding",
      "shoulder bouncing", "foot tapping", "hand waving", "arm swinging",
      "body rolling", "spinning while dancing", "jumping to the beat",
      "performing on stage", "singing into microphone", "rapping into microphone",
      "playing guitar", "playing piano", "playing drums", "DJing", "crowd surfing",
    ],
  },
  {
    label: "Gestures",
    options: [
      "pointing", "waving", "clapping", "snapping fingers", "giving thumbs up",
      "crossing arms", "raising arms", "reaching out", "holding hands up",
      "covering face", "touching chest", "touching head", "brushing hair back",
      "adjusting jacket", "adjusting sunglasses", "putting hands in pockets",
      "throwing hands up", "making hand signs", "beckoning", "saluting",
    ],
  },
  {
    label: "Facial Expression / Head Movement",
    options: [
      "smiling", "laughing", "crying", "frowning", "smirking", "shouting",
      "whispering", "looking at camera", "looking away",
      "looking down", "looking up", "turning head", "tilting head", "nodding",
      "shaking head", "closing eyes", "opening eyes", "blinking",
      "staring intensely",
    ],
  },
  {
    label: "Environment Interaction",
    options: [
      "opening a door", "closing a door", "leaning on a wall", "sitting on a chair",
      "standing up", "sitting down", "lying down", "kneeling", "picking something up",
      "dropping something", "throwing something", "pushing something",
      "pulling something", "carrying something", "leaning over a railing",
      "looking out a window", "walking through smoke", "walking through rain",
      "splashing through water", "kicking dust", "touching a wall",
      "running fingers along a surface",
    ],
  },
  {
    label: "Object Interaction",
    options: [
      "holding microphone", "holding phone", "looking at phone", "taking a photo",
      "recording video", "holding flowers", "holding money", "counting money",
      "holding a drink", "drinking", "smoking", "lighting a cigarette",
      "wearing headphones", "putting on headphones", "removing sunglasses",
      "putting on sunglasses", "holding a weapon prop", "holding a bag",
      "carrying luggage", "tossing keys", "spinning keys", "reading a note",
    ],
  },
  {
    label: "Emotional Action",
    options: [
      "collapsing to knees", "reaching toward camera", "running away",
      "chasing someone", "being chased", "searching for someone", "hiding",
      "waiting", "hesitating", "reacting in shock", "celebrating", "arguing",
      "fighting", "hugging", "pushing away", "walking alone",
      "standing in silence", "looking heartbroken", "looking confident",
      "looking angry", "looking lost",
    ],
  },
  {
    label: "Camera-Facing Motion",
    options: [
      "walking toward camera", "walking past camera", "turning to face camera",
      "looking directly into lens", "reaching toward lens", "pointing at camera",
      "singing to camera", "dancing toward camera", "moving in slow motion",
      "freezing in place", "silhouette movement", "hair blowing in wind",
      "clothing flowing in wind", "walking through frame", "entering frame",
      "exiting frame", "crossing foreground", "moving in background",
    ],
  },
  {
    label: "Group Movement",
    options: [
      "crowd dancing", "crowd jumping", "crowd waving arms", "crowd clapping",
      "people walking around", "people running past", "group marching",
      "group circling character", "group following character",
      "group surrounding character", "backup dancers performing",
      "band performing", "audience cheering", "friends walking together",
      "couple dancing", "couple arguing", "couple embracing",
    ],
  },
  {
    label: "Vehicle / Travel",
    options: [
      "driving", "riding in car", "getting into car", "getting out of car",
      "leaning out car window", "walking beside car", "sitting on car hood",
      "riding motorcycle", "riding bicycle", "skateboarding", "roller skating",
      "riding elevator", "walking down stairs", "walking up stairs",
      "riding escalator", "running through tunnel", "walking across street",
    ],
  },
  {
    label: "Stylized / Surreal Motion",
    options: [
      "floating", "levitation", "falling in slow motion", "spinning in place",
      "walking in reverse", "glitching", "teleporting", "duplicating",
      "morphing pose", "freeze-frame pose", "dramatic turn", "slow-motion walk",
      "wind-swept pose", "hero pose", "shadow dancing", "silhouette dancing",
      "smoke reveal", "light reveal", "walking through sparks",
      "dancing in rain", "falling backward into darkness", "reaching through light",
      "moving like a puppet", "robotic movement", "fluid dreamlike movement",
    ],
  },
];

export function normalizeStoryboardCustomCameraFlowSequence(input) {
  let source = input;
  if (typeof source === "string") {
    const raw = source.trim();
    if (!raw) return [];
    try {
      source = JSON.parse(raw);
    } catch (_error) {
      source = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    }
  }
  if (source && !Array.isArray(source) && typeof source === "object") {
    source = source.shots || source.sequence || source.candidates || source.list || source.items || [];
  }
  if (!Array.isArray(source)) return [];
  return source.map((item) => {
    if (typeof item === "string") {
      const parts = item.trim().replace(/^\s*(?:[-*•]|\d+[.)])\s*/, "").split(/\s+(?:\||—|–|-|->|=>)\s+/);
      return { shot: String(parts[0] || "").trim(), camera: String(parts.slice(1).join(" | ") || "").trim() };
    }
    if (!item || typeof item !== "object") return null;
    return {
      shot: String(item.shot || item.framing || item.type || item.description || item.name || "").trim(),
      camera: String(item.camera || item.camera_motion || item.movement || item.motion || "").trim(),
    };
  }).filter((item) => item?.shot).map((item) => ({ shot: item.shot, camera: item.camera }));
}

export const STORYBOARD_CAMERA_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill missing shot or camera motion fields.",
    sequence: [],
  },
  balanced: {
    label: "Balanced cinematic flow",
    description: "Alternates wide, medium, close, lateral, reveal, and reset shots without repeating inward zooms.",
    guidance: "Use the selected starting shot as the literal first generated frame. Do not add a wider, farther-away, establishing, or full-body lead-in before it. Preserve the selected framing unless the selected camera move explicitly changes scale. Treat inward moves as rare accents, never as a default pattern.",
    sequence: [
      { shot: "wide shot", camera: "slow cinematic drift" },
      { shot: "medium close-up shot", camera: "pull back" },
      { shot: "tracking shot", camera: "side-follow shot" },
      { shot: "close-up shot", camera: "slow orbit left" },
      { shot: "medium wide shot", camera: "dolly right" },
      { shot: "profile shot", camera: "pan reveal" },
      { shot: "low-angle shot", camera: "crane up" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "over-the-shoulder shot", camera: "reveal right" },
      { shot: "full-body shot", camera: "track backward" },
    ],
  },
  intimate_closeups: {
    label: "Intimate close-ups",
    framing_candidates: true,
    description: "Uses only distinct frame-filling close-ups, body-detail reveals, tight seated poses, and tight upper-body compositions.",
    guidance: "Every shot remains close, intimate, and frame-filling. The furthest framing is a tightly composed upper-body shot. Never use a wide shot, distant shot, full-body shot, small-in-frame composition, or full environment view. Each shot must use a distinct framing, angle, subject detail, or camera movement.",
    sequence: [
      { shot: "extreme close-up of one eye, slowly pulling back to reveal the full face", camera: "slow pullback" },
      { shot: "extreme close-up of the mouth, slowly pulling back to the upper body", camera: "slow pullback" },
      { shot: "close-up of both eyes", camera: "slight sideways camera slide" },
      { shot: "tight face close-up from a three-quarter angle", camera: "slow three-quarter orbit" },
      { shot: "tight profile close-up of the face", camera: "gentle lateral drift" },
      { shot: "close-up of the hand resting on the hip, slowly panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of fingers brushing through the hair, tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "close-up of the shoulder and neck, panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of the lips, then tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "close-up of the eyes, slowly tilting downward to the hands", camera: "slow downward tilt" },
      { shot: "close-up of the feet walking, panning upward along the body to the face", camera: "slow upward pan" },
      { shot: "close-up of the feet standing still, slowly tilting upward to the upper body", camera: "slow upward tilt" },
      { shot: "close-up of one hand reaching toward the camera, revealing the face behind it", camera: "slow reveal" },
      { shot: "close-up of hands gripping clothing, panning upward to the face", camera: "slow upward pan" },
      { shot: "close-up of a hand touching the chest, tilting upward to the eyes", camera: "slow upward tilt" },
      { shot: "tight seated portrait with the knees, torso, and face filling the frame", camera: "slow lateral drift" },
      { shot: "seated side-profile shot with the body filling the entire frame", camera: "slow side pan" },
      { shot: "seated curled-up pose framed from knees to face", camera: "slow push-in" },
      { shot: "tight upper-body shot with the subject leaning toward the camera", camera: "slow push-in" },
      { shot: "tight upper-body shot from behind the shoulder, revealing the face in profile", camera: "slow shoulder reveal" },
      { shot: "low-angle close-up from the waist upward, keeping the face near the top of frame", camera: "slow upward tilt" },
      { shot: "high-angle close-up looking down at the subject’s face and upper body", camera: "slow downward drift" },
      { shot: "tight overhead shot of the subject lying down, filling the frame", camera: "slow overhead drift" },
      { shot: "close-up of the subject lying on their side, slowly panning from feet to face", camera: "slow lateral pan" },
      { shot: "close-up from behind as the subject turns their head toward the camera", camera: "slow turn reveal" },
      { shot: "tight front-facing upper-body shot with a slow push-in toward the eyes", camera: "slow push-in" },
      { shot: "tight side shot with a slow horizontal pan from shoulder to face", camera: "slow horizontal pan" },
      { shot: "close-up framed through the subject’s moving hair", camera: "gentle hair reveal" },
      { shot: "reflection close-up in a mirror, slowly moving from the reflection’s hands to face", camera: "slow reflection pan" },
      { shot: "tight silhouette close-up with the face and shoulders filling the frame", camera: "slow silhouette drift" },
    ],
  },
  music_video: {
    label: "Music video shots",
    framing_candidates: true,
    description: "Uses performance, movement, location, reveal, tracking, and rhythmic music-video shot ideas.",
    guidance: "Use only the selected music-video candidate framing for each shot. Choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [
      { shot: "wide performance shot with the subject centered in the environment", camera: "slow performance drift" },
      { shot: "full-body shot walking toward the camera", camera: "track backward" },
      { shot: "side-tracking shot following the subject's movement", camera: "side track" },
      { shot: "low-angle full-body performance shot", camera: "low-angle tracking move" },
      { shot: "high-angle shot looking down at the subject", camera: "high-angle crane drift" },
      { shot: "slow push-in from wide to medium framing", camera: "slow push-in" },
      { shot: "pull-back revealing the full location", camera: "slow pull-back reveal" },
      { shot: "circular camera move around the subject", camera: "full circular orbit" },
      { shot: "profile shot while the subject walks", camera: "profile side track" },
      { shot: "rear tracking shot following the subject from behind", camera: "rear follow" },
      { shot: "overhead shot of the subject lying or moving on the ground", camera: "overhead tracking drift" },
      { shot: "Dutch-angle performance shot", camera: "tilted handheld drift" },
      { shot: "static shot with the subject moving through the frame", camera: "locked-off composition" },
      { shot: "camera crossing from behind the subject to the front", camera: "wraparound reveal" },
      { shot: "full-body dancing shot with rhythmic camera motion", camera: "rhythmic orbit" },
      { shot: "walking past the camera in profile", camera: "profile pass-by track" },
      { shot: "camera following the subject through a doorway", camera: "doorway follow-through" },
      { shot: "wide shot with the subject isolated in the environment", camera: "slow environmental drift" },
      { shot: "slow-motion full-body movement shot", camera: "smooth-motion follow" },
      { shot: "handheld roaming performance shot", camera: "roaming handheld move" },
      { shot: "silhouette shot against strong backlighting", camera: "slow silhouette reveal" },
      { shot: "mirror or reflection shot", camera: "reflection slide" },
      { shot: "long-lens shot compressing the background", camera: "compressed lateral drift" },
      { shot: "ground-level shot looking upward as the subject approaches", camera: "ground-level push-in" },
      { shot: "elevated crane-style reveal", camera: "crane rise" },
      { shot: "continuous one-take shot following the subject", camera: "continuous tracking move" },
      { shot: "whip-pan transition between movements or locations", camera: "whip pan" },
      { shot: "match cut connecting two poses or actions", camera: "match-cut reframing" },
    ],
  },
  fisheye_distorted: {
    label: "Fisheye and Distorted-Lens Shots",
    framing_candidates: true,
    description: "Uses only fisheye, warped-perspective, curved-reflection, and distorted-lens compositions.",
    guidance: "Use only the selected fisheye or distorted-lens candidate framing for each shot. Keep the warped perspective visibly intentional and choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [
      { shot: "extreme fisheye shot with the subject leaning toward the lens", camera: "dramatic fisheye push-in" },
      { shot: "full-body fisheye shot with exaggerated perspective", camera: "wide fisheye drift" },
      { shot: "low-angle fisheye shot as the subject looks down into the camera", camera: "low fisheye tilt-up" },
      { shot: "crouching close to the camera and staring into it", camera: "fisheye push-in" },
      { shot: "reaching one hand toward the fisheye lens", camera: "hand-to-lens reveal" },
      { shot: "slowly circling around the stationary camera", camera: "orbit around fixed lens" },
      { shot: "camera placed on the floor as the subject walks around it", camera: "ground-level fisheye rotation" },
      { shot: "camera tilted upward as the subject bends toward the lens", camera: "upward fisheye tilt" },
      { shot: "distorted wide shot with the environment curving around the subject", camera: "curved-perspective drift" },
      { shot: "face close to the lens while the subject's body recedes behind it", camera: "fisheye pullback" },
      { shot: "camera rotating slightly as the subject stares into the lens", camera: "rolling fisheye rotation" },
      { shot: "moving past the lens with warped motion", camera: "warped pass-by move" },
      { shot: "camera positioned between the subject's feet as the subject looks down", camera: "upward fisheye tilt" },
      { shot: "fisheye shot from inside a doorway as the subject approaches", camera: "doorway fisheye push-in" },
      { shot: "looking through glass directly into the lens", camera: "glass distortion drift" },
      { shot: "framing the camera with both hands", camera: "hands-around-lens reveal" },
      { shot: "low fisheye shot with the subject's hair falling toward the camera", camera: "low fisheye tilt" },
      { shot: "leaning into the lens, then suddenly pulling away", camera: "rapid fisheye pullback" },
      { shot: "distorted reflection in curved glass or a mirror", camera: "curved reflection slide" },
      { shot: "camera placed on the ground while the subject walks around it", camera: "ground-level orbit" },
      { shot: "camera pushed toward the subject's face, then pulled back to full body", camera: "fisheye push-pull" },
      { shot: "fisheye lens pointed upward as the subject spins above it", camera: "upward spinning fisheye" },
      { shot: "singing directly into the lens with exaggerated perspective", camera: "fisheye vocal push-in" },
      { shot: "handheld camera held at arm's length as the subject looks into it", camera: "handheld fisheye sway" },
      { shot: "centered fisheye shot with the entire background bending around the subject", camera: "centered warped drift" },
    ],
  },
  custom: {
    label: "Custom",
    framing_candidates: true,
    description: "Uses the project-specific camera-shot list imported by the user.",
    guidance: "Use only the selected framing from the user's custom camera-shot list. Choose the strongest fit for the lyrics, story, character motion, performance, and location. Preserve distinct shot variety across the scene and avoid unnecessary repetition across scenes; sensible reuse is allowed when the action clearly calls for it or the candidate pool is exhausted.",
    sequence: [],
  },
  quiet: {
    label: "Quiet dramatic",
    description: "Slower restrained camera choices for emotional, eerie, or cinematic scenes.",
    sequence: [
      { shot: "establishing shot", camera: "slow cinematic drift" },
      { shot: "medium wide shot", camera: "locked-off shot" },
      { shot: "profile shot", camera: "subtle handheld movement" },
      { shot: "intimate close-up shot", camera: "slow zoom out" },
      { shot: "reflection shot", camera: "focus pull" },
      { shot: "centered shot", camera: "pull back" },
      { shot: "silhouette shot", camera: "tilt up" },
      { shot: "close-up shot", camera: "drifting camera move" },
    ],
  },
  energetic: {
    label: "Fast energetic",
    description: "Bigger changes between scenes with fast moves, reveals, tracking, and punchier reframing.",
    sequence: [
      { shot: "wide shot", camera: "whip pan transition" },
      { shot: "medium shot", camera: "track left" },
      { shot: "close-up shot", camera: "whip right" },
      { shot: "low-angle shot", camera: "orbit reveal" },
      { shot: "full-body shot", camera: "dolly left" },
      { shot: "Dutch angle shot", camera: "push-through transition" },
      { shot: "medium wide shot", camera: "crane up" },
      { shot: "reaction shot", camera: "rack focus" },
      { shot: "tracking shot", camera: "chase shot" },
      { shot: "detail shot", camera: "snap zoom" },
    ],
  },
};

function storyboardMotionFamily(motion = "") {
  const text = String(motion || "").toLowerCase();
  if (/push|dolly in|zoom in|track forward|crash zoom|snap zoom/.test(text)) return "in";
  if (/pull|dolly out|zoom out|track backward/.test(text)) return "out";
  if (/orbit|arc|circle|rotation|rotate/.test(text)) return "orbit";
  if (/track|follow|dolly left|dolly right|truck/.test(text)) return "track";
  if (/reveal|tilt|crane|jib|rise|descend/.test(text)) return "reveal";
  if (/focus|rack/.test(text)) return "focus";
  return text.split(/\s+/).slice(0, 2).join(" ");
}

export function storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion = "", customSequence = []) {
  const preset = STORYBOARD_CAMERA_FLOW_PRESETS[profileKey] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
  const customFlow = profileKey === "custom";
  const sequence = profileKey === "custom"
    ? normalizeStoryboardCustomCameraFlowSequence(customSequence)
    : (preset.sequence || []);
  if (!sequence.length) return null;
  let entry = sequence[sceneIndex % sequence.length];
  // A custom list is an authored shot-by-shot plan. Preserve its exact order;
  // the repetition-avoidance rule is only for generated preset sequences.
  if (!customFlow && previousMotion && storyboardMotionFamily(entry.camera) === storyboardMotionFamily(previousMotion)) {
    entry = sequence[(sceneIndex + 1) % sequence.length] || entry;
  }
  return entry;
}

export const STORYBOARD_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill still-image shot/composition fields.",
    sequence: [],
  },
  intimate: {
    label: "Intimate character shots",
    description: "Close, emotional stills for faces, hands, expressions, and quiet character moments.",
    sequence: [
      "intimate close-up shot",
      "medium close-up shot",
      "eyes shot",
      "hands shot",
      "profile shot",
      "head-and-shoulders shot",
      "reflection shot",
      "moody close-up shot",
    ],
  },
  music_video_stills: {
    label: "Music video stills",
    description: "Album-cover and performance-friendly framing with cinematic variety but no camera movement.",
    sequence: [
      "medium shot",
      "low-angle shot",
      "wide shot",
      "hero shot",
      "Dutch angle shot",
      "silhouette shot",
      "full-body shot",
      "dramatic low-angle shot",
      "centered shot",
      "beauty shot",
    ],
  },
  editorial: {
    label: "Editorial fashion",
    description: "Stylized portrait, fashion, and magazine-like compositions.",
    sequence: [
      "editorial fashion composition",
      "beauty shot",
      "full-body shot",
      "profile shot",
      "wide environmental still",
      "centered symmetrical composition",
      "negative space composition",
      "commercial product shot",
    ],
  },
  cinematic_story: {
    label: "Cinematic story frames",
    description: "Film-still composition for locations, story beats, and emotionally readable scenes.",
    sequence: [
      "establishing shot",
      "medium wide shot",
      "over-the-shoulder shot",
      "frame-within-a-frame shot",
      "environment shot",
      "reflection shot",
      "silhouette shot",
      "detail shot",
      "wide shot",
    ],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for story-heavy music videos: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led music-video scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const ID_LORA_IMAGE_SHOT_FLOW_PRESETS = {
  off: {
    label: "Off",
    description: "Do not auto-fill film-still composition fields.",
    sequence: [],
  },
  film_dialogue_coverage: {
    label: "Film dialogue coverage",
    description: "Short-film coverage for dialogue scenes: readable faces, eyelines, reactions, and location context.",
    sequence: [
      "medium close-up dialogue shot",
      "over-the-shoulder shot",
      "reaction close-up",
      "two-shot dialogue frame",
      "profile close-up",
      "medium shot with foreground framing",
      "insert detail shot",
      "wide establishing film still",
    ],
  },
  intimate_drama: {
    label: "Intimate drama frames",
    description: "Close emotional film stills for confessions, quiet tension, and character-led scenes.",
    sequence: [
      "tight close-up",
      "intimate medium close-up",
      "profile close-up",
      "hands and face detail shot",
      "reflection close-up",
      "seated conversation frame",
      "shallow-focus reaction shot",
      "low-key portrait frame",
    ],
  },
  noir_story_frames: {
    label: "Noir story frames",
    description: "Moody dramatic coverage with shadows, silhouettes, foregrounds, and tense blocking.",
    sequence: [
      "low-key medium shot",
      "silhouette dialogue frame",
      "over-the-shoulder noir shot",
      "frame-within-a-frame shot",
      "side-lit profile shot",
      "wide empty-space composition",
      "reflection shot",
      "detail insert shot",
    ],
  },
};

export const STORYBOARD_IMAGE_AESTHETIC_PRESETS = [
  { value: "", label: "Default cinematic still", description: "Balanced cinematic lighting, color, and texture for a polished text-to-image prompt.", prompt_guidance: "Create a polished cinematic still with clear subject placement, believable wardrobe and environment details, purposeful lighting, readable composition, lens/framing detail, and a strong music-video production still feeling." },
  { value: "music_video_gloss", label: "Glossy music video", description: "Glossy high-production music-video still, dramatic color contrast, stylish lighting, album-cover polish.", prompt_guidance: "Build a glossy high-production music-video still. Specify stylized wardrobe, intentional pose, dramatic color contrast, polished hair and makeup, expensive-looking lighting, reflective or atmospheric set details, album-cover composition, crisp lens choice, and cinematic depth. Do not merely say glossy music video." },
  { value: "dark_neon", label: "Dark neon", description: "Dark cinematic neon lighting, saturated color accents, glossy reflections, smoky atmosphere, night-club energy.", prompt_guidance: "Build a dark neon cinematic still. Use saturated colored light sources, glossy reflections, wet or polished surfaces, smoke/haze, rim light, deep shadows, vivid accent colors on the subject, and a nightlife or futuristic music-video atmosphere. Describe where the neon comes from and how it shapes the face, outfit, and environment." },
  { value: "editorial_fashion", label: "Editorial fashion", description: "High-fashion editorial photography, intentional posing, refined wardrobe detail, magazine-grade lighting.", prompt_guidance: "Build an editorial fashion photograph, not a plain portrait. Give the subject a deliberate model pose with body angles, hand placement, posture, and gaze. Describe refined wardrobe styling, fabric behavior, accessories, hair/makeup direction, fashion-magazine lighting, background styling, composition, lens/framing, and a strong art-directed theme." },
  { value: "editorial_fashion_photography", label: "Editorial fashion photography", description: "Editorial fashion photography with confident model posing, dramatic styling, creative wardrobe themes, magazine-grade composition, bold makeup and hair, and polished high-resolution lighting.", prompt_guidance: "Build a detailed editorial fashion photograph. Include a confident model pose, strong body line, hand/shoulder/hip placement, dramatic styling choices, creative wardrobe concept, fabric texture and silhouette, bold hair and makeup, accessories, modern magazine composition, art-directed setting, high-resolution studio or location lighting, and a clear fashion story. Do not just write 'editorial fashion composition'." },
  { value: "conceptual_portrait_photography", label: "Conceptual portrait photography", description: "Conceptual portrait photography built around a clear visual idea, symbolic prop, emotional pose, controlled environment, cinematic lighting, and a strong central portrait composition.", prompt_guidance: "Build a conceptual portrait around one clear visual idea. Choose a symbolic prop, object arrangement, or environmental metaphor that fits the scene. Describe the subject's pose, relation to the prop, wardrobe, hair/makeup, controlled setting, color palette, lighting direction, mood, lens/framing, and how the composition communicates the concept visually without explaining it." },
  { value: "avant_garde_fashion_photography", label: "Avant-garde fashion photography", description: "Avant-garde fashion photography with unusual makeup, sculptural hair, strange or powerful poses, experimental styling, abstract studio or surreal setting, and bold high-contrast lighting.", prompt_guidance: "Build an avant-garde fashion photograph. Use unusual makeup, sculptural or geometric hair, experimental wardrobe shape, exaggerated silhouette, strange powerful pose, asymmetrical composition, abstract studio or surreal set design, hard shadows or high-contrast light, unexpected materials, and a bold futuristic or theatrical fashion mood. Make it visually daring, not casual." },
  { value: "beauty_editorial_photography", label: "Beauty editorial photography", description: "Beauty editorial photography focused on close-up makeup, hair, skin texture, eyes, lips, jewelry or face details, soft luxury lighting, and clean magazine beauty composition.", prompt_guidance: "Build a beauty editorial photograph. Use close-up or tight portrait framing focused on eyes, lips, makeup, hair texture, jewelry, nails, skin glow, and face-framing styling. Describe makeup colors, glossy or matte finish, hair placement, accessories near the face, soft diffused lighting, clean backdrop, shallow depth of field, and luxury magazine composition." },
  { value: "high_fashion_editorial", label: "High fashion editorial", description: "High fashion editorial photography inspired by dramatic fashion competition shoots: couture wardrobe, expressive posing, epic location, wind or fabric movement, glamorous styling, and cinematic full-body framing.", prompt_guidance: "Build a high fashion editorial shoot like a dramatic fashion competition photo. Use couture-level wardrobe, exaggerated fabric movement, strong full-body or three-quarter pose, elongated body line, expressive hands and face, wind or motion in hair/fabric, glamorous accessories, bold makeup, epic location styling, low or cinematic camera angle, dramatic natural or studio lighting, and a clear fashion-story payoff. The prompt must describe the actual fashion shoot details, not just name the style." },
  { value: "creative_portrait_photography", label: "Creative portrait photography", description: "Creative portrait photography with a posed subject, strong visual theme, props or animals when appropriate, colorful art direction, expressive styling, and a memorable environment.", prompt_guidance: "Build a creative portrait photograph with a strong visual theme. Include a posed subject, purposeful prop or themed object if appropriate, color-directed wardrobe, expressive hair/makeup, layered environment details, playful or artistic composition, lens/framing, lighting style, and a memorable subject-environment relationship. If an animal or prop is used, make it clearly integrated into the scene concept." },
  { value: "gritty_analog", label: "Gritty analog", description: "Gritty analog film look, visible texture, natural imperfections, moody documentary realism.", prompt_guidance: "Build a gritty analog film still with imperfect realism: visible film grain, practical lighting, worn textures, imperfect surfaces, muted color response, handheld or documentary-feeling framing, natural body posture, atmospheric shadows, and a lived-in environment. Avoid overly polished studio language." },
  { value: "soft_dream_pop", label: "Soft dream pop", description: "Soft dreamy pop aesthetic, gentle bloom, pastel color, romantic haze, delicate cinematic lighting.", prompt_guidance: "Build a soft dream-pop still with gentle bloom, pastel color palette, romantic haze, delicate backlight, floating or soft fabric details, dreamy hair/makeup styling, graceful pose, shallow depth of field, soft environment edges, and a light emotional music-video mood." },
  { value: "high_contrast_drama", label: "High-contrast drama", description: "Bold shadows, sculpted highlights, intense facial emotion, dramatic production-still lighting.", prompt_guidance: "Build a high-contrast dramatic still with sculpted highlights, deep shadows, strong key light direction, visible tension in posture, intense facial emotion, dramatic wardrobe silhouette, textured environment, cinematic contrast ratio, and a composition that creates visual pressure." },
  { value: "surreal_symbolic", label: "Surreal symbolic", description: "Surreal symbolic music-video still, heightened atmosphere, poetic objects, dreamlike composition.", prompt_guidance: "Build a surreal symbolic music-video still. Use poetic visual motifs, dreamlike composition, unusual scale or placement of objects, symbolic set dressing, atmospheric light, controlled color palette, and a subject pose that feels ritualistic or uncanny. Keep the imagery visual and concrete rather than explanatory." },
  { value: "clean_studio", label: "Clean studio", description: "Clean studio photography, crisp subject detail, controlled lighting, uncluttered composition.", prompt_guidance: "Build a clean studio photograph with crisp subject detail, controlled lighting setup, precise wardrobe styling, polished hair/makeup, uncluttered backdrop, intentional pose, clear silhouette, lens/framing detail, and professional commercial or editorial clarity." },
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished film-style music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, emotionally readable composition, and a cinematic story-frame finish." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film music-video still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp, street, stage, or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish unless the scene asks for it." },
];

export const ID_LORA_IMAGE_AESTHETIC_PRESETS = [
  { value: "film_default", label: "Default film still", description: "Balanced short-film still lighting, believable production design, natural texture, and cinematic composition.", prompt_guidance: "Build a polished short-film still, not a music-video still. Use believable character blocking, grounded wardrobe, practical lighting, lens/framing detail, textured production design, natural color contrast, and emotionally readable composition." },
  { value: "indie_film_naturalism", label: "Indie film naturalism", description: "Naturalistic indie-drama still with lived-in details, imperfect realism, and intimate character focus.", prompt_guidance: "Build an indie-film still with naturalistic lighting, lived-in wardrobe, imperfect textures, believable posture, intimate framing, subtle emotional detail, muted color response, and environment details that feel observed rather than staged." },
  { value: "neo_noir_dialogue", label: "Neo-noir dialogue", description: "Low-key shadows, practical neon, suspicious glances, dramatic contrast, and noir-style tension.", prompt_guidance: "Build a neo-noir dialogue still with low-key lighting, practical neon or sodium light, deep shadows, hard rim light, reflective surfaces, guarded facial expression, tense blocking, and a controlled color palette. Keep it cinematic and grounded." },
  { value: "gritty_punk_bar", label: "Gritty punk bar", description: "Worn bar textures, punk attitude, practical stage/neon light, smoky atmosphere, and analog grit.", prompt_guidance: "Build a gritty punk-bar film still with worn leather or denim styling, messy lived-in hair/makeup, scratched tables, stickers, posters, dim practical lights, colored neon spill, smoky air, visible texture, defiant posture, and a raw 35mm cinematic finish." },
  { value: "psychological_thriller", label: "Psychological thriller", description: "Uneasy framing, controlled color, negative space, tense facial detail, and subtle dread.", prompt_guidance: "Build a psychological-thriller still with uneasy composition, negative space, controlled color palette, tense facial detail, practical low light, slightly off-balance framing, foreground obstruction, and environmental details that imply pressure without explaining it." },
  { value: "warm_dialogue_drama", label: "Warm dialogue drama", description: "Warm practical interiors, soft skin tones, intimate framing, and emotionally readable acting.", prompt_guidance: "Build a warm dialogue-drama still with practical lamp or bar light, gentle skin tones, shallow depth of field, intimate framing, small emotional facial detail, believable wardrobe, textured surroundings, and a quiet cinematic finish." },
  { value: "35mm_analog_film", label: "35mm analog film", description: "Film grain, practical lighting, imperfect texture, grounded color, and documentary-like realism.", prompt_guidance: "Build a 35mm analog film still with visible grain, practical lighting, imperfect surfaces, grounded color response, natural posture, textured wardrobe, shallow lens character, and a lived-in environment. Avoid glossy music-video polish." },
];

export function storyboardImageShotFlowEntry(profileKey, sceneIndex) {
  const preset = STORYBOARD_IMAGE_SHOT_FLOW_PRESETS[profileKey] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  const sequence = preset.sequence || [];
  if (!sequence.length) return "";
  return sequence[sceneIndex % sequence.length] || "";
}

export function storyboardImageAestheticPreset(value = "") {
  return STORYBOARD_IMAGE_AESTHETIC_PRESETS.find((item) => item.value === value) || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
}

export function storyboardImageAestheticGuidance(value = "", options = {}) {
  const presets = options.idLoraMode ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const preset = presets.find((item) => item.value === value) || presets[0] || storyboardImageAestheticPreset(value);
  return preset.prompt_guidance || preset.description || "";
}

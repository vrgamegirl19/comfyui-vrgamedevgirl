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
    framing_candidates: true,
    description: "Alternates wide masters, medium anchors, close portraits, lateral tracks, and spatial reveals without relying on repetitive inward zooms.",
    guidance: "Use the selected starting shot as the literal first generated frame without adding unprompted lead-ins. Maintain strict scale alternation between wide, medium, and close framing. Avoid consecutive push-ins; prioritize lateral tracks, pullbacks, pedestal rises, smooth orbits, and held compositions to create a natural, rhythmic editorial rhythm.",
    sequence: [
      // Wide & Establishing Resets (Dolly-out, Drifts & Pedestals)
      { shot: "wide cinematic master shot with the subject positioned off-center against balanced architectural or natural geometry", camera: "slow lateral dolly glide" },
      { shot: "wide-angle environmental shot with rich depth of field, subject framed clearly in the midground", camera: "slow pullback reveal" },
      { shot: "wide landscape or interior master, soft ambient light filling the frame with balanced negative space", camera: "static locked-off master" },
      { shot: "elevated wide shot from crane height looking across the entire location, subject anchoring the composition", camera: "slow descending pedestal" },

      // Medium Shots & Full-Body Anchors (Tracking & Lateral Moves)
      { shot: "medium-wide shot framed from the knees up, subject standing centered under balanced directional lighting", camera: "parallel tracking slide" },
      { shot: "full-body shot with clean headroom and ground perspective, subject stationary as camera recedes smoothly", camera: "backward dolly track" },
      { shot: "medium shot framed from the waist up, shallow background separation with soft architectural bokeh", camera: "slow horizontal tracking glide" },
      { shot: "medium three-quarter portrait, lighting defining the contours of the torso and jawline", camera: "smooth arc track" },

      // Intimate & Portrait Scales (Orbits, Pullbacks & Holds)
      { shot: "close-up portrait framed from chest to crown, sharp focus on facial features with soft atmospheric falloff", camera: "slow orbital arc" },
      { shot: "intimate choker close-up focused strictly on the eyes and brow, held in sharp focus", camera: "static locked close-up" },
      { shot: "tight profile close-up against a muted background, rim light tracing the edge of the cheek and neck", camera: "slow lateral drift" },
      { shot: "tight close-up on the face, the camera gently easing backward to reveal the full shoulders and neckline", camera: "slow subtle pullback" },

      // Angles, Heights & Perspective Shifts
      { shot: "low-angle medium shot looking up from waist height, giving the subject an imposing, heroic silhouette", camera: "low-angle crane rise" },
      { shot: "high-angle medium shot looking down at an elegant downward angle, capturing floor textures and shadows", camera: "slow high-angle drift" },
      { shot: "canted Dutch-angle medium close-up, sharp diagonal composition with contrasting side key light", camera: "subtle canted slide" },
      { shot: "straight-on eye-level medium shot with perfect bilateral symmetry, subject framed dead-center", camera: "held static composition" },

      // Occlusion & Depth Reveals
      { shot: "over-the-shoulder medium shot framed past a foreground silhouette, sharp focus held on the subject ahead", camera: "slow lateral reveal right" },
      { shot: "medium close-up framed through an open interior archway or architectural frame, creating layered depth", camera: "slow tracking pan" },
      { shot: "rear three-quarter medium shot looking past the subject toward the open environment beyond", camera: "slow forward tracking glide" },
      { shot: "wide shot framed from behind foreground foliage or columns, revealing the subject centered in clear space", camera: "smooth parallax slide" }
    ],
  },
    intimate_closeups: {
      label: "Intimate close-ups",
      framing_candidates: true,
      description: "Tactile, frame-filling macro and medium close-ups emphasizing emotional presence, subtle movement, and shallow depth of field.",
      guidance: "Maintain an intimate, immediate proximity at all times, with framing never wider than a tight bust shot. Prioritize shallow depth of field, natural rack focuses, soft handheld drifting, and subject-motivated camera moves over mechanical tilts. Every shot must feature distinct focal planes, atmospheric light falloff, or subtle organic movement.",
      sequence: [
        // Macro & Sensory Details
        { shot: "macro close-up on one eye catching soft ambient light, shallow depth of field, holding focus as subject blinks", camera: "static macro with micro-drift" },
        { shot: "extreme close-up of parting lips, gentle breath catching the light, drifting softly toward the jawline", camera: "organic handheld drift" },
        { shot: "macro shot of fingertips brushing strands of hair away, rack focus from the hair in foreground to the soft expression behind it", camera: "rack focus" },
        { shot: "close-up of hands resting on collarbone and neck, feeling a subtle pulse, shallow focus on skin texture", camera: "static intimate frame" },
        { shot: "tight focus on knuckles resting against soft fabric, background softly blurred, gentle ambient breathing movement", camera: "slow lateral glide" },

        // Profile, Angle & Light Play
        { shot: "tight three-quarter profile of the face with rim lighting tracing the contour of the cheek and brow", camera: "subtle rotational orbit" },
        { shot: "profile close-up framed tightly from chin to forehead, capturing soft eye movements as the gaze shifts", camera: "gentle tracking push" },
        { shot: "tight silhouette of facial profile and neck against warm backlight, emphasizing soft contours and eyelashes", camera: "slow ambient drift" },
        { shot: "tight Dutch angle close-up of the temple and ear, capturing subtle shifting expressions in soft shadow", camera: "subtle canted tilt" },
        { shot: "tight portrait framed through foreground out-of-focus elements, creating natural emotional depth", camera: "slow creeping push-in" },

        // Occlusion & Reveals
        { shot: "shallow-focus close-up partially obscured by soft hair in the extreme foreground, slowly revealing the eyes", camera: "slow parallax slide" },
        { shot: "hand enters extreme foreground close to lens, creating soft bokeh before settling to reveal the focused face beyond", camera: "deep-focus rack" },
        { shot: "intimate mirror reflection close-up with soft condensation or dust texture on glass, focusing on the reflected gaze", camera: "slow tilt across surface" },
        { shot: "over-the-shoulder macro perspective, the nape of the neck soft in the foreground as the jaw and cheekbone catch light", camera: "gentle over-the-shoulder push" },
        { shot: "close-up from behind as subject slowly turns chin toward the frame, landing into soft three-quarter key light", camera: "static frame catching subject movement" },

        // Seated & Postural Intimacy
        { shot: "tight seated composition, knees pulled close to frame, face resting lightly against forearm in sharp focus", camera: "slow creeping push-in" },
        { shot: "compact seated profile filling the frame, focus held on the curve of the shoulder and quiet facial composure", camera: "organic handheld drift" },
        { shot: "curled pose framed tightly around shoulders, cheek, and folded arms, soft directional window light", camera: "slow lateral pan" },
        { shot: "tight top-down perspective of subject resting head on a pillow or surface, hair pooling around the frame", camera: "slow downward pedestal" },
        { shot: "side-lying close-up, cheek against surface, slow focus pull from resting hand in foreground to calm eyes", camera: "rack focus with slow push" },

        // Movement & Subject Dynamics
        { shot: "tight bust shot as subject leans slightly closer into the lens sweet spot, falling into sharp focus", camera: "static lock-off letting subject drive depth" },
        { shot: "low-angle intimate close-up tracking throat to chin as the head tilts upward toward the light", camera: "subtle upward tilt" },
        { shot: "high-angle tight portrait looking down into upturned eyes, frame filled entirely from brow to chest", camera: "slow descending pull" },
        { shot: "tight tracking shot moving with subject's slow step, staying framed strictly on the collar and chin", camera: "matched speed tracking glide" },
        { shot: "tight frontal close-up with eyes locked directly down the barrel of the lens, shallow falloff", camera: "imperceptible slow push-in" }
      ]
  },
  music_video: {
    label: "Music video shots",
    framing_candidates: true,
    description: "High-energy, stylized music video setups covering dynamic performance, kinetic tracking, heroic angles, and atmospheric narrative beats.",
    guidance: "Match camera dynamics to musical energy and tempo. Favor motivated motion—Steadicam tracking, rhythmic push-pulls, dynamic jib rises, and stylized focal lengths—over passive drifts. Avoid static mid-shots unless deliberately locked for graphic choreography; keep camera height, focal length, and movement varied across sequential cuts.",
    sequence: [
      // Hero & Performance Anchors
      { shot: "wide-angle hero performance shot centered in a grand architectural space, dramatic ceiling and floor perspective lines", camera: "slow forward dolly glide" },
      { shot: "low-angle full-body performance shot looking up at the subject against sky or overhead lighting grid", camera: "push-in low tilt" },
      { shot: "tight, eye-level singing performance shot with fast background bokeh movement, capturing raw vocal delivery", camera: "kinetic handheld float" },
      { shot: "high-energy full-body dance routine shot, framing tight choreography with clear spatial perspective", camera: "rhythmic pulse push-in" },
      { shot: "Dutch-angle mid-performance shot, stylized tilt creating diagonal tension across the composition", camera: "canted orbital drift" },

      // Dynamic Movement & Lead/Follow
      { shot: "full-body tracking shot moving backward at matched pace as subject strides aggressively toward lens", camera: "backward Steadicam lead" },
      { shot: "lateral tracking shot pacing subject moving across a textured urban or studio wall", camera: "parallel tracking slide" },
      { shot: "over-the-shoulder follow shot chasing closely behind the subject moving through a narrow corridor or crowd", camera: "forward tracking follow" },
      { shot: "sweeping 360-degree circular orbit maintaining subject dead-center while the background spins into blur", camera: "rapid circular orbit" },
      { shot: "dynamic wraparound shot transitioning from subject's back to a full frontal face reveal mid-stride", camera: "curved arc wrap" },

      // Scale, Elevation & Jib
      { shot: "overhead bird's-eye shot looking straight down at subject lying or rotating across patterned flooring", camera: "descending rotational crane" },
      { shot: "wide environmental shot with the subject small in frame, emphasizing stark negative space and isolation", camera: "slow creeping pull-back" },
      { shot: "dramatic elevated reveal starting from subject's feet, booming vertically up to an expansive wide master", camera: "fast jib rise" },
      { shot: "high-angle boom shot looking down from above head height, slowly descending to an intimate eye-line", camera: "descending crane boom" },
      { shot: "ground-skimming low perspective as boots step past the lens, throwing the camera into foreground blur", camera: "low-angle pass-by glide" },

      // Lens Character & Stylized Framing
      { shot: "ultra-wide lens tracking shot close to the subject's face, exaggerated barrel distortion and dynamic depth", camera: "forward wide push" },
      { shot: "long-lens telephoto shot compressing distance, subject walking in slow motion as heat shimmer or dust distorts the background", camera: "compressed lateral track" },
      { shot: "vertigo dolly zoom (zolly), subject size remains fixed while the background perspective expands dramatically", camera: "dolly zoom" },
      { shot: "locked-off graphic framing with subject bursting into the edge of frame, performing, then exiting clear", camera: "static lock-off" },
      { shot: "silhouette performance framed against intense strobe, blinding sun, or saturated neon lightbox", camera: "slow backlit push-in" },

      // Narrative Beats & Kinetic Transitions
      { shot: "tight mirror or puddle reflection shot, focus pulling from the glossy surface to the subject entering real space", camera: "reflection rack focus" },
      { shot: "subject passing through an illuminated doorway into darkness, backlit halo outlining their silhouette", camera: "portal follow-through" },
      { shot: "floating one-take sequence weaving around multiple band members or dancers across continuous set pieces", camera: "continuous fluid Steadicam" },
      { shot: "high-speed snap-zoom punching directly from a wide performance shot to an extreme close-up on the eyes", camera: "crash zoom" },
      { shot: "kinetic whip pan snapping away from the subject's abrupt hand motion into solid motion blur", camera: "fast directional whip" }
    ],
  },
  fisheye_distorted: {
    label: "Fisheye and Distorted-Lens Shots",
    framing_candidates: true,
    description: "Curvilinear, ultra-wide barrel distortion, spherical optical warping, curved-surface reflections, and skate-style low-rigging.",
    guidance: "Harness the extreme field-of-view (180° circular or diagonal fisheye) intentionally. Exploit edge-of-frame barrel compression, extreme foreground/background size disparity, and curved perspective lines. Vary between skate-cam low chases, Hype Williams 90s-style performance centers, convex reflections, and Dutch-angle rotational rolls. Avoid redundant static leans into the glass.",
    sequence: [
      // 90s Hype Williams & Performance Center
      { shot: "dead-center circular fisheye performance shot, architecture bending radially around the subject as walls curve inward", camera: "slow forward dolly glide" },
      { shot: "extreme fisheye close-up with the nose and forehead bulged near the optical center, body tapering steeply into distant vanishing point", camera: "subtle optical breathing push" },
      { shot: "subject drops suddenly from standing into a tight squat inches from the dome element, delivering lyrics straight down the barrel", camera: "rapid downward tilt follow" },
      { shot: "high-energy performance with hands gesturing across the peripheral edge, fingers stretching and warping with barrel distortion", camera: "rhythmic rotational sway" },
      { shot: "dramatic roll with the camera rotating 180 degrees on the optical axis, turning the curved horizon upside down while keeping the subject centered", camera: "full barrel roll" },

      // Low-Angle, Skate-Cam & Ground-Skim
      { shot: "ground-level skate-rigged fisheye skimming asphalt, tracking inches behind sneakers stepping forward into frame", camera: "low-angle dynamic follow" },
      { shot: "upward-facing fisheye locked flat on the floor, subject straddling the lens as oversized boot soles step around the frame boundary", camera: "static floor lock-off" },
      { shot: "ultra-low upward tilt capturing subject standing tall above the lens, torso warping toward the sky with curved clouds radiating outward", camera: "low-angle rising arc" },
      { shot: "camera mounted to a low stabilizer carving in tight S-curves around the performer, horizon line wobbling dynamically", camera: "slalom tracking pass" },
      { shot: "performer spinning rapidly directly over the upward-facing dome lens, hair and loose clothing sweeping through the extreme edges", camera: "static ground upward gaze" },

      // Physical Proximity, Hands & Occlusion
      { shot: "subject reaches an exaggerated, massive palm forward to shield or tap the front element before snapping hand back", camera: "snap push-in with lens flare" },
      { shot: "both hands frame the circular bezel of the fish-eye lens, peering through the gap with intense warped focus", camera: "macro-proximity float" },
      { shot: "performer passes tight laterally across the frame, their silhouette dramatically bloating into the curve before compressing out of frame", camera: "whip-pan pass-by" },
      { shot: "subject leans cheek inches from the glass, breathing mist onto the lens element before wiping it away with a sleeve", camera: "static intimate lock-off" },

      // Curved Surfaces & Physical Distortions
      { shot: "extreme convex distortion filmed through a chrome security mirror or polished hubcap, world warped into a sphere", camera: "curved reflection tilt" },
      { shot: "subject viewed through thick ribbed glassware or a glass prism, fragmenting facial features into distorted streaks", camera: "prismatic rack drift" },
      { shot: "fisheye framed looking out from inside a cramped washing machine or circular tunnel, rim of the opening framing the distorted performer", camera: "portal pull-back" },
      { shot: "reflection on sunglasses or visor curved like an orb, revealing an ultra-wide distorted landscape in the reflection", camera: "macro push-in to reflection" },

      // Environmental & Narrative Compression
      { shot: "claustrophobic hallway shot where straight fluorescent light fixtures bend into severe arcs around the performer", camera: "backward hallway track" },
      { shot: "performer running toward the camera from far distance, covering ground rapidly as the ultra-wide lens exaggerates forward velocity", camera: "static head-on catch" },
      { shot: "Dutch-tilt fisheye looking down from a high ceiling corner, security-camera aesthetic with circular vignetted borders", camera: "high-angle surveillance drift" },
      { shot: "abrupt snap-pull starting with eyelashes filling the frame, pulling back into a full-body wide in a single second", camera: "whip zoom pullback" },
      { shot: "subject holding the camera on an extended selfie pole at arm's length, running through an environment as the background swoops behind them", camera: "performer-mounted snorricam track" }
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
    framing_candidates: true,
    description: "Restrained, atmospheric cinematic framing prioritizing heavy negative space, stillness, shadows, and creeping camera moves.",
    guidance: "Zero abrupt or fast movements. Frame subjects as static compositional anchors within large, moody environments. Let slow creeping dollies, held lock-offs, and light falloff create psychological weight without relying on character performance.",
    sequence: [
      { shot: "wide establishing shot of an empty architectural space, subject seated motionless in the lower-third, heavy negative space", camera: "static locked-off master" },
      { shot: "medium-wide shot framed through a distant interior doorway, observing the motionless subject in cold ambient light", camera: "slow imperceptible push-in" },
      { shot: "severe side-profile framed against a soft out-of-focus background, clean cinematic contrast and deep shadows", camera: "static tripod lock-off" },
      { shot: "dead-center symmetrical wide shot, subject seated still under a single overhead pool of downlight", camera: "slow crawling dolly-in" },
      { shot: "sharp silhouette of the subject framed against twilight window light, surrounded by pitch darkness", camera: "slow rising pedestal" },
      { shot: "intimate choker close-up locked strictly on the eyes, razor-thin depth of field with static framing", camera: "held static close-up" },
      { shot: "low-key medium shot with the subject positioned at the far edge of the frame, balance tipped heavily toward dark negative space", camera: "slow lateral creeping track" },
      { shot: "deep-focus shot with rain-streaked window glass sharp in the foreground, subject resting out of focus in the room beyond", camera: "static split-depth frame" },
      { shot: "rear three-quarter medium shot looking past the subject toward a distant desolate vista or dark room", camera: "slow forward creeping push" },
      { shot: "dimly lit interior mirror reflection, subject motionless while foreground shadows frame the composition", camera: "static locked-off reflection" }
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

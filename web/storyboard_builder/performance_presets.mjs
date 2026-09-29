export const PERFORMANCE_STYLE_PRESETS = [
  {
    value: "off",
    label: "Off",
    direction: "",
  },
  {
    value: "",
    label: "Default cinematic",
    direction: "Use a natural cinematic music-video performance with visible emotion, expressive face, motivated body language, and camera energy that fits the scene.",
  },
  {
    value: "rock_punk",
    label: "Rock / punk",
    direction: "Use raw rock performance energy: intense facial emotion, head movement, sharp gestures, defiant posture, and gritty stage-like body language.",
  },
  {
    value: "metal_screaming",
    label: "Metal / screaming",
    direction: "Use aggressive high-intensity performance energy: fierce expression, powerful stance, forceful gestures, hair and clothing reacting to motion, and heavy dramatic presence.",
  },
  {
    value: "rap_hiphop",
    label: "Rap / hip-hop",
    direction: "Use rap-style energy instead of soft singing: confident direct-to-camera presence, expressive hand gestures, head nods, shoulder movement, and sharper body language.",
  },
  {
    value: "pop_performance",
    label: "Pop performance",
    direction: "Use polished pop performance energy: expressive singing, clean confident movement, controlled gestures, direct eye contact, stylish body language, and camera-friendly emotion.",
  },
  {
    value: "ballad_emotional",
    label: "Ballad / emotional",
    direction: "Use emotional ballad performance energy: vulnerable facial expression, slower gestures, longing eyes, subtle hand movement, restrained body language, and intimate camera presence.",
  },
  {
    value: "rnb_smooth",
    label: "R&B / smooth",
    direction: "Use smooth R&B performance energy: relaxed confident expression, controlled sensual movement, gentle hand gestures, soft rhythmic body motion, and close emotional intensity.",
  },
  {
    value: "edm_club",
    label: "EDM / club",
    direction: "Use energetic club performance energy: rhythmic movement, dance-like gestures, bright reactive expression, beat-driven body language, and dynamic camera motion.",
  },
  {
    value: "spoken_word",
    label: "Spoken word",
    direction: "Use spoken-word energy instead of singing: focused eyes, intentional gestures, restrained intensity, and poetic performance presence.",
  },
  {
    value: "no_vocals_broll",
    label: "No vocals / B-roll",
    direction: "Do not include singing, rapping, speaking, lip-sync, mouth movement, microphones, or vocal performance. Use visual action, environment interaction, and mood-driven movement only.",
  },
];

export function storyboardPerformancePreset(value = "") {
  return PERFORMANCE_STYLE_PRESETS.find((item) => item.value === value) || PERFORMANCE_STYLE_PRESETS[0];
}

export const FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "off",
    label: "Off",
    description: "Do not attach facial direction",
    direction: "",
  },
  {
    value: "",
    label: "Default natural",
    description: "Natural expressive face",
    direction: "Use natural expressive facial performance: engaged eyes, subtle natural eye movement, active brows, subtle cheek and jaw movement, visible emotion that fits the lyric or scene, and occasional natural blinking.",
  },
  {
    value: "pop_polished",
    label: "Pop / polished stage",
    description: "Camera-ready pop emotion",
    direction: "Use polished pop-star facial performance: bright eyes, subtle natural eye movement, direct camera gaze, soft confident smile, playful smirk, relaxed brows, slight head tilts, lips slightly parted while singing, charming camera-ready expression, and occasional natural blinking.",
  },
  {
    value: "pop_flirty",
    label: "Pop / playful flirty",
    description: "Playful, charming pop face",
    direction: "Use playful pop facial performance: flirty smile, coy glance, subtle natural eye movement, light pout, glossy pout, raised brows, charming direct gaze, playful smirk, subtle head tilt, lips slightly parted while singing, and occasional natural blinking.",
  },
  {
    value: "love_tender",
    label: "Love song / tender",
    description: "Soft romantic expression",
    direction: "Use tender love-song facial performance: softened eyes, subtle natural eye movement, warm smile, affectionate gaze, raised inner brows, gentle head tilt, relaxed cheeks, subtle vulnerable emotion, and occasional natural blinking.",
  },
  {
    value: "sad_wounded",
    label: "Sad / wounded",
    description: "Grief, hurt, vulnerability",
    direction: "Use wounded sad-song facial performance: lowered gaze, heavy or watery eyes, subtle natural eye movement, raised inner brows, pinched brows, downturned mouth, trembling lips or chin when appropriate, defeated expression, and occasional natural blinking.",
  },
  {
    value: "happy_joyful",
    label: "Happy / joyful",
    description: "Bright and joyful",
    direction: "Use joyful facial performance: bright smile, smiling eyes, subtle natural eye movement, raised cheeks, delighted expression, playful gaze, lifted mouth corners, relaxed brows, head tilt with smile, and occasional natural blinking.",
  },
  {
    value: "rock_intense",
    label: "Rock / intense",
    description: "Gritty rock intensity",
    direction: "Use intense rock facial performance: focused stare, subtle natural eye movement, furrowed brows, defiant smirk, clenched jaw, gritty emotional strain, sharp eye contact, forceful singing expression, and occasional natural blinking.",
  },
  {
    value: "metal_rage",
    label: "Metal / rage",
    description: "Aggressive heavy metal face",
    direction: "Use aggressive heavy metal facial performance: fierce stare, subtle natural eye movement, furrowed brows, wild eyes, clenched jaw, snarling mouth shapes during vocals, bared teeth on powerful notes, flared nostrils, strained neck intensity, raw emotional scream expression, and occasional natural blinking.",
  },
  {
    value: "rap_high_intensity",
    label: "Rap / high intensity",
    description: "Sharp rap delivery",
    direction: "Use high-intensity rap facial performance: intense stare, sharp eye contact, subtle natural eye movement, furrowed brows, animated eyes, confident smirk, tight jaw, mouth open mid-verse, fast-moving mouth during delivery, challenging look, victory grin, and occasional natural blinking.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

export function storyboardFacialPerformancePreset(value = "") {
  return FACIAL_PERFORMANCE_PRESETS.find((item) => item.value === value) || FACIAL_PERFORMANCE_PRESETS[0];
}

export const ID_LORA_PERFORMANCE_STYLE_PRESETS = [
  {
    value: "dialogue_naturalism",
    label: "Dialogue naturalism",
    direction: "Use grounded short-film acting: conversational timing, motivated gestures, lived-in posture, subtle emotional shifts, and behavior that feels observed rather than performed.",
  },
  {
    value: "tense_confrontation",
    label: "Tense confrontation",
    direction: "Use restrained confrontation energy: clipped gestures, guarded posture, controlled anger, charged pauses, and body language that suggests pressure under the surface.",
  },
  {
    value: "indie_drama",
    label: "Indie drama",
    direction: "Use intimate indie-film acting: small revealing gestures, vulnerable stillness, natural imperfections, quiet tension, and emotionally specific reactions.",
  },
  {
    value: "noir_restraint",
    label: "Noir restraint",
    direction: "Use noir-style restraint: low-key confidence, suspicious glances, minimal gestures, guarded delivery, and tension carried through posture and eyes.",
  },
  {
    value: "comedic_awkwarness",
    label: "Comedic awkward",
    direction: "Use dry comedic acting: awkward pauses, slightly mismatched reactions, contained embarrassment, small nervous gestures, and believable conversational timing.",
  },
  {
    value: "emotional_confession",
    label: "Emotional confession",
    direction: "Use confession-scene acting: exposed emotion, hesitant gestures, wavering confidence, visible vulnerability, and a line delivery that feels personally risky.",
  },
  {
    value: "suspense_dread",
    label: "Suspense dread",
    direction: "Use suspense-film tension: alert posture, careful stillness, anxious scanning, controlled breathing, and reactions that imply something important is about to break.",
  },
  {
    value: "punk_bar_attitude",
    label: "Punk bar attitude",
    direction: "Use gritty punk-bar acting: defiant posture, sharp side-eye, casual toughness, impatient gestures, and messy lived-in confidence without turning it into a stage performance.",
  },
];

export const ID_LORA_FACIAL_PERFORMANCE_PRESETS = [
  {
    value: "",
    label: "Default screen acting",
    description: "Natural film face",
    direction: "Use grounded screen-acting facial detail: attentive eyes, small brow changes, readable thought, subtle jaw tension, natural mouth shapes for speech, and emotion that fits the dialogue.",
  },
  {
    value: "curious_inquisitive",
    label: "Curious / inquisitive",
    description: "Curious screen expression",
    direction: "Use curious facial performance: bright attentive eyes, slight head angle, lifted brow, searching gaze, relaxed mouth between words, and a sense of active listening.",
  },
  {
    value: "guarded_suspicious",
    label: "Guarded / suspicious",
    description: "Guarded tension",
    direction: "Use guarded facial performance: narrowed eyes, tight jaw, controlled mouth, skeptical brow, held gaze, and restrained suspicion under the dialogue.",
  },
  {
    value: "defiant_controlled",
    label: "Defiant / controlled",
    description: "Controlled defiance",
    direction: "Use controlled defiance: steady eye contact, tense mouth corners, lifted chin, compressed jaw, and a look that refuses to back down.",
  },
  {
    value: "vulnerable_confession",
    label: "Vulnerable confession",
    description: "Exposed emotion",
    direction: "Use vulnerable confession facial performance: softened eyes, raised inner brows, small uncertain mouth movements, visible hesitation, and emotion barely held together.",
  },
  {
    value: "dry_comedic",
    label: "Dry comedic",
    description: "Subtle comedy face",
    direction: "Use dry comedic facial performance: tiny reaction beats, restrained disbelief, awkward half-smile, quick eye shifts, and understated embarrassment.",
  },
  {
    value: "custom",
    label: "Custom",
    description: "Use custom facial text",
    direction: "",
  },
];

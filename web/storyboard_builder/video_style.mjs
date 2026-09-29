import { normalizeStoryboardMiniMaxH3Mode, normalizeStoryboardProjectVideoEngine } from "./scenes.mjs";

const MINIMAX_VIDEO_STYLE_LABELS = [
  "Cinematic realism", "Gothic romance", "Dark fantasy", "Ethereal dreamscape", "Surrealism", "Cosmic horror",
  "Psychological horror", "Found footage", "Analog horror", "Body horror", "Occult ritual", "Silent Hill-inspired",
  "Cyberpunk", "Biopunk", "Dieselpunk", "Steampunk", "Post-apocalyptic", "Dystopian sci-fi", "Retro-futurism",
  "Y2K futurism", "Vaporwave", "Synthwave", "Dreamcore", "Weirdcore", "Liminal space", "Dark academia",
  "Cottagecore", "Fairycore", "Angelcore", "Goblincore", "Whimsigoth", "Baroque", "Rococo", "Art Nouveau",
  "Art Deco", "Victorian gothic", "Renaissance-inspired", "Medieval fantasy", "Mythological epic", "Film noir",
  "Neo-noir", "Expressionism", "Giallo horror", "Grindhouse", "1970s psychedelic", "1980s music video",
  "1990s grunge", "Early-2000s pop", "Indie sleaze", "Lo-fi VHS", "Super 8 film", "Vintage Hollywood",
  "High-fashion editorial", "Avant-garde fashion", "Runway glamour", "Luxury commercial", "Beauty campaign",
  "Pop-star music video", "Industrial metal", "Gothic metal", "Alternative rock", "Punk rock", "Dark pop",
  "Hyperpop", "K-pop-inspired", "R&B glamour", "Eerie claymation", "Stop-motion", "Paper-cut animation",
  "Hand-painted animation", "Anime-inspired", "Graphic novel", "Comic-book", "Cel-shaded 3D", "Photorealistic CGI",
  "Low-poly 3D", "Miniature diorama", "Dollhouse surrealism", "Liquid chrome", "Holographic iridescence",
  "Neon noir", "Monochrome minimalism", "High-key white studio", "Low-key chiaroscuro", "Soft pastel",
  "Desaturated melancholy", "Crimson-and-black", "Teal-and-orange blockbuster", "Golden-hour nostalgia",
  "Moonlit blue", "Underwater ethereal", "Elemental fantasy", "Nature mysticism", "Apocalyptic biblical",
  "Glitch art", "Datamosh", "CRT distortion", "Kaleidoscopic", "Double exposure", "Infrared", "Thermal vision",
  "Fisheye distortion", "Security-camera footage", "Documentary realism", "Social-media selfie", "TikTok transformation",
  "Dreamlike slow motion", "Frenetic montage", "One-take immersive", "Music-video performance",
  "Narrative short film", "Movie-trailer aesthetic",
];

const MINIMAX_VIDEO_STYLE_VERBIAGE = {
  "Cinematic realism": "Naturalistic practical lighting, restrained color grading, balanced contrast, subtle film grain, realistic skin texture, neutral tones, believable materials, and polished cinematic clarity throughout.",
  "Gothic romance": "Deep burgundy, black, and ivory tones, soft shadows, luminous highlights, ornate details, rich velvet and lace textures, candlelit atmosphere, and melancholic visual softness throughout.",
  "Dark fantasy": "Shadow-heavy lighting, desaturated earth tones, metallic accents, dramatic contrast, weathered textures, monumental fantasy production design, atmospheric haze, and richly cinematic grading throughout.",
  "Ethereal dreamscape": "Pastel colors, diffused highlights, soft focus, glowing edges, translucent layers, pearlescent haze, low contrast, and weightless dreamlike beauty throughout.",
  "Surrealism": "Unexpected proportions, distorted perspective, symbolic abstraction, unnatural colors, impossible spatial relationships, uncanny objects, and deliberately illogical dream imagery throughout.",
  "Cosmic horror": "Near-black palettes, cold highlights, immense scale, distorted geometry, oppressive shadows, ancient textures, unsettling negative space, and incomprehensible otherworldly detail throughout.",
  "Psychological horror": "Sickly restrained color, oppressive shadow, uneasy negative space, subtly distorted interiors, harsh practical light, clammy skin tones, ambiguous background details, and persistent visual dread throughout.",
  "Found footage": "Authentic in-world consumer-camera imagery with practical available light, imperfect exposure, mild autofocus softness, sensor noise, compression artifacts, clipped highlights, noisy shadows, subdued color, and an unpolished documentary texture. Keep faces readable; avoid glossy grading, studio polish, pristine sharpness, and cinematic glamour.",
  "Analog horror": "Degraded videotape imagery, faded color, tracking noise, scan lines, chromatic bleed, warped edges, crushed blacks, blown highlights, timestamp-like visual language without readable text, and ominous broadcast-era texture throughout.",
  "Body horror": "Visceral organic textures, pallid flesh tones, wet highlights, anatomical distortion, diseased surfaces, clinical details, bruised color accents, harsh close detail, and deeply unsettling physical materiality throughout.",
  "Occult ritual": "Candlelit darkness, ceremonial symbols, weathered stone, smoke, wax, ash, deep red and black accents, antique ritual objects, symmetrical arrangements, and secretive sacred atmosphere throughout.",
  "Silent Hill-inspired": "Dense pale fog, rusted industrial surfaces, damp concrete, peeling walls, muted gray-green color, dirty amber light, corroded metal, abandoned spaces, and oppressive psychological decay throughout.",
  "Cyberpunk": "Neon magenta, cyan, and electric blue light, rain-slick surfaces, holographic signage shapes without readable text, dense urban technology, reflective synthetic materials, high contrast, and gritty futuristic detail throughout.",
  "Biopunk": "Organic technology, translucent membranes, bone-like structures, cultured tissue, surgical hardware, sickly green and amber light, wet biological surfaces, laboratory grime, and engineered-life detail throughout.",
  "Dieselpunk": "Oil-stained metal, riveted machinery, soot, heavy industrial architecture, military-era styling, muted olive and rust colors, hard smoky light, analog gauges, and imposing mechanical detail throughout.",
  "Steampunk": "Aged brass, copper pipes, leather, polished wood, intricate gears, Victorian tailoring, warm amber light, steam-filled atmosphere, engraved ornament, and handcrafted mechanical detail throughout.",
  "Post-apocalyptic": "Sun-bleached ruins, scavenged materials, dust, rust, broken infrastructure, weathered clothing, harsh natural light, muted earth colors, and layered environmental decay throughout.",
  "Dystopian sci-fi": "Monumental controlled architecture, cold gray-blue palettes, severe uniforms, sterile surfaces, surveillance motifs without readable text, stark artificial light, rigid visual order, and oppressive technological detail throughout.",
  "Retro-futurism": "Optimistic vintage future design, chrome, molded plastic, analog controls, bold geometric forms, saturated period color, glowing panels, clean illustrative surfaces, and nostalgic speculative detail throughout.",
  "Y2K futurism": "Glossy silver, translucent plastics, icy blue and white palettes, bubble-like interfaces without readable text, chrome accessories, soft digital glow, clean synthetic surfaces, and early-digital optimism throughout.",
  "Vaporwave": "Pastel pink, lavender, aqua, and sunset gradients, marble surfaces, retro computer textures, classical-statue motifs, soft haze, luminous grid-like design, and nostalgic digital unreality throughout.",
  "Synthwave": "Hot magenta, violet, and electric cyan palettes, deep black silhouettes, neon grids, glossy reflections, dramatic sunset gradients, chrome accents, and polished retro-electronic atmosphere throughout.",
  "Dreamcore": "Soft familiar spaces, hazy pastel light, washed color, low-detail backgrounds, uncanny childhood objects, gentle bloom, empty interiors, and comforting yet disorienting dream imagery throughout.",
  "Weirdcore": "Low-resolution digital texture, awkward cropping, mismatched color, uncanny ordinary objects, liminal interiors, crude graphic shapes without readable text, visual noise, and deliberately unsettling internet-era imagery throughout.",
  "Liminal space": "Empty transitional architecture, fluorescent or sodium lighting, repetitive corridors, vacant rooms, muted institutional color, dated surfaces, deep vanishing points, and eerily familiar stillness throughout.",
  "Dark academia": "Deep brown, charcoal, forest green, and oxblood tones, old books, dark wood, worn leather, classical architecture, window light, dust, tweed textures, and scholarly melancholy throughout.",
  "Cottagecore": "Warm natural light, wildflowers, handmade fabrics, rustic wood, ceramics, baskets, soft earth colors, pastoral interiors, gentle weathering, and cozy rural detail throughout.",
  "Fairycore": "Mossy forests, tiny flowers, luminous dust, translucent wings, dewdrops, soft green and pastel color, miniature natural details, glowing mushrooms, and delicate enchanted atmosphere throughout.",
  "Angelcore": "Ivory and pale gold palettes, luminous white fabric, soft clouds, radiant backlight, delicate feathers, sacred ornament, pearlescent highlights, and serene celestial atmosphere throughout.",
  "Goblincore": "Muddy greens and browns, moss, mushrooms, stones, bones, jars, tarnished trinkets, damp forest textures, cluttered natural collections, and earthy mischievous detail throughout.",
  "Whimsigoth": "Midnight blue, plum, black, and antique gold tones, celestial patterns, velvet, candles, stained glass, ornate jewelry, mystical clutter, and romantic witchy atmosphere throughout.",
  "Baroque": "Deep jewel tones, dramatic light and shadow, gilded ornament, rich fabric, carved architecture, elaborate decoration, painterly highlights, and theatrical seventeenth-century grandeur throughout.",
  "Rococo": "Powder pink, pale blue, cream, and gold palettes, delicate florals, curved ornament, silk, porcelain, airy light, playful luxury, and ornate eighteenth-century elegance throughout.",
  "Art Nouveau": "Flowing botanical lines, stained glass, wrought metal, muted jewel colors, floral ornament, organic symmetry, decorative illustration, and elegant turn-of-the-century craftsmanship throughout.",
  "Art Deco": "Bold geometry, black and gold contrast, polished stone, lacquer, chrome, stepped forms, symmetrical ornament, rich jewel tones, and glamorous machine-age luxury throughout.",
  "Victorian gothic": "Black lace, dark carved wood, aged stone, gaslight, heavy drapery, mourning attire, tarnished silver, deep wine tones, and haunted nineteenth-century atmosphere throughout.",
  "Renaissance-inspired": "Warm earth pigments, rich red and blue fabric, classical architecture, fresco-like color, soft directional light, fine textile detail, balanced humanist elegance, and old-master visual richness throughout.",
  "Medieval fantasy": "Weathered stone, timber halls, chainmail, wool, leather, heraldic color, torchlight, misty landscapes, handcrafted props, and grounded legendary-world detail throughout.",
  "Mythological epic": "Monumental temples, heroic silhouettes, carved stone, bronze and gold accents, dramatic skies, ceremonial fabric, divine light, vast landscapes, and timeless legendary grandeur throughout.",
  "Film noir": "High-contrast black-and-white imagery, hard key light, venetian-blind shadows, wet streets, cigarette haze, deep blacks, bright highlights, period interiors, and morally shadowed atmosphere throughout.",
  "Neo-noir": "Deep shadows, saturated neon accents, reflective night surfaces, controlled color contrast, urban grime, practical light, smoky atmosphere, and sleek contemporary darkness throughout.",
  "Expressionism": "Angular sets, exaggerated shadows, distorted architecture, stark color or monochrome contrast, theatrical makeup, painted surfaces, and emotionally warped visual design throughout.",
  "Giallo horror": "Saturated red, yellow, blue, and green light, glossy black surfaces, ornate interiors, sharp shadows, glamorous styling, lurid practical effects, and stylish Italian horror atmosphere throughout.",
  "Grindhouse": "Faded color, heavy grain, scratched film, dirty highlights, crushed shadows, cheap practical effects, lurid wardrobe, distressed print texture, and raw exploitation-era finish throughout.",
  "1970s psychedelic": "Burnt orange, avocado, violet, and acid color, bold patterns, soft film grain, optical layering, warped graphic forms, warm haze, and richly hallucinatory period design throughout.",
  "1980s music video": "Saturated neon color, glossy highlights, smoky studio atmosphere, dramatic backlight, bold makeup, metallic wardrobe, soft diffusion, analog video texture, and theatrical pop imagery throughout.",
  "1990s grunge": "Muted dirty color, fluorescent interiors, distressed denim and flannel, photocopied graphic texture without readable text, harsh flash, visible grain, urban wear, and unpolished alternative-era realism throughout.",
  "Early-2000s pop": "Glossy candy color, icy highlights, metallic accessories, low-rise era styling, bright studio surfaces, soft skin diffusion, digital-camera crispness, and playful Y2K polish throughout.",
  "Indie sleaze": "Direct-flash nightlife imagery, blown skin highlights, deep black backgrounds, messy styling, grainy digital texture, smoky clubs, saturated accents, and deliberately careless downtown glamour throughout.",
  "Lo-fi VHS": "Soft analog resolution, tape grain, color bleed, scan lines, tracking instability, crushed blacks, clipped whites, oversaturated consumer color, and worn home-video texture throughout.",
  "Super 8 film": "Warm faded color, pronounced small-gauge grain, soft focus, halation, light leaks, flickering exposure texture, rounded highlights, and intimate home-movie character throughout.",
  "Vintage Hollywood": "Elegant studio lighting, luminous skin, rich black-and-white or restrained Technicolor tones, soft diffusion, tailored wardrobe, painted-set refinement, and classic star-era glamour throughout.",
  "High-fashion editorial": "Sculptural wardrobe, immaculate makeup, controlled color, premium fabric detail, bold graphic styling, clean luxury surfaces, precise beauty lighting, and magazine-grade visual polish throughout.",
  "Avant-garde fashion": "Experimental silhouettes, unexpected materials, abstract makeup, severe color blocking, sculptural sets, conceptual styling, high-detail fabric texture, and art-gallery fashion imagery throughout.",
  "Runway glamour": "Luxury garments, luminous skin, glossy hair, dramatic show lighting, polished surfaces, rich color, crisp textile detail, and elevated fashion-week spectacle throughout.",
  "Luxury commercial": "Pristine product-grade surfaces, controlled highlights, rich neutral color, immaculate materials, elegant reflections, premium environments, clean contrast, and expensive advertising polish throughout.",
  "Beauty campaign": "Luminous skin, refined makeup detail, soft controlled highlights, clean backgrounds, flattering color, glossy hair, delicate texture, and premium cosmetic-advertising finish throughout.",
  "Pop-star music video": "Bold saturated color, glamorous wardrobe, luminous skin, dramatic set lighting, glossy production design, metallic accents, atmospheric haze, and polished superstar imagery throughout.",
  "Industrial metal": "Cold steel, concrete, rust, oil, black leather, harsh white and red light, smoke, abrasive texture, heavy machinery, and severe high-contrast atmosphere throughout.",
  "Gothic metal": "Black leather and lace, deep crimson accents, cathedral stone, silver ornament, smoke, dramatic pale skin, low-key light, and dark romantic grandeur throughout.",
  "Alternative rock": "Lived-in rehearsal spaces, worn instruments, denim and leather, practical stage light, muted color, visible grain, textured walls, and grounded independent-band authenticity throughout.",
  "Punk rock": "Photocopied texture without readable text, torn fabric, studs, leather, raw club interiors, harsh flash, red and black accents, grime, and confrontational DIY visual energy throughout.",
  "Dark pop": "Deep black palettes, jewel-tone accents, glossy shadows, dramatic beauty light, surreal luxury details, refined makeup, controlled haze, and sleek ominous pop polish throughout.",
  "Hyperpop": "Acid neon color, chrome, glossy plastic, exaggerated digital texture, candy gradients, iridescent makeup, maximal graphic detail, and intensely synthetic internet-pop imagery throughout.",
  "K-pop-inspired": "Immaculate styling, vivid coordinated color, glossy sets, luminous skin, detailed fashion, polished hair and makeup, clean highlights, and high-budget pop perfection throughout.",
  "R&B glamour": "Warm bronze skin tones, black and gold accents, soft practical light, satin and velvet textures, elegant interiors, luminous highlights, and intimate luxury throughout.",
  "Eerie claymation": "Hand-sculpted clay surfaces, visible fingerprints, miniature sets, muted uncanny color, uneven handmade forms, soft practical miniature lighting, and tactile unsettling charm throughout.",
  "Stop-motion": "Tactile handcrafted materials, miniature practical sets, visible fabrication seams, slightly stepped pose character, controlled tabletop lighting, and charming physical-animation texture throughout.",
  "Paper-cut animation": "Layered cut-paper shapes, visible fibers, flat illustrated color, crisp silhouettes, handmade edges, shadowed paper depth, decorative patterns, and crafted collage texture throughout.",
  "Hand-painted animation": "Visible brushwork, layered pigment, painterly backgrounds, softened outlines, rich handcrafted color, canvas or watercolor texture, and expressive illustrated detail throughout.",
  "Anime-inspired": "Clean expressive linework, stylized facial features, cel-painted color, luminous eyes, graphic shadows, detailed illustrated backgrounds, controlled highlights, and polished animation-art finish throughout.",
  "Graphic novel": "Bold ink lines, dramatic shadow blocks, limited accent color, textured paper, illustrated crosshatching, high contrast, and sophisticated sequential-art atmosphere throughout.",
  "Comic-book": "Crisp outlines, saturated primary colors, halftone texture, graphic shadow shapes, stylized anatomy, printed-paper character, and energetic illustrated spectacle throughout.",
  "Cel-shaded 3D": "Three-dimensional forms with clean graphic outlines, flat color regions, stepped shadows, controlled highlights, simplified materials, and polished illustrated-game rendering throughout.",
  "Photorealistic CGI": "Physically accurate materials, realistic global illumination, detailed skin and hair, precise reflections, volumetric atmosphere, clean high-resolution rendering, and seamless digital realism throughout.",
  "Low-poly 3D": "Faceted geometry, simplified forms, flat-shaded surfaces, restrained texture, clean geometric color, stylized lighting, and intentionally economical digital design throughout.",
  "Miniature diorama": "Clearly handcrafted miniature environments, tiny scaled props, model-making textures, shallow miniature depth, painted surfaces, practical tabletop lighting, and charming physical detail throughout.",
  "Dollhouse surrealism": "Miniature domestic rooms, toy-like furniture, porcelain or plastic textures, artificial pastel color, uncanny scale relationships, pristine tiny details, and dreamlike domestic unease throughout.",
  "Liquid chrome": "Mirror-bright silver surfaces, fluid metallic forms, warped reflections, cool specular highlights, deep black contrast, futuristic polish, and glossy sculptural abstraction throughout.",
  "Holographic iridescence": "Prismatic rainbow highlights, pearlescent surfaces, translucent layers, shifting cyan-magenta color, glossy reflections, soft luminous haze, and futuristic iridescent finish throughout.",
  "Neon noir": "Near-black environments, saturated neon red, blue, and violet accents, wet reflections, hard silhouettes, smoky atmosphere, glossy urban surfaces, and brooding futuristic contrast throughout.",
  "Monochrome minimalism": "Single-hue or black-and-white palette, clean negative space, simple materials, restrained contrast, sparse production design, precise tonal separation, and elegant visual reduction throughout.",
  "High-key white studio": "Bright seamless white surroundings, soft wraparound light, low shadow density, clean neutral color, crisp product-grade detail, airy surfaces, and immaculate studio clarity throughout.",
  "Low-key chiaroscuro": "Deep black shadows, narrow pools of directional light, sculpted facial highlights, rich tonal contrast, restrained color, and dramatic painterly darkness throughout.",
  "Soft pastel": "Powder pink, pale blue, lavender, mint, and cream color, diffused light, gentle contrast, matte surfaces, delicate texture, and calm airy softness throughout.",
  "Desaturated melancholy": "Muted color, cool gray and faded earth tones, soft overcast light, low saturation, restrained highlights, subtle grain, weathered surfaces, and quiet visual sadness throughout.",
  "Crimson-and-black": "Dominant black surfaces with vivid crimson accents, deep shadows, hard red highlights, dark wardrobe, severe contrast, and intense graphic drama throughout.",
  "Teal-and-orange blockbuster": "Cool teal shadows, warm amber skin and highlights, strong complementary contrast, polished surfaces, atmospheric depth, controlled saturation, and large-scale commercial cinema finish throughout.",
  "Golden-hour nostalgia": "Warm amber sunlight, long soft shadows, gentle haze, faded earth color, glowing skin, subtle grain, sunlit dust, and tender memory-like warmth throughout.",
  "Moonlit blue": "Deep navy and cobalt tones, cool silver highlights, soft night haze, pale skin light, subdued warm accents, dark silhouettes, and luminous nocturnal atmosphere throughout.",
  "Underwater ethereal": "Aqua and deep blue palettes, diffused caustic light, suspended particles, translucent fabric, softened detail, pearlescent highlights, and immersive aquatic beauty throughout.",
  "Elemental fantasy": "Visually dominant fire, water, air, earth, ice, or lightning motifs, richly textured natural materials, luminous energy, dramatic atmospheric light, and mythic environmental detail throughout.",
  "Nature mysticism": "Ancient forests, moss, stone, roots, mist, filtered natural light, symbolic organic details, muted green and earth color, and sacred wilderness atmosphere throughout.",
  "Apocalyptic biblical": "Monumental skies, ash and fire, stark divine light, ancient stone, distressed earth tones, ceremonial silhouettes, vast destruction, and solemn prophetic grandeur throughout.",
  "Glitch art": "Digital fragmentation, RGB channel separation, block corruption, pixel noise, scan errors, broken color fields, displaced image sections, and deliberate electronic artifacting throughout.",
  "Datamosh": "Compressed digital smearing, macroblock trails, color displacement, broken codec texture, fragmented silhouettes, melted pixel fields, and aggressive corrupted-video appearance throughout.",
  "CRT distortion": "Curved glass-screen appearance, scan lines, phosphor glow, chromatic fringing, barrel distortion, soft analog resolution, bloom, static noise, and vintage monitor texture throughout.",
  "Kaleidoscopic": "Mirrored geometric repetition, radial symmetry, jewel-like color, layered reflections, intricate patterning, luminous fragments, and hypnotic prismatic imagery throughout.",
  "Double exposure": "Layered translucent imagery, overlapping silhouettes, blended environments, luminous tonal merging, photographic grain, controlled negative space, and poetic composite texture throughout.",
  "Infrared": "False-color foliage, pale luminous skin, dark skies, unusual magenta or cyan tonal mapping, high contrast, bright vegetation, and uncanny infrared-photography texture throughout.",
  "Thermal vision": "Heat-map color ranging from deep violet and blue through red, orange, yellow, and white, simplified surface detail, glowing warm bodies, and sensor-like thermal imaging throughout.",
  "Fisheye distortion": "Pronounced barrel distortion, curved edges, expanded center perspective, compressed borders, close spatial exaggeration, and distinctive ultra-wide optical appearance throughout.",
  "Security-camera footage": "Fixed surveillance-system image quality, high or corner-mounted viewpoint appearance, wide utilitarian lens, low resolution, digital noise, flat exposure, limited color, and institutional monitoring texture without readable overlays.",
  "Documentary realism": "Available practical light, natural skin and material texture, restrained color, modest contrast, believable environments, subtle sensor grain, and honest unembellished observational realism throughout.",
  "Social-media selfie": "Front-facing phone-camera appearance, close personal perspective, wide phone-lens facial character, automatic exposure, digital sharpening, casual available light, and immediate user-generated authenticity throughout.",
  "TikTok transformation": "Bright mobile-video color, crisp phone-camera detail, bold styling contrast, clean vertical-content polish without requiring a vertical aspect ratio, beauty-filter sheen, and highly legible before-and-after visual design throughout.",
  "Dreamlike slow motion": "Soft temporal blur, luminous highlight bloom, gentle pastel or muted color, floating particles, delicate fabric detail, low contrast, and romantic dreamlike image softness throughout.",
  "Frenetic montage": "Punchy high-contrast imagery, varied but coordinated color treatments, bold graphic details, sharp texture changes, intense highlights, and fragmented editorial visual energy throughout.",
  "One-take immersive": "Naturalistic spatial continuity, consistent practical lighting, coherent production design, believable environmental depth, uninterrupted visual realism, and an immediate lived-in atmosphere throughout.",
  "Music-video performance": "Expressive stage styling, dramatic practical and colored light, atmospheric haze, polished wardrobe and makeup, rich contrast, glossy highlights, and premium performance-world production design throughout.",
  "Narrative short film": "Grounded production design, believable wardrobe, motivated practical lighting, restrained cinematic grading, detailed environments, natural skin texture, and cohesive story-world realism throughout.",
  "Movie-trailer aesthetic": "Large-scale cinematic contrast, dramatic skies and practical light, rich production design, deep blacks, luminous highlights, atmospheric depth, premium color grading, and event-film visual polish throughout.",
};

export const MINIMAX_VIDEO_STYLE_PRESETS = [
  {
    value: "",
    label: "Default / let prompt decide",
    description: "No additional global video aesthetic is imposed.",
    prompt_guidance: "",
  },
  ...MINIMAX_VIDEO_STYLE_LABELS.map((label) => ({
    value: label.toLowerCase().replace(/&/g, "and").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""),
    label,
    description: `${label} visual direction for MiniMax video generation.`,
    prompt_guidance: MINIMAX_VIDEO_STYLE_VERBIAGE[label],
  })),
  {
    value: "custom",
    label: "Custom — type exact wording",
    description: "Use custom visual-style wording exactly as entered in every eligible prompt.",
    prompt_guidance: "",
  },
];

export function storyboardMiniMaxVideoStylePreset(value = "") {
  return MINIMAX_VIDEO_STYLE_PRESETS.find((item) => item.value === value) || MINIMAX_VIDEO_STYLE_PRESETS[0];
}

export function storyboardMiniMaxVideoStyleVerbiage(value = "", custom = "") {
  const preset = storyboardMiniMaxVideoStylePreset(value);
  const direction = preset.value === "custom" ? String(custom || "").trim() : String(preset.prompt_guidance || "").trim();
  if (!direction) return "";
  return preset.value === "custom" ? direction : `${preset.label}: ${direction}`;
}

export function storyboardSceneSupportsVideoStyle(scene = {}) {
  const engine = normalizeStoryboardProjectVideoEngine(scene.project_video_engine || scene.projectVideoEngine);
  if (engine === "ltx") return true;
  return ["text_to_video", "reference_to_video"].includes(normalizeStoryboardMiniMaxH3Mode(scene.minimax_h3_mode || scene.minimaxH3Mode));
}

export const MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS = [
  { value: "", label: "Off / natural time", description: "All people and the environment move in the same natural time." },
  { value: "realtime_subjects_timelapse_world", label: "Real-time characters / time-lapse world", description: "Mapped characters stay natural while anonymous extras, location activity, and optional light changes race around them.", prompt_guidance: "Create a clearly separated two-speed reality: protected characters remain in natural real time while only the unprotected background world moves in smooth accelerated time-lapse." },
  { value: "frozen_world", label: "Characters move / world frozen", description: "Protected characters move naturally through a world held almost perfectly still.", prompt_guidance: "Protected characters move naturally through a world frozen at one instant. Unprotected people, particles, vehicles, liquids, smoke, weather, and environmental motion remain suspended unless the scene explicitly releases one element." },
  { value: "reverse_world", label: "Characters forward / world reverses", description: "Protected characters continue normally while unprotected background action runs backward.", prompt_guidance: "Protected characters move and perform forward in natural time while unprotected background people and environmental events visibly run in reverse, including recoverable spills, retracing traffic, returning debris, and reversed weather or smoke." },
  { value: "day_night_sweep", label: "Real-time characters / day-to-night sweep", description: "Characters stay natural while daylight, shadows, windows, and practical lights rapidly change.", prompt_guidance: "Protected characters remain in natural time while the environment passes rapidly through a readable day-to-night or night-to-day cycle, with accelerated sky color, sunlight angle, shadow travel, window light, and practical lights switching on or off." },
  { value: "seasonal_passage", label: "Real-time characters / seasons pass", description: "The location visibly crosses seasons around stable real-time characters.", prompt_guidance: "Protected characters remain in natural time while the environment transitions through accelerated seasonal change: vegetation, weather, ground cover, atmospheric color, and daylight evolve coherently without changing the characters' identities or wardrobe unless explicitly requested." },
  { value: "crowd_flow", label: "Real-time characters / crowd river", description: "Anonymous extras stream around the referenced cast while the cast remains readable.", prompt_guidance: "Protected characters remain sharply readable in natural time while anonymous unreferenced extras flow around them as an accelerated crowd river, forming continuous directional streams without duplicating, replacing, or obscuring the protected cast." },
  { value: "looping_background", label: "Real-time characters / looping background", description: "Background actions repeat in visible cycles while the referenced cast continues normally.", prompt_guidance: "Protected characters continue naturally while unprotected background actions repeat in deliberate seamless temporal loops. Keep each loop spatially anchored and visually distinct from the protected characters' unrepeated performance." },
  { value: "delayed_world", label: "Characters lead / world echoes behind", description: "The environment responds a beat late, creating a temporal echo.", prompt_guidance: "Protected characters move in natural time while unprotected environmental reactions lag behind them in visible delayed echoes, as though the world responds one beat late. Preserve physical readability and avoid duplicating the protected characters." },
  { value: "living_shadows", label: "Real-time characters / living shadows", description: "Characters remain normal while unprotected shadows move independently or at accelerated speed.", prompt_guidance: "Protected characters remain in natural time while cast shadows and environmental shadows move independently at accelerated speed, changing direction and shape without changing the protected characters' bodies, faces, or identity." },
  { value: "reflection_delay", label: "Real-time characters / delayed reflections", description: "Mirrors and reflective surfaces lag behind the real-time cast.", prompt_guidance: "Protected characters move in natural time while their reflections and the environment's reflections respond with a deliberate temporal delay. Keep the real characters singular and stable; the delayed imagery exists only inside physically plausible reflective surfaces." },
  { value: "gravity_separation", label: "Real-time characters / altered-gravity world", description: "The cast stays grounded while loose environmental objects behave in surreal gravity.", prompt_guidance: "Protected characters remain grounded and move in natural time while unprotected loose objects, dust, fabric scraps, droplets, leaves, and environmental debris rise, fall, or drift under visibly altered gravity." },
  { value: "custom", label: "Custom temporal effect", description: "Use the user's exact temporal-effect wording while retaining the selected protection and extras rules." },
];

// FX are injected by the Builder into each timestamped shot after prompt
// generation. They are intentionally separate from camera flow and temporal
// world effects so the LLM does not have to invent, place, or repeat them.
export const STORYBOARD_FX_PRESETS = [
  { value: "", label: "Off", description: "Do not add builder-managed FX.", cues: [] },
  { value: "lighting", label: "Lighting FX", description: "Practical-light flicker, sweeps, pulses, exposure and color changes.", cues: ["A brief practical-light flicker cascade sweeps through the environment and resolves into clean subject light.", "A controlled red-to-blue light sweep travels across the set, creating a readable cinematic exposure pulse.", "Neon reflections pulse once across the surfaces on the musical accent, without obscuring the subject's face."] },
  { value: "camera_lens", label: "Camera / Lens FX", description: "Lens flare, bloom, parallax, focus breathing, motion blur and optical artifacts.", cues: ["A restrained lens flare and foreground-parallax accent passes through the frame as the camera moves.", "A subtle lens-bloom pulse and focus-breathing shift accent the camera move while the subject remains sharp.", "A controlled motion-blur streak catches the edge of the frame during the camera movement, preserving subject readability."] },
  { value: "glitch", label: "Glitch / Digital FX", description: "Scan lines, signal tearing, RGB separation, frame stutter and digital distortion.", cues: ["A restrained scan-glitch briefly tears across the background, leaving the subject's face and body stable.", "A short RGB-separation and signal-noise burst flickers at the musical accent, then clears completely.", "A controlled digital frame stutter affects the environment for a moment without duplicating or deforming the subject."] },
  { value: "atmospheric", label: "Atmospheric FX", description: "Fog, smoke, dust, ash, rain, sparks and environmental particles.", cues: ["A thin layer of atmospheric mist curls through the background, catching the existing light without hiding the subject.", "Small airborne particles drift through the light beam and briefly sparkle around the environment.", "A restrained veil of smoke and dust crosses the deeper background, preserving the foreground performance clearly."] },
  { value: "energy", label: "Energy / Impact FX", description: "Beat flashes, sparks, shockwave-like light and controlled energy accents.", cues: ["A compact beat-synchronized light burst radiates through the environment and quickly settles.", "A brief ring of sparks and reflected light marks the musical accent without touching or altering the subject.", "A controlled impact pulse ripples through loose environmental particles while the subject continues naturally."] },
  { value: "film_texture", label: "Film / Texture FX", description: "Film grain, halation, gate weave, light leaks and shutter trails.", cues: ["A subtle film-grain and halation texture becomes visible in the highlights for this shot.", "A restrained analog gate-weave and soft light leak add a brief tactile film accent.", "A short shutter-trail texture catches the brightest movement while keeping the composition and subject readable."] },
  { value: "distortion", label: "Distortion FX", description: "Fisheye warp, heat haze, ripples, refraction and reality-bending lens effects.", cues: ["A localized lens distortion gently bends the outer edges of the environment while the subject remains stable.", "A brief heat-haze ripple passes through the background, refracting light without changing the subject's identity.", "A controlled wide-angle warp accentuates the perspective at the frame edges, then returns to a clean image."] },
  { value: "supernatural", label: "Supernatural FX", description: "Living shadows, aura, reality fractures, floating debris and uncanny visual accents.", cues: ["The environment's shadows shift independently for a brief uncanny accent while the subject remains physically natural.", "A faint supernatural glow gathers in the background atmosphere and fades without changing the subject's face or clothing.", "A few loose environmental fragments hover briefly in the air, creating a controlled reality-fracture accent around the subject."] },
  { value: "music_video", label: "Music Video FX", description: "A curated rhythmic mix of lighting, lens and restrained post-production accents.", cues: ["A rhythmic neon light pulse combines with a restrained lens flare on the musical accent.", "A brief bloom and light-streak accent sweeps across the frame, then resolves into clean cinematic contrast.", "A controlled mix of atmospheric particles and subtle chromatic color separation marks the beat without obscuring the subject."] },
  { value: "custom", label: "Custom FX JSON", description: "Use the exact custom FX JSON wording and Builder placement rules." },
];

export function storyboardFxPreset(value = "") {
  return STORYBOARD_FX_PRESETS.find((item) => item.value === String(value || "")) || STORYBOARD_FX_PRESETS[0];
}

export function normalizeStoryboardCustomFxJson(input) {
  let source = input;
  if (typeof source === "string") {
    try { source = JSON.parse(source); } catch { return null; }
  }
  if (Array.isArray(source)) source = { cues: source };
  if (!source || typeof source !== "object") return null;
  const rawCues = Array.isArray(source.cues) ? source.cues : (Array.isArray(source.effects) ? source.effects : []);
  const cues = rawCues.map((item) => {
    if (typeof item === "string") return item.trim();
    if (item && typeof item === "object") return String(item.text || item.cue || item.description || "").trim();
    return "";
  }).filter(Boolean).slice(0, 12);
  const primary = String(source.primary || source.primary_effect || "").trim();
  const secondary = String(source.secondary || source.secondary_effect || "").trim();
  if (primary) cues.unshift(primary);
  if (secondary) cues.push(secondary);
  if (!cues.length) return null;
  return {
    label: String(source.label || source.name || "Custom FX").trim().slice(0, 120) || "Custom FX",
    cues: Array.from(new Set(cues)).slice(0, 12),
    timing: String(source.timing || "on the strongest musical or action accent").trim().slice(0, 240),
    intensity: Math.max(0, Math.min(10, Number(source.intensity ?? 6))) || 6,
    avoid: String(source.avoid || source.avoid_text || "Do not obscure, duplicate, deform, or alter the mapped subject.").trim().slice(0, 500),
  };
}

export function storyboardFxContract(value = "", customInput = "", shotIndex = 0) {
  const preset = storyboardFxPreset(value);
  const custom = value === "custom" ? normalizeStoryboardCustomFxJson(customInput) : null;
  const source = custom || preset;
  if (!source?.cues?.length) return null;
  const cue = source.cues[Math.max(0, Number(shotIndex) || 0) % source.cues.length];
  const timing = custom ? custom.timing : "on a readable musical, lyric, or action accent when one is present";
  const intensity = custom ? custom.intensity : 6;
  const avoid = custom ? custom.avoid : "Keep the mapped subject's face, identity, body, wardrobe, performance, and lip sync stable and readable.";
  return {
    label: source.label || preset.label,
    cue,
    timing,
    intensity,
    exact_verbiage: `FX placement — ${source.label || preset.label}: Insert this FX directly inside the timestamped shot after the camera direction and before the lyric or performance action. Use one clear ${intensity}/10 visual accent: ${cue} Timing: ${timing} ${avoid}`,
  };
}

export function storyboardTemporalWorldEffectPreset(value = "") {
  return MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.find((item) => item.value === value) || MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS[0];
}

export function storyboardTemporalProtectedMode(value = "") {
  return ["all_referenced", "lead_only", "custom"].includes(String(value || "")) ? String(value) : "all_referenced";
}

export function storyboardTemporalIntensity(value = 8) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, Math.round(number))) : 8;
}

function storyboardTemporalLocationExamples(scene = {}) {
  const location = scene.location_ref || scene.locationRef || {};
  const context = [
    location.name,
    location.description,
    scene.setting,
    scene.location,
    scene.story_beat,
    scene.prompt_summary,
  ].map((value) => String(value || "").toLowerCase()).join(" ");
  if (/store|shop|market|liquor|grocery|retail|checkout/.test(context)) {
    return "customers and staff crossing aisles, checkout activity, shelf restocking, changing window light, and accelerated reflections";
  }
  if (/kitchen|dining|restaurant|cafe|bar|diner/.test(context)) {
    return "location-appropriate patrons or household activity, staff movement, changing practical light, drifting steam, and accelerated shadows";
  }
  if (/street|road|alley|sidewalk|parking|overpass|city|urban/.test(context)) {
    return "pedestrians, passing traffic, moving reflections, changing signs or practical lights, fast clouds, and traveling shadows";
  }
  if (/laundromat|laundry/.test(context)) {
    return "customers cycling through the room, spinning machines, baskets changing position, shifting fluorescent light, and window reflections";
  }
  if (/bedroom|apartment|living room|house|home|hallway|corridor|stair/.test(context)) {
    return "location-appropriate household or neighbor activity, rapidly shifting window light, traveling shadows, changing practical lights, weather, and moving reflections";
  }
  if (/forest|woods|field|garden|park|outdoor|beach|mountain/.test(context)) {
    return "location-appropriate passersby when permitted, fast clouds, traveling sunlight or moonlight, moving shadows, weather, vegetation, smoke, and airborne particles";
  }
  return "location-appropriate anonymous activity when permitted, changing light and shadows, weather, traffic or reflections, smoke, particles, and moving environmental details";
}

export function storyboardTemporalWorldEffectForScene(scene = {}, state = {}) {
  const override = String(scene.temporal_world_effect_override || scene.temporalWorldEffectOverride || "global").trim();
  const globalKey = String(state.temporalWorldEffect || state.temporal_world_effect || "").trim();
  const key = override === "off" ? "" : (override && override !== "global" ? override : globalKey);
  const preset = storyboardTemporalWorldEffectPreset(key);
  const custom = String(
    key === "custom" && override && override !== "global"
      ? (scene.temporal_world_effect_custom || scene.temporalWorldEffectCustom || "")
      : (state.temporalWorldEffectCustom || state.temporal_world_effect_custom || ""),
  ).trim();
  const baseDirection = key === "custom" ? custom : String(preset.prompt_guidance || "").trim();
  if (!key || !baseDirection) return null;

  const protectedMode = storyboardTemporalProtectedMode(state.temporalProtectedCharacters || state.temporal_protected_characters);
  const protectedCustom = String(state.temporalProtectedCustom || state.temporal_protected_custom || "").trim();
  const protectedDirection = protectedMode === "lead_only"
    ? "Protect only the first mapped/reference character at natural 1x real-time speed; other mapped characters may receive the selected temporal effect."
    : protectedMode === "custom" && protectedCustom
      ? `Protect only these named mapped/reference characters at natural 1x real-time speed: ${protectedCustom}.`
      : "Protect every mapped/reference character in the scene at natural 1x real-time speed, including secondary referenced characters. Never accelerate, freeze, reverse, echo, duplicate, or temporally distort any protected character.";
  const allowExtras = state.temporalAllowBackgroundExtras !== false && state.temporal_allow_background_extras !== false;
  const extrasDirection = allowExtras
    ? "Anonymous unreferenced background extras are allowed. Infer only extras that naturally belong in the mapped location—for example customers or staff in a store, family or household activity in a kitchen, and pedestrians or traffic on a street. Keep them clearly secondary; never turn an extra into a principal character or duplicate a mapped/reference character."
    : "Do not add anonymous background people. Apply the temporal effect only to existing unprotected scene elements and the environment.";
  const environmentTimePassage = state.temporalEnvironmentTimePassage !== false && state.temporal_environment_time_passage !== false;
  const environmentDirection = environmentTimePassage
    ? "Environmental time passage is enabled: when appropriate, accelerate or transform daylight, shadows, practical lighting, weather, traffic, smoke, particles, and location activity while preserving spatial continuity."
    : "Do not add a day/night, lighting, weather, or seasonal time passage unless the scene notes explicitly request it.";
  const intensity = storyboardTemporalIntensity(state.temporalBackgroundIntensity ?? state.temporal_background_intensity ?? 8);
  const intensityDirection = intensity <= 3
    ? `Use a subtle ${intensity}/10 background-effect intensity with restrained, readable temporal separation.`
    : intensity <= 6
      ? `Use a clear ${intensity}/10 background-effect intensity that is immediately visible but does not overpower the protected characters.`
      : intensity <= 8
        ? `Use a strong ${intensity}/10 background-effect intensity with unmistakable temporal separation while keeping protected faces and actions readable.`
        : `Use an extreme ${intensity}/10 background-effect intensity with dramatic temporal contrast, while protected characters remain stable, singular, and readable.`;
  const audioDirection = "Temporal speed separation is visual only. Keep supplied or generated dialogue, singing, lip sync, facial timing, and primary audio at normal speed; never time-stretch, reverse, or accelerate protected voices.";
  const locationExamples = storyboardTemporalLocationExamples(scene);
  const cueCount = intensity >= 9 ? "at least two concrete, clearly visible effect cues" : "at least one concrete, clearly visible effect cue";
  const stagingRequirement = `TIMESTAMP STAGING REQUIREMENT: Every timestamp block must actively show ${cueCount} affecting only the unprotected background/world while protected characters continue at natural 1x speed. Use the mapped location to choose actions such as ${locationExamples}. At intensity 7 or higher, subtle flicker, ambience, drifting particles, or a vague mention of time passage alone does not satisfy this requirement. The temporal contrast must be immediately visible in the action itself. Any phrase such as “no people” inside a location-reference description describes only the source image and does not prohibit anonymous background extras when this contract permits them.`;
  const timestampAction = `Visibly enact this temporal layer at ${intensity}/10: ${baseDirection} Use concrete location-appropriate activity such as ${locationExamples}. Protected mapped/reference characters remain singular and move at natural 1x speed; only the permitted unprotected background/world receives the effect.`;
  const continuityRequirement = allowExtras
    ? "CONTINUITY RULE: Do not add any new named, principal, mapped, or referenced characters. Anonymous unreferenced background extras are explicitly permitted and must remain secondary and subject to the selected temporal effect; never prohibit them with a generic ‘no new characters’ or ‘no people’ rule."
    : "CONTINUITY RULE: Do not add new named, mapped, referenced, principal, or anonymous characters.";
  // This is a builder-owned contract. Gemma should not spend tokens recreating
  // temporal rules, and the final renderer prompt should receive one canonical
  // block instead of repeated per-shot instructions.
  const verbiage = `Temporal / World Effect — ${preset.label} — Mandatory:\n\n${baseDirection}\n\n${protectedDirection}\n\nOnly the unprotected background/world may receive this effect, including location-appropriate environmental motion or anonymous extras when permitted. Keep the effect secondary; never alter, duplicate, replace, or obscure a mapped/reference character.\n\nTimestamp staging requirement: every shot must visibly show at least one concrete, readable effect affecting only the unprotected background/world while protected characters continue moving naturally at normal 1x speed.\n\n${audioDirection}`;
  return {
    enabled: true,
    key,
    label: preset.label,
    exact_verbiage: verbiage,
    protected_characters: protectedMode,
    protected_custom: protectedCustom,
    allow_background_extras: allowExtras,
    background_intensity: intensity,
    environment_time_passage: environmentTimePassage,
    timestamp_staging_requirement: stagingRequirement,
    timestamp_action: timestampAction,
    continuity_requirement: continuityRequirement,
  };
}

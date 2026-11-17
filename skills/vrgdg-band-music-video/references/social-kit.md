# Social media kit

**When:** after the final is approved, or when the user asks ("what about my social media sharing stuff").

**Where it goes:** `SocialMediaSharing/` inside the project the user treats as theirs. Tonight that was the alternates
copy.

## 1. Facts and poster references

```
python scripts/social_kit.py --work W [--project "<their project>"] --final "<final.mp4>"
```
- **poster_refs:** the character images in `POSTER_CARDS` order (1_singer, 2_drummer, ...), for attaching to the
  image model in that order.
- **facts.json:** scenes, runtime, render count and GPU hours from logs/render.log, re-shoot notes, locations and
  voices.
- **Use these numbers and never invent any.** Tonight's were 28 scenes, 2:52, 45 renders and 8.0 GPU hours.

## 2. The four text files Claude writes

- **`poster_prompt.txt`:** a ChatGPT image prompt. A music video gets **16:9 single or album key art that reads as a
  YouTube thumbnail** (a short film gets a theatrical poster).
  1. Open with what to make and that the images are attached.
  2. Name each attached image by number with that character's look.
  3. Composition: one image that tells the story. Tonight: the singer mid-scream on the kitchen counter in the stolen
     aviators, the drummer chugging by the broken table, the guitarist hugging his jacket, the bassist peeking from
     the wardrobe, police lights through the window. Keep the faces big for the thumbnail.
  4. Style: match the video's art style.
  5. The exact text: the title, and a tagline taken from the lyrics ("NEVER GOING BACK").
  6. End with "Use only the text listed above."
- **`youtube_description.txt`:**
  - a title line, a one-line hook, and a short synopsis of the story;
  - a ━━━ divider, then 🎬 HOW THIS WAS MADE with "▶ MY PART" (the user's brief, song, art, settings and mid-render
    fixes, written in the user's voice) and "▶ WHAT CLAUDE DID ON ITS OWN" (lyric timing, vocal-free track, scene plan,
    every prompt, the intro and outro voices, the overnight renders with real numbers, the real re-shoots, the
    checks, and the alternates);
  - 🛠 TOOLS, a 👥 CAST line, and the user's links (always the GitHub node pack
    https://github.com/vrgamegirl19/comfyui-vrgamedevgirl, plus the skill download if they gave one).
- **`reddit_post.txt`:** the same content as plain text for Reddit's rich text editor: emoji section headings, "•"
  bullets, bare URLs, no Markdown symbols, and a `[VIDEO LINK]` placeholder.
- **`short_post.txt`:** one post for X, at most 280 characters (count a link as 23; tonight's was 224), with the hook,
  a `[LINK]` placeholder and 2–4 hashtags. Then a 2–3 sentence version for Instagram, TikTok, Facebook and Threads.

## 3. Rules

- **Facts:** only from facts.json, the logs and the spec.
- **Credit the user's tools honestly:** "song made with Suno", the user's own character and location art.
- **Never describe the characters as real people.**
- **Ask before posting about:**
  - **The artist or band name:** if you don't have it, leave it out and ask, never invent one.
  - **A censored title:** offer "F*CK THE STATIC" when the title has profanity, because thumbnails and titles with
    profanity can limit monetization.
- **The user's links:** fill in only the ones they gave; leave placeholders for the rest.

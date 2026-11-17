"""refs_spec.py for make_refs.py: the cards and Krea 2 prompts of NO PERMISSION TO EXIST (2026-11-13).

Copy to <work>/refs_spec.py and rewrite for the new video. Picked seeds there: singer 212, guitarist 231,
bassist 222, drummer 232, clerk 221; every location 301 except roof 302.

What made these work (see references/lessons.md "Reference pictures"):
- Name an exact young age and a fresh face: "a handsome young man of 24 with a fresh, smooth youthful face".
  "In his mid-20s" alone gave a man of about 40 (the bassist and clerk lines below still say mid-20s and came out
  fine, but the men aged most).
- Spell out the genre look item by item (hair colour and cut, each garment, boots, jewellery); "a punk look" alone
  gave a plain drummer.
- Plain garments ("a plain ripped white t-shirt") so no logos or text appear.
- Empty hands on the sheet: instruments come from the scene prompt, and a sheet instrument leaks into every scene.
- Locations: name every feature you will want to use in a scene (the stamp rack, the turnstiles, the amber buttons).
"""

SUBJECTS = {
    "singer": ("Vex (lead singer)",
               "a beautiful young woman of 25 with a soft heart-shaped face, full cheeks, big grey-green eyes and delicate feminine features, the lead singer of a British punk rock band, slim build, fair skin, a choppy "
               "shoulder-length bleached platinum-blonde shag haircut with dark roots and a jagged fringe, smudged black "
               "eyeliner, deep red lipstick and a small silver nose ring; she wears a black leather biker jacket with "
               "silver studs on the shoulders and lapels over a plain ripped white sleeveless t-shirt, a black studded "
               "choker, a short red tartan kilt skirt with a silver chain at the hip, ripped black fishnet tights and "
               "scuffed black lace-up combat boots"),
    "guitarist": ("Rook (guitarist)",
                  "a handsome young man of 24 with a fresh, smooth youthful face and bright eyes, the guitarist of a "
                  "British punk rock band, tall and slim, a tall bright scarlet-red mohawk with shaved sides and a "
                  "silver hoop in one ear; he wears a sleeveless black denim vest covered in silver studs and safety "
                  "pins, open over a plain black-and-white horizontally striped t-shirt, black leather wristbands, "
                  "tight black jeans ripped at the knees with a studded belt, and black high-top boots"),
    "bassist": ("Jinx (bassist)",
                "a woman in her mid-20s, the bassist of a British punk rock band, warm brown skin, a jet-black "
                "chin-length bob with a blunt fringe dyed vivid hot pink, winged black eyeliner and red lipstick; she "
                "wears a cropped leopard-print faux-fur jacket over a plain black slip dress, a silver padlock chain "
                "necklace, black fishnet tights and chunky black platform creeper shoes"),
    "drummer": ("Moz (drummer)",
                "a young man of 24 with a fresh youthful face, the drummer of a British punk rock band, sturdy and "
                "muscular with broad shoulders and strong arms, a bleached-blond buzz cut and a cheeky look; he wears a "
                "tight plain white ribbed cotton tank top, thin bright red suspenders over his shoulders clipped to "
                "rolled-up pale blue bleach-splattered jeans, a black studded leather wristband, and oxblood-red "
                "lace-up boots"),
    "clerk": ("The Clerk",
              "a man in his mid-20s, a prim government office clerk, slim and narrow-shouldered, neatly side-parted "
              "slick brown hair, clean-shaven, round thin wire-rimmed glasses; he wears a slightly too large pale grey "
              "three-piece suit, a white shirt buttoned to the collar, a mustard-yellow tie with a silver tie clip, and "
              "polished brown lace-up shoes"),
}

LOCATIONS = {
    "hall": ("Permit office waiting hall",
             "the vast public waiting hall of a 1970s British government permit office at night: brutalist concrete "
             "pillars and beige wood-panelled walls, a long row of service counter windows with frosted glass screens "
             "along the back wall, rows of bolted orange plastic chairs pushed to the left and right sides leaving a "
             "wide open stretch of scuffed speckled linoleum floor in the middle, a ticket-number dispenser on a metal "
             "post, round wall clocks, tall rubber plants in pots, loose paper forms scattered on the floor, and long "
             "cold fluorescent tube lights overhead with a few glowing warm amber"),
    "office": ("Clerk's office",
               "a cramped clerk's office in a 1970s British government building: a heavy wooden desk in the middle "
               "crowded with rubber stamps on a rack, ink pads, a green glass banker's lamp, a beige rotary telephone "
               "and towering stacks of paper forms, a worn brown leather swivel chair behind the desk, grey metal "
               "filing cabinets and shelves of ring binders along beige walls, venetian blinds over a small window, "
               "brown carpet tiles, a buzzing fluorescent ceiling light"),
    "archive": ("Archive corridor",
                "an endless narrow archive corridor in the basement of a government building: both walls lined from "
                "floor to ceiling with grey metal filing cabinets and open shelves of bulging manila folders, a row of "
                "bare fluorescent tube lights running along the ceiling and receding into the far distance, worn "
                "green linoleum floor, a few loose sheets of paper on the floor, exposed pipes overhead"),
    "lobby": ("Entrance lobby",
              "the entrance lobby of a 1970s British government office block in the evening: a row of waist-high "
              "chrome turnstiles across the middle of the room, a queue lane made of thick red rope barriers on brass "
              "posts in front of them, a polished speckled terrazzo floor, a long wood-veneer reception desk on one "
              "side, concrete pillars, large potted palms, and tall glass entrance doors at the back with orange "
              "street light glowing beyond"),
    "lift": ("Office lift",
             "the inside of an old office lift seen from its doorway looking in: a small square cabin with brown "
             "wood-effect laminate walls, a brass handrail running around the walls at waist height, a brushed steel "
             "button panel with two columns of round glowing amber buttons on the right-hand wall, a worn burgundy "
             "carpet floor and a square frosted ceiling light"),
    "roof": ("Rooftop at dusk",
             "the flat rooftop of a brutalist concrete office block in London at dusk: a wide gravel and tar surface "
             "with puddles, squat air vents, a rusty water tank and a tangle of aerials, a low concrete parapet "
             "around the edge, and beyond it a wide skyline of London rooftops, chimneys and tower blocks under a "
             "glowing orange and magenta sunset sky with dark clouds, the first city lights coming on"),
}

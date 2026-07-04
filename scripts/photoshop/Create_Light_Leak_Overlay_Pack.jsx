/*
  VRGDG Light Leak Overlay Pack Generator

  Run in Adobe Photoshop:
    File > Scripts > Browse... > Create_Light_Leak_Overlay_Pack.jsx

  Output:
    LightLeak_WarmDream/
      frame_0001.png ... frame_0008.png
      manifest.json

  These frames intentionally use a pure black background for Screen/Add
  blending in video compositing.
*/

#target photoshop

(function () {
  app.displayDialogs = DialogModes.NO;

  var PACK_NAME = "LightLeak_WarmDream";
  var WIDTH = 1920;
  var HEIGHT = 1080;
  var FRAME_COUNT = 8;
  var DPI = 72;
  var VERSION = "1.0";

  var previousRulerUnits = app.preferences.rulerUnits;
  app.preferences.rulerUnits = Units.PIXELS;

  function pad4(value) {
    var text = String(value);
    while (text.length < 4) text = "0" + text;
    return text;
  }

  function clamp(value, minValue, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
  }

  function makeRng(seed) {
    var state = seed % 2147483647;
    if (state <= 0) state += 2147483646;
    return function () {
      state = (state * 16807) % 2147483647;
      return (state - 1) / 2147483646;
    };
  }

  function rgbColor(r, g, b) {
    var color = new SolidColor();
    color.rgb.red = clamp(Math.round(r), 0, 255);
    color.rgb.green = clamp(Math.round(g), 0, 255);
    color.rgb.blue = clamp(Math.round(b), 0, 255);
    return color;
  }

  function fillSelection(doc, color, opacity) {
    doc.selection.fill(color, ColorBlendMode.NORMAL, opacity, false);
    doc.selection.deselect();
  }

  function addBlackBackground(doc) {
    var layer = doc.artLayers.add();
    layer.name = "pure_black_screen_blend_background";
    doc.activeLayer = layer;
    doc.selection.select([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]);
    fillSelection(doc, rgbColor(0, 0, 0), 100);
    layer.move(doc, ElementPlacement.PLACEATEND);
  }

  function addSoftEllipse(doc, name, x, y, w, h, color, opacity, blurPx) {
    var layer = doc.artLayers.add();
    layer.name = name;
    layer.blendMode = BlendMode.SCREEN;
    layer.opacity = opacity;
    doc.activeLayer = layer;
    doc.selection.select([
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ]);
    fillSelection(doc, color, 100);
    layer.applyGaussianBlur(blurPx);
    return layer;
  }

  function addSoftRect(doc, name, x, y, w, h, color, opacity, blurPx) {
    var layer = doc.artLayers.add();
    layer.name = name;
    layer.blendMode = BlendMode.SCREEN;
    layer.opacity = opacity;
    doc.activeLayer = layer;
    doc.selection.select([
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ]);
    fillSelection(doc, color, 100);
    layer.applyGaussianBlur(blurPx);
    return layer;
  }

  function addNoiseHaze(doc, rng, frameIndex) {
    var layer = doc.artLayers.add();
    layer.name = "subtle_film_haze_noise_" + pad4(frameIndex);
    layer.blendMode = BlendMode.SCREEN;
    layer.opacity = 8 + Math.round(rng() * 5);
    doc.activeLayer = layer;
    doc.selection.select([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]);
    fillSelection(doc, rgbColor(28, 16, 10), 100);
    layer.applyAddNoise(9 + rng() * 8, NoiseDistribution.GAUSSIAN, true);
    layer.applyGaussianBlur(1.2 + rng() * 1.8);
  }

  function addFrameVariation(doc, frameIndex) {
    var rng = makeRng(9000 + frameIndex * 137);
    var drift = (frameIndex - 1) / Math.max(1, FRAME_COUNT - 1);
    var warmA = rgbColor(255, 128 + rng() * 70, 28 + rng() * 28);
    var warmB = rgbColor(255, 56 + rng() * 58, 88 + rng() * 42);
    var amber = rgbColor(255, 186 + rng() * 42, 64 + rng() * 34);
    var cyan = rgbColor(42 + rng() * 38, 210 + rng() * 35, 255);

    addBlackBackground(doc);

    var edgeSide = frameIndex % 4;
    if (edgeSide === 0) {
      addSoftRect(doc, "left_film_burn_edge", -140 + drift * 160, -80, 330 + rng() * 180, HEIGHT + 160, warmA, 68, 95 + rng() * 35);
    } else if (edgeSide === 1) {
      addSoftRect(doc, "right_film_burn_edge", WIDTH - 260 - drift * 130, -80, 390 + rng() * 180, HEIGHT + 160, warmB, 64, 110 + rng() * 35);
    } else if (edgeSide === 2) {
      addSoftRect(doc, "top_film_burn_edge", -100, -130 + drift * 100, WIDTH + 200, 290 + rng() * 140, amber, 58, 100 + rng() * 45);
    } else {
      addSoftRect(doc, "bottom_film_burn_edge", -100, HEIGHT - 250 - drift * 110, WIDTH + 200, 340 + rng() * 140, warmA, 58, 120 + rng() * 40);
    }

    addSoftEllipse(
      doc,
      "large_warm_bloom",
      -260 + rng() * 460 + drift * 170,
      -140 + rng() * 430,
      760 + rng() * 440,
      420 + rng() * 320,
      warmA,
      44 + rng() * 22,
      105 + rng() * 45
    );

    addSoftEllipse(
      doc,
      "pink_amber_haze",
      WIDTH * 0.45 + rng() * 420 - drift * 160,
      HEIGHT * 0.08 + rng() * 530,
      560 + rng() * 520,
      260 + rng() * 320,
      warmB,
      28 + rng() * 18,
      95 + rng() * 50
    );

    if (frameIndex % 2 === 0) {
      addSoftRect(
        doc,
        "subtle_cyan_streak",
        WIDTH * 0.48 + rng() * 360,
        HEIGHT * 0.16 + rng() * 500,
        520 + rng() * 420,
        18 + rng() * 26,
        cyan,
        20 + rng() * 13,
        18 + rng() * 20
      );
    }

    var streakCount = 2 + Math.floor(rng() * 3);
    for (var i = 0; i < streakCount; i++) {
      addSoftRect(
        doc,
        "warm_lens_streak_" + (i + 1),
        -120 + rng() * WIDTH * 0.62,
        110 + rng() * (HEIGHT - 220),
        520 + rng() * 820,
        8 + rng() * 22,
        i % 2 === 0 ? amber : warmB,
        13 + rng() * 18,
        10 + rng() * 24
      );
    }

    var orbCount = 3 + Math.floor(rng() * 4);
    for (var j = 0; j < orbCount; j++) {
      var size = 80 + rng() * 250;
      addSoftEllipse(
        doc,
        "soft_bokeh_leak_" + (j + 1),
        rng() * WIDTH,
        rng() * HEIGHT,
        size,
        size * (0.55 + rng() * 0.8),
        j % 3 === 0 ? amber : warmA,
        10 + rng() * 16,
        35 + rng() * 45
      );
    }

    addNoiseHaze(doc, rng, frameIndex);
  }

  function savePng(doc, file) {
    var options = new PNGSaveOptions();
    options.interlaced = false;
    doc.saveAs(file, options, true, Extension.LOWERCASE);
  }

  function writeTextFile(file, text) {
    file.encoding = "UTF8";
    file.open("w");
    file.write(text);
    file.close();
  }

  function createManifest() {
    var frames = [];
    for (var i = 1; i <= FRAME_COUNT; i++) {
      frames.push('    "frame_' + pad4(i) + '.png"');
    }
    return [
      "{",
      '  "name": "' + PACK_NAME + '",',
      '  "type": "overlay_fx_pack",',
      '  "version": "' + VERSION + '",',
      '  "width": ' + WIDTH + ",",
      '  "height": ' + HEIGHT + ",",
      '  "frame_count": ' + FRAME_COUNT + ",",
      '  "background": "pure_black",',
      '  "recommended_blend_mode": "screen",',
      '  "default_opacity": 0.45,',
      '  "default_speed": 0.65,',
      '  "default_motion": {',
      '    "pan_x": 0.05,',
      '    "pan_y": 0.02,',
      '    "scale": 1.08,',
      '    "rotation_degrees": 1.5',
      "  },",
      '  "frames": [',
      frames.join(",\n"),
      "  ]",
      "}",
      "",
    ].join("\n");
  }

  try {
    var parentFolder = Folder.selectDialog("Choose where to save the VRGDG overlay FX pack");
    if (!parentFolder) {
      app.preferences.rulerUnits = previousRulerUnits;
      return;
    }

    var packFolder = new Folder(parentFolder.fsName + "/" + PACK_NAME);
    if (!packFolder.exists) packFolder.create();

    for (var frame = 1; frame <= FRAME_COUNT; frame++) {
      var doc = app.documents.add(WIDTH, HEIGHT, DPI, PACK_NAME + "_" + pad4(frame), NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
      addFrameVariation(doc, frame);
      doc.flatten();
      var outputFile = new File(packFolder.fsName + "/frame_" + pad4(frame) + ".png");
      savePng(doc, outputFile);
      doc.close(SaveOptions.DONOTSAVECHANGES);
    }

    writeTextFile(new File(packFolder.fsName + "/manifest.json"), createManifest());
    alert("Created overlay FX pack:\n" + packFolder.fsName + "\n\nUse Screen/Add blending over video.");
  } catch (error) {
    alert("Could not create light leak overlay pack:\n" + error);
  } finally {
    app.preferences.rulerUnits = previousRulerUnits;
  }
})();

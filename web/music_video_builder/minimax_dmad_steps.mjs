// The DMAD row's step boxes edit the same settings as the Sampler (single pass) and the Pass 1 / Pass 2 sections,
// so each pair is kept in step in both directions. Pass 1 is the single pass when 2 Pass is off.
export function wireMiniMaxDmadSteps({ stepsControls, twoPassControls, singlePassSteps, state, save, persist }) {
  for (const { pass, input } of stepsControls) {
    const twoPassSteps = twoPassControls[pass - 1].steps;
    const target = () => (state.miniMaxH3TwoPassEnabled ? twoPassSteps : pass === 1 ? singlePassSteps : null);
    input.addEventListener("input", () => {
      const field = target();
      if (!field) return;
      field.value = input.value;
      save();
    });
    input.addEventListener("change", persist);
    for (const source of [twoPassSteps, ...(pass === 1 ? [singlePassSteps] : [])]) {
      source.addEventListener("input", () => {
        if (target() === source) input.value = source.value;
      });
    }
  }
}

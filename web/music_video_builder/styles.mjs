const builderResponsiveStyle = document.createElement("style");
builderResponsiveStyle.textContent = `
  .vrgdg-builder-tools-pane {
    container-type: inline-size;
  }
  .vrgdg-builder-tool-row {
    display: grid;
    grid-template-columns: minmax(132px, max-content) minmax(150px, 1fr);
    gap: 8px;
    align-items: center;
    border: 1px solid #3f3f46;
    border-radius: 7px;
    background: #27272a;
    padding: 8px;
    margin-bottom: 8px;
    min-width: 0;
  }
  .vrgdg-builder-tool-row > button {
    width: 100%;
    min-width: 0;
    min-height: 30px;
    justify-content: center;
    white-space: normal;
  }
  .vrgdg-builder-tool-hint {
    min-width: 0;
    font-size: 11px;
    line-height: 1.25;
    color: #a1a1aa;
    overflow-wrap: normal;
    word-break: normal;
  }
  @container (max-width: 360px) {
    .vrgdg-builder-tool-row {
      grid-template-columns: minmax(0, 1fr);
      align-items: stretch;
    }
    .vrgdg-builder-tool-hint {
      padding: 0 4px 2px;
    }
  }
`;
document.head.appendChild(builderResponsiveStyle);

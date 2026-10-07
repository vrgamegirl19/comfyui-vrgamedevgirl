const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { functionSource, readBuilderModule } = require('./builder_source.cjs');

const source = functionSource(readBuilderModule('timeline_tool_windows.mjs'), 'createTimelineToolWindows');
const availabilitySource = functionSource(readBuilderModule('timeline_tool_windows.mjs'), 'timelineDeleteAvailability');

class Element {
  constructor(label = '') {
    this.label = label;
    this.children = [];
    this.listeners = new Map();
    this.style = {
      set cssText(value) { this.display = /display:([a-z]+)/.exec(value)?.[1] || this.display; },
    };
    this.offsetWidth = label === 'window' ? 340 : 190;
    this.offsetHeight = label === 'window' ? 220 : 130;
  }
  append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  contains(target) { return this === target || this.children.some((child) => child.contains(target)); }
  getBoundingClientRect() { return { left: 100, top: 500 }; }
  get offsetLeft() { return Number.parseFloat(this.style.left) || 0; }
  get offsetTop() { return Number.parseFloat(this.style.top) || 0; }
  click() { this.listeners.get('click')?.({ target: this }); this.onclick?.({ target: this }); }
}

test('timeline tools open as a draggable window and bulk deletes stay behind one button', () => {
  const overlay = new Element('overlay');
  const toolButtons = Array.from({ length: 8 }, (_, index) => new Element(`tool-${index}`));
  const deleteButtons = ['images', 'videos', 'segments'].map((label) => new Element(label));
  const windowListeners = new Map();
  const context = vm.createContext({
    document: { createElement: () => new Element('window') },
    window: {
      innerWidth: 1200, innerHeight: 800,
      addEventListener(name, callback) { windowListeners.set(name, callback); },
      removeEventListener(name) { windowListeners.delete(name); },
    },
    makeButton: (label) => new Element(label),
  });
  vm.runInContext(`${source}\nglobalThis.create = createTimelineToolWindows;`, context);
  let availability = { images: true, videos: true, segments: true };
  const { toolsButton, deleteAllButton, refreshDeleteActions } = context.create({
    overlay, toolButtons, deleteButtons, getDeleteAvailability: () => availability,
  });
  const [toolsWindow, deleteMenu] = overlay.children;
  assert.deepEqual(toolsWindow.children[1].children, toolButtons);
  assert.deepEqual(deleteMenu.children, deleteButtons);
  assert.equal(toolsWindow.style.display, 'none');
  assert.equal(deleteAllButton.style.display, 'none');
  refreshDeleteActions();
  assert.equal(deleteAllButton.style.display, '');
  toolsButton.click();
  assert.equal(toolsWindow.style.display, 'block');
  const header = toolsWindow.children[0];
  const startLeft = toolsWindow.offsetLeft;
  header.listeners.get('pointerdown')({ button: 0, target: header.children[0], clientX: 100, clientY: 100, preventDefault() {} });
  windowListeners.get('pointermove')({ clientX: 145, clientY: 120 });
  windowListeners.get('pointerup')();
  assert.equal(toolsWindow.offsetLeft, startLeft + 45);
  header.children[1].click();
  assert.equal(toolsWindow.style.display, 'none');
  deleteAllButton.click();
  assert.equal(deleteMenu.style.display, 'flex');
  deleteButtons[0].click();
  assert.equal(deleteMenu.style.display, 'none');
  availability = { images: false, videos: false, segments: true };
  refreshDeleteActions();
  assert.deepEqual(deleteButtons.map((button) => button.style.display), ['none', 'none', '']);
  deleteAllButton.click();
  assert.equal(deleteMenu.style.display, 'flex');
  availability = { images: false, videos: false, segments: false };
  refreshDeleteActions();
  assert.equal(deleteMenu.style.display, 'none');
  assert.equal(deleteAllButton.style.display, 'none');
});

test('bulk-delete availability follows assigned images, videos, and segments', () => {
  const context = vm.createContext({});
  vm.runInContext(`${availabilitySource}\nglobalThis.availability = timelineDeleteAvailability;`, context);
  const availability = (segments) => ({ ...context.availability(segments) });
  assert.deepEqual(availability([]), { images: false, videos: false, segments: false });
  assert.deepEqual(availability([{}]), { images: false, videos: false, segments: true });
  assert.deepEqual(availability([{ first_last_frame_end_image_data: 'data:image/png;base64,abc' }]),
    { images: true, videos: false, segments: true });
  assert.deepEqual(availability([{ image_history: [' '] }, { video_backup_paths: ['saved.mp4'] }]),
    { images: false, videos: true, segments: true });
  assert.deepEqual(availability([{ custom_image_path: 'still.png', video_path: 'clip.mp4' }]),
    { images: true, videos: true, segments: true });
});

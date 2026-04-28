## 2025-04-24 - Custom Element Focus Rings Match Global Border Radii

**Learning:** When a global `:focus-visible` pseudo-class defines a specific `border-radius`, custom interactive elements (like tabs, pills, and toggles) that declare their own `border-radius` must explicitly declare their own `:focus-visible` styles with a matching `border-radius`. Failing to do so causes the global focus ring to inherit the custom element's shape but clip or mismatch the global outline definition, leading to a broken or inconsistent visual focus state for keyboard navigators.

**Action:** Always explicitly define `:focus-visible` on custom interactive components (especially those modifying `border-radius` or using `border: none`/`outline: none`) to ensure the focus outline explicitly matches the shape of the custom component and provides an accessible, clear focus indicator.

## 2025-04-20 - Restoring Accessibility Focus Rings Overridden by Specific CSS
**Learning:** Type-specific input resets (e.g., `input[type="range"] { outline: none; }`) inherently possess higher CSS specificity than global pseudo-class selectors like `:focus-visible`. This causes interactive elements like range sliders to lose their accessibility focus rings entirely when navigating via keyboard, making the UI inaccessible to screen reader and keyboard users.
**Action:** Always explicitly re-declare `:focus-visible` styles on elements that have their outlines removed or modified via highly specific selectors (e.g., `input[type="range"]:focus-visible { outline: 2px solid var(--accent); }`) to ensure keyboard focus remains visible.

## 2025-04-20 - ARIA Tabbed Navigation
**Learning:** Native button-based tab groups in vanilla HTML lack intrinsic semantics for screen readers to understand them as an interconnected tablist. Users relying on assistive technology won't know how many tabs exist or which panel they correspond to.
**Action:** When building or updating tabbed navigation in vanilla HTML, always implement the standard ARIA tab pattern: assign `role="tablist"` to the container, `role="tab"` to the buttons along with dynamic `aria-selected` tracking, and `role="tabpanel"` to the respective panels. Cross-reference them explicitly using `aria-controls` on the tabs and `aria-labelledby` on the panels.

## 2026-04-28 - ARIA Patterns for Tabbed Navigation
**Learning:** Tabbed interfaces in vanilla HTML require explicit ARIA roles (tablist, tab, tabpanel), relationships (aria-controls, aria-labelledby), and dynamic state management (aria-selected) to be accessible. Visual indicators like an 'active' class are not communicated to screen readers.
**Action:** Always implement the full ARIA tab pattern, linking tabs and panels explicitly and updating the aria-selected attribute symmetrically when the active tab changes.

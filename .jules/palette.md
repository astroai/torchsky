## 2025-04-24 - Custom Element Focus Rings Match Global Border Radii

**Learning:** When a global `:focus-visible` pseudo-class defines a specific `border-radius`, custom interactive elements (like tabs, pills, and toggles) that declare their own `border-radius` must explicitly declare their own `:focus-visible` styles with a matching `border-radius`. Failing to do so causes the global focus ring to inherit the custom element's shape but clip or mismatch the global outline definition, leading to a broken or inconsistent visual focus state for keyboard navigators.

**Action:** Always explicitly define `:focus-visible` on custom interactive components (especially those modifying `border-radius` or using `border: none`/`outline: none`) to ensure the focus outline explicitly matches the shape of the custom component and provides an accessible, clear focus indicator.
## 2026-04-28 - ARIA Patterns for Tabbed Navigation
**Learning:** Tabbed interfaces in vanilla HTML require explicit ARIA roles (tablist, tab, tabpanel), relationships (aria-controls, aria-labelledby), and dynamic state management (aria-selected) to be accessible. Visual indicators like an 'active' class are not communicated to screen readers.
**Action:** Always implement the full ARIA tab pattern, linking tabs and panels explicitly and updating the aria-selected attribute symmetrically when the active tab changes.

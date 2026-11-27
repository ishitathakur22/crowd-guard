"""Define exit_capacity_per_sec, which the discharge code already references."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")

if "self.exit_capacity_per_sec" in text and "self.exit_capacity_per_sec = " not in text:
    anchor = "        self.turnover_scale = 1.0"
    n = text.count(anchor)
    print(f"anchor occurrences: {n}")
    if n:
        text = text.replace(
            anchor,
            anchor
            + "\n\n"
            "        # Discharge capacity of one exit cell, people per second.\n"
            "        #\n"
            "        # The continuum model used to do `rho[exit_mask] = 0.0`, removing\n"
            "        # everyone who reached an exit no matter how many arrived at once. A\n"
            "        # door is not infinitely wide: the SFPE/Fruin specific flow through a\n"
            "        # 1 m opening is about 1.3 people per second. Without this limit the\n"
            "        # forecast drains a crowded venue faster than reality, which is the\n"
            "        # measured failure - at the hotspot scenario it loses 476-826 people\n"
            "        # the truth retains.\n"
            "        #\n"
            "        # Surplus now queues at the threshold rather than disappearing, which\n"
            "        # is what a real crowd does and what makes a bottleneck a bottleneck.\n"
            "        self.exit_capacity_per_sec = 1.3",
            1,
        )
        path.write_text(text, encoding="utf-8")
        print("attribute defined")
else:
    print("nothing to do")

print(f"  'self.exit_capacity_per_sec =' present: "
      f"{'self.exit_capacity_per_sec = ' in text}")

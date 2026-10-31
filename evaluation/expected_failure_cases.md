# Expected failure cases

These cases are already expected from the Check-in 1 baseline trials and from
the literature. They will seed the discussion section once the HPC grids exist.
They are not scored results.

1. Ignored thin strokes. The pretrained scribble ControlNet often drops weak or
   broken freehand lines that a Canny map would have kept.
2. Extra objects invented to satisfy the prompt. Classifier-free guidance can
   add furniture, people, or background detail that the sketch never drew.
3. Copied strokes that hurt realism after fine-tuning. On a small hand-drawn
   set, LoRA can overfit and leave sketch-like marks in the photo.

Side-by-side grids should keep at least one example of each failure once the
baseline and adapter outputs are available. No fabricated images are stored here.

# Introduction and motivation (draft)

Turning a rough sketch into a photorealistic image is useful for early design,
concept art, and tools that help people who can draw the idea but cannot
describe it well in words. Pretrained ControlNet models are strongest on clean
edge maps. Real freehand drawings have broken strokes, uneven pressure, and
missing detail. This project studies how far a Stable Diffusion 1.5 plus
ControlNet pipeline can move from synthetic edges to that kind of drawing when
only a small set of real sketches is available for adaptation.

The baseline is Stable Diffusion 1.5 with the public canny and scribble
ControlNet checkpoints. We keep Stable Diffusion frozen and train a rank-8
LoRA adapter on the ControlNet attention projections using 50 to 100
hand-drawn sketch, photo, and prompt triples. The same sketches are then used
to compare the pretrained scribble baseline with the adapted model. We ablate
the ControlNet conditioning scale and the classifier-free guidance scale
because both change the balance between sketch fidelity and photorealism.

Quality is measured three ways. FID compares a larger COCO-based generation
set with held-out COCO photographs. CLIP score measures agreement with the
text prompt. A pairwise human survey asks which photo follows the sketch more
closely. Early baseline trials already showed that clean synthetic edges are
easier for the pretrained model than incomplete hand drawings. Closing that
gap, without inventing objects that the sketch never drew, is the claim this
project will test once the HPC runs and survey votes are finished.

This section will expand with the related-work citations and the final claim
after the Check-in 4 measurements are recorded.

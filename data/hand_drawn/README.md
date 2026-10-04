# Hand-drawn check-in 1 set

Each example uses the same filename stem in three folders:

- `images/` is the photo target
- `sketches/` is the hand-drawn sketch
- `prompts/` is one text prompt, saved as `<stem>.txt`

`src/train_controlnet_lora.py` skips a sketch unless both the matching photo and the prompt file exist. The check-in set is the 50 to 100 hand-drawn sketches from the proposal.

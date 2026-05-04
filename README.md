# deep-learning
Repository for Deep Learning Methods course.

## Minimal web UI

This workspace now includes a small Gradio app for multi-label image prediction.

### Run

1. Train your model in the notebook and save a checkpoint as a PyTorch state dict.
2. Set the checkpoint path:
   - `MODEL_CKPT=/path/to/model.pth`
3. Start the UI:
   - `python app.py`

### Notes

- The model uses `sigmoid` outputs with a configurable threshold.
- Predictions are multi-label: multiple classes can be returned for one image.
- Class names currently used by the app are: `Tree`, `Tomato`, `Pizza`.


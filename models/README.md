# Local AI model weights

The app intentionally does not download models while it is running. Put these pretrained weights in a `models/` directory beside `main.py`:

- `yolo26m.pt` — general-purpose pretrained detector for common categories.
- `yolov8l-worldv2.pt` — larger YOLO-World open-vocabulary detector; accepts text category prompts.

To perform the one-time downloads through Ultralytics, install the dependencies and explicitly run:

```bash
python download_models.py
```

That script needs an internet connection only while fetching missing checkpoints. Afterward, the app loads the local files and runs inference on CPU. You can also obtain checkpoints from the official model pages:

- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/models/yolo-world/

Model size affects latency and RAM use. These are relatively substantial checkpoints; the first load can take time on CPU. Review the Ultralytics licensing terms before redistributing the model weights or packaging the application commercially.

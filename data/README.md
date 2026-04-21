# Data Directory

EuroSAT is downloaded automatically at runtime via `torchgeo` when you run `main.py` or `test_phase1.py`.

```
data/
├── eurosat/          ← downloaded by torchgeo (do not commit)
└── samples/          ← extracted 640×640 RGB PNGs (do not commit)
    ├── AnnualCrop_0.png
    ├── Forest_0.png
    ├── ...
    ├── before.png    ← AnnualCrop sample used for change detection
    └── after.png     ← Industrial sample used for change detection
```

Both `data/` directories are excluded from git via `.gitignore`.

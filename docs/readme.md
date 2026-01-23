# Model naming + file layout (Pilot Academy)

## Model name convention (used for configs, checkpoints, runs)
**Pattern**
```

<backbone>-<temporal>-<conditioning>-<task>[-<variant>]

```

### Tokens
- **backbone**: `pilotnet` | `vit` | `vit_tiny` | `vit_small`
- **temporal**: `sf` (single-frame) | `mf` (multi-frame window) | `seq` (sequence / temporal transformer)
- **conditioning**: `bc` (plain) | `navbc` (navigation-command conditioned)
- **task**: `steer` | `steer+vel` | `ctrl` | `wp` | `traj`
- **variant** (optional): `cached` | `aug` | `ablateX` | `debug`

### Examples
- `pilotnet-sf-bc-steer`
- `vit-sf-bc-steer`
- `vit-mf-bc-steer`
- `vit-sf-navbc-steer`
- `vit-mf-navbc-wp`
- `vit-mf-navbc-wp-cached`

---

## Where model code lives + how to name files
Store models in:
```

src/pilot_academy/models/

```

**File naming rule**
- one file per family: `<backbone>.py` or `<backbone>_<variant>.py`
- the exported builder matches the model name intent

### Suggested files
```

src/pilot_academy/models/
pilotnet.py            # PilotNet family (steer / ctrl heads)
vit.py                 # ViT single-frame
vit_mf.py              # ViT multi-frame/sequence encoder variants
heads.py               # shared heads: steer, steer+vel, wp, traj
registry.py            # model registry (name -> builder)

```

---

## Configs: where and what to store
Keep configs in:
```

configs/models/
configs/train/

```

### Model config (example: `configs/models/vit-mf-navbc-wp.yaml`)
- name: `vit-mf-navbc-wp`
- backbone: vit / vit_tiny / vit_small
- temporal: sf / mf / seq
- conditioning: bc / navbc
- task: wp / steer / steer+vel / traj
- init / architecture params:
  - `image_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`
  - `seq_len` (if mf/seq)
  - `nav_vocab_size` or `n_commands` (if navbc)
  - `dropout`, `mlp_ratio`, etc.

### Training config (example: `configs/train/default.yaml`)
- data params: `n_events`, `n_straight`, caching flags
- pipeline: batch size, shuffle buffer, image H/W/C
- optimizer/loss: lr, weight decay, loss type, metrics
- run params: `seed`, `steps_per_epoch`, `epochs`, output dir

---

## Architecture info: where to document it
Put a short architecture card per model name in:
```

docs/architectures/<model_name>.md

```

Recommended contents:
- inputs / outputs
- backbone + temporal encoding
- conditioning (nav commands) mechanism
- head definition (steer / wp / traj)
- parameter count + expected input shapes
- training notes (losses, normalization)

Example file:
```

docs/architectures/vit-mf-navbc-wp.md

````

---

## How to use models (registry pattern)
### 1) Register builders
In `src/pilot_academy/models/registry.py`:
- map model name → builder function
- builder consumes a parsed config dict

### 2) Build from config in training
Example usage:
```python
from pilot_academy.models.registry import build_model
from pilot_academy.config import load_yaml

cfg = load_yaml("configs/models/vit-mf-navbc-wp.yaml")
model = build_model(cfg["name"], cfg)
````

### 3) Save outputs with the model name

Use the model name as the run folder:

```
artifacts/checkpoints/<model_name>/<run_stamp>/
artifacts/logs/<model_name>/<run_stamp>/
```

---

## Minimal checklist

* Model name follows: `backbone-temporal-conditioning-task[-variant]`
* Code lives in: `src/pilot_academy/models/`
* Model config lives in: `configs/models/<model_name>.yaml`
* Training config lives in: `configs/train/*.yaml`
* Architecture notes live in: `docs/architectures/<model_name>.md`


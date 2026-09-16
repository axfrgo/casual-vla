# Fortifiers Aegis — Competition Submission and Reproduction Record

**Author:** Alexander Ferguson  
**Roles:** researcher, scientist, developer  
**Handle:** `axfrgo`

This document is the submission-facing record for the dual-arm dinner-table
system. It describes what is implemented, what is measured, what is still
open, and how a reviewer can reproduce each result. It deliberately separates
browser visualization, native MuJoCo physics, neural policy inference,
OpenVINO inference, and physical Intel hardware evidence.

## Executive status

| Evidence item | Current result | Boundary |
| --- | --- | --- |
| Native MuJoCo deterministic reference | **10/10** on seeds `1001`–`1010` | Real MuJoCo task gates; not neural or hardware evidence |
| Direct camera/state/language joint policy, v7 | **0/10** on seeds `1001`–`1010` | Real neural checkpoint; failed at `retrieve_fork` |
| Direct camera/state/language joint policy, fresh v9 | **0/10** on held-out seeds `4001`–`4010` | More native demonstrations did not fix out-of-distribution grasping |
| Neural-guided intent policy + measured controller | **10/10** on seeds `1001`–`1010` | Neural high-level intent plus verified low-level controller; not end-to-end neural joint control |
| OpenVINO compact-policy CPU benchmark | **2.39 ms mean / 3.18 ms p95 / 417.7 FPS** | Inference-only on the current Intel host; not Core Ultra |
| Intel Core Ultra Series 2/3 benchmark | **Open** | Requires the physical target machine |
| Required neural VLA video | **Open** | Existing MP4 is native MuJoCo deterministic-baseline evidence |
| Cloud GPU training | **Completed** on one temporary NVIDIA L4 VM | Training acceleration/provenance; not Intel edge evidence |

The only defensible full-task score currently available is the deterministic
native MuJoCo reference and the separately labeled neural-guided system. The
direct joint-target policy is a genuine neural artifact, but its measured
score is `0/10`; it must not be presented as a competition pass.

## Solution architecture

```text
natural-language command
          |
          v
three MuJoCo RGB cameras + 21-value robot/task state
          |
          +--> compact camera/state/language BC policy ------+
          |                                                  |
          +--> neural task-intent gate ---------------------+--> 13-channel action contract
                                                             |
                                     fail-closed schema/range checks
                                                             |
                                                             v
                                           native MuJoCo SO-101 actuators
                                                             |
                                measured contact, lift, retention, release, handoff
                                                             |
                                                             v
                                                      task success gates
```

The native environment is authoritative for object attachment, contact,
placement, hand-off, and final completion. A predicted token, a browser
animation, or a low inference latency number cannot advance the task by itself.
Every accepted action is checked against the 13-channel actuator contract and
the environment verifies the physical consequence in MuJoCo.

### Stack mapping

| Layer | Selected technology | Role in this repository | Status |
| --- | --- | --- | --- |
| Scene interchange/authoring | OpenUSD | Recommended authoring/interchange layer for a richer asset pipeline; USD is not currently the authoritative runtime scene | Architecture option, not claimed evidence |
| Physics scene | MuJoCo 3.13.0 + MJCF | Authoritative dual-SO-101 dinner-table physics, cameras, contacts, perturbations, task gates | Implemented |
| Robot asset | Official SO-101 MJCF/URDF/STL assets | Namespaced left and right arm instances | Implemented |
| Motion planning boundary | MoveIt 2 + OMPL | Recommended hardware-side planner and collision-checking boundary; not silently substituted into MuJoCo scores | Integration plan |
| Data format | LeRobotDataset plus a transparent NumPy staging manifest | Reproducible RGB/state/action collection and training hand-off | Implemented |
| Training | PyTorch compact BC and intent heads | Reproducible first neural policy path; current direct policy is a research baseline, not a foundation VLA | Implemented |
| High-capacity VLA | LeRobot `SmolVLAAdapter` boundary | Permitted checkpoint can be connected without changing the MuJoCo action contract | Adapter boundary; checkpoint/run open |
| Edge deployment | OpenVINO static FP32 IR | Export, native task evaluation, and inference benchmark for the compact policy | Implemented on current CPU |
| Target hardware | Intel Core Ultra Series 3 preferred; Series 2 fallback | Final physical benchmark target with CPU/GPU/NPU device evidence | Hardware run open |
| Training accelerator | Temporary GCP NVIDIA L4 | Short-lived CUDA experiment only; VM is deleted after artifact retrieval | Completed and cleaned up after run |

The current compact network uses three RGB views, a 21-value state vector, a
17-token bag-of-words instruction vector, and a bounded 13-dimensional
relative actuator target. This is intentionally smaller and more reproducible
than a foundation VLA. The repository contains the adapter boundary for a
permitted SmolVLA checkpoint, but no pretrained foundation checkpoint is
invented or implied by the compact-policy evidence.

## MuJoCo task and perturbation contract

The authoritative runtime scene is
`sim/mujoco/models/official_dual_so101_dinner_table.xml`. It contains two
namespaced official SO-101 arms, a drawer and tray, fork, spoon, plate, cup,
placement zones, and camera views. The six task phases are:

1. open the drawer;
2. retrieve the fork from the drawer tray;
3. retrieve the spoon from the drawer tray;
4. place the plate;
5. hand the cup from the right arm to the left arm;
6. place the cup and verify the final table state.

The evaluation sweep deterministically varies placement, mass, friction,
object scale/shape, lighting, and background. The perturbation is seeded and
is included in every JSON report, so a result can be reproduced rather than
described qualitatively.

The environment uses measured fingertip contact and lift gates for grasp
acceptance, post-contact retention for long transports, explicit release
verification, and two-arm source/receiver contact for the cup hand-off. The
native baseline uses a controller with calibrated motion primitives; that is
why its result is a reference controller score and not a direct VLA score.

## Reproduction from a clean checkout

The commands below are written from the repository root. PowerShell is used on
Windows; the same commands work in a compatible shell after replacing path
separators.

### 1. Install and smoke-test MuJoCo

```powershell
uv sync --project sim/mujoco
uv lock --check --offline --project sim/mujoco
uv run --project sim/mujoco fortifiers-mujoco-eval --episodes 10 --output sim/mujoco/results/mujoco-smoke.json
```

### 2. Reproduce the deterministic native reference

```powershell
uv run --project sim/mujoco python -m fortifiers_mujoco.evaluate_policy `
  --episodes 10 --start-seed 1001 --max-actions 180 `
  --output sim/mujoco/results/mujoco-policy-eval.json
```

Expected result: `10/10`, seeds `1001`–`1010`, zero failed task actions, and
approximately 23–24 controller actions per episode. This is the reference
physics result, not a neural-policy or Intel-hardware result.

### 3. Collect demonstrations and export LeRobot data

```powershell
uv sync --project sim/mujoco --extra vla
uv run --project sim/mujoco fortifiers-mujoco-collect-vla `
  --episodes 10 --start-seed 3001 `
  --output tmp/vla-demonstrations-10-fresh
uv run --project sim/mujoco fortifiers-mujoco-export-lerobot `
  tmp/vla-demonstrations-10-fresh/manifest.json `
  --repo-id local/fortifiers-dinner-table-v9 `
  --root tmp/lerobot-dinner-table-v9
```

The fresh dataset used for the GPU run contains 10 expert episodes, 230 RGB /
state / action frames, three camera views, 21 state values, and 13 actuator
targets. The manifest records the instruction, camera names, state names,
actuator names, control ranges, perturbation metadata, and source episode
paths. Teacher demonstrations are training data, not independent success
evidence.

### 4. Train the compact neural policy

```powershell
uv run --project sim/mujoco --extra vla fortifiers-mujoco-train-imitation `
  tmp/vla-demonstrations-10-fresh/manifest.json `
  --output tmp/imitation-policy-v9.pt `
  --epochs 80 --augmentation-repeats 1 --seed 13 --device auto
```

`--device auto` selects CUDA when available and otherwise CPU. Use
`--device cuda` only after checking `torch.cuda.is_available()` on the target
machine. The checkpoint stores its training device, seed, split, loss, model
dimensions, language vocabulary, action mode, and source manifest.

Evaluate the actual checkpoint through native task gates:

```powershell
uv run --project sim/mujoco --extra vla fortifiers-mujoco-eval-imitation `
  tmp/imitation-policy-v9.pt --episodes 10 --start-seed 4001 `
  --max-actions 100 --patience-actions 24 --device cpu `
  --output tmp/imitation-evaluation-v9-heldout-10seed.json
```

Expected result for the checked-in fresh v9 experiment: `0/10`. Each episode
reached `retrieve_fork` and stopped after 40 actions for lack of task progress.
This is a valuable negative result: supervised frame loss and more teacher
episodes did not establish closed-loop grasp generalization.

### 5. Run the separately labeled neural-guided path

```powershell
uv run --project sim/mujoco --extra vla fortifiers-mujoco-train-neural-intent `
  tmp/vla-demonstrations-10-phase-onehot/manifest.json `
  --output tmp/neural-intent-policy.pt
uv run --project sim/mujoco --extra vla fortifiers-mujoco-eval-neural-guided `
  tmp/neural-intent-policy.pt --episodes 10 --start-seed 1001 `
  --confidence-threshold 0.5 `
  --output tmp/neural-guided-evaluation.json
```

Expected result: `10/10` with zero intent rejections. The report explicitly
attributes low-level motion to `closed-loop-dinner-policy-v1`. This proves a
neural high-level intent gate connected to a measured controller; it does not
prove that a neural joint-target policy solved the task.

### 6. Export and benchmark OpenVINO

```powershell
uv sync --project sim/mujoco --extra edge --extra vla
uv run --project sim/mujoco --extra edge --extra vla `
  fortifiers-mujoco-export-imitation-openvino `
  tmp/imitation-policy-v9.pt --output tmp/imitation-policy-v9.xml
uv run --project sim/mujoco --extra edge --extra vla `
  fortifiers-mujoco-eval-openvino tmp/imitation-policy-v9.xml `
  --episodes 1 --device CPU --output tmp/openvino-evaluation-v9-smoke.json
uv run --project sim/mujoco --extra edge --extra vla `
  fortifiers-mujoco-benchmark-openvino tmp/imitation-policy-v9.xml `
  --device CPU --output tmp/openvino-benchmark-v9.json
```

The benchmark reports model input precision, device, warm-up count, latency
percentiles, and throughput. It is inference-only. Task success comes only
from the separate OpenVINO-connected MuJoCo evaluation report.

### 7. Record the simulation video artifact

```powershell
uv sync --project sim/mujoco --extra video
uv run --project sim/mujoco --extra video fortifiers-mujoco-record-video `
  --output public/evidence/dinner-table-10-seed-demo.mp4 `
  --episodes 10 --start-seed 1001
```

The current MP4 is a native MuJoCo deterministic-baseline overview recording
with a JSON sidecar. It is not the required neural VLA video until the frame
source is changed to the selected neural checkpoint and the six-item video
checklist below is visibly covered.

## Cloud GPU provenance

The compact trainer was executed once on the existing `casualdat` Google Cloud
project using a temporary, labeled, Spot `g2-standard-8` VM in
`us-central1-a` with one NVIDIA L4. The CUDA 12.9 Ubuntu 24.04 image reported
NVIDIA driver `580.173.02` and 23,034 MiB of GPU memory. The job used the locked
repository dependencies, the 10-episode native dataset, fixed seed `13`, 80
epochs, one augmentation repeat, and `--device cuda`.

The VM was authorized for this one-shot experiment and is deleted after
artifact retrieval. The GPU result is reproducible as a training provenance
record, but it is not Core Ultra evidence and does not upgrade the policy's
task score. The retrieved checkpoint remains in the local ignored `tmp/`
directory; binary weights are not committed as public website evidence.

To repeat the experiment on a new authorized VM, capture all of the following
before cleanup:

```text
gcloud compute instances describe <instance> --project=casualdat --zone=us-central1-a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv pip freeze --project /home/alexj/aegis
<exact training command>
<checkpoint metadata JSON>
```

Never leave a temporary GPU VM running after the artifact and provenance have
been retrieved. Verify that the named instance is absent and that unrelated
project instances were not changed.

## Competition video checklist

The final video must make the following six facts visible, in order or with
clear overlays and timestamps:

| # | Required demonstration | Evidence rule | Current state |
| --- | --- | --- | --- |
| 1 | Natural-language command under a randomized initial scene | Show seed/perturbation and the exact instruction before motion | Browser/native baseline exists; neural video open |
| 2 | Perception/policy inference from simulated camera observations | Show live RGB observations and the selected policy/checkpoint | Direct neural inference exists; final selected VLA video open |
| 3 | Coordinated two-arm action with hand-off or complementary action | Both official SO-101 arms and source/receiver hand-off must be visible | Native baseline verified; neural-selected run open |
| 4 | Completed dinner-table state | Show final placements, held-object state, and task completion | Native baseline verified; neural-selected run open |
| 5 | Ten randomized seeds and success summary | Show seeds, per-seed result, aggregate success rate | Native baseline and guided report exist; direct policy is 0/10 |
| 6 | Core Ultra Series 2/3 OpenVINO benchmark | Show physical machine, device selection, precision, latency, throughput | Open until run on actual Core Ultra hardware |

The video must not splice a browser animation into a native MuJoCo claim or
show a deterministic controller trace while labeling it as VLA inference.
Overlays should name the active model, policy mode, seed, device, and evidence
boundary.

## Intel hardware recommendation and mapping

For the final physical evidence, use an Intel Core Ultra Series 3 system with
NPU and integrated Arc graphics as the preferred target, and keep a Core Ultra
Series 2 system as the compatibility fallback. Series 3 is the stronger target
for a new benchmark because current OpenVINO release notes explicitly include
Core Ultra Series 3 CPU scheduling optimization; Series 2 remains a valid
OpenVINO NPU target. The final report must still state the exact SKU, RAM,
driver/runtime versions, selected device (`CPU`, `GPU`, or `NPU`), precision,
latency, throughput, and thermal/power policy.

Recommended mapping:

| Work | Core Ultra target |
| --- | --- |
| USD/MJCF asset authoring and MuJoCo scene validation | CPU; GPU may accelerate visualization |
| PyTorch/LeRobot fine-tuning of compact or small VLA | GPU when available; cloud GPU is acceptable for development |
| Camera preprocessing and policy runtime | CPU or integrated GPU depending measured path |
| Quantized edge VLA inference | NPU first when the model is supported, then GPU/CPU fallback |
| OpenVINO benchmark | Run the same exported IR on each selected device and report each separately |
| Hardware motion planning | MoveIt 2 + OMPL with the calibrated SO-101 model and collision scene |

OpenUSD is useful as the asset composition/interchange layer; MJCF remains the
runtime contract for this MuJoCo evaluation. MoveIt/OMPL is a hardware-side
planning integration and must not be credited for native MuJoCo policy success
unless the same policy and action trace are actually connected.

Official references for this mapping are Intel's [Robotics AI
Suite](https://builders.intel.com/intel-technologies/software/edge-ai-suites/robotics-ai-suite),
the [OpenVINO NPU documentation](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html),
the [OpenVINO release notes](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html),
the [LeRobot imitation-learning guide](https://huggingface.co/docs/lerobot/il_robots),
the [OpenUSD introduction](https://openusd.org/24.08/intro.html),
[MuJoCo's XML reference](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html),
[OMPL](https://ompl.kavrakilab.org/), and
[MoveIt motion-planning concepts](https://github.com/moveit/moveit2_tutorials/blob/main/doc/concepts/motion_planning.rst).

## Remaining work to claim a full competition pass

1. Select or fine-tune a permitted high-capacity VLA, preferably with temporal
   action chunks and more varied corrective demonstrations, then connect that
   checkpoint through `SmolVLAAdapter` or the permitted official interface.
2. Make the direct policy solve the native grasp/transport loop without the
   measured controller silently taking over. The current direct score is
   `0/10`; this is the central neural gap.
3. Generate the required video from that selected neural checkpoint and cover
   all six checklist items with visible evidence overlays.
4. Run the same OpenVINO IR on a real Core Ultra Series 2 or 3 system and
   preserve the raw benchmark JSON plus machine/runtime provenance.
5. Bind and calibrate the official challenge observation/action/reset schema,
   if it differs from this repository's MuJoCo contract.

Until those five items are complete, the correct submission language is
“native MuJoCo reference plus measured neural-guided system, with direct VLA
and Core Ultra evidence still in progress,” not “fully validated physical VLA.”

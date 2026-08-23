# LeWM-JAX + GCIQL-Chunk

## Training

LeWM-JAX and GCIQL-Chunk are trained from the same offline task data. The independent policy can be trained in parallel with LeWM. Representation-sharing policies require a pretrained LeWM checkpoint, which is loaded once and frozen throughout policy training.

| Mode | Q input | V input | Policy input | Trainable representation |
|---|---|---|---|---|
| `independent` | independent pixel encoder | independent pixel encoder | independent pixel encoder | Q, V, and policy encoders |
| `pi` | independent pixel encoder | independent pixel encoder | frozen LeWM latent | Q and V encoders |
| `qv` | frozen LeWM latent | frozen LeWM latent | independent pixel encoder | policy encoder |
| `all` | frozen LeWM latent | frozen LeWM latent | frozen LeWM latent | none; heads only |

Q, V, and policy downstream heads are always separate. In `pi`, `qv`, and `all`, the LeWM encoder, projector, predictor, and batch statistics receive no gradients.

The controlled representation ablation uses `p_aug=0.0`. If augmentation is enabled, one shared crop is applied before both pixel and LeWM encoding so the representation mode is the only architectural difference.

## Execution

The main guided controller uses the deterministic GCIQL-Chunk proposal as the initial mean of the first CEM action block. LeWM then optimizes the action sequence with min-over-horizon latent goal distance. The canonical setting is CEM 300×5, horizon 5, receding horizon 1, and action block 5.

Evaluation reports four modes:

- `policy`: execute GCIQL-Chunk directly;
- `lewm`: LeWM-CEM without a policy proposal;
- `guided`: deterministic policy initialization followed by LeWM-CEM;
- `native_q`: sample policy-supported chunks, select one through the policy's native Q interface, then run LeWM-CEM.

LeWM-4Tasks and OGBench-Env-8Tasks keep separate evaluation entrypoints because their dataset-goal protocol, environment reset API, and action scaling differ.

For `qv` and `all`, native-Q evaluation checks that the policy and planner name the same normalized LeWM checkpoint path. This is a lightweight configuration check; checkpoint contents are not hashed.

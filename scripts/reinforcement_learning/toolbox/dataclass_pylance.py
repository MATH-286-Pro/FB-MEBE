import dataclasses


@dataclasses.dataclass
class AGENT_CFG:
    compile:    bool = False
    cudagraphs: bool = False

    train: "TRAIN_CFG" = dataclasses.field(default_factory=lambda: TRAIN_CFG())
    model: "MODEL_CFG" = dataclasses.field(default_factory=lambda: MODEL_CFG())

@dataclasses.dataclass
class TRAIN_CFG:
    discount: float = 0.98

    batch_size:          int = 1024
    use_mix_rollout:     bool = True
    update_z_every_step: int = 150

    weight_decay:    float = 0.0
    clip_grad_norm:  float | None = None

    fb_target_tau:     float = 0.01
    critic_target_tau: float = 0.005

    fb_pessimism_penalty:     float = 0.0
    actor_pessimism_penalty:  float = 0.5
    critic_pessimism_penalty: float = 0.5

    z_buffer_size: int = 10_000

    stddev_clip:             float = 0.3

    lr_f:      float = 1e-4
    lr_b:      float = 1e-5
    lr_actor:  float = 1e-4
    lr_critic: float = 1e-4

    ortho_coef:       float = 100
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    relabel_ratio:    float = 0.8        #ff0000??? 这是什么
    reg_coeff:        float = 0.01
    q_loss_coef:      float = 0.1

    # discriminator
    grad_penalty_discriminator: float = 10
    weight_decay_discriminator: float = 0.0
    lr_discriminator:           float = 1e-5

    # critic
    critic_coef: float = 0.003

@dataclasses.dataclass
class MODEL_CFG:
    device: str = "cpu"

    policy_dim:     int = 0
    obs_dim:        int = 0
    goal_dim:       int = 0
    critic_dim:     int = 0
    action_dim:     int = 0

    norm_obs: bool = True
    momentum: float = 0.0005

    inference_batch_size: int = 500_000

    actor_std: float = 0.2

    # FB-CPR
    seq_length: int = 8

    archi: "ARCHI" = dataclasses.field(default_factory=lambda: ARCHI())

@dataclasses.dataclass
class ARCHI:
    z_dim:  int = 256
    norm_z: bool = True

    f:      "F_ARCHI" = dataclasses.field(default_factory=lambda: F_ARCHI())
    b:      "B_ARCHI" = dataclasses.field(default_factory=lambda: B_ARCHI())
    actor:  "ACTOR_ARCHI" = dataclasses.field(default_factory=lambda: ACTOR_ARCHI())
    critic: "CRITIC_ARCHI" = dataclasses.field(default_factory=lambda: CRITIC_ARCHI())
    discriminator: "DISCRIMINATOR_ARCHI" = dataclasses.field(default_factory=lambda: DISCRIMINATOR_ARCHI())

@dataclasses.dataclass
class F_ARCHI:
    model: str = "simple"
    hidden_dim:    int = 1024
    hidden_layers: int = 2
    embedding_layers: int = 2
    num_parallel:     int = 2
    ensemble_mode:    str = "batch"

@dataclasses.dataclass
class B_ARCHI:
    norm:           bool = True
    hidden_dim:    int = 256
    hidden_layers: int = 1

@dataclasses.dataclass
class ACTOR_ARCHI:
    model: str = "simple"
    hidden_dim:    int = 1024
    hidden_layers: int = 2

    embedding_layers: int = 2

@dataclasses.dataclass
class CRITIC_ARCHI:
    enable: bool = False
    model: str = "simple"
    hidden_dim:    int = 1024
    hidden_layers: int = 2

    embedding_layers: int = 2
    num_parallel:     int = 2
    ensemble_mode:    str = "batch"
    
@dataclasses.dataclass
class DISCRIMINATOR_ARCHI:
    hidden_layers: int = 3
    hidden_dim:    int = 1024
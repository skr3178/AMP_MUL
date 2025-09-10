# AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

## English | 中文

### Environment Setup | 安装环境

**English:**
Ensure the runtime environment is Linux operating system with Python 3.8 programming environment; and ensure there is a connected display (or use a virtual display with 3D acceleration) for rendering.

First, download the physics simulation environment Isaac Gym Preview 4 [click here](https://developer.nvidia.com/isaac-gym), follow the instructions to install it, then ensure the example programs can run normally: `python/examples`, such as `joint_monkey.py`.

Then, install the following required packages:

**中文:**
确保运行软件环境为 Linux 操作系统、Python 3.8 编程环境；并确保有插入显示器（或使用具有3D加速的虚拟显示器），以便渲染。

首先下载物理仿真环境 Isaac Gym Preview 4 [点击此处](https://developer.nvidia.com/isaac-gym)，按照其中的说明安装好，然后确保实例程序能够正常运行：`python/examples` ，比如 `joint_monkey.py`。

然后，安装如下所需的包：

```
"gym==0.23.1",
"torch",
"omegaconf",
"termcolor",
"jinja2",
"hydra-core>=1.2",
"rl-games>=1.6.0",
"pyvirtualdisplay",
"urdfpy==0.0.22",
"pysdf==0.1.9",
"warp-lang==0.10.1",
"trimesh==3.23.5",
```

### Training Models | 训练模型

**English:**
For training the original model, set the HumanoidAMPPO.yaml configuration to `reward_combine: 'add'`, then use the following command:

```bash
python launch.py task=HumanoidAMP headless=True
```

After training is complete, test the training results as follows:

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

For the improved model, set the HumanoidAMPPO.yaml configuration to `reward_combine: 'mul'`, then use the following command:

```bash
python launch.py task=HumanoidAMP headless=True
```

After training is complete, test the training results as follows:

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

**中文:**
对于训练原始的模型，设置 HumanoidAMPPO.yaml 配置为 `reward_combine: 'add' ` ，然后使用下面的命令行：

```bash
python launch.py task=HumanoidAMP headless=True
```

在训练好以后，测试训练的结果如下：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

对于改进后的模型，设置 HumanoidAMPPO.yaml 配置为 `reward_combine: 'mul' ` ，然后使用下面的命令行：

```bash
python launch.py task=HumanoidAMP headless=True
```

在训练好以后，测试训练的结果如下：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

### Rendering Results to Video | 渲染结果到视频

**English:**
To render the trained model results to video, you can use the following command:

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn capture_video=True
```

The rendered video will be saved to the current working directory.

**中文:**
将训练好的模型结果渲染到视频，可以使用如下命令行：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn capture_video=True
```

所渲染的视频会保存到当前工作目录下。

### Converting to Physics-Based Motion Capture Files | 转换为基于动力学的动捕文件

**English:**
You can refer to the following function (for more details, see the various modules in the `fmbvh` library):

`./isaacgymenvs/tasks/amp/utils_amp/motion_lib.export_bvh`

When running the above training, testing, and other commands, motion capture files will be automatically exported to the `./runs/` directory.

**中文:**
可以参考如下函数（其余详见库 `fmbvh` 中的各个模块）：

`./isaacgymenvs/tasks/amp/utils_amp/motion_lib.export_bvh`

在运行上面的训练、测试等指令，都会在 `./runs/` 目录下自动导出动捕文件。

# add this to the conda variable everytime before running the script

export LD_LIBRARY_PATH=/home/skr/miniconda3/envs/amp_env/lib:$LD_LIBRARY_PATH

There are .npy and .npz file types.

## The Issue with .npy Files for Animation

The .npy file contains a different data structure than what the motion viewer expects:

### What the .npy file contains:

* rotation: Quaternion rotations for each body part (shape: 29 frames × 15 bodies × 4 quaternion components)
* root_translation: Root position over time (shape: 29 frames × 3 coordinates)
* global_velocity: Global velocities (shape: 29 frames × 15 bodies × 3 velocity components)
* global_angular_velocity: Global angular velocities (shape: 29 frames × 15 bodies × 3 angular velocity components)
* skeleton_tree: Skeleton hierarchy information
* fps: Frame rate (16.0)

### What the motion viewer expects (from .npz files):

* dof_positions: Joint angles/DOF positions
* dof_velocities: Joint velocities
* body_positions: Body positions in 3D space
* body_rotations: Body rotations as quaternions
* body_linear_velocities: Body linear velocities
* body_angular_velocities: Body angular velocities
* dof_names: Names of the degrees of freedom
* body_names: Names of the body parts
* fps: Frame rate

#### Network Architecture

* Actor-Critic Network: Standard PPO-based policy and value networks
* Discriminator Network: Distinguishes between agent-generated motion and reference motion data
* AMP Observation Space: Specialized observations including:
* Root height, rotation, velocity, angular velocity
* Joint positions and velocities
* Key body positions (relative to root)

!pseudocode.png(pseudocode.png)

Main algorithm utilized for training is AMP based 'amp_continuous.py'

which has

1. data collection part

   ```
   # From amp_continuous.py lines 131-200
   def play_steps(self):
       for n in range(self.horizon_length):
           # Collect observations and actions
           obs, rewards, dones, infos = self.env_step(res_dict['actions'])
           # Store AMP observations for discriminator training
           self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
   ```
2. reward combined part

   ```
   # From amp_continuous.py lines 610-621
   def _combine_rewards(self, task_rewards, amp_rewards):
       disc_r = amp_rewards['disc_rewards']
       if self._reward_combine == 'add':
           combined_rewards = self._task_reward_w * task_rewards + \
                            + self._disc_reward_w * disc_r
       elif self._reward_combine == 'mul':
           combined_rewards = self._task_reward_w * task_rewards * \
                            + self._disc_reward_w * disc_r
   ```
3. Loss total

   ```
   # From amp_continuous.py lines 337-338
   loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + \
          self.bounds_loss_coef * b_loss + self._disc_coef * disc_loss
   ```
4. * from config.yaml:
   * disc_coef: 5 - Discriminator loss weight
   * disc_reward_scale: 2 - Discriminator reward scaling
   * amp_batch_size: 512 - AMP observation batch size
   * reward_combine: 'mul' - Reward combination method (multiplicative vs additive)
   * task_reward_w: 1.0 - Task reward weight
   * disc_reward_w: 1.0 - Discriminator reward weight
5. Discriminator loss:
6. def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):

   Binary cross-entropy loss for agent vs demo classification:
   disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
   disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
   disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)
   Regularization terms
   disc_loss += self._disc_logit_reg * disc_logit_loss
   disc_loss += self._disc_grad_penalty * disc_grad_penalty
   disc_loss += self._disc_weight_decay * disc_weight_decay

Key components and entry point:

Entry point:

```
# From launch.py lines 80-89
import isaacgym
from isaacgymenvs.train import launch_rlg_hydra
launch_rlg_hydra()
```

Env Creation:

```
# From isaacgymenvs/train.py lines 115-139
def create_isaacgym_env(**kwargs):
    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        **kwargs,
    )
```

Physics Sim:

```

# From isaacgymenvs/tasks/humanoid_amp.py lines 57-69
def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    # Isaac Gym initialization
    super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, 
                     graphics_device_id=graphics_device_id, headless=headless, 
                     virtual_screen_capture=virtual_screen_capture, force_render=force_render)
```

Env Config:

```
# From HumanoidAMP.yaml lines 47-67
sim:
  dt: 0.0166 # 1/60 s
  substeps: 2
  up_axis: "z"
  use_gpu_pipeline: ${eq:${...pipeline},"gpu"}
  gravity: [0.0, 0.0, -9.81]
  physx:
    num_threads: ${....num_threads}
    solver_type: ${....solver_type}
    use_gpu: ${contains:"cuda",${....sim_device}}
    num_position_iterations: 4
    num_velocity_iterations: 0
    contact_offset: 0.02
    rest_offset: 0.0
    bounce_threshold_velocity: 0.2
    max_depenetration_velocity: 10.0
```

The fmbvh folder is only used for post-processing and visualization purposes, specifically.

1. Motion Export (Post-Training):

* The export_bvh() function in motion_lib.py uses fmbvh to convert trained motion data into BVH files
* This happens after training is complete, when you want to export the learned motions
* The exported BVH files are saved to the ./runs/ directory

Visualization Scripts:

* render_bvh_files.py - Renders BVH files to images
* simple_render.py - Simple rendering utilities
* These are standalone scripts for visualizing motion data

Network structure

Policy Network (Actor):
State → [1024] → ReLU → [512] → ReLU → Action_μ, Action_σ

Critic Network:
State → [1024] → ReLU → [512] → ReLU → Value

Discriminator Network:
AMP_Obs → [1024] → ReLU → [512] → ReLU → Linear → Logit

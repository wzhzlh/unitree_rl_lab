# Unitree RL Lab 完整开发指南

## 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [验证安装](#验证安装)
4. [系统架构](#系统架构)
5. [核心模块详解](#核心模块详解)
6. [强化学习训练流程](#强化学习训练流程)
7. [强化学习部署流程](#强化学习部署流程)
8. [如何添加新机器人](#如何添加新机器人)
9. [如何添加新策略](#如何添加新策略)
10. [配置文件详解](#配置文件详解)
11. [API参考](#api参考)
12. [开发最佳实践](#开发最佳实践)
13. [常见问题](#常见问题)

---

## 项目概述

Unitree RL Lab 是一个基于 Isaac Lab 构建的强化学习部署框架，用于将训练好的策略部署到 Unitree 机器人上。项目支持 Go2、H1、G1-29dof 等多款机器人。

### 技术栈

| 组件 | 版本要求 |
|-----|---------|
| 仿真环境 | Isaac Sim 5.1.0 + Isaac Lab 2.3.0 |
| 推理引擎 | ONNX Runtime |
| 机器人通信 | Unitree SDK2 (DDS) |
| 编程语言 | C++17 / Python 3.10+ |
| 依赖库 | Eigen3, yaml-cpp, spdlog, Boost |

### 分支说明

| 分支名 | Isaac Lab | Isaac Sim | 说明 |
|-------|-----------|-----------|------|
| `main` | 2.3.0 | 5.1.0 | 主分支（推荐） |
| `isaaclab3.0` | 3.0 | - | 实验分支 |
| `lab21` | 2.1 | 4.5.0 | 旧版本兼容 |
| `sim2sim` | - | - | Sim2Sim 仿真 |

### 项目结构

```
unitree_rl_lab/
├── deploy/                          # 部署代码
│   ├── include/                     # 公共头文件
│   │   ├── FSM/                     # 有限状态机
│   │   │   ├── BaseState.h          # 状态基类
│   │   │   ├── CtrlFSM.h            # FSM控制器
│   │   │   ├── FSMState.h           # FSM状态基类
│   │   │   ├── State_Passive.h      # 被动状态
│   │   │   ├── State_FixStand.h     # 站立状态
│   │   │   └── State_RLBase.h       # RL策略状态
│   │   ├── isaaclab/                # Isaac Lab部署模块
│   │   │   ├── algorithms/          # 算法模块
│   │   │   ├── assets/              # 资产模块
│   │   │   ├── devices/             # 设备模块
│   │   │   ├── envs/                # 环境模块
│   │   │   │   ├── mdp/             # MDP组件
│   │   │   │   │   ├── actions/     # 动作空间
│   │   │   │   │   └── observations/# 观测空间
│   │   │   │   └── manager_based_rl_env.h
│   │   │   ├── manager/             # 管理器
│   │   │   └── utils/               # 工具函数
│   │   ├── param.h                  # 参数解析
│   │   └── unitree_articulation.h   # 机器人关节接口
│   └── robots/                      # 机器人特定实现
│       ├── g1_29dof/                # G1 29自由度机器人
│       ├── go2/                     # Go2 四足机器人
│       ├── h1/                      # H1 人形机器人
│       └── ...
├── source/                          # 源代码
│   └── unitree_rl_lab/              # Python包
│       ├── assets/                  # 资产文件
│       ├── tasks/                   # 任务定义
│       └── config/                  # 配置文件
└── README.md
```

---

## 环境配置

### 硬件要求

#### RTX 5060 规格

| 项目 | 规格 |
|-----|------|
| 显存 | 8GB GDDR7 |
| 架构 | Blackwell |
| CUDA版本 | 12.8+ |
| Tensor Cores | 第5代 |

#### 实测配置 (RTX 5060 Laptop)

| 项目 | 实测值 |
|-----|--------|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 显存 | 8151 MiB |
| 驱动版本 | 580.126.09 |
| Python | 3.11.15 |

#### 推荐配置

- **CPU**: 8核以上 (推荐 Intel i7/i9 或 AMD Ryzen 7/9)
- **内存**: 32GB RAM (最低 16GB)
- **存储**: 100GB+ SSD (NVMe推荐)
- **GPU**: NVIDIA RTX 5060 (8GB VRAM)

### 系统要求

#### 操作系统

- **推荐**: Ubuntu 22.04 LTS
- **支持**: Ubuntu 20.04 LTS / Windows 10/11 (WSL2)

#### 驱动要求

```bash
# 检查NVIDIA驱动版本
nvidia-smi

# 要求驱动版本 >= 570.00 (Blackwell架构支持)
```

### 安装步骤

#### 步骤1: 安装NVIDIA驱动

```bash
# 添加NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 安装推荐驱动 (Blackwell架构需要最新驱动)
sudo apt install nvidia-driver-575

# 重启系统
sudo reboot

# 验证安装
nvidia-smi
```

#### 步骤2: 安装CUDA Toolkit 12.8

```bash
# 下载CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装CUDA Toolkit
sudo apt install cuda-12-8 cuda-toolkit-12-8

# 配置环境变量
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

#### 步骤3: 安装Miniconda

```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载shell
source ~/.bashrc
```

#### 步骤4: 创建Conda虚拟环境

```bash
# 创建Python 3.11环境 (项目要求 >= 3.10)
conda create -n isaaclab python=3.11 -y

# 激活环境
conda activate isaaclab

# 验证Python版本
python --version  # 应显示 Python 3.11.x
```

> **注意**: 始终确保在执行任何操作前已激活 `isaaclab` 环境：
> ```bash
> conda activate isaaclab
> ```

#### 步骤5: 安装Isaac Sim 5.1.0

Isaac Sim 是 Isaac Lab 的基础，需要先安装。

##### 方法A: 通过Omniverse Launcher安装 (推荐)

```bash
# 1. 下载NVIDIA Omniverse Launcher
cd ~/Downloads
wget https://install.omniverse.nvidia.com/launcher/linux-x86_64/omniverse-launcher-linux.AppImage

# 2. 添加执行权限
chmod +x omniverse-launcher-linux.AppImage

# 3. 启动Launcher
./omniverse-launcher-linux.AppImage

# 4. 在Launcher中安装:
#    - 登录NVIDIA账号
#    - 进入 "Exchange" 标签
#    - 搜索 "Isaac Sim"
#    - 安装 Isaac Sim 5.1.0
#    - 缓存位置建议选择SSD

# 5. 配置环境变量 (安装完成后)
echo 'export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-5.1.0' >> ~/.bashrc
source ~/.bashrc
```

##### 方法B: 通过pip安装 (适用于服务器)

```bash
# 激活conda环境
conda activate isaaclab

# 安装Isaac Sim Python包
pip install isaacsim --extra-index-url https://pypi.nvidia.com
```

#### 步骤6: 安装PyTorch

在安装 Isaac Lab 之前，需要先安装 PyTorch。由于官方源下载可能超时，建议使用国内镜像。

```bash
# 激活conda环境
conda activate isaaclab

# 方法1: 使用清华镜像安装 (国内推荐)
pip install torch==2.7.0 torchvision==0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法2: 使用官方源安装 (延长超时时间)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --timeout 300

# 安装缺失的依赖
pip install pyyaml jinja2 typeguard

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

> **常见问题**: 如果安装过程中出现 `ReadTimeoutError`，请使用清华镜像或延长超时时间。

#### 步骤7: 安装Isaac Lab 2.3.0+

```bash
# 克隆Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 确保conda环境激活
conda activate isaaclab

# 安装Isaac Lab
./isaaclab.sh -i

# 验证安装
./isaaclab.sh -p -c "import isaaclab; print(isaaclab.__version__)"
```

> **注意**: 当前最新版本为 Isaac Lab 0.54.3

#### 步骤8: 安装Unitree RL Lab

```bash
# 克隆项目
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab

# 确保conda环境激活
conda activate isaaclab

# 安装项目
./unitree_rl_lab.sh -i

# 安装git-lfs (用于下载大文件)
sudo apt install -y git-lfs
git lfs install

# 重启shell以激活环境变量
source ~/.bashrc
conda activate isaaclab
```

#### 步骤9: 下载机器人模型文件

```bash
# 方法1: 使用URDF文件 (推荐)
git clone https://github.com/unitreerobotics/unitree_ros.git

# 配置路径
# 编辑 source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
# 设置 UNITREE_ROS_DIR = "/path/to/unitree_ros/unitree_ros"

# 方法2: 使用USD文件
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model

# 配置路径
# 设置 UNITREE_MODEL_DIR = "/path/to/unitree_model"
```

### 版本兼容性矩阵

| 组件 | 推荐版本 | 最低版本 | 实测版本 |
|-----|---------|---------|---------|
| Ubuntu | 22.04 LTS | 20.04 LTS | 22.04 LTS |
| NVIDIA Driver | 575.x | 570.x | 580.126.09 |
| CUDA | 12.8 | 12.4 | 12.8 (驱动支持) |
| Python | 3.11 | 3.10 | 3.11.15 |
| PyTorch | 2.7.0+cu128 | 2.3.0 | 2.7.0+cu128 |
| Isaac Sim | 5.1.0 | 5.0.0 | - |
| Isaac Lab | 2.3.0 | 2.3.0 | 0.54.3 |
| Unitree RL Lab | 0.2.1 | - | 0.2.1 |
| RSL-RL | 2.3.1 | 2.2.0 | - |

---

## 验证安装

### 1. 检查CUDA和GPU

```bash
nvidia-smi
# 应显示 RTX 5060 和 CUDA 12.8

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. 检查已安装包

```bash
conda activate isaaclab
pip list | grep -E "torch|isaac|gym|numpy"
```

预期输出:
```
gymnasium                 1.2.1
isaaclab                  0.54.3
isaaclab_assets           0.2.4
isaaclab_mimic            1.0.16
isaaclab_rl               0.5.0
isaaclab_tasks            0.11.14
numpy                     1.26.4
torch                     2.7.0+cu128
torchvision               0.22.0+cu128
```

### 3. 列出可用任务

```bash
conda activate isaaclab
cd unitree_rl_lab
./unitree_rl_lab.sh -l
```

预期输出:
```
+----+-------------------------------+-----------------------------------------------+
| S. | Task Name                     | Entry Point                                   |
+----+-------------------------------+-----------------------------------------------+
| 1  | Unitree-G1-29dof-Velocity     | unitree_rl_lab.tasks.locomotion.robots.g1_29 |
| 2  | Unitree-Go2-Velocity          | unitree_rl_lab.tasks.locomotion.robots.go2   |
| 3  | Unitree-H1-Velocity           | unitree_rl_lab.tasks.locomotion.robots.h1    |
| ...                               |                                               |
+----+-------------------------------+-----------------------------------------------+
```

### 4. 运行测试任务

```bash
# 测试G1机器人 (无头模式)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless --num_envs 64

# 如果成功，将开始训练并输出日志
```

### 5. 可视化测试

```bash
# 带GUI运行
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 16
```

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Application                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      CtrlFSM (状态机)                        ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    ││
│  │  │ Passive  │  │ FixStand │  │ Velocity │  │  Mimic   │    ││
│  │  │ (被动)   │→ │ (站立)   │→ │ (RL控制) │  │ (模仿)   │    ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              ManagerBasedRLEnv (强化学习环境)                ││
│  │  ┌──────────────────┐    ┌──────────────────┐              ││
│  │  │ ObservationManager│    │  ActionManager   │              ││
│  │  │  (观测管理器)     │    │  (动作管理器)    │              ││
│  │  └──────────────────┘    └──────────────────┘              ││
│  │              ↓                    ↓                         ││
│  │  ┌──────────────────────────────────────────────────────┐   ││
│  │  │              OrtRunner (ONNX推理引擎)                 │   ││
│  │  │         policy.onnx → 神经网络策略推理                │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Articulation (机器人关节接口)                  ││
│  │  ┌──────────────────────────────────────────────────────┐   ││
│  │  │              Unitree SDK2 (DDS通信)                   │   ││
│  │  │         LowCmd (命令) ←→ LowState (状态)              │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
传感器数据 → LowState → Articulation.update() → 观测计算
                                                      ↓
                                              ObservationManager
                                                      ↓
                                              OrtRunner.act()
                                                      ↓
                                              ActionManager
                                                      ↓
                                              LowCmd → 电机控制
```

---

## 核心模块详解

### 1. 有限状态机 (FSM)

#### BaseState - 状态基类

```cpp
class BaseState
{
public:
    BaseState(int state, std::string state_string);
    
    // 生命周期方法
    virtual void enter() {}      // 进入状态时调用
    virtual void pre_run() {}    // 每次运行前
    virtual void run() {}        // 主运行逻辑
    virtual void post_run() {}   // 每次运行后
    virtual void exit() {}       // 退出状态时调用
    
    // 状态检查注册
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
};
```

**状态注册宏**:
```cpp
#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) { \
        return std::make_shared<Derived>(s, ss); \
    } \
    inline struct __registrar_##Derived { \
        __registrar_##Derived() { \
            getFsmMap()[#Derived] = __factory_##Derived; \
        } \
    } __registrar_instance_##Derived;
```

#### CtrlFSM - 状态机控制器

```cpp
class CtrlFSM
{
public:
    CtrlFSM(YAML::Node cfg);  // 从配置文件初始化
    void start();             // 启动状态机
    
private:
    void run_();              // 主循环 (1kHz)
    std::shared_ptr<BaseState> currentState;
    std::vector<std::shared_ptr<BaseState>> states;
};
```

**状态转换机制**:
- 每个状态注册 `registered_checks` 条件检查
- 条件满足时自动切换到目标状态
- 支持手柄按键触发转换

### 2. 强化学习环境

#### ManagerBasedRLEnv

```cpp
class ManagerBasedRLEnv
{
public:
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_);
    
    void reset();  // 重置环境
    void step();   // 执行一步
    
    // 核心组件
    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;  // ONNX推理引擎
    
    float step_dt;  // 控制周期 (通常20ms)
};
```

**step() 执行流程**:
```cpp
void step() {
    episode_length += 1;
    robot->update();                    // 更新机器人状态
    auto obs = observation_manager->compute();  // 计算观测
    auto action = alg->act(obs);        // 神经网络推理
    action_manager->process_action(action);     // 处理动作
}
```

### 3. 观测管理器 (ObservationManager)

#### 观测项配置结构

```cpp
struct ObservationTermCfg
{
    YAML::Node params;         // 参数
    ObsFunc func;              // 计算函数
    std::vector<float> clip;   // 裁剪范围
    std::vector<float> scale;  // 缩放因子
    int history_length = 1;    // 历史帧数
    bool scale_first = false;  // 是否先缩放后裁剪
};
```

#### 注册自定义观测

```cpp
#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env, YAML::Node params); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env, YAML::Node params)
```

**示例 - 基座角速度观测**:
```cpp
REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}
```

### 4. 动作管理器 (ActionManager)

#### 动作项基类

```cpp
class ActionTerm 
{
public:
    virtual int action_dim() = 0;                          // 动作维度
    virtual std::vector<float> raw_actions() = 0;          // 原始动作
    virtual std::vector<float> processed_actions() = 0;    // 处理后动作
    virtual void process_actions(std::vector<float> actions) = 0;
    virtual void reset() {};
};
```

#### JointPositionAction - 关节位置动作

```cpp
class JointPositionAction : public JointAction
{
    // 处理流程:
    // 1. raw_actions = actions (神经网络输出)
    // 2. processed = raw * scale + offset
    // 3. clamp(processed, clip[0], clip[1])
};
```

### 5. ONNX推理引擎

```cpp
class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path);
    
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        // 1. 创建输入张量
        // 2. 执行ONNX模型推理
        // 3. 返回动作输出
    }
};
```

---

## 强化学习训练流程

### 1. 训练策略

```bash
# 激活环境
conda activate isaaclab
cd unitree_rl_lab

# 训练G1机器人速度控制任务
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 512

# 无头模式训练 (推荐，性能更好)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless --num_envs 512

# 指定最大迭代次数
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless --num_envs 512 --max_iterations 10000
```

### 2. RTX 5060 显存优化配置

由于RTX 5060有8GB显存，需要合理配置并行环境数量:

```bash
# G1-29dof (人形机器人) - 显存占用较大
--num_envs 512   # 推荐 (约6GB显存)
--num_envs 1024  # 极限 (可能接近8GB上限)
--num_envs 256   # 保守 (约4GB显存)

# Go2 (四足机器人) - 显存占用较小
--num_envs 1024  # 推荐
--num_envs 2048  # 可尝试

# H1 (人形机器人)
--num_envs 512   # 推荐
```

### 3. PyTorch优化

在训练脚本开头添加:

```python
import torch

# 启用TF32加速 (Blackwell架构优化)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 禁用确定性模式以提升性能
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

### 4. 测试策略

```bash
# 测试训练好的策略
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity --num_envs 16
```

### 5. 导出ONNX模型

训练完成后，将模型导出为ONNX格式:
```
config/policy/velocity/v0/exported/policy.onnx
```

---

## 强化学习部署流程

### 1. 创建部署配置

在 `config/policy/velocity/v0/params/deploy.yaml` 中配置:

```yaml
# 关节映射
joint_ids_map: [0, 6, 12, 1, 7, 13, ...]

# 控制参数
step_dt: 0.02
stiffness: [100.0, 100.0, ...]
damping: [2.0, 2.0, ...]
default_joint_pos: [-0.1, -0.1, ...]

# 动作配置
actions:
  JointPositionAction:
    scale: [0.25, 0.25, ...]
    offset: [-0.1, -0.1, ...]

# 观测配置
observations:
  base_ang_vel:
    scale: [0.2, 0.2, 0.2]
    history_length: 5
  projected_gravity:
    scale: [1.0, 1.0, 1.0]
    history_length: 5
  # ...
```

### 2. Sim2Sim测试

```bash
# 启动Mujoco仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco

# 启动控制器
cd deploy/robots/g1_29dof/build
./g1_ctrl
```

### 3. Sim2Real部署

```bash
./g1_ctrl --network eth0
```

---

## 如何添加新机器人

### 步骤1: 创建机器人目录

```bash
mkdir -p deploy/robots/my_robot/{config,include,src}
```

### 步骤2: 定义类型 (include/Types.h)

```cpp
#pragma once
#include "unitree/dds_wrapper/robots/my_robot/my_robot.h"

using LowCmd_t = unitree::robot::my_robot::publisher::LowCmd;
using LowState_t = unitree::robot::my_robot::subscription::LowState;
```

### 步骤3: 创建配置文件 (config/config.yaml)

```yaml
FSM:
  _:
    Passive:
      id: 1
    FixStand:
      id: 2
    Velocity:
      id: 3
      type: RLBase

  Passive:
    transitions: 
      FixStand: LT + up.on_pressed
    mode: [1, 1, 1, ...]
    kd: [3, 3, 3, ...]

  FixStand:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity: RB + X.on_pressed
    kp: [100, 100, ...]
    kd: [2, 2, ...]
    ts: [0, 3]
    qs: [[], [q0, q1, ...]]

  Velocity:
    transitions: 
      Passive: LT + B.on_pressed
    policy_dir: config/policy/velocity
```

### 步骤4: 实现主程序 (main.cpp)

```cpp
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "Types.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);
    
    // 初始化DDS
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    
    // 初始化状态
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    FSMState::lowstate->wait_for_connection();
    
    // 启动FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();
    
    while (true) { sleep(1); }
    return 0;
}
```

### 步骤5: 实现RL状态 (src/State_RLBase.cpp)

```cpp
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

### 步骤6: 创建CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(my_robot_ctrl)

set(CMAKE_CXX_STANDARD 17)

find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(ONNXRuntime REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_executable(my_robot_ctrl
    main.cpp
    src/State_RLBase.cpp
)

target_link_libraries(my_robot_ctrl
    unitree_sdk2
    yaml-cpp
    Eigen3::Eigen
    spdlog::spdlog
    ONNXRuntime::ONNXRuntime
)
```

---

## 如何添加新策略

### 添加新的观测项

在 `deploy/include/isaaclab/envs/mdp/observations/observations.h` 中添加:

```cpp
REGISTER_OBSERVATION(my_custom_obs)
{
    // 计算自定义观测
    std::vector<float> obs;
    // ...
    return obs;
}
```

### 添加新的动作类型

在 `deploy/include/isaaclab/envs/mdp/actions/joint_actions.h` 中添加:

```cpp
class MyCustomAction : public ActionTerm
{
public:
    MyCustomAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : ActionTerm(cfg, env) {}
    
    int action_dim() override { return _action_dim; }
    std::vector<float> raw_actions() override { return _raw_actions; }
    std::vector<float> processed_actions() override { return _processed_actions; }
    void process_actions(std::vector<float> actions) override {
        // 自定义处理逻辑
    }
};

REGISTER_ACTION(MyCustomAction);
```

### 添加新的FSM状态

```cpp
// include/State_MyCustom.h
#pragma once
#include "FSM/FSMState.h"

class State_MyCustom : public FSMState
{
public:
    State_MyCustom(int state, std::string state_string = "MyCustom");
    
    void enter() override;
    void run() override;
    void exit() override;
};

REGISTER_FSM(State_MyCustom)
```

---

## 配置文件详解

### config.yaml 结构

```yaml
FSM:
  _:                           # 启用的状态列表
    Passive:                   # 状态名称
      id: 1                    # 状态ID
      type: RLBase             # 状态类型 (可选,默认为状态名)
  
  Passive:                     # Passive状态配置
    transitions:               # 状态转换条件
      FixStand: LT + up.on_pressed  # 目标状态: 触发条件
    mode: [1, 1, ...]          # 电机模式
    kd: [3, 3, ...]            # 阻尼增益
  
  Velocity:                    # RL状态配置
    transitions:
      Passive: LT + B.on_pressed
    policy_dir: config/policy/velocity  # 策略目录
```

### deploy.yaml 结构

```yaml
# 关节映射 (SDK顺序 → 策略顺序)
joint_ids_map: [0, 6, 12, ...]

# 控制参数
step_dt: 0.02                  # 控制周期 (秒)
stiffness: [100.0, ...]        # 关节刚度
damping: [2.0, ...]            # 关节阻尼
default_joint_pos: [-0.1, ...] # 默认关节位置

# 速度命令范围
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]
      lin_vel_y: [-0.3, 0.3]
      ang_vel_z: [-0.2, 0.2]

# 动作配置
actions:
  JointPositionAction:
    scale: [0.25, ...]         # 动作缩放
    offset: [-0.1, ...]        # 动作偏移
    clip: null                 # 动作裁剪

# 观测配置
observations:
  base_ang_vel:                # 观测名称
    params: {}                 # 计算参数
    scale: [0.2, 0.2, 0.2]     # 缩放因子
    clip: null                 # 裁剪范围
    history_length: 5          # 历史帧数
```

---

## API参考

### ArticulationData

```cpp
struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W;        // 世界坐标系重力向量
    Eigen::Vector3f FORWARD_VEC_B;        // 基座前向向量
    
    std::vector<float> joint_stiffness;   // 关节刚度
    std::vector<float> joint_damping;     // 关节阻尼
    Eigen::VectorXf joint_pos;            // 关节位置
    Eigen::VectorXf joint_vel;            // 关节速度
    Eigen::VectorXf default_joint_pos;    // 默认关节位置
    
    Eigen::Vector3f root_ang_vel_b;       // 基座角速度
    Eigen::Vector3f projected{}
_gravity_b;  // 投影重力
    Eigen::Quaternionf root_quat_w;       // 基座四元数
    
    std::vector<float> joint_ids_map;     // 关节ID映射
    UnitreeJoystick* joystick;            // 手柄接口
};
```

### 内置观测函数

| 观测名称 | 描述 | 维度 |
|---------|------|------|
| `base_ang_vel` | 基座角速度 | 3 |
| `projected_gravity` | 投影重力 | 3 |
| `joint_pos` | 关节位置 | N |
| `joint_pos_rel` | 相对关节位置 | N |
| `joint_vel_rel` | 相对关节速度 | N |
| `last_action` | 上一步动作 | N |
| `velocity_commands` | 速度命令 | 3 |
| `gait_phase` | 步态相位 | 2 |

### 内置动作类型

| 动作名称 | 描述 |
|---------|------|
| `JointPositionAction` | 关节位置控制 |
| `JointVelocityAction` | 关节速度控制 |

---

## 开发最佳实践

### 1. 代码风格

- 使用 C++17 标准
- 遵循 Google C++ 代码风格
- 使用 `spdlog` 进行日志记录
- 使用 `Eigen` 进行矩阵运算

### 2. 线程安全

```cpp
// 策略线程与主线程共享数据时使用互斥锁
std::lock_guard<std::mutex> lock(act_mtx_);
```

### 3. 错误处理

```cpp
// 使用spdlog记录错误
spdlog::error("Failed to load config: {}", e.what());

// 关键错误使用critical并退出
spdlog::critical("Unmatched robot type.");
exit(-1);
```

### 4. 性能优化

- 控制循环运行在独立线程,频率由 `step_dt` 控制
- 使用 `std::this_thread::sleep_until` 精确控制时序
- 避免在控制循环中进行内存分配

### 5. 调试技巧

```cpp
// 启用调试日志
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

// 记录日志到文件
./g1_ctrl --log
```

### 6. 安全检查

```cpp
// 检查机器人姿态
if (bad_orientation(env, 1.0)) {
    // 切换到Passive状态
}

// 检查通信超时
if (lowstate->isTimeout()) {
    // 切换到Passive状态
}
```

---

## 常见问题

### Q1: CUDA版本不匹配

```bash
# 检查CUDA版本
nvcc --version
nvidia-smi  # 右上角显示的CUDA Version是驱动支持的最高版本

# 如果版本不匹配，重新安装CUDA
sudo apt purge cuda-*
sudo apt install cuda-12-8
```

### Q2: 显存不足 (OOM)

```bash
# 降低并行环境数量
--num_envs 256  # 或更低

# 使用CPU卸载 (不推荐，会显著降低性能)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Q3: Isaac Sim启动失败

```bash
# 检查Omniverse缓存
rm -rf ~/.cache/ov
rm -rf ~/.local/share/ov/data/Kit/Isaac-Sim

# 重新启动
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

### Q4: 找不到机器人模型

```bash
# 检查路径配置
cat source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py

# 确保UNITREE_ROS_DIR或UNITREE_MODEL_DIR指向正确路径
```

### Q5: 训练速度慢

```bash
# 确保使用headless模式
--headless

# 增加环境数量 (在显存允许范围内)
--num_envs 1024

# 检查是否使用了GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Q6: Blackwell架构兼容性

RTX 5060使用Blackwell架构，需要确保:

```bash
# 1. 驱动版本 >= 570.00
nvidia-smi

# 2. CUDA >= 12.8
nvcc --version

# 3. PyTorch支持Blackwell
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu128
```

### Q7: PyTorch安装超时 (ReadTimeoutError)

安装PyTorch时出现 `ReadTimeoutError` 是常见问题，解决方案：

```bash
# 方法1: 使用国内镜像 (推荐)
pip install torch==2.7.0 torchvision==0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法2: 延长超时时间
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --timeout 300

# 方法3: 使用较低版本
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
```

### Q8: 缺少依赖包

如果提示缺少 `pyyaml`, `jinja2`, `typeguard` 等包：

```bash
pip install pyyaml jinja2 typeguard
```

### Q9: rl-games 安装失败

```bash
# 直接使用 pip 安装
pip install rl-games

# 或使用国内镜像
pip install rl-games -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q10: git lfs 未安装

```bash
# 安装 git-lfs
sudo apt install -y git-lfs
git lfs install

# 下载大文件
git lfs pull
```

---

## 快速启动脚本

为了方便快速启动训练，创建以下脚本。

### 创建 ~/train_go2.sh

```bash
#!/bin/bash
# GO2 机器人训练脚本 (RTX 5060 优化)

set -e  # 错误时退出

CONDA_ENV="isaaclab"
TASK="Unitree-Go2-Velocity"
NUM_ENVS=512
HEADLESS=true

echo "================================"
echo "Unitree RL Lab - GO2 训练脚本"
echo "================================"

# 1. 加载环境
echo "📦 加载环境..."
source ~/.bashrc
conda activate $CONDA_ENV

# 2. 检查环境变量
echo "🔍 检查环境变量..."
if [ -z "$EXP_PATH" ]; then
    echo "❌ EXP_PATH 未设置！"
    exit 1
fi
echo "   EXP_PATH: $EXP_PATH"

# 3. 进入项目目录
cd ~/unitree_rl_lab

# 4. 启动训练
echo "🚀 启动 GO2 训练..."
echo "   任务: $TASK"
echo "   环境数: $NUM_ENVS"

if [ "$HEADLESS" = true ]; then
    ./unitree_rl_lab.sh -t --task $TASK --headless --num_envs $NUM_ENVS
else
    ./unitree_rl_lab.sh -t --task $TASK --num_envs $NUM_ENVS
fi

echo ""
echo "✅ 训练启动完成！"
```

保存为 `~/train_go2.sh` 并添加执行权限：

```bash
chmod +x ~/train_go2.sh
```

使用方法：

```bash
# 启动GO2训练
~/train_go2.sh

# 或直接使用命令
cd ~/unitree_rl_lab
conda activate isaaclab
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless --num_envs 512
```

### 创建 ~/train_g1.sh

```bash
#!/bin/bash
# G1 机器人训练脚本 (RTX 5060 优化)

set -e

CONDA_ENV="isaaclab"
TASK="Unitree-G1-29dof-Velocity"
NUM_ENVS=512
HEADLESS=true

echo "================================"
echo "Unitree RL Lab - G1 29DOF 训练脚本"
echo "================================"

source ~/.bashrc
conda activate $CONDA_ENV

if [ -z "$EXP_PATH" ]; then
    echo "❌ EXP_PATH 未设置！"
    exit 1
fi

cd ~/unitree_rl_lab

echo "🚀 启动 G1-29DOF 训练..."
echo "   环境数: $NUM_ENVS"

if [ "$HEADLESS" = true ]; then
    ./unitree_rl_lab.sh -t --task $TASK --headless --num_envs $NUM_ENVS
else
    ./unitree_rl_lab.sh -t --task $TASK --num_envs $NUM_ENVS
fi

echo "✅ 训练启动完成！"
```

保存为 `~/train_g1.sh`：

```bash
chmod +x ~/train_g1.sh
~/train_g1.sh
```

### 环境变量配置检查清单

在启动训练前，确保以下检查全部通过：

```bash
conda activate isaaclab

python << 'EOF'
import os, sys

checklist = [
    ("EXP_PATH 已设置", bool(os.environ.get('EXP_PATH'))),
    ("KIT_PATH 已设置", bool(os.environ.get('KIT_PATH'))),
    ("isaacsim 可导入", False),
    ("isaaclab 可导入", False),
    ("torch 可导入", False),
    ("GPU 可用", False),
]

try:
    from isaacsim import SimulationApp
    checklist[2] = ("isaacsim 可导入", True)
except:
    pass

try:
    from isaaclab.app import AppLauncher
    checklist[3] = ("isaaclab 可导入", True)
except:
    pass

try:
    import torch
    checklist[4] = ("torch 可导入", True)
    checklist[5] = ("GPU 可用", torch.cuda.is_available())
except:
    pass

print("\n✅ 启动前检查清单:\n")
for item, status in checklist:
    symbol = "✓" if status else "✗"
    print(f"  [{symbol}] {item}")

all_ok = all(status for _, status in checklist)
print(f"\n{'✅ 可以启动训练！' if all_ok else '❌ 需要修复上述问题'}\n")
EOF
```

### RTX 5060 显存优化建议

| 任务 | 环境数 | 显存占用 | 说明 |
|-----|-------|--------|------|
| Go2-Velocity | 512 | ~5-6GB | 推荐 (轻量级) |
| Go2-Velocity | 1024 | ~7-8GB | 最大 |
| G1-29dof-Velocity | 512 | ~6-7GB | 推荐 (重量级) |
| G1-29dof-Velocity | 256 | ~4-5GB | 保守 |
| H1-Velocity | 512 | ~6-7GB | 推荐 |

如果出现 OOM (Out of Memory)，降低 `--num_envs`:

```bash
# 显存不足时
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless --num_envs 256
```

### 监控训练进度

在另一个终端运行：

```bash
# 方式1: 使用 TensorBoard
tensorboard --logdir ~/unitree_rl_lab/runs

# 方式2: 实时查看日志
tail -f ~/unitree_rl_lab/runs/*/events.out.tfevents.* | head -100
```

### 完整的环境配置总结

你的 `~/.bashrc` 应该包含以下内容（特别注意 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 的路径顺序和范围）：

```bash
# ===== CUDA 环境 =====
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# ===== Isaac Sim & Lab 环境配置 =====
# 1. 基础路径 (必须包含 ISAAC_PATH，供 Python 脚本定位包)
export ISAACSIM_PATH=~/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
export ISAAC_PATH=${ISAACSIM_PATH}   
export EXP_PATH=${ISAACSIM_PATH}
export KIT_PATH=${ISAACSIM_PATH}/kit
export CARB_APP_PATH=${ISAACSIM_PATH}/kit

# 2. PYTHONPATH (只放纯净的第三方库和项目库，绝对不要放 kit/python/lib 以免污染标准库产生 SRE 错误)
export PYTHONPATH=${ISAACSIM_PATH}/python_packages:$PYTHONPATH
export PYTHONPATH=~/IsaacLab/source/isaaclab:$PYTHONPATH
export PYTHONPATH=~/unitree_rl_lab/source/unitree_rl_lab:$PYTHONPATH

# 3. LD_LIBRARY_PATH (包含所有 kit、plugins 路径以支持 carb C++ 底层插件加载)
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/kernel/plugins:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/plugins:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/plugins/carb_gfx:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/plugins/rtx:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/plugins/gpu.foundation:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/kit/python/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ISAACSIM_PATH}/extscache/omni.physics.physx/lib:$LD_LIBRARY_PATH

export OMNI_TELEMETRY=0
export MESA_GL_VERSION_OVERRIDE=4.6
```

### 故障排除

| 问题 | 现象与原因 | 解决方案 |
|-----|----------|--------|
| `ModuleNotFoundError: isaacsim` | 终端未加载环境变量 | 重新加载环境: `source ~/.bashrc` |
| `TypeError: 'NoneType' object is not callable` | `SimulationApp` 加载失败。原因是缺少 `ISAAC_PATH`，或者底层 C++ 插件 carb 加载失败（缺少 `LD_LIBRARY_PATH`）。 | 确保 `~/.bashrc` 中有完整的 `LD_LIBRARY_PATH` 列表，并包含 `export ISAAC_PATH=${ISAACSIM_PATH}`。 |
| `AssertionError: SRE module mismatch` | `PYTHONPATH` 污染。加入 `kit/python/lib` 等内部库会导致 Conda 环境自身的 `_sre.so` C扩展与 Isaac Python3 标准库冲突。 | 从 `PYTHONPATH` 中移除所有 Isaac 的 Python 内置标准库路径，仅保留 `python_packages` 等第三方包路径。 |
| `KeyError: 'CARB_APP_PATH'` | Python 初始化时被跳过导致变量未注册 | 在 `~/.bashrc` 中手动加上 `export CARB_APP_PATH=${ISAACSIM_PATH}/kit` |
| `KeyError: 'class_name'` 或 `unexpected keyword argument 'stochastic'` | `rsl_rl` 版本不兼容（>= 4.0.0）。旧版中的 `RslRlPpoActorCriticCfg` 已经被弃用，其自带的 `stochastic`, `init_noise_std` 等参数在传给底层 `MLPModel` 字典时会崩溃。 | 1. 将配置改为独立的 `actor` 和 `critic`（使用 `RslRlMLPModelCfg`）；2. 在传给 `OnPolicyRunner` 时将所有的废弃参数从字典中 `.pop()` 清除掉。 |
| 想要运行带 GUI 的可视化，但不出画面 | 使用 `-t` 时 `unitree_rl_lab.sh` 脚本内硬编码了 `--headless`。 | 1. 观看推理使用 `./unitree_rl_lab.sh -p ...`<br> 2. 带GUI训练使用原生的 `python scripts/rsl_rl/train.py --task ...` (不用脚本包装)。 |
| GPU 显存不足 (OOM) | 环境过多或使用 GUI | 降低 `--num_envs`: (无头模式推荐 512, 带GUI推荐 16 或 64) |

---

## 参考链接

- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab)
- [Isaac Sim 下载](https://developer.nvidia.com/isaac-sim)
- [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Eigen 库](https://eigen.tuxfamily.org/)

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0.0 | 2025-01 | 初始版本 |
| 1.1.0 | 2025-04 | 支持 Isaac Sim 5.1.0 + Isaac Lab 2.3.0 |
| 1.2.0 | 2026-04 | 整合环境配置和开发指南，适配 RTX 5060 |

---

*本文档由 Unitree Robotics 维护*
*当前分支: main*
*文档更新日期: 2026-04-18*
*适用于: NVIDIA RTX 5060 Laptop (Blackwell架构)*
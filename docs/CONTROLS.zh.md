<!-- tracks: CONTROLS.md @ sha256:27b850f34f9e911d -->

# 操作方式 —— 手柄、键盘，以及由谁决定它们的含义

一个人怎样驾驶这台机器人：controller 从手柄和键盘读到什么，一个任务怎样说明它的摇杆和按键做什么，
一个 bundle 怎样说明哪个按键在模式之间切换。本文讲的是设计和背后的理由，并标出实现每一部分的文件。
具体操作步骤 —— 写一个 `controls.yaml`、加一个模式切换 —— 以
[`controls` skill](../.claude/skills/controls/SKILL.md) 为清单；bundle 的文件格式见
[`deploy/BUNDLE.md`](../deploy/BUNDLE.md)。

---

## 1. 五条设计原则

1. **词是固定的，含义不是。** 只有一个文件
   [`controller/vocabulary.json`](../controller/vocabulary.json) 列出手柄或键盘能说的一切 ——
   `A` 按下了、左摇杆在 (0.4, −0.9)、`key_w` 按下了 —— 而不涉及其中任何东西的含义。含义归使用输入的
   那一方，而且恰好在两个地方（第 2 条）。字典里没有的名字在任何地方都会在加载时被拒绝，因为一个
   编造的名字能解析、什么也不绑定，看上去和一个坏掉的按键一模一样。

2. **两个文件，加起来就是全部。** 一个任务在驾驶时摇杆和按键做什么，是这个任务的
   `controls.yaml`（§4）。哪个按键*切换进*某个模式，是 bundle 在 `deploy/manifests.json`
   里的条目（§5）。谁也不能写对方那一半：任务不点模式的名，manifest 不把摇杆绑到命令轴上。

3. **手柄和键盘是两条路。** 各自把自己的输入直接绑到它要做的事上 —— 命令轴的一个方向、任务自留的
   一个控制、一次模式切换、释放 —— 谁也不点对方控制的名字，所以两者可以有意地不一样：Space 在行走时是
   跳跃，在爪子模式里是爪子。该一致的地方之所以一致，是因为任务文件给两者列的是同一批轴，而键盘要是
   不绑某个轴，就必须写明原因。2026-09-29 之前，键盘是一个*虚拟手柄* —— 按住一个键就是推一根没人握着
   的手柄上的摇杆，再走真手柄的映射 —— 这让键盘的布局成了手柄布局的副本。而那不是一只放在键盘上的手
   想要的布局：机器人自己的操作指南 Control-agent 3.1 用 J 和 L 转向、用 H 和 ; 扭身，手柄却是在
   一根摇杆上做这两件事；它把 N 和 M 当作高、低两种站姿，而手柄是移动高度并停在那里；它还用 Shift、
   Alt 和 Ctrl 把爪子的机械臂伸出去。

4. **每个模式通过自己任务的操作方式读取操作者。** 一个 bundle 跑好几个策略，不同的策略用不同的操作
   方式：在 `jumper` bundle 里，右摇杆按下再往上推，在行走时抬高机身，在爪子模式里是低头；右扳机在
   爪子模式里合拢爪子，在行走时什么都不做；摇杆推满在一个模式里是 0.8 m/s，在另一个里是 0.5 m/s。
   controller 给每个模式一个自己的 operator，由这个模式的契约构建，从不强迫两个任务一致。

5. **一个按键同一时刻只有一个含义。** 一个既切换模式、又是当前模式自己操作的按键，按一下会做两件事。
   这类重叠一律拒绝；一个切换用 `from` 写明它能在哪些模式里被按 —— Space 就是这样在一个模式里是跳跃、
   在另一个模式里是爪子的。唯一允许的共用，是手柄上在按下那一刻就离开当前模式的组合键（§5.7）。另外，
   按住修饰键时，和它组合的那个控制只响应组合：Menu + 上是一支舞，绝不同时也是单按上的那个动作。

五条背后是这个仓库围绕的那种失败：**输入的 bug 是静默的。** 正负号反了，机器人就往后走，而每块屏幕
看起来都正常；一个什么都没绑的键，看上去就像一个没触发的键。所以每个值都注明来源，没有哪个正负号
有默认值，任何含糊的东西都在加载时被拒绝，而不是被猜。

---

## 2. 字典

```bash
python -m controller --vocabulary          # 在没有 Python 的板子上：controller --vocabulary
```

`controller/vocabulary.json`（`kk-control-vocabulary/2`）被三个程序读，没有一个写它：

| 读者 | 方式 |
|---|---|
| `controller/vocabulary.py` | `python -m controller --vocabulary` 打印它 |
| `deploy/fsm/src/vocabulary.rs` | 用 `include_str!` 编译进去，所以板子上不需要这个文件 |
| `tasks/jumper/common/mdp/controls.py` | 校验任务 `controls.yaml` 里的每一个名字 |

它列出的内容：

| 部分 | 名字 | 说明 |
|---|---|---|
| `buttons` | `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3` | 机器人的手柄服务发布这十个。`view` 列在 `absent` 下：手柄有这个键，服务不发布它。`X` 和 `Y` 标为*未确认* —— 服务和 `controller/xbox.py` 对哪个 evdev 编码是哪个键有分歧，还没人按一下去确认 |
| `dpad` | `dpad_up` `dpad_down` `dpad_left` `dpad_right` | 线上传的是取值 −1/0/+1 的 `dpad_x`/`dpad_y`，没有手势能读，所以拆成四个布尔量 |
| `axes` | `Lx` `Ly` `Rx` `Ry` `LT` `RT` | 摇杆在 [−1, 1]，用 evdev 的约定 —— **`Ly` 和 `Ry` 向下为正**，`Lx` 和 `Rx` 向右为正 —— 扳机在 [0, 1]。手柄服务施加 0.15 的死区**并重新缩放**，所以推满仍能到 ±1 |
| `keys` | 小键盘；主键盘区的 `key_1` 到 `key_4`；`key_w s a d i j k l m n o u q b g h`；`key_space`；四个方向键；左右 Ctrl、左右 Shift、左右 Alt | 每个都带 GLFW 编码（MuJoCo 窗口）和 `KeyboardEvent.code`（浏览器） |
| `modifiers` | `ctrl` `shift` `alt` | 每个都是它两个键中的任一个：左 Ctrl 或右 Ctrl 按着时，`ctrl` 就算按着 |
| `gestures` | `rise` `fall` `hold` `toggle` `single` `double` `triple` `quadruple` `quintuple` | 见下文"手势" |

**按键组合（keystroke）** 是键盘绑定写明按下什么的方式：一个键（`key_j`）、单独一个修饰键
（`ctrl`），或者按住修饰键再按一个键（`shift+key_j`，即按住 Shift 时的 J）。J 和 Shift + J 是两个按键
组合，永远不会同时起作用：修饰键按着时，和它有组合的那个键只响应组合，不响应它自己。

**为什么是这些键。** MuJoCo 的 viewer 在调用用户按键回调的*同时*也执行它自己的快捷键，而每个字母都是
快捷键 —— `W` 把场景切成线框，`G` 是雾，`H` 是凸包，Space 暂停。小键盘是它不占用的那一组。字母仍然在
字典里，是因为一个按手的习惯排布的键盘被认为值得这点闪烁；jumper 的各任务按 Control-agent 3.1 的方式
排布（2026-09-29）：`W A S D` 和方向键行走，`I K` 俯仰，`J L` 转向，Shift + `J L` 扭身，`U O` 横滚，
`N M` 调高度，`B` 全部松开；在爪子模式里 Space 合拢爪子，Shift、Alt 和 Ctrl 把机械臂伸出去。那天之前
`Q` 和 `O` 是两个扳机，`M` 是按下右摇杆；现在 `Q` 什么也没绑。Space、`G`、`H`、Ctrl 和数字键 `1` 到
`4` 切换 `jumper` bundle 的模式，每个都是一个独立的切换（§5.4）；数字键在主键盘区和小键盘上都一样。
主键盘区的 `1` 到 `4` 还会切换 MuJoCo 的 geom 分组；小键盘的不会。Alt 对浏览器有代价：单独按下再
松开时，Windows 会把焦点移到浏览器的菜单上，所以页面必须对它 `preventDefault`。

### 手势

一个用作开关的控制是一个手柄按钮或一个键，加上一个手势。四种读一次按下：

| 手势 | 何时触发 |
|---|---|
| `rise` | 按下去的那一 tick，一次 |
| `fall` | 弹回来的那一 tick，一次 |
| `hold` | 按住的每一 tick —— 死人开关，不是模式 |
| `toggle` | 从这次按下起一直开着，直到下一次按下 —— 进入模式的默认手势 |

五种数按下的次数 —— **连击**：

| 手势 | 何时触发 |
|---|---|
| `single` | 按一次，并且窗口内没有第二次 |
| `double` | 两次，每次都在上一次的窗口之内 |
| `triple` | 三次 |
| `quadruple` | 四次 |
| `quintuple` | 五次 |

一个控制可以用两种手势绑两次 —— 按下预备、松开出发，就是同一个按钮上的 `rise` 和 `fall`。
连击的规则见 §5.3。

**一个键的两个沿。** 浏览器会报告键的按下和松开。MuJoCo 的 viewer 只报告按下，所以
`rl/mjrl/viewer/keys.py` 在 viewer 前面装了一个自己的 GLFW 按键回调，把两个沿都交给每个读者，
并丢掉自动重复；在每一种宿主上，一个键从按下到松开都算按住。（2026-09-29 之前，松开是从永远不会
到来的按键重复里推断的，每个键都在按下 0.75 s 后自动松开。）

---

## 3. 输入怎样到达 controller

controller —— `deploy/fsm` 里的 Rust crate —— 在三种宿主上是同一份代码，每种宿主用自己的方式把输入
交给它。第四条路径 `play --task` 根本不跑 controller。

| 宿主 | 手柄 | 键盘 | 时钟 |
|---|---|---|---|
| **板子**（`runtime/board/controller`） | 通过 DDS 收手柄服务 `control-pod-svc` 发来的 `RobotControlRaw_ControlRaw`，约 50 Hz，domain 0 —— 原始手柄，不是别人解读过的（`deploy/fsm/src/dds.rs`） | 没有：手柄是唯一来源 | controller 自己的 |
| **浏览器**（`WebFsm`，`runtime/web/controller.wasm`） | Gamepad API 的原始值：`setAxis`、`setPad`，然后 `padFrame(nowUs)` | `KeyboardEvent.code` 经 `setKeyCode(code, down, repeat, nowUs)`；重复被忽略 | 页面的 `nowUs` |
| **`play --app`**（`runtime/mjlab/<platform>/controller.so`） | Xbox 手柄，Linux 上经 evdev、Windows 上经 XInput 读取（`controller/`），每一步用 `set_pad_frame` 送进去 | viewer 的按键经 `mjrl.viewer.keys`，`set_key(name, down, now_us)` | 仿真时间 |
| **`play --task`**（没有 controller） | 同一个读取器，送进 Python 的 operator | 同样的 viewer 按键，送进 Python 的 operator | 墙钟时间 |

三条在每种宿主上都成立：

- **正负号保持设备报告的样子。** `controller/xbox.py` 不翻转 `Ly` —— "那是偷偷塞进设备驱动里的机器人
  观点" —— 手柄服务也不翻转。从摇杆到命令的翻转就是任务 `controls.yaml` 里的 `sign`，别处都没有。
- **没有第二个死区。** 每个 `controls.yaml` 都写 `deadzone: device_reported_rescaled`：设备已经施加过
  一个死区并重新缩放到 ±1（机器人的服务是 0.15，台架上的 evdev 驱动是它自己的 `flat`，XInput 是微软
  文档给出的死区），使用方再加一个，会让人能要求的范围变窄，而且自己察觉不到。Rust 的 operator 拒绝
  任何其他取值。
- **操作者离开，一切都放掉。** 手柄输入旧于 `command_timeout_ms`（`jumper` bundle 里是 500 ms）时，
  所有锁存的开关被释放，每个模式的命令回到静止值，每个沿重新布防，于是级联回到默认模式；输入恢复时
  仍按着的按钮不会被当成一次新的按下。

`play --task` 是训练侧的回放，不是部署宿主。在那里，命令项被换成 operator 项
（`tasks/jumper/common/mdp/operator.py`），读同一个 `controls.yaml`，两个设备各走各的绑定。operator 在
被碰之前不起作用 —— 随机采样的命令照常通过 —— 按 `B` 把命令交还给采样器。训练本身从不读手柄：命令是
采样的，`controls.yaml` 只决定人怎样够到它们。

在 Windows 上，Python 侧经 XInput 读手柄（`controller/xinput.py`）：Xbox 手柄以及自称是 Xbox 手柄的
手柄，每次读数都换算成 Linux 的 `xpad` 驱动会给出的值，所以正负号和按键名与 Linux 相同。macOS 上
不读手柄，`play` 用键盘驾驶。`python -m controller` 不带参数时，会连续打印已连接手柄的轴和按钮，以及
它预期的正负号 —— 这是台架上检查一个还没在机器人上实测过的正负号的办法。

---

## 4. 任务的操作方式：`controls.yaml`

### 4.1 谁读它

```
tasks/<task>/controls.yaml
   │  load_controls()  (tasks/jumper/common/mdp/controls.py) —— 下面所有的拒绝
   ├──▶ play --task：Python 的 operator
   └──▶ scripts/export.py：契约里的 `controller` 块（layout.json，operator_controller/2）
            └──▶ 这个模式在 controller 里的 operator：板子上、浏览器里、play --app 里
```

`velocity_env_cfg` 以 `controls=` 接收这个文件，没有它会报错，除非任务传了 `operator_command=False`
（`jumper.swing` 就是这样；`jump`、`ref_free_jump`、各支舞和各个固定动作都不建在它上面，只靠模式切换
驾驶）。导出的块带着文件的 sha256 和 §4.5 的命令整形，契约里在它旁边还有策略训练时的命令范围，所以
机器人永远不需要这个文件本身。

### 4.2 格式

```yaml
schema_version: 2                 # 1 会被点名拒绝

command:                          # operator 驱动的每个命令项一条
  - term: twist                   # 它在 cfg.commands 里的键
    feeds: velocity_commands      # 它落进的 observation 项
    frame: body
    axes:                         # 按命令项写入的顺序
      - { name: lin_vel_x, unit: m/s, positive: forward }
      - ...
    scaling:
      rule: full_deflection_is_range_edge

task:                             # 可选：任务自己响应的控制，按名字
  <名字>: { kind: amount | press, does: <任务用它做什么> }

devices:
  gamepad:
    scheme: absolute              # 摇杆位置就是命令
    layout: xbox
    axes: { <轴>: <绑定> | [<绑定>, ...] }
    shift: { button: R3, gesture: hold }        # 可选；hold 或 toggle
    reset: R3                                   # 可选：点按一下，让每个被移动的轴回到静止值
    deadzone: device_reported_rescaled
    release_button: B
    task: { <名字>: <一个手柄控制> }             # `task:` 里有控制时，每一个都要有
  keyboard:
    scheme: keys
    full_after_s: 2.0             # 按住一个键多少秒到推满
    axes: { <轴>: { "+": [<按键组合>, ...], "-": [<按键组合>, ...] } | { unbound: <原因> } }
    release: [key_escape]         # 放掉一切的按键组合
    task: { <名字>: [<按键组合>, ...] }          # `task:` 里有控制时，每一个都要有
```

**缩放。** 推满就是策略**训练时**所用范围的边缘，两端分别缩放 —— 摇杆的意思是"这个策略被要求过的最快"，
不可能要求训练分布之外的东西。居中值不是零的轴，要点出保存这个值的配置字段 `rest: neutral_height`，
并从那里开始缩放：手离开手柄就是站立高度，而不是一个趴在地上的机身。

**一条手柄绑定**是 `{ source, sign, travel, shifted }`：

| 字段 | |
|---|---|
| `source` | `Lx` `Ly` `Rx` `Ry` `LT` `RT` |
| `sign` | `1` 或 `-1`，必填 —— 方向唯一被决定的地方 |
| `travel` | 可选，`[from, to]`：到 `from` 之前为零，到 `to` 为满。四个数 `[from, to, back, off]` 还会在 `back` 到 `off` 之间回落到零 |
| `shifted` | `true`：只在 `shift` 按钮打开的那一层时生效 |

一个轴上的一组绑定会相加。两条绑定不能在同一层里*爬升*过同一个控制的同一段行程；一段回落可以和下一条
绑定的爬升共用一段行程，这就是把一根摇杆从一个轴交给另一个轴的方式。

**键盘上的一个轴**，是把它往两个方向推的那些按键组合：`"+"` 是这个轴自己的正方向，就是它的
`positive:` 所说的那个 —— 前、左、逆时针、低头、升高 —— `"-"` 是反方向。所以键盘不带正负号：没有
摇杆可以让正负号相对于它。每个命令轴都在 `keyboard.axes` 里，要么有绑定，要么写 `{ unbound: <原因> }`：
一个悄悄缺掉的轴，就是一个做不到手柄能做的事、而且没有任何东西说出来的键盘。按住一个键，它的方向在
`full_after_s` 内线性推到满，键一抬起立即回到静止值；同一方向上的两个按键组合相加再钳位。

**一个轴可以被移动，而不是被摆到某处。** 轴上的 `integrate_s` 让偏转变成速度：推满时，这个轴在这么多
秒内从静止值走到范围的任一端，松手后它就停在那里。这是手柄的：在键盘上，同一个轴和键盘上所有的轴
一样是被摆到某处的。`reset` 点出一个手柄按钮，
点按它 —— 按下再松开，中间没有动过它可能要去够的摇杆 —— 就让每个这样的轴回到静止值；释放也会。

### 4.3 最简单的情况：`jumper.tripod`

四个步态任务带的是同一个文件，两根摇杆驱动一个速度命令：

| 命令轴 | 手柄 | 符号 | 键盘，`"+"` / `"-"` |
|---|---|---|---|
| `lin_vel_x`（向前为正） | `Ly` | −1 | `W` 或 ↑ / `S` 或 ↓ |
| `lin_vel_y`（向左为正） | `Lx` | −1 | `A` 或 ← / `D` 或 → |
| `ang_vel_z`（逆时针为正） | `Rx` | −1 | `J` / `L` |
| 交还命令 | `B`（`release_button`） | | `B`（`release`） |

```yaml
devices:
  gamepad:
    scheme: absolute
    layout: xbox
    axes:
      lin_vel_x: { source: Ly, sign: -1 }
      lin_vel_y: { source: Lx, sign: -1 }
      ang_vel_z: { source: Rx, sign: -1 }
    deadzone: device_reported_rescaled
    release_button: B
  keyboard:
    scheme: keys
    full_after_s: 2.0
    axes:
      lin_vel_x: { "+": [key_w, key_up], "-": [key_s, key_down] }
      lin_vel_y: { "+": [key_a, key_left], "-": [key_d, key_right] }
      ang_vel_z: { "+": [key_j], "-": [key_l] }
    release: [key_escape]
```

注意正负号在哪里。在手柄上，`sign: -1` 说的是摇杆的负端是前 —— 按 evdev 的约定，往上推就是 `Ly` 为负。
键盘上没有可以弄错的正负号：`W` 在 `"+"` 下面，是因为 `W` 是前，而前是 `lin_vel_x` 的正方向，轴自己
这么说。各任务的范围不同（tripod ±0.5 m/s 和 ±0.75 rad/s，flat ±0.8 和 ±1.0），文件却相同，因为推满
就是各任务自己的范围边缘。

### 4.4 两个命令、分段的摇杆、一个切换层和一个被移动的轴：`jumper.posture`

`jumper.posture` 驱动一个速度命令和一个姿态命令，共七个通道，任一设备都能驾驶：

| 做什么 | 手柄 | 键盘 |
|---|---|---|
| 前进 / 后退 | 左摇杆上 / 下 | `W` / `S`，或 ↑ / ↓ |
| 左移 / 右移 | 左摇杆左 / 右 | `A` / `D`，或 ← / → |
| 低头 / 抬头（`pitch`） | 右摇杆上 / 下 | `I` / `K` |
| 逆时针 / 顺时针扭身 | 右摇杆左 / 右，前半程 | Shift + `J` / Shift + `L` |
| 逆时针 / 顺时针转向 | 右摇杆左 / 右，后半程 | `J` / `L` |
| 横滚，左侧低 / 左侧高 | 按住 `R3`，右摇杆左 / 右 | `U` / `O` |
| 升高 / 降低 —— 是速度；松手就停在那里 | 按住 `R3`，右摇杆上 / 下 | `N` / `M` |
| 高度回到站立高度 | 点按 `R3` | `B`，连同其他一切 |
| 全部松开 | `B` | `B` |
| 站着不动、水平端正，停在最后设定的高度 | 手离开 | 松开 |

```yaml
    axes:
      lin_vel_x: { source: Ly, sign: -1 }
      lin_vel_y: { source: Lx, sign: -1 }
      twist:     { source: Rx, sign: -1, travel: [0.0, 0.5, 0.5, 0.75] }
      ang_vel_z: { source: Rx, sign: -1, travel: [0.5, 1.0] }
      pitch:     { source: Ry, sign: -1 }
      roll:      { source: Rx, sign: 1, shifted: true }
      height:    { source: Ry, sign: -1, shifted: true }
    shift: { button: R3, gesture: hold }
    reset: R3
  keyboard:
    axes:
      ang_vel_z: { "+": [key_j], "-": [key_l] }
      twist:     { "+": [key_h], "-": [key_semicolon] }
      pitch:     { "+": [key_i], "-": [key_k] }
      roll:      { "+": [key_o], "-": [key_u] }
      height:    { "+": [key_n], "-": [key_m] }
      # ……以及行走，同 §4.3
```

高度轴在 `rest: neutral_height` 旁边带着 `integrate_s: 1.0`：摇杆推满时一秒从站立走到任一端
（2026-09-29；两秒太慢）。

- **`travel` 把一个控制分给几个轴。** `twist` 在右摇杆行程的前一半爬升，在 0.5 到 0.75 之间回落到零，
  而 `ang_vel_z` 在后一半爬升：机身先扭向转弯方向，转得快之前又回正。
- **`shifted: true`** 把一条绑定放到按住 `R3` 时才生效的那一层上（`gesture: hold`；若是 `toggle`，
  则从一次点按持续到下一次）。生效时，右摇杆的左右行程让机身横滚，上下行程移动高度；扭身、转向和
  俯仰都为零。松开 `R3`，转向和俯仰立即恢复。
- **在手柄上，高度是被移动的，而不是被摆到某处的**（`integrate_s`，2026-09-29 提出）：推动摇杆，机身
  以一定速度升高或降低 —— 推满时一秒从站立高度走到范围的任一端 —— 松手后机身就停在那个高度。不碰摇杆、
  点按一下 `R3`，高度回到站立高度（`reset: R3`）；按住 `R3` 并推动摇杆是在调高度或横滚，那一次按下不算
  点按。手柄上的 `B` 和键盘上的 Esc 连同其他一切把它放回去。在那之前，高度是被摆到某处的：当天早上
  之前在扳机上，之后在 `R3` 和摇杆上，`R3` 一松开就回到站立高度。
- **在键盘上，高度是被摆到某处的**，照文档的写法：`N` 和 `M` 是高、低两种站姿 —— 按住时机身沿按键的
  斜坡走过去，松开就回到站立高度，就像 `U` 和 `O` 是两种横滚。键盘从不移动手柄调出的高度：键盘驾驶时，
  手柄留下的高度被保留，手柄一被碰就又是那个高度。
- **键盘没有切换层。** 横滚和高度有自己的键，扭身也有：`J`、`L` 旁边的 `H` 和 `;`。文档
  2026-09-29 修订之前，扭身是按住 Shift 时的 `J` 和 `L` —— 一个组合键，字典里仍写作 `shift+key_j`，
  修饰键按着时只有它响应。
- **列表把几个控制相加**：同一个轴上的两条手柄绑定相加后截断。现在没有任务用到；2026-09-29 之前
  `height` 上的两个扳机就是这样。

俯仰约定 —— **低头为正** —— 与 `jumper.five_foot` 共用，因为两者都写机器人唯一的俯仰通道；
2026-09-26 之前训练的 posture checkpoint 读到的俯仰是反的。

### 4.5 行走时命令能做什么：`bands` 和 `max_rate`

一个命令项可以带一个 `moving` 区间和一个 `stand_threshold` —— 速度命令要求移动时某个轴最多能到多远 ——
以及一个 `max_rate`。导出把它们写进契约，controller 的 `CommandShaper` 在每种宿主上、在每次
observation 之前应用它们，先钳位再限速：不管人怎么推摇杆，策略看到的都是它训练时见过的那种命令。
`jumper.posture` 行走时把姿态限制在 ±15°；`jumper.five_foot` 还以 30°/s 的速率让机身姿态缓变。

### 4.6 任务自留的控制：`jumper.five_foot`

`jumper.five_foot` 的行走和倾身与 `jumper.posture` 相同（没有高度），并自留了五个没有命令去读的控制，
在文件顶部声明一次，在每个设备上按名字绑定：

```yaml
task:
  claw_left:      { kind: amount, does: closes the left claw as far as it is held }
  claw_right:     { kind: amount, does: closes the right claw as far as it is held }
  arm_thumb_up:   { kind: press,  does: "holds the carried arm straight out, thumb up, while held" }
  arm_thumb_down: { kind: press,  does: "holds the carried arm straight out, thumb down, while held" }
  arm_web_up:     { kind: press,  does: "holds the carried arm straight out, thumb-web up, while held" }
  arm_to_face:    { kind: press,  does: "holds the carried arm bent in, the claw before the visor, while held" }

devices:
  gamepad:
    task: { claw_left: LT, claw_right: RT, arm_thumb_up: dpad_up,
            arm_thumb_down: dpad_down, arm_web_up: dpad_left, arm_to_face: dpad_right }
  keyboard:
    task: { claw_left: [key_space], claw_right: [key_space], arm_thumb_up: [shift],
            arm_thumb_down: [alt], arm_web_up: [ctrl], arm_to_face: [key_q] }
```

`amount` 取 0 到 1，按多深就是多少：在手柄上是一个轴 —— 扳机的行程 —— 在键盘上是一个键的斜坡。`press`
在按住时为 1：在手柄上是一个十字键方向，因为十个按钮留给 bundle 切换模式、留给操作者全部松开；在键盘
上可以是任何按键组合。任务的控制之间可以共用一个按键组合 —— 两只爪子都在 Space 上合拢，而一个模式只读
自己那一侧的 —— 但不能和轴的方向或释放共用。`does` 里的话就是操作说明书里印出来的那些。

自留控制*做*什么，是任务自己的代码；在机器人上是它的部署 hook
`tasks/jumper/five_foot/deploy/lib.rs`，它按名字拿到自留的控制 —— 来自正在驾驶的那个设备 —— 别的都
拿不到。hook 声明它读哪些名字，controller 拒绝两个方向的不一致：hook 读了文件没有声明的名字，或文件
声明了没有 hook 响应的控制。回放时，任务自己的项响应它能响应的：`mdp/gripper.py` 经
`Operator.task_control` 在 `claw_left` 上合拢左爪。

`jumper.five_foot` 的机械臂在它的控制按住期间伸到一个预设位，什么都不按的那一刻就回到收起位 ——
这是 Control-agent 3.1 的点动开关，2026-09-29 提出；在那之前，十字键方向在松开时才被读取，再按一次才
收起，这是写 hook 时经由 MuJoCo 的 viewer 的键所能做到的全部。两个同时按住，就是两者之间的姿态；大拇指
朝上和大拇指朝下同时按住，两者都不算；唯一向内弯的预设 `arm_to_face` 不参与混合 —— 按住，就是这个姿态。在爪子模式里 Shift 把机械臂伸出去，而一个为某件事按住的修饰键
不能同时修饰另一个键去做另一件事，所以键盘在那里的扭身是 `unbound`，扭身只归手柄。

### 4.7 哪个设备在驾驶

任一设备都能独自驱动所有通道，而且**最后被碰的那个在驾驶** —— 命令和任务自留的控制都一样。碰一下
手柄（摇杆离开中位、一个按钮、十字键）就让它成为来源，并清掉键盘的按住状态：一个仍然按着的键在重新
按下之前什么也不驱动，所以松开手柄不会让之前按着的键复活。按下一个文件绑定了的键，就从空闲的手柄
接管。按住的键在 `full_after_s` 内线性升到推满，一松开立即回到静止值 —— 所有轴都是，被移动的轴也
一样：键只会摆放，只有手柄会移动。

释放 —— 手柄上的 `release_button`、键盘上的一个 `release` 按键组合，每个文件里分别是 `B` 和 Esc ——
放掉一切：
键盘的按住状态被清掉，点按式的切换层关闭（按住式的跟随按键本身），每个被移动的轴回到静止值，命令回到
静止值（在 `play --task` 里是回到随机采样器）。

不管哪个模式在跑，每个模式的 operator 都听到每一帧、每一个键，所以切换进去的模式已经知道 shift 和
释放的状态。各模式不同的只是解读：每个 operator 读它自己契约里的绑定，按它自己的范围缩放。

### 4.8 加载时拒绝什么

`load_controls`，以及每种宿主上 controller 的 `OperatorSpec::check`，拒绝的是同一批东西，每一样都是
一个能加载、然后驾驶出错的文件：

- 字典不认识的名字 —— 一个 source、一个按钮、一个键、一个修饰键 —— 或者一个既不是键、也不是修饰键、
  也不是修饰键加键的按键组合；
- 没有手柄绑定的命令轴，或键盘 `axes` 里缺了、又没写 `unbound:` 和原因的命令轴；绑在不存在的命令轴上
  的手柄绑定或键盘条目；列了两次的命令项或轴；YAML 里写了两次的键；
- 不是 ±1 的 `sign`、顺序错乱的 `travel`、两条绑定爬升过同一段行程；
- 既不是 `hold` 也不是 `toggle` 的 `shift`、和释放键是同一个的 shift、没有任何绑定 `shifted` 的
  shift、没有 shift 的 `shifted` 绑定；
- 不是正秒数的 `integrate_s`；没有可放回的被移动轴的 `reset`，或放在释放键上的 `reset`；
- 同一个按键组合落在轴的方向和释放之中的两件事上；键盘块里某个修饰键既单独绑定、又在那里修饰一个键 ——
  为其中一件按住它，就会在伸手去够另一件时做了它；
- 声明了、但没在两个设备上都绑定的任务控制，或绑定了、却没声明的；`kind` 不是 `amount` 或 `press`，
  或没有 `does`；手柄上不在十字键方向上的 `press`、不在轴上的 `amount`、一个手柄控制上的两个任务控制，
  或同时被某个命令、释放、shift 或 reset 读的手柄控制；
- 已退役的 schema 1、分档键盘的 `step`，以及 `centre` 键。

### 4.9 怎样写一个

新任务用 `new-task` skill，它会写全部四个文件；只写操作方式用 `controls` skill。最简单的情况可以生成：

```bash
python .claude/skills/controls/scripts/write_controls.py --task jumper.<name> --print
```

`--bind AXIS=SOURCE±` 改一条手柄绑定（结尾的 `+`/`-` 是符号，必须写），`--release` 改手柄的释放键，
`--force` 覆盖已有文件。它写出的键盘就是每个 jumper 任务共用的那一套：`W S A D` 和方向键、`J L`、`B`。
第二个命令、分段的摇杆、切换层、修饰键组合、被移动的轴或任务自留的控制，都要在它写出的文件上手工编辑，
以 `jumper.posture` 和 `jumper.five_foot` 的文件为范例。`tests/test_controls.py` 只加载现有任务的文件，
所以新写或改过的文件要直接检查：

```bash
python -c "from pathlib import Path; from tasks.jumper.common.mdp.controls import load_controls; load_controls(Path('tasks/jumper/<name>/controls.yaml'))"
```

---

## 5. 在 bundle 里切换模式

### 5.1 两个文件

| | 负责 | 例子 |
|---|---|---|
| `deploy/manifests.json` | 有哪些模式、切换进这些模式的开关 —— 每个设备一个 —— 以及离开它们的时机 | `jump` 在 `A` 和 Space 上，从 `locomotion` 按 |
| `deploy/jumper.controller.toml` | 级联、安全限值、不跑模型的状态，以及 `[fsm]` 设置（包括连击窗口和 `exclusive`） | 第一条 `feedback_stale` → `safe` |

`scripts/deploy.py` 把两者合成为 bundle 的 `controller.toml`，并拒绝声明了绑定、`[fsm.keyboard]` 表或
模型状态的级联文件 ——"两个文件点同一个键的名，就是它们开始不一致的方式"。manifest 不写规则，所以级联
文件必须为 manifest 绑定的每个模式写一条 `when = "button:<mode>"`；没有规则读的绑定会被当作死键拒绝。

### 5.2 一个模式的条目

```json
"jump": {
  "task": "jumper.jump", "policy": "tasks/jumper/jump/out/<dir>",
  "pad":  {"button": "A", "on": "toggle", "from": ["locomotion"]},
  "keys": {"key": ["key_space"], "on": "toggle", "from": ["locomotion"]}
}
```

| 字段 | |
|---|---|
| `task`、`policy` | 必填：任务 id 和导出目录（从不是 checkpoint） |
| `pad` | 手柄的开关：`{button, on, with, from}`。`button` 是一个手柄按钮或一个 `dpad_*` 方向；`with` 是必须按住的一个手柄按钮 —— `"with": "menu"` 加 `"button": "dpad_up"` 就是按住 Menu 再按上 |
| `keys` | 键盘的开关：`{key, on, with, from}`。`key` 是一个列表 —— jumper 的舞在 `["key_1", "keypad_1"]` 上，两排都行 —— 也可以点一个当作键用的修饰键（`"ctrl"`）；`with` 是一个修饰键，`ctrl`、`shift` 或 `alt`，它两个键中的任一个 |
| `on` | 两种开关里都是手势；省略时为 `toggle` |
| `from` | 两种开关里都是它能在哪些模式里被按；省略时为任何模式 |
| `hook` | 任务的部署配置，交给它的 `deploy/lib.rs` —— `{"side": "right"}` |

既没有 `pad` 也没有 `keys` 的模式没有绑定：它由一条不需要按键的规则到达，通常是级联最后的 `always`，
这就让它成了默认模式。每个开关变成以模式命名的 `[[fsm.button]]` 条目 —— 手柄的一条，键盘的每个键
各一条 —— 带 `pad` 或 `key`、`on`、`with` 和 `from`。**同名的条目共用一个锁存**，所以任一设备都能把
模式打开，任一设备也都能把它关掉。2026-09-29 之前，一个模式为两个设备只带一个 `button`、一个 `key`、
一个手势和一个修饰键。

`from` 只管**进入**：级联处在 `from` 没列出的状态时，锁存不能打开，一次性手势、事件或离开也不能触发。
已经打开的锁存总能被它自己的控制关掉，从它进入的那个模式里面关 —— 再按一次 `A` 离开跳跃，尽管跳跃并不
在它自己的 `from` 里。一个开关正是用它来说明：自己不在某个模式里被按，因为那个模式自己的操作用到了
这个控制（§5.9）。

### 5.3 手势、锁存和连击

`button:<name>` 在这条绑定**激活**时为真。什么让它激活，取决于手势：

- **`toggle`** 按一下锁存，再按一下解锁：按一下进入，再按一下离开。模式切换几乎总是要这个，因为一个模式
  要在它的过渡和动作期间一直保持进入。
- **`rise`** 和 **`fall`** 只在一次观测中激活；**`hold`** 在控制按住时激活（死人开关）。
- **连击**数的是同一个控制 —— 同一个手柄按钮，或同一个修饰键下的同一个键 —— 的按下次数，每一次都要在
  上一次之后的 `click_window_ms` 之内；它像 `toggle` 一样切换模式：`double` 进入，再 `double` 离开。作为
  `event` 时，连击只触发一次。

连击次数怎样判定：

- **一个控制上绑定的最长连击，在凑齐它的那一次按下时立即触发。** `A` 上只绑了 `double` 时，第二次按下
  立即进入；只绑了 `single` 时，第一次按下就进入 —— 对任何拿手柄的人来说它就是一个 `toggle`。
- **更短的次数要等窗口过去**，因为再按一次还可能让次数变长。`A` 上同时绑了 `single` 和 `double` 时，
  按一次，要在 300 ms 之后才进入 `single` 的模式；按两次，则立即进入 `double` 的模式。等待是共用一个
  控制的代价，也只在共用时才有。
- **没有绑定的次数什么也不触发** —— 在绑了 `single` 和 `double` 的控制上按三次，什么都不发生。
- **窗口是时间，不是 tick 数。** 它用每种宿主本来就交给 controller 的时间戳来量，所以以手柄 50 Hz 观测
  的板子和以帧率观测的浏览器，对同样的按下数出同样的结果。只要绑了任何连击，`[fsm]` 里就必须有
  `click_window_ms`；`jumper` 的级联文件设为 300 —— 拍定的，不是实测的。
- **键盘数它自己的。** 一个键和一个手柄按钮是两个控制，各数各的，所以按一次 Space、再按一次 `A` 不算
  双击。（2026-09-29 之前，bundle `keyboard` 表里的键*就是*它对应的手柄按钮，两者合在一起数。）

crate 拒绝三种组合，每一种都是会误触发的连击：同一个控制上连击与读单次按下的手势并存（每次连击都会先
是一次 `toggle` —— 应把单击绑成 `single`）；在同时是别的绑定的修饰键的控制上用连击（手伸向组合键时
连击就会触发）；以及在任务自留的十字键方向上用连击（§5.7）。

**修饰键被它修饰的东西消耗掉。** Menu + 上一旦触发，Menu 自己的绑定在它松开之前都保持安静，所以
"单独松开 Menu 离开舞蹈"不会在手松开选中这支舞的那个组合键时触发。键盘上的 Ctrl 也同样被消耗。

**按住的修饰键让单独的控制静默。** 按住 Menu 时，十字键上只响应和 Menu 组合的那条绑定 —— 舞蹈 ——
不响应单按上的那个固定动作；按住 Ctrl 时，`1` 就是 Ctrl + `1`。这就是 operator 对组合键的规则
（Shift 按着时只有 `shift+key_j` 响应），用在切换上。

**锁存不会比该有的活得久。** 一段录制播完会释放它所在模式的锁存，级联随之交还（`jumper.jump` 在开始
1.17 s 后）；过期的手柄输入会释放全部锁存（§3）。一段录制在播完*之前*就被级联离开——倾倒回退、反馈过期
——同样算结束，它的锁存随之释放。2026-10-15 之前并非如此：一次把机器人掀翻的冲刺在 `safe` 期间仍然锁存着，
机器人一回到水平，级联就从头进入冲刺，没有任何人按键（一场螃蟹对战的第二回合开场就是一只螃蟹从起点冲了出去）。
在真机上，这就是摔倒、被人扶起、然后再把摔倒时的那个动作做一遍。以同样方式离开的步态保留锁存：`carry` 是按住
的 LB，松开死人开关是 operator 的事。

### 5.4 用于切换的键盘

键盘在它自己的键上切换：模式的 `keys` 点出这些键，`with` 是一个修饰键，键盘上没有任何东西指向手柄按钮。
舞蹈在键盘上是 Ctrl + `1` `2` `3` `4`、在手柄上是 Menu + 十字键，是因为 manifest 把两者都写了，而不是
因为一个是另一个的副本 —— 而且它们在必须不同的地方就不同：键盘的舞蹈不能从爪子模式里按，因为那里的
Ctrl 把机械臂伸出去，而手柄的 Menu 组合键可以。

一个键盘开关，要对照它能被按的每个模式的键盘块来判定：它的键和它的修饰键，都不能是那个模式的键盘用到
的键或修饰键。键盘上没有组合键的例外（手柄的例外见 §5.7）；一个键盘开关靠 `from` 离开用到它的模式。
板子上没有键盘；这些键是给浏览器和 `play --app` 用的，后者把 viewer 的每个键的按下和松开都交给它的
controller，同时 Space 会暂停 viewer，`G`、`H` 会画出雾和凸包。

### 5.5 离开

bundle 层的 `leave` 列表列出放掉某些模式的时机，放掉之后级联落到它的 `always` 模式 —— 默认模式：

```json
"leave": [
  {"name": "leave_motion",
   "leaves": ["dance_crab", "...", "gesture_salute"],
   "pad":  {"button": "menu", "on": "fall", "from": ["dance_crab", "...", "gesture_salute"]},
   "keys": {"key": ["ctrl"], "on": "fall", "from": ["dance_crab", "...", "gesture_salute"]}}
]
```

离开是一个时刻 —— `rise`、`fall` 或连击 —— 从不是 `toggle`、`hold` 或事件；它本身不锁存，所以没有规则
可以读它，它离开的每个名字都必须是某条会锁存的绑定的名字。在 `jumper` bundle 里，单独松开 Menu 或
Ctrl，就打断任何一支舞或一个固定动作，机器人重新走起来（Control-agent 3.1：单击Menu退出舞蹈回到初始
姿态）。它是 `fall`，而选中舞蹈的那个组合键已经消耗掉了修饰键，所以伸手去按 Menu + 上的途中不会离开。

### 5.6 级联

`[[fsm.rule]]` 按顺序求值，**第一个匹配的胜出**；这个顺序就是安全论证。`jumper` 的级联：

```toml
[[fsm.rule]]  when = "feedback_stale"      enter = "safe"      # 没有反馈，就不跑策略
[[fsm.rule]]  when = "tilted"              enter = "safe"      # 超过 50 度
[[fsm.rule]]  when = "in_state:safe"       enter = "@initial"
[[fsm.rule]]  when = "button:jump"         enter = "jump"
[[fsm.rule]]  when = "button:dance_crab"   enter = "dance_crab"     # ……以及另外三支舞
[[fsm.rule]]  when = "button:gesture_hello" enter = "gesture_hello" # ……以及另外三个固定动作
[[fsm.rule]]  when = "button:claw_right"   enter = "claw_right"
[[fsm.rule]]  when = "button:claw_left"    enter = "claw_left"
[[fsm.rule]]  when = "always"              enter = "locomotion"
```

最后一条必须是 `always`，其他任何一条都不能是，所以级联是完全的，而它的目标就是默认模式。`@stay` 匹配后
什么都不改；`gripper_active` 会被拒绝，因为没有宿主提供它。

**一次一个模式。** 除非 `[fsm] exclusive = true`，锁存是一个集合；设了它，一个锁存打开时会释放其他所有
锁存：最后按下的那个开关就是当前模式。`jumper` 的级联文件设了它（2026-09-29）：没有它，在右爪模式里按
`LB` 会锁存在右爪*后面*，从爪子模式进入的舞蹈结束时会交还给爪子，而不是交还给行走。有了它，操作者那些
规则之间的顺序只决定 §5.7 的组合键。

**切入过渡。** 进入一个跑模型的模式时，设定点在 `mode_switch_ramp_s` 内从实测位姿滑到这个模式的 home
位姿，增益为 `ramp_kp`/`ramp_kd`；每个关节都进入 `pose_reach_tol` 以内时策略才开始 —— 看的是实测位姿，
从不是计时器。

### 5.7 任务和切换共用一个十字键方向

`jumper.five_foot` 把十字键的上、下、左留给它的机械臂，而舞蹈是 Menu + 十字键。两者只能以**在按下那一刻
就离开当前模式的组合键**共用一个方向，`FsmConfig::chord_leaves_first` 拒绝其他一切：这个切换必须带
`with`，必须是 `toggle`（`rise` 只在一个 tick 为真，级联会离开又回来），而且它的规则必须排在任何让级联
留在这个模式的规则之前。这就是舞蹈的规则排在两条爪子规则之上的原因。键盘没有这样的例外：Ctrl 在爪子
模式里属于机械臂，所以键盘的舞蹈在 `from` 里把爪子模式排除在外。

### 5.8 事件：模式内部的控制

顶层的 `buttons` 列表声明不是模式切换的绑定 —— 一段录制动作的 `go`，由人在模式内部选择时机：

```json
"buttons": [{ "name": "<go_event>", "pad": {"button": "A", "on": "fall"} }]
```

它们用同样的 `pad` 和 `keys` 开关，以 `event = true` 合成进去，由契约在 `reference.go_event` 里点了它
名字的那个模式读取；controller 拒绝没有模式读的事件，也拒绝没有按钮声明的 `go_event`。录制也可以改为
声明 `starts_on_entry` —— `jumper` 的跳跃、舞蹈和固定动作都是这样，所以这个 bundle 没有 `buttons` ——
两者互斥。

### 5.9 每一种重叠，以及在哪里被拒绝

| 重叠 | 由谁拒绝 | 何时 |
|---|---|---|
| 字典之外的名字；规则点了没有声明的按钮；没有规则读的绑定；同一控制、同一手势、同一修饰键绑了两次；同时带 `pad` 和 `key` 的绑定；不是其绑定所在设备的修饰键的 `with`；点了不存在的状态的 `from` | `FsmConfig::parse` | 每种宿主，加载时 |
| 连击与读单次按下的手势并存；修饰键上的连击；有连击却没有 `click_window_ms` | `FsmConfig::parse` | 每种宿主，加载时 |
| 是 `toggle`、`hold` 或事件的离开，放掉一个没人锁存的名字的离开，或被规则读的离开；`[fsm]` 不认识的任何键 | `FsmConfig::parse` | 每种宿主，加载时 |
| 能在某个模式里被按的开关，放在那个模式的手柄块用到的控制上 —— 命令的摇杆、shift、reset、释放 —— 或放在它的键盘块用到的键或修饰键上 | `operator::check_switches`，由 `Bundle::open` 和 `Operators::beside` 调用 | 打包器的检查、板子，以及每种宿主加载时 |
| 放在任务自留的十字键方向上的手柄开关，且不是先离开的组合键 | 同上 → `chord_leaves_first` | 同上 |
| 级联文件里出现绑定、`[fsm.keyboard]` 或模型状态 | `scripts/deploy.py` | 构建时 |

一个开关，在它**能被按**的每个模式里判定 —— 它的 `from` 点了这个模式，或者它没有 `from` —— 也在它锁存
的那个模式里判定，因为它在那里总能把自己关掉。

### 5.10 加一个切换

让双击左摇杆（`L3`）成为从行走进入新模式 `crab` 的方式：

1. 导出这个模式的策略，并在 manifest 里给它一个条目：
   `"crab": { "task": "jumper.<task>", "policy": "tasks/jumper/<task>/out/<dir>", "pad": {"button": "L3", "on": "double", "from": ["locomotion"]} }`。
   在这个开关能被按的地方，选一个没有模式的 `controls.yaml` 在用的按钮 —— 不能是 `B`（每个任务的
   释放键），也不能是 `R3`（姿态类任务的切层键）—— 也别用 `X` 或 `Y`，字典把它们标为未确认。
2. 在级联文件里加 `[[fsm.rule]] when = "button:crab" enter = "crab"`，放在它应当胜过的规则之上、安全规则
   之下。`click_window_ms` 已经在那里了。
3. 键盘：给它 `"keys": {"key": ["keypad_5"], "on": "double", "from": ["locomotion"]}`；没有别的东西绑定
   `keypad_5`。它是第二条名为 `crab` 的绑定，有自己的计数。
4. 构建：`python scripts/deploy.py --allow-incomplete` 不用 docker 就能检查两半；完整构建是
   `python scripts/deploy.py`。在 `manual.en.json` 里读一下新增的那几行。

### 5.11 `jumper` bundle

按 Control-agent 3.1 第二节（按键及其功能）的排布，2026-09-29 提出。每个开关都是 `toggle`；数字键
在主键盘区和小键盘上都一样。

| 模式 | 任务 | 手柄 | 键盘 | 能从哪些模式按 |
|---|---|---|---|---|
| `locomotion` | `jumper.posture` | —（级联的 `always`：默认模式） | — | |
| `jump` | `jumper.jump` | `A` | Space | `locomotion` |
| `claw_left` | `jumper.five_foot`，hook side left | `LB` | `V` | `locomotion`、`claw_right` |
| `claw_right` | `jumper.five_foot`，hook side right | `RB` | `B` | `locomotion`、`claw_left` |
| `dance_crab`（螃蟹舞） | `jumper.dance` | Menu + 十字键上 | Ctrl + `1` | `locomotion`、任一支舞或任一个固定动作；手柄上还有两个爪子模式 |
| `dance_brazilian` | `jumper.dance_brazilian` | Menu + 十字键下 | Ctrl + `2` | 同上 |
| `dance_maze` | `jumper.dance_maze` | Menu + 十字键左 | Ctrl + `3` | 同上 |
| `dance_dream_wings` | `jumper.dance_dream_wings` | Menu + 十字键右 | Ctrl + `4` | 同上 |
| `gesture_hello` | `jumper.gesture_hello` | 十字键上 | `1` | `locomotion` |
| `gesture_bow` | `jumper.gesture_bow` | 十字键下 | `2` | `locomotion` |
| `gesture_paw` | `jumper.gesture_paw` | 十字键左 | `3` | `locomotion` |
| `gesture_salute` | `jumper.gesture_salute` | 十字键右 | `4` | `locomotion` |
| `leave_motion`（一个离开） | 放掉任何一支舞或一个固定动作 | 单独按下再松开 Menu | 单独按下再松开 Ctrl | 任一支舞或任一个固定动作 |

模式内部：`locomotion` 见 §4.4，爪子模式见 §4.6；跳跃、各支舞和各个固定动作不读摇杆也不读键 ——
由一段录制驱动，从进入一直到它结束。

- **一次一个模式**（`exclusive`）。再按一次爪子自己的开关，交还给 `locomotion`；在右爪模式里按 `LB`
  立即换成左爪，在左爪模式里按 `RB` 立即换成右爪。
- **舞蹈和固定动作会自己结束**，级联随之交还给 `locomotion`。单独松开 Menu 或 Ctrl 会打断它；再按一次
  它自己的开关也会结束它，按另一支舞的组合键则换到那支舞。
- **跳跃不能被打断**：在空中离开跳跃，机器人会落不到它的落地姿态上，所以 `leave_motion` 不包括它，只有
  再按一次 `A`（Space）才能提前结束一次跳跃。
- **在爪子模式里**，`A` 什么都不做，Space 是爪子，键盘和手柄都一样（跳跃只能从 `locomotion` 按）；单按
  十字键是机械臂，1 到 4 什么都不做。手柄的 Menu 组合键能从那里进入舞蹈，并在按下时离开爪子模式；键盘的
  不能，因为 Ctrl 把机械臂伸出去。
- `jumper.dance_waist` 有导出，但不在第二节里，所以不在这个 bundle 里。

---

## 6. 操作说明书

每个 bundle 都带 `manual.en.json`（`kk-bundle-manual/1`）：每个模式 —— 它的任务、是不是默认模式、怎样
进入和离开 —— 连同每个手柄控制在那个模式下做什么（`controls`），以及单独成列的每个键在那个模式下做什么
（`keys`）；还有每个切换，手柄的和键盘的分开列（`device`），并写明各自能从哪些模式按（`from`）。它在
构建时由 controller 对自己各项控制的描述 `pad_guide()`（`deploy/fsm/src/guide.rs`）生成，所以不会和
绑定不一致；一个键以它做的事出现，从不以一个手柄按钮出现，而且一个模式只列出它的 hook 响应的那些任务
控制 —— 左爪模式列 `LT`，不列 `RT`。连击以人的动作出现 —— 手柄的开关上是
`"pad": "double-click A"`，两种开关上都是 `"does": "switch to jump; double-click again to switch back"`。
浏览器从 `padGuide()` 实时画出同样的数据，`play --app` 启动时也从它打印两个设备的切换键。

中文说明书是它的译本，由 `bundle-manual` skill 用 `--translate` 加进去；构建会检查只有文字不同。

---

## 7. 各部分在哪里

| 部分 | 文件 |
|---|---|
| 字典 | `controller/vocabulary.json`；`controller/vocabulary.py`；`deploy/fsm/src/vocabulary.rs` |
| 在台架上读手柄 | `controller/device.py`（Linux）、`controller/xinput.py`（Windows）、`controller/xbox.py`；`python -m controller` |
| viewer 按键的两个沿 | `rl/mjrl/viewer/keys.py` |
| `controls.yaml` 及其检查 | `tasks/jumper/common/mdp/controls.py` |
| 回放用的 operator | `tasks/jumper/common/mdp/operator.py` |
| 契约的 `controller` 块 | `controls.py` 里的 `controller_contract`，由 `scripts/export.py` 写出 |
| 每个模式的 operator、键盘的按键组合、`CommandShaper`、`check_switches` | `deploy/fsm/src/operator.rs` |
| 手势、锁存、连击、`from`、离开、一次一个模式 | `deploy/fsm/src/control.rs` 里的 `Buttons` |
| `[fsm]`、绑定、加载时的拒绝 | `deploy/fsm/src/config.rs` |
| 级联 | `deploy/fsm/src/fsm.rs` |
| bundle 层的冲突检查 | `deploy/fsm/src/bundle.rs` 里的 `Bundle::open` |
| 任务的部署 hook | `deploy/fsm/src/hook.rs`、`tasks/<task>/deploy/lib.rs` |
| 合成 manifest 和级联文件；说明书 | `scripts/deploy.py` 里的 `compose_fsm`、`write_manual` |
| 给页面用的手柄和键盘描述 | `deploy/fsm/src/guide.rs` 里的 `pad_guide` |

钉住这些行为的测试都以它们抓的那种失败命名 —— `tests/test_controls.py`、`tests/test_controller.py`、
`tests/test_viewer_keys.py`、`tests/test_web_host.py`、`tests/test_app_play.py`，以及每个模块旁边的 Rust
测试（在 `deploy/fsm` 里 `cargo test`），其中包括 `a_click_is_counted_in_time_not_in_frames`。

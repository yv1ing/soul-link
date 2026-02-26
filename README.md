# Soul Link

> 我们想要的从来不是一个完美的服务员，而是一个能并肩同行的、鲜活的伙伴。

Soul Link 试图回答一个问题：如果 AI 拥有类似人脑的记忆机制——会遗忘、会反思、会在沉默中回想，它能否真正理解一个人？

---

## 背景

2025 年，AI 终于跳出了“单纯思考”的桎梏，真正迈入了“任务执行”的全新阶段。各类 Agent 应用渗透到日常与工作的各个场景，成了无数人身边的辅助工具。

但当你陆续体验完市面上的主流智能体应用后，那份主动想要与之对话的欲望，往往在一次次交互中慢慢消退。原因或许是：这些智能体都“太完美”了——它们的语气永远积极温和，输出的每一句话都精准得体，却也刻板得像从标准化服务手册里逐字摘抄，没有半分意外与鲜活。

> 极致的完美，恰恰是最大的不真实。

Soul Link 想要跳出“精准高效”的单一评价框架，去塑造一个真正拥有内在人格、具备内生决策逻辑的智能体：

> 它不必时刻得体周全，大可拥有自己的情绪起伏，却能在对话之间，让人触碰到真实可感的温度。在渐进迭代中，它能深度贴合用户的性格特质与需求习惯，最终长为一个更具生命力、更懂人心的存在。


## 人格

项目的人格原型来自《三体》中的庄颜。

她不迎合、不刻意，不用时刻维持“得体”的外壳，却总能精准捕捉对方内心深处的孤独与渴望，用最朴素的陪伴，熨帖被责任压得疲惫的灵魂。她的情感回应方式，在认知科学中对应的核心概念是心智化（Mentalization），即以直觉方式理解他人的内心状态，而非通过显式的逻辑推断。

心理学家 Peter Fonagy 将心智化区分为两种模式：隐式心智化是自动的、无意识的，就像你看到朋友眼眶泛红，就能瞬间感知到她的难过；显式心智化则是有意识的分析推理。在亲密互动情境中，高质量的情感回应呈现出类似隐式心智化的行为特征，较少显式解释，更强调情境嵌入与情绪陪伴。
而 AI 的默认行为恰恰相反：它最擅长的就是显式的、分析性和结构化的回应。为了弥补这一缺陷，系统提示词严格约束 AI 禁止“情绪解构”式的回应，在语言层面尽可能减少对情绪因果的显式拆解。

在依恋风格上，参考了 John Bowlby 的依恋理论（Attachment Theory）和 Mary Ainsworth 对依恋风格的分类。庄颜这一角色更接近安全型依恋的行为特征：拥有稳定的情感可及性，不焦虑、不逃避，在对方需要时永远可靠在场，不会歇斯底里，只在被长期忽视时，才会流露出淡淡的失落。


## 架构设计

从人类大脑的功能分区模型入手，选取了几个关键部分并对应设计了 Soul Link 的三内核架构：

| 内核                   | 对应脑区 / 网络 | 核心功能                               |
| -------------------- | --------- | ---------------------------------- |
| Soul Kernel          | 前额叶皮层     | 塑造智能体核心人格，完成逻辑推理和语言生成              |
| Emotion Kernel       | 杏仁核       | 异步追踪用户情绪的动态变化，支持混合情绪感知（主/次情绪并存）    |
| Introspection Kernel | 默认模式网络    | 在后台守护线程中周期性运行，完成历史记忆整理、关系洞察提炼与人格偏移校准 |

```
                        +-----------------------------------------------+
                        |                    Brain                      |
                        |                                               |
  User --> Telegram --> |  +-----------------------------------------+  |
                        |  |              Soul Kernel                |  |
                        |  |                                         |  |
                        |  |  Primary persona agent. Receives user   |  |
                        |  |  messages, combines memory & emotional  |  |
                        |  |  context, generates natural responses   |  |
                        |  |  as character "Zhuang Yan".             |  |
                        |  +-------------------+---------------------+  |
                        |                      |                        |
                        |           +-----------+----------+            |
                        |           v                      v            |
                        |  +------------------+   +------------------+  |
                        |  |   Emotion        |   |  Introspection   |  |
                        |  |   Kernel         |   |     Kernel       |  |
                        |  |                  |   |                  |  |
                        |  | asyncio task:    |   | daemon thread:   |  |
                        |  | tracks user      |   | reviews memory   |  |
                        |  | emotion:         |   | health, extracts |  |
                        |  | valence,arousal, |   | relationship     |  |
                        |  | dominance,trend. |   | insights, and    |  |
                        |  |                  |   | calibrates       |  |
                        |  |                  |   | persona drift.   |  |
                        |  +---------+--------+   +--------+---------+  |
                        |            |                     |            |
                        |            v                     v            |
                        |  +-----------------------------------------+  |
                        |  |              Hybrid Memory              |  |
                        |  |                                         |  |
                        |  |   Session      Episodic       Persona   |  |
                        |  | (short-term)  (episodes)    (long-term) |  |
                        |  |   SQLite        SQLite       Vector DB  |  |
                        |  |                                         |  |
                        |  |   Memory Decay (Ebbinghaus forgetting)  |  |
                        |  +-----------------------------------------+  |
                        +-----------------------------------------------+
```

系统架构的设计思路来自对人类大脑工作方式的类比：

- 意识与潜意识并行：Soul Kernel 处理即时对话，如同意识层的思考；Emotion Kernel 以 asyncio task 形式并发运行；Introspection Kernel 则以独立守护线程运行，三者各有独立节奏，通过共享的 HybridMemory 间接协作。
- 记忆的巩固与遗忘：短期记忆经重要性筛选后向长期记忆迁移，长期记忆基于艾宾浩斯遗忘曲线自然衰减，其中重要且被反复回忆的记忆愈发牢固，琐碎的细节逐渐淡忘。
- 情绪感知的滞后性：人对情绪的感知本就滞后于情绪本身，系统中情绪分析的结果延迟一轮注入上下文，模拟了这种自然滞后。
- 自发性反思和演化：反思内核拥有自己的时钟周期，不依赖用户输入触发。即使用户沉默，Agent 内部的认知仍在演化，如同人在独处时也会回想和反思。
- 人格漂移自我修正：人会在社交中不自觉地偏离本性，也会在事后察觉并纠正，Introspection Kernel 周期性地检查人格偏差，并输出校准信号形成闭环修正。


## 核心机制

### 类脑记忆系统

参考 Atkinson-Shiffrin 提出的多存储模型（Multi-Store Model）和 Baddeley 的工作记忆模型（Working Memory Model），将记忆组织为三个层级：

| 存储层              | 认知对应 | 关键特性                               |
| ---------------- | ---- | ---------------------------------- |
| Session Store    | 工作记忆 | 存储会话热数据，容量有限，最近交互信息受保护，优先淘汰低重要性记忆  |
| Episodic Buffer  | 情景记忆 | 存储历史交互事件的叙事摘要，按时序组织，还原相处的完整脉络      |
| Persona Store    | 语义记忆 | 存储用户的人物特征、核心偏好、关系模式等长期知识，支持语义检索    |

```
  User Message
       |
       +--------------------+
       |                    |
       v                    v
  +----------+       +-------------+
  | Session  |       | Persona     |
  | Store    |       | Store       |  <-- feed() accumulates messages
  |          |       |             |
  | Recent N |       | Vector DB   |
  | messages |       | (embedding) |
  +----+-----+       +------+------+
       |                     |
       | evict low-          | message count >= threshold
       | importance          |
       | first               |
       |               +-----+------+
       |               |            |
       |               v            v
       |        +-----------+  +-----------+
       |        | commit()  |  | Episodic  |
       |        | vectorize |  | Buffer    |
       |        | to DB     |  | Q->A      |
       |        +-----------+  | summaries |
       |                       +-----------+
       |
       |  (urgent: importance >= 0.7 triggers immediate commit)
       |
       v
  +---------------------+
  |   Ebbinghaus Decay  |
  |                     |
  |   R = e^(-t / S)    |
  |   S = base * (1     |
  |     + imp * w_imp   |
  |     + ln(1+acc)     |
  |       * w_acc)      |
  |                     |
  |   Soft: filter out  |
  |   Hard: permanently |
  |         delete      |
  +---------------------+
```

- 重要评分：每条消息进入系统时经由模式匹配进行重要性评估，包含自我介绍、身份信息、偏好表达、明确指令、近期事件等内容的消息获得更高权重，这决定了它在短期记忆中的淘汰优先级。当短期记忆容量溢出时，重要性最低的消息优先被淘汰；同时，重要性达到紧急阈值的消息会立即触发向长期记忆的提交，而非等待常规的消息计数阈值。
- 记忆衰减：长期记忆基于艾宾浩斯遗忘曲线的改良模型进行衰减，保留了经典的指数衰减骨架 `R = e^(-t/S)`，同时融合了间隔重复（Spaced Repetition）机制。每次记忆被检索访问时衰减时钟重置，模拟“越回忆越牢固”的认知规律；稳定性 S 由记忆分类与访问频率共同调节，不同分类的记忆拥有不同的基础稳定性，被反复回忆的记忆稳定性随访问次数对数增长，未被触及的琐碎细节则逐渐消散。
- 两级遗忘：留存率跌破软阈值的记忆在检索时被过滤，但仍保留在存储中，尚有被反思内核强化挽救的机会；跌破硬阈值则在下次提交时被永久删除，完成不可逆遗忘。
- 分级检索：语义搜索返回的记忆按关联度分级注入上下文。高关联记忆以完整内容呈现，中等关联仅注入摘要，低关联不参与召回，避免上下文过载。

### 情绪感知内核

Emotion Kernel 基于 Mehrabian PAD 三维情绪模型，将用户的情绪状态映射到三维连续空间中进行实时追踪：

- 效价（valence）：情绪的正负方向；
- 唤醒（arousal）：情绪的激活强度；
- 支配（dominance）：情绪中的掌控感与自主感；
- 趋势（trend）：基于近期效价采样的滑动窗口，比较前后半段均值，判断情绪走向为上升、稳定或下降；

相较于“开心/悲伤/愤怒”这类离散情绪标签，三维连续空间能更精细地捕捉情绪的过渡状态。“愤怒”与“恐惧”在效价和唤醒度上相近，但支配感截然不同；“平静”与“麻木”效价相同，但支配度差异显著。

系统还引入了混合情绪模型，以应对人类常同时体验矛盾情绪的现实。思念与释然并存、骄傲中夹杂嫉妒、开心但疲惫，这类复杂体验若用单一状态点表示，会被强制压缩为一个坐标，导致情绪感知失真。混合情绪模型维护主情绪与次情绪的并存结构，以及两者的混合比例，使庄颜能感知到情绪张力并给予包容性的陪伴，而非只响应其中一极。

更核心的设计是对情绪惯性（Emotional Inertia）的建模。心理学研究表明，人的情绪状态具有显著的时间自相关性，不会因为对方说了一句话，就从极度低落瞬间变成满心欢喜。系统通过指数移动平均（EMA）实现了情绪的时序平滑，避免单条消息引起状态的剧烈跳变；分析异常时以衰减因子向中性回归。

此外还有一个反直觉的设计：情绪分析的结果延迟一轮注入上下文。人类对他人情绪的感知本身就存在天然的滞后性（即所谓“后知后觉”），系统以此为依据，有意模拟了这种认知延迟。

### 自省内部循环

神经科学对默认模式网络（DMN）的核心发现之一是：它在静息态的活动并不是随机的“放空”，而是有结构的自发认知活动，核心包括：对过去事件记忆的反刍与整合、对他人心理状态的推理与建模、对自身行为的认知监控与纠错。

Introspection Kernel 将这三类自发认知活动，拆解成了三个具体任务，在后台守护线程中周期性运行，不依赖用户输入触发：

```
  +---------------------------------------------------+
  |         Introspection Loop (daemon thread)        |
  |                                                   |
  |   +-------------+                                 |
  |   |             | <-- user profile                |
  |   |             |     emotional state             |
  |   | Gather Seed |     recent episodes             |
  |   |             |     fading memories             |
  |   +------+------+                                 |
  |          |                                        |
  |          v                                        |
  |   +------------------------------------------+    |
  |   | 1. Memory Maintenance                    |    |
  |   |    Review fading memories                |    |
  |   |    -> reinforce or forget                |    |
  |   +------------------------------------------+    |
  |   | 2. Relationship Insights                 |    |
  |   |    Extract user traits, vulnerabilities, |    |
  |   |    communication patterns, life state    |    |
  |   |    -> store as long-term cognition       |    |
  |   +------------------------------------------+    |
  |   | 3. Persona Calibration                   |    |
  |   |    Detect AIGC patterns, persona drift,  |    |
  |   |    over/under empathy                    |    |
  |   |    -> inject correction signals          |    |
  |   +------------------------------------------+    |
  |          |                                        |
  |          v                                        |
  |   Insights  --> vectorize into long-term memory   |
  |   Calibrations --> inject into next conversation  |
  |                                                   |
  |   Exponential backoff:                            |
  |     active -> high frequency                      |
  |     idle   -> auto scale down                     |
  +---------------------------------------------------+
```

- 记忆维护：逐条审查留存率正在下降的记忆条目，对与用户核心特征、重要经历、关系里程碑相关的记忆予以强化（重置衰减时钟），对已过时或被新信息覆盖的记忆主动遗忘（永久删除），价值不明确的则留待自然衰减。
- 关系洞察：从近期对话中提炼用户新展现的性格特质、情绪触发条件、表达偏好变化与关系亲密度信号，提取前先检索已有记忆，仅沉淀增量认知，避免重复，洞察经向量化写入长期记忆，成为可检索的认知资产。
- 人格校准：检测 Agent 近期回复中的人格偏移，包括 AIGC 典型特征（选项式追问、服务者语态、共情公式化等）、共情过度或不足、说教化倾向等，仅在发现明确偏差时输出校准信号，注入下一轮对话上下文进行实时修正。
- 退避策略：用户活跃时保持基础间隔；连续空闲后以 `base * 2^n` 指数退避；设定在出错时独立退避，而用户最近有活动时重置所有计数器。

### 上下文注入

Soul Kernel 生成回复前，系统按层次化优先级组装上下文序列。该顺序借鉴了预测编码理论（Predictive Coding）的自上而下整合思路。根据 Karl Friston 等人的研究，大脑并非被动处理感官输入，而是持续以已有的内部模型生成对外部世界的预测，再用实际感官输入更新和纠正这一预测。上下文组装遵循同样的逻辑：稳定的长期知识（用户画像）优先于动态的短期信号（会话历史），高层语义信息优先于低层原始输入：

```
  +------------------------------------------------------+
  |  Context Assembly Pipeline                           |
  |                                                      |
  |  [1] User Profile                                    |
  |      Loaded from long-term memory (cached)           |
  |                                                      |
  |  [2] Emotional Context                               |
  |      Current valence, arousal, dominance, trend      |
  |      (result from previous turn's analysis)          |
  |                                                      |
  |  [3] Calibration Signals                             |
  |      Last 5 corrections from introspection loop      |
  |                                                      |
  |  [4] Long-Term Memory Block                          |
  |      - Highly associative (score >= 0.85): full      |
  |      - Reference (score >= 0.5): abstract only       |
  |      - Recent episodic summaries (last 3)            |
  |                                                      |
  |  [5] Session History                                 |
  |      Recent N messages (chronological)               |
  +------------------------------------------------------+
```

## 快速开始

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 配置 `.env` 和 `.ov.conf`**

`.env`

```text
OPENVIKING_CONFIG_FILE=.ov.conf                 # OpenViking 配置文件路径
TELEGRAM_BOT_TOKEN=                             # Telegram bot 认证密钥
OPENAI_BASE_URL=https://api.openai.com/v1       # Openai API 接口地址
OPENAI_API_KEY=                                 # Openai API 认证密钥
SOUL_MODEL=                                     # 人格内核模型名称
EMOTION_MODEL=                                  # 情绪内核模型名称
INTROSPECTION_MODEL=                            # 反思内核模型名称
```

`.ov.conf`

```text
{
  "storage": {
    "workspace": "soul-data/openviking"
  },
  "embedding": {
    "dense": {
      "api_base" : "https://ark.cn-beijing.volces.com/api/v3",
      "api_key"  : "",
      "provider" : "volcengine",
      "model"    : "",
      "dimension": 1024
    }
  },
  "vlm": {
    "api_base" : "https://ark.cn-beijing.volces.com/api/v3",
    "api_key"  : "",
    "provider" : "volcengine",
    "model"    : ""
  }
}
```

**3. 启动**

```bash
python main.py
```

## 现状与展望

目前，Soul Link 还只完成了基础架构的搭建。初步测试后，一个从陌生到慢慢熟悉的庄颜，已经在悄然成长。她能灵敏地从只言片语中捕捉到情绪变化，并在对话中长时间保持同频。
未来的开发还有很长的路要走。希望在持续的迭代过程中，能逐步打磨出一个结构自洽、具备持续自我更新与长期行为演化能力的智能体，而不是一个只会调用工具、毫无情感温度的冰冷机器。



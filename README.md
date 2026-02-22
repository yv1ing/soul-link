# Soul Link

> 灵感来自《三体》中的角色"庄颜"——一个温柔、纯粹、自洽的理想型伴侣。

Soul Link 试图回答一个问题：如果 AI 拥有类似人脑的记忆机制——会遗忘、会反思、会在沉默中回想——它能否真正理解一个人？

每一次对话中，悄悄捕捉用户的性格特质、情绪模式与表达习惯，渐进地构建用户画像。为了避免数据爆炸，记忆不是无限堆积的，它会随着时间衰减、会被反思筛选、会在反复回忆中巩固。随着时间推移，Agent 对用户的理解将不断深化，逐渐成为真正"懂你"的 AI 伴侣。

## 架构设计

采用三内核协同的"类脑"架构，各内核独立运行、异步协作：

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
                        |  | Async tracks     |   | Periodically     |  |
                        |  | user emotion:    |   | reviews memory   |  |
                        |  | valence,arousal, |   | health, extracts |  |
                        |  | trend.           |   | relationship     |  |
                        |  |                  |   | insights, and    |  |
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

架构的设计思路来自对人类大脑工作方式的类比：

- **意识与潜意识并行**：Soul Kernel 处理即时对话，如同意识层的思考；Emotion Kernel 和 Introspection Kernel 在后台异步运行，如同潜意识对情绪的感知和对经历的反刍，三者各有独立节奏，通过共享记忆间接协作；
- **记忆的巩固与遗忘**：短期记忆经重要性筛选后向长期记忆迁移，长期记忆基于艾宾浩斯遗忘曲线自然衰减——重要且被反复回忆的记忆愈发牢固，琐碎的细节逐渐淡忘；
- **情绪感知的滞后性**：人对情绪的感知本就滞后于情绪本身，系统中情绪分析的结果延迟一轮注入上下文，模拟了这种自然滞后；
- **自发性反思和演化**：反思内核拥有自己的时钟周期，不依赖用户输入触发——即使用户沉默，Agent 内部的认知仍在演化，如同人在独处时也会回想和反思；
- **人格漂移自我修正**：人会在社交中不自觉地偏离本性，也会在事后察觉并纠正，Introspection Kernel 周期性地检查人格偏差，并输出校准信号形成闭环修正；

## 核心机制

### 类脑记忆系统

人类大脑中的记忆并非一成不变的存档，而是一个不断筛选、巩固和淡忘的动态过程。Soul Link 的记忆系统模拟了这一过程，将记忆组织为三个层级：

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
       |  (urgent: importance >= 0.x triggers immediate commit)
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

- **重要评分**：每条消息进入系统时经由模式匹配进行重要性评估，包含身份信息、偏好表达、明确指令等内容的消息获得更高权重，这决定了它在短期记忆中的淘汰优先级——当短期记忆容量溢出时，重要性最低的消息优先被淘汰；同时，重要性超过紧急阈值的消息会立即触发向长期记忆的提交，而非等待常规的消息计数阈值；
- **记忆衰减**：长期记忆基于艾宾浩斯遗忘曲线的改良模型进行衰减——保留了经典的指数衰减骨架 `R = e^(-t/S)`，同时融合了间隔重复（Spaced Repetition）机制：每次记忆被检索访问时衰减时钟重置，模拟"越回忆越牢固"的认知规律；稳定性 S 由记忆分类与访问频率共同调节——不同分类的记忆拥有不同的基础稳定性（如用户画像永不遗忘，偏好类记忆比事件类记忆更持久），被反复回忆的记忆稳定性随访问次数对数增长，未被触及的琐碎细节则逐渐消散；
- **两级遗忘**：留存率跌破软阈值的记忆在检索时被过滤，但仍保留在存储中，尚有被反思内核强化挽救的机会；跌破硬阈值则被永久删除，完成不可逆遗忘；
- **分级检索**：语义搜索返回的记忆按关联度分级注入上下文——高关联记忆以完整内容呈现，中等关联仅注入摘要，低关联不参与召回，避免上下文过载；

### 自省内部循环

Introspection Kernel 模拟人类对经历的自发性反刍——它在后台守护线程中周期性运行，不依赖用户输入触发，而是自主地从已有记忆与近期对话中提炼更深层的认知：

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

- **记忆维护**：逐条审查留存率正在下降的记忆条目，对与用户核心特征、重要经历、关系里程碑相关的记忆予以强化，对已过时或被新信息覆盖的记忆主动遗忘，而价值不明确的不做干预，留待自然衰减；
- **关系洞察**：从近期对话中提炼用户新展现的性格特质、情绪触发条件、表达偏好变化与关系亲密度信号，提取前先检索已有记忆，仅沉淀增量认知，避免重复，洞察经向量化写入长期记忆，成为可检索的认知资产；
- **人格校准**：检测 Agent 近期回复中的人格偏移——包括 AIGC 典型特征（选项式追问、服务者语态、共情公式化等）、共情过度或不足、说教化倾向等，仅在发现明确偏差时输出校准信号，注入下一轮对话上下文进行实时修正；

### 情绪感知内核

Emotion Kernel 基于 Russell 环形情绪模型（Circumplex Model of Affect），将用户的情绪状态映射到二维连续空间中进行实时追踪：

- **效价 (valence)**：情绪的正负方向，从极度负面到极度正面；
- **唤醒 (arousal)**：情绪的激活强度，从极度平静到极度激动；
- **趋势 (trend)**：基于近期效价采样的滑动窗口，判断情绪走向为上升、稳定或下降；

情绪状态的更新采用指数移动平均（EMA）进行时序平滑，避免单条消息引起状态的剧烈跳变；分析异常时以衰减因子向中性回归。情绪分析以异步任务运行，不阻塞主回复生成——当前消息的分析结果延迟一轮注入上下文，模拟人类情绪感知固有的时间滞后性，使情感响应呈现自然的连续性。

### 上下文注入

Soul Kernel 生成回复前，系统将多源信息组装为层次化的上下文序列：

```
  +------------------------------------------------------+
  |  Context Assembly Pipeline                           |
  |                                                      |
  |  [1] User Profile                                    |
  |      Loaded from long-term memory (cached)           |
  |                                                      |
  |  [2] Emotional Context                               |
  |      Current valence, arousal, trend snapshot        |
  |                                                      |
  |  [3] Calibration Signals                             |
  |      Last N corrections from introspection loop      |
  |                                                      |
  |  [4] Long-Term Memory Block                          |
  |      - Highly associative: full overview             |
  |      - Reference: abstract only                      |
  |      - Recent episodic summaries                     |
  |                                                      |
  |  [5] Session History                                 |
  |      Recent N messages (chronological)               |
  +------------------------------------------------------+
```

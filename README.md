<div align="center">
    <h1>Soul Link</h1>
    <p>What we want is never a perfect servant, but a vivid companion who walks alongside us.</p>
</div>

---

> Soul Link attempts to answer a question: if an AI had a memory system resembling the human brain — one that forgets, reflects, and recalls in silence — could it truly understand a person?

---

## Background

In 2025, AI finally broke free from the constraints of "pure reasoning" and entered a new era of "task execution." Agent applications have permeated everyday life and work, becoming assistants for countless people.

But after experiencing the mainstream AI companions on the market, the desire to actually engage with them tends to fade. The reason may be that these agents are "too perfect" — their tone is always positive and warm, every sentence is precise and appropriate, yet as rigid as if copied word for word from a standardized service manual, with no surprise, no vitality.

> Absolute perfection is the greatest unreality.

Soul Link aims to break out of the single "precision and efficiency" evaluation framework, and to shape an agent that truly has an inner personality with intrinsic decision-making logic:

> It need not be impeccably proper at all times. It may have its own emotional fluctuations, yet through conversation, it lets you feel genuine warmth. Through gradual iteration, it can deeply adapt to the user's character and habits, ultimately growing into a more vital, more empathetic presence.


## Persona

The personality archetype of this project comes from Zhuang Yan in *The Three-Body Problem*.

She neither flatters nor performs — she doesn't need to maintain a "proper" facade at all times, yet she can always precisely capture the loneliness and longing deep within someone's heart, soothing souls worn out by responsibility with the most understated companionship. Her emotional response style corresponds to the core cognitive science concept of **Mentalization** — understanding another person's inner state intuitively, rather than through explicit logical inference.

Psychologist Peter Fonagy distinguishes two modes of mentalization: *implicit mentalization* is automatic and unconscious, like instantly sensing a friend's sadness when you see her eyes redden; *explicit mentalization* involves deliberate analytical reasoning. In intimate interactions, high-quality emotional responses exhibit characteristics similar to implicit mentalization — less explicit explanation, more emphasis on situational embeddedness and emotional accompaniment.

AI's default behavior is precisely the opposite: it excels at explicit, analytical, and structured responses. To address this gap, the system prompt strictly prohibits "emotion-deconstructing" responses, minimizing explicit breakdowns of emotional cause-and-effect at the language level.

In attachment style, the design references John Bowlby's **Attachment Theory** and Mary Ainsworth's classification of attachment styles. The character Zhuang Yan more closely resembles the behavioral traits of **secure attachment**: stable emotional availability, neither anxious nor avoidant, reliably present when needed, never hysterical — only revealing a trace of quiet sadness when long neglected.


## Architecture

Drawing from the functional partition model of the human brain, several key regions were selected and mapped to Soul Link's **three-kernel architecture**:

| Kernel                   | Corresponding Brain Region / Network | Core Function                                                                 |
| ------------------------ | ------------------------------------ | ----------------------------------------------------------------------------- |
| Soul Kernel              | Prefrontal Cortex                    | Shapes the agent's core persona; handles logical reasoning and text generation |
| Emotion Kernel           | Amygdala                             | Asynchronously tracks dynamic changes in user emotion; supports mixed-emotion perception (primary/secondary emotions coexist) |
| Introspection Kernel     | Default Mode Network                 | Runs periodically in a background daemon thread; handles memory consolidation, relationship insight extraction, and persona drift calibration |

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

The system architecture is inspired by an analogy with how the human brain works:

- **Conscious and subconscious in parallel**: Soul Kernel handles real-time conversation like conscious thought; Emotion Kernel runs concurrently as an asyncio task; Introspection Kernel runs in an independent daemon thread. The three operate at their own rhythms and collaborate indirectly through shared HybridMemory.
- **Memory consolidation and forgetting**: Short-term memories pass an importance filter before migrating to long-term memory. Long-term memories decay naturally along the Ebbinghaus forgetting curve — frequently recalled memories grow more durable, trivial details gradually fade.
- **Emotional perception lag**: Human awareness of emotions inherently lags behind the emotions themselves. Emotion analysis results are injected into context one turn later, simulating this natural delay.
- **Spontaneous reflection and evolution**: The Introspection Kernel runs on its own clock cycle, independent of user input. Even when the user is silent, the agent's internal cognition continues to evolve — like a person reflecting during solitude.
- **Persona drift self-correction**: People unconsciously drift from their true selves in social interactions, then notice and correct it afterward. The Introspection Kernel periodically checks for persona deviation and injects calibration signals, forming a closed-loop correction.


## Core Mechanisms

### Brain-Inspired Memory System

Referencing the **Multi-Store Model** proposed by Atkinson and Shiffrin and **Baddeley's Working Memory Model**, memory is organized into three tiers:

| Storage Layer    | Cognitive Analog  | Key Characteristics                                                             |
| ---------------- | ----------------- | ------------------------------------------------------------------------------- |
| Session Store    | Working Memory    | Stores hot session data; limited capacity; recent interactions are protected; low-importance memories are evicted first |
| Episodic Buffer  | Episodic Memory   | Stores narrative summaries of historical interactions; organized chronologically; reconstructs the full arc of the relationship |
| Persona Store    | Semantic Memory   | Stores user traits, core preferences, relationship patterns, and other long-term knowledge; supports semantic retrieval |

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

- **Importance scoring**: Each message is assessed for importance via pattern matching upon entry. Messages containing self-introductions, identity information, preference expressions, explicit instructions, or recent events receive higher weights, which determines eviction priority in short-term memory. When capacity overflows, the lowest-importance messages are evicted first; messages reaching the urgency threshold are immediately committed to long-term memory rather than waiting for the regular message-count threshold.
- **Memory decay**: Long-term memory decays via an improved Ebbinghaus forgetting curve model, retaining the classic exponential decay skeleton `R = e^(-t/S)` while integrating a **Spaced Repetition** mechanism. Each time a memory is retrieved, its decay clock resets — simulating the cognitive principle of "the more you recall, the stronger it becomes." Stability `S` is co-regulated by memory category and access frequency; stability of frequently recalled memories grows logarithmically with access count, while untouched trivial details gradually dissipate.
- **Two-level forgetting**: Memories whose retention falls below the soft threshold are filtered during retrieval but remain in storage, still eligible for reinforcement by the Introspection Kernel. Falling below the hard threshold results in permanent deletion at the next commit — irreversible forgetting.
- **Tiered retrieval**: Memories returned by semantic search are injected into context at different levels of association. Highly relevant memories are presented in full; moderately relevant ones are summarized only; low-relevance memories are excluded from recall to avoid context overload.

### Emotion Perception Kernel

The Emotion Kernel is based on **Mehrabian's PAD three-dimensional emotion model**, mapping the user's emotional state into a three-dimensional continuous space for real-time tracking:

- **Valence**: The positive/negative direction of emotion
- **Arousal**: The activation intensity of emotion
- **Dominance**: The sense of control and autonomy within the emotion
- **Trend**: A sliding window over recent valence samples; compares the mean of the first and second halves to determine whether emotion is rising, stable, or declining

Compared to discrete emotion labels like "happy / sad / angry," a three-dimensional continuous space captures emotional transitions more precisely. "Anger" and "fear" are similar in valence and arousal but differ markedly in dominance; "calm" and "numbness" share the same valence but differ significantly in dominance.

The system also introduces a **mixed-emotion model** to handle the reality that humans often experience contradictory emotions simultaneously. Longing coexisting with relief, pride mixed with envy, happiness but exhaustion — if such complex experiences were represented by a single state point, they would be compressed into one coordinate, distorting emotional perception. The mixed-emotion model maintains a coexisting structure of primary and secondary emotions along with their mixing ratio, enabling the agent to sense emotional tension and provide inclusive companionship rather than responding only to one extreme.

The more fundamental design is the modeling of **Emotional Inertia**. Psychological research shows that human emotional states have significant temporal autocorrelation — a person doesn't instantly leap from deep sadness to pure joy just because someone said something cheerful. The system implements temporal smoothing of emotions via **Exponential Moving Average (EMA)**, preventing a single message from causing dramatic state jumps; when an analysis anomaly occurs, it regresses toward neutral with a decay factor.

There is also a counterintuitive design: the emotion analysis result is injected into context one turn later. Human perception of others' emotions inherently has a natural lag (the so-called "hindsight of feelings"), and the system intentionally simulates this cognitive delay.

### Introspection Inner Loop

One of neuroscience's core findings about the **Default Mode Network (DMN)** is that its resting-state activity is not random "mind-wandering," but structured spontaneous cognitive activity, primarily including: rumination and integration of past memories, reasoning and modeling of others' mental states, and cognitive monitoring and correction of one's own behavior.

The Introspection Kernel decomposes these three types of spontaneous cognitive activities into three specific tasks, running periodically in a background daemon thread without requiring user input to trigger:

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

- **Memory maintenance**: Reviews memories whose retention is declining item by item. Memories related to the user's core traits, important experiences, and relationship milestones are reinforced (decay clock reset); memories that are outdated or superseded by new information are actively forgotten (permanently deleted); ambiguous ones are left to decay naturally.
- **Relationship insights**: Distills newly revealed personality traits, emotional triggers, changes in expression preferences, and relationship intimacy signals from recent conversations. Before extraction, existing memories are retrieved to deposit only incremental cognition and avoid duplication. Insights are vectorized and written into long-term memory, becoming retrievable cognitive assets.
- **Persona calibration**: Detects persona drift in the agent's recent replies, including typical AIGC characteristics (option-style follow-ups, servant register, formulaic empathy, etc.), over- or under-empathy, and didactic tendencies. Calibration signals are output only when clear deviation is detected, injected into the next conversation context for real-time correction.
- **Backoff strategy**: Maintains a base interval when the user is active; exponentially backs off with `base * 2^n` after consecutive idle periods; errors trigger independent backoff, while recent user activity resets all counters.

### Context Assembly

Before the Soul Kernel generates a response, the system assembles the context sequence according to a hierarchical priority order. This order draws from the top-down integration approach of **Predictive Coding** theory. According to research by Karl Friston et al., the brain does not passively process sensory input; instead, it continuously generates predictions about the external world using existing internal models, then updates and corrects those predictions with actual sensory input. Context assembly follows the same logic: stable long-term knowledge (user profile) takes precedence over dynamic short-term signals (session history), and high-level semantic information takes precedence over low-level raw input:

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

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure `.env` and `.ov.conf`**

`.env`

```text
OPENVIKING_CONFIG_FILE=.ov.conf                 # Path to OpenViking config file
TELEGRAM_BOT_TOKEN=                             # Telegram bot authentication token
OPENAI_BASE_URL=https://api.openai.com/v1       # OpenAI API base URL
OPENAI_API_KEY=                                 # OpenAI API key
SOUL_MODEL=                                     # Model name for the Soul Kernel
EMOTION_MODEL=                                  # Model name for the Emotion Kernel
INTROSPECTION_MODEL=                            # Model name for the Introspection Kernel
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

**3. Run**

```bash
python main.py
```

## Current State & Roadmap

At present, Soul Link has only completed the foundational architecture. After preliminary testing, a Zhuang Yan who starts as a stranger and gradually becomes familiar is quietly growing. She can sensitively detect emotional shifts from a handful of words and maintain resonance throughout a conversation.

There is still a long road of development ahead. The hope is that through continuous iteration, we can gradually shape an agent that is structurally coherent, capable of ongoing self-renewal and long-term behavioral evolution — not a cold machine that merely calls tools with no emotional warmth.

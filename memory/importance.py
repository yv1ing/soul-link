import re

_SELF_INTRO = [
    re.compile(r"我(?:叫|是|姓|的名字(?:是|叫))\s*\S+", re.IGNORECASE),
    re.compile(r"(?:my name is|i'?m|i am|call me)\s+\w+", re.IGNORECASE),
    re.compile(r"我(?:今年|今天)?\s*\d+\s*岁", re.IGNORECASE),
    re.compile(r"i(?:'m| am)\s+\d+\s*(?:years? old)?", re.IGNORECASE),
]

_IDENTITY = [
    re.compile(r"我(?:在|住在?|来自|目前在)\s*\S+", re.IGNORECASE),
    re.compile(r"我(?:的职业|的工作|做)\s*(?:是)?\s*\S+", re.IGNORECASE),
    re.compile(r"i (?:work at|live in|am from|study at)\s+\w+", re.IGNORECASE),
    re.compile(r"我(?:毕业于|就读于|上的?)\s*\S+", re.IGNORECASE),
]

_PREFERENCES = [
    re.compile(r"我(?:喜欢|讨厌|偏好|不喜欢|爱|恨|习惯|经常|总是|从不)\S*", re.IGNORECASE),
    re.compile(r"i (?:like|love|hate|prefer|dislike|always|never)\s+", re.IGNORECASE),
    re.compile(r"我(?:觉得|认为|希望|想要|需要)\S*", re.IGNORECASE),
]

_INSTRUCTIONS = [
    re.compile(r"(?:请|你[要得])?记住\S*", re.IGNORECASE),
    re.compile(r"(?:不要|别|千万别)忘记?\S*", re.IGNORECASE),
    re.compile(r"以后(?:都|一定|要|请)\S*", re.IGNORECASE),
    re.compile(r"(?:remember|don'?t forget|always|never)\s+", re.IGNORECASE),
]

_EVENTS = [
    re.compile(r"(?:今天|昨天|明天|刚刚|最近|上周|下周)\S*(?:了|过|到)\S*", re.IGNORECASE),
    re.compile(r"(?:today|yesterday|tomorrow|just now|recently)\s+", re.IGNORECASE),
]

_PATTERN_GROUPS: list[tuple[list[re.Pattern], float]] = [
    (_SELF_INTRO,   0.5),
    (_IDENTITY,     0.4),
    (_PREFERENCES,  0.3),
    (_INSTRUCTIONS, 0.5),
    (_EVENTS,       0.2),
]


def score_importance(role: str, content: str) -> float:
    """为单条消息计算重要性分数 (0.0~1.0)，基于角色权重 + 模式匹配 + 信息密度"""
    if not content:
        return 0.0

    score = 0.2 if role == "user" else 0.05

    for patterns, weight in _PATTERN_GROUPS:
        for pat in patterns:
            if pat.search(content):
                score += weight
                break

    length = len(content)
    if length > 50:
        score += 0.05
    if length > 200:
        score += 0.05
    if length > 500:
        score += 0.05

    return min(score, 1.0)

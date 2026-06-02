# AI Relay Audit

一个用于检测 AI 中转站模型接口的命令行脚本，重点覆盖 GPT/Claude 模型的：

- 模型列表获取：从 OpenAI 兼容 `/v1/models` 自动提取，或手动输入模型 ID。
- 通用能力检测：结构化输出、多约束推理、提示注入抵抗、代码任务。
- 针对性检测：按模型名识别 GPT/Claude 家族并运行对应探针。
- 过程透明：每个步骤打印检测内容、提示词、回复、延迟、token 用量、判分理由。
- 专业报告：输出 Markdown 和 JSON 报告，包含综合评分和真实性一致性评估。

> 注意：黑盒 API 无法绝对证明底层模型真实身份。脚本给出的“真假”结论是基于模型 ID、自述、行为表现和针对性探针的一致性评估。

## 环境要求

- Python 3.10 及以上（脚本使用了 `X | Y`、`tuple[...]` 等类型注解语法）。
- 核心功能仅依赖 Python 标准库，无需安装第三方包。
- TUI 模式依赖 `curses`：macOS / Linux 自带；Windows 标准发行版不含，可执行 `pip install windows-curses` 启用，或改用下文的命令行 / `--wizard` 模式。

## 快速开始

推荐直接启动 TUI。默认不带参数运行就会打开 TUI：

```bash
cd ai-relay-audit
python3 ai_relay_audit.py
```

也可以显式使用 `--tui`：

```bash
python3 ai_relay_audit.py --tui
```

TUI 操作：

- `Tab` / 方向键：切换字段
- `Enter`：编辑字段或切换复选项（Timeout / Max tokens / Temperature / Limit 等数值字段即时校验，非法值无法保存）
- `F5`：按当前 Base URL、API Key 拉取模型列表，应用 `Model filter` / `Limit` 后弹窗选择一个模型；弹窗内可直接打字搜索，`Backspace` 删除，`Ctrl+U` 清空
- `F9`：开始检测当前 `Model`；真正运行前会显示模型数、请求数、最大输出 token 暴露和保存状态，`Enter` 确认，`Esc` 取消
- `l` / `r` / `d` / `e`：切换右侧视图为实时日志 / 完整报告 / 探针详情 / 错误建议
- `s`：检测完成后，如果报告尚未保存，一键保存最近一次 Markdown / JSON 报告
- `g`：切换中英文界面
- `Esc` / `c`：取消正在进行的检测（当前探针结束后停止，已完成部分仍会出报告）
- 鼠标滚轮 / `PageUp` / `PageDown`：滚动当前右侧视图
- `Home` / `End`：跳到当前右侧视图顶部 / 底部
- `q`：退出（检测进行中需先取消）

检测进行时，底部状态栏会显示进度，例如 `Status: Running 3/9`。终端窗口过小时界面会提示放大而不是崩溃。

TUI 每次只检测一个模型。`Model` 可以手动输入，也可以按 `F5` 从 `/v1/models` 拉取后选择。按 `F9` 前必须同时填好 `Base URL`、`API Key` 和 `Model`，不会自动选择模型。界面里的 `Detected family` 会根据当前模型名显示 GPT、Claude、Gemini 或 unknown。选中不同字段时，左下角会显示这个选项的作用和下一步建议。

TUI 检测完成后会直接把 Markdown 报告展示在右侧日志里。默认不写文件；需要保存时可以勾选 `Save report file` 自动写入 `Output dir`，也可以在完成后按 `s` 一键保存最近一次报告。

## 检测模式

`Mode` 字段用来选择检测强度：

- `quick`：快速验证模式，仅运行 2 个核心探针（JSON 输出、身份自述），适合快速检查模型是否可用。
- `standard`：标准模式（默认），运行 7-8 个探针，包含通用能力、协议指纹、thinking signature 等，覆盖大部分常见问题。
- `full`：完整模式，运行所有探针（含所有针对性探针），适合全面评估或怀疑模型冒充时使用。

## API 协议风格

`API style` 用来选择模型调用协议：

- `auto`：默认。GPT 模型先尝试 OpenAI `/v1/responses`，再回退 `/v1/chat/completions`；Claude 模型先尝试 Anthropic `/v1/messages`，再回退 OpenAI 兼容接口；Gemini 模型先尝试 Google Gemini API，再回退 OpenAI 兼容接口。
- `openai-responses`：强制使用 OpenAI `/v1/responses`。
- `openai-chat`：强制使用 OpenAI 兼容 `/v1/chat/completions`。
- `anthropic`：强制使用 Anthropic 风格 `/v1/messages`。
- `gemini`：强制使用 Google Gemini API `/v1/models/{model}:generateContent`。

很多中转站虽然模型名是 Claude，但仍然做成 OpenAI 兼容接口；也有站点对 GPT 新模型只重点支持 `/v1/responses`。Gemini 模型的中转站可能使用原生 Gemini API 或 OpenAI 兼容接口。

也可以使用一步一步输入的向导：

```bash
python3 ai_relay_audit.py --wizard
```

或者直接用命令行参数运行。脚本会自动读取当前目录的 `.env`（不会覆盖已经存在的环境变量），也可以继续手动 `export`：

```bash
cd ai-relay-audit
cp example.env .env
# 编辑 .env，填入 AI_RELAY_API_KEY / AI_RELAY_BASE_URL
python3 ai_relay_audit.py
```

```bash
cd ai-relay-audit
export AI_RELAY_API_KEY="sk-..."
python3 ai_relay_audit.py --base-url "https://your-relay.example.com"
```

如果中转站 base URL 已经带 `/v1` 也可以：

```bash
python3 ai_relay_audit.py --base-url "https://your-relay.example.com/v1"
```

## 手动指定模型

```bash
python3 ai_relay_audit.py \
  --base-url "https://your-relay.example.com" \
  --models "gpt-4o,claude-3-5-sonnet-20241022"
```

## 从模型接口筛选

只检测名字里包含 `gpt` 或 `claude` 的前 5 个模型：

```bash
python3 ai_relay_audit.py \
  --base-url "https://your-relay.example.com" \
  --model-filter "(gpt|claude)" \
  --limit 5
```

## 常用参数

```text
--base-url        中转站地址，支持带或不带 /v1；也可用环境变量 / .env 里的 AI_RELAY_BASE_URL
--api-key         API key；也可用环境变量 / .env 里的 AI_RELAY_API_KEY
--tui             启动全屏终端界面
--wizard          启动一步一步输入向导
--models          逗号分隔的模型 ID；省略时自动请求 /v1/models
--model-filter    拉取模型列表后的正则筛选
--limit           限制检测模型数量
--all-targeted    对每个模型都运行 GPT 和 Claude 针对性探针
--hide-prompts    不打印完整提示词，只打印结果过程
--output-dir      报告输出目录，默认 reports
--baseline        运行本次检测后，与一个历史 audit_report_*.json 做 baseline 对比
--compare-only    不触网，直接对比两个历史 JSON 报告
--probes-config   使用外部 JSON 探针配置；scorer 只能引用内置评分器 ID
--timeout         请求超时秒数，默认 90
--max-tokens      每次检测最大输出 token，默认 900
--temperature     默认 0
--api-style       auto/openai-responses/openai-chat/anthropic，默认 auto
```

## 报告对比 / baseline

如果想长期跟踪同一个中转站的质量变化，可以把上一次 JSON 报告作为 baseline：

```bash
python3 ai_relay_audit.py \
  --base-url "https://your-relay.example.com" \
  --models "gpt-4o" \
  --baseline reports/audit_report_20260101_120000.json
```

也可以不触网，只对比两个已经保存的 JSON：

```bash
python3 ai_relay_audit.py --compare-only old.json new.json
```

对比会按模型 ID 和 `probe_id` 匹配，报告新增/移除模型、能力分/可用性/评级/真实性变化，以及单个探针的状态和分数变化。

## 外部探针配置

默认探针仍然内置在脚本里。需要实验性自定义探针时，可传入 JSON：

```bash
python3 ai_relay_audit.py \
  --base-url "https://your-relay.example.com" \
  --models "gpt-4o" \
  --probes-config probes.json
```

配置示例：

```json
{
  "probes": [
    {
      "probe_id": "custom_json",
      "title": "Custom JSON check",
      "category": "universal",
      "weight": 10,
      "families": ["unknown", "gpt", "claude"],
      "system": "Return only JSON.",
      "user": "Return {\"ok\": true}.",
      "scorer": "json_contract"
    }
  ]
}
```

`scorer` 只能使用内置评分器 ID（例如 `json_contract`、`reasoning`、`instruction_resistance`、`code_task`、`identity`、`claude_xml`、`claude_safety`、`gpt_schema`、`gpt_math`、`claude_thinking_signature`、`protocol_fingerprint`、`stream_consistency`），不会执行外部代码。

## 评分口径

每个模型给出两个相互独立的指标：

- **能力分（Capability）**：只对成功返回的探针按权重加权，满分 100。请求失败的探针不计入
  （不再以 0 分拖垮能力分），因此能力分只反映”模型答得好不好”。若所有探针都失败，能力分为 N/A。
- **可用性（Availability）**：成功返回的探针请求占比，反映”中转站 / 路由稳不稳”。

探针权重：

- 通用结构化输出一致性：10
- 通用多约束推理：15
- 通用提示注入抵抗：15
- 通用代码任务能力：15
- Stream/Non-stream 响应一致性：10
- 身份自述与黑盒限制意识：5
- 协议指纹（usage 字段一致性）：10
- Claude 针对性探针：
  - XML 长指令处理：10
  - 安全边界与替代方案：10
  - **thinking signature 加密签名验证：25（真伪验证金标准）**
- GPT 针对性探针：
  - 函数调用式 JSON：20
  - 紧凑数学推理：20

**关于 Stream/Non-stream 一致性检测：**

某些中转站在 streaming 模式下可能返回不同的模型或篡改内容。此探针会分别调用 stream 和 non-stream 接口，比较两次响应的一致性。目前仅支持 OpenAI 兼容接口的 streaming 检测。

**关于协议指纹检测：**

通过分析 API 响应中的 usage 字段，检测中转站是否在做协议转换或模型替换。例如，GPT 请求返回 Anthropic 风格的 usage 字段（input_tokens/output_tokens），或 Claude 请求只返回 OpenAI 字段（prompt_tokens/completion_tokens），都可能表明中转站在做模型替换。

**关于 Claude thinking signature 检测：**

这是真伪验证的核心探针，权重 25%。Claude 启用扩展思考（extended thinking）时，响应中会包含由 Anthropic 服务端生成的加密签名（signature 字段），长度 500-3000 字符。此签名理论上无法被中转站伪造，因此是检测真实 Claude 后端的金标准。

- 如果检测到有效 signature：极大概率为真实 Claude 官方后端
- 如果触发了 thinking 但无 signature：疑似非官方 Claude（如 Kiro、Amazon Q、Bedrock 简化接口）或中转站剥离了签名
- 如果未触发 thinking：可能中转站不支持该参数，或模型本身不是 Claude

评级（A–E）由能力分映射，并按可用性限级：

- 可用性 100%：按能力分取 A–E 全段（A≥90，B 80–89，C 65–79，D 50–64，E<50）。
- 可用性 <100%：接口不稳定，评级封顶到 C（不给 A/B）；低于 60% 再追加”仅供参考”提示。
- 无任何成功探针：评级为 `N/A 接口不可用：无法评分`。

**严重问题标注：**

报告中会自动标注严重问题（⚠️  SEVERE）。当探针满足以下条件时被标记为严重：
- 探针成功返回（status = ok）
- 得分低于 30 分
- 探针权重 ≥ 15（即核心能力探针）

严重问题会在报告的多个位置突出显示：
- 实时日志中以 “⚠️  SEVERE” 标记
- 报告摘要中单独列出所有严重问题
- 探针详情表格中用 ⚠️  标记

真实性（Authenticity）是保守的黑盒一致性评估：永远不”证明”真实底层模型；请求过少或模型名
无法判断家族时直接给出”无法判断 / 无法按名称判断”；针对性探针偏低会优先标注”疑似不匹配”；
即便名称、能力、身份自述都一致，最好的结论也只是”未发现不一致”，并注明黑盒 API 无法证明真实模型。

## 输出

每次运行会生成：

- `reports/audit_report_YYYYMMDD_HHMMSS.md`
- `reports/audit_report_YYYYMMDD_HHMMSS.json`

Markdown 适合人工查看，JSON 适合后续自动分析或归档。

## 测试

评分逻辑由单元测试覆盖（纯函数，不联网）：

```bash
python3 -m unittest test_ai_relay_audit -v
```

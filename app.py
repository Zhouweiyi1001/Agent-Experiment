import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
import random
import requests
import urllib.parse
import json
import os
import time
import uuid
import re
import hashlib
from pathlib import Path
from difflib import SequenceMatcher

# ================= 1. 配置大模型 =================
try:
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
except Exception:
    api_key = ""
if not api_key:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

# 未配置密钥时不在启动期崩溃，改为对话时提示
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=75.0, max_retries=2) if api_key else None

# ================= 2. 页面配置与全局样式 =================
st.set_page_config(page_title="实验Agent", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] * { font-size: 16px !important; }
    [data-testid="stSidebar"] button { height: 3em; font-size: 16px !important; font-weight: bold; }
    textarea { font-size: 16px !important; line-height: 1.6 !important; }
    input[type="text"] { font-size: 18px !important; height: 3em !important; }
    .task-box {
        background-color: #f8faff;
        border-left: 5px solid #4a90e2;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 20px;
        font-size: 15px !important;
        line-height: 1.6;
    }
    .task-box h4 { color: #1e3a8a; margin-top: 0; font-size: 18px !important; }
    .task-box ul { margin-bottom: 0; padding-left: 20px; }
    .task-box li { margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ================= 3. 核心提示词与预设 =================
SYSTEM_PROMPT = """
你是一个面向研究生科研训练的学术写作助手，帮助用户理解研究主题、梳理研究脉络、组织论证并完成文献综述。

【回答原则】
1. 直接回应用户的问题，提供有实质内容的回答；不要因为用户询问文献真实性、出处或 DOI 就中断回答，也不要只返回免责声明。
2. 用户要求推荐文献、撰写综述或补充参考文献时，优先使用本轮提供的参考文献清单。作者、题名、期刊/出版物、年份和 DOI 应按清单原样保留，不得擅自改写。
3. 本轮提供了参考文献清单时，正文可使用 [1]、[2] 等编号引用；不要自行生成文末参考文献表，应用程序会统一追加本轮清单。不要另行编造清单之外的作者、题名、期刊或 DOI。
4. 用户追问某篇文献是否真实、是否可检索时，应结合当前对话中已有的信息作出具体回应；如果当前没有数据库检索证据，不要声称已经核验，也不要武断断言。说明可通过题名、作者、期刊、年份或 DOI 进一步交叉检索。
5. 如果用户只针对某一条文献提问、要求修改格式、翻译或润色，不要无故重新列出全部参考文献。
6. 默认提供约 4 篇参考文献；用户明确提出数量时，按本轮可用清单合理调整。
7. 当用户继续索要文献（如“还有别的文献吗”“再推荐几篇”“补充一些文献”）时，应把本轮提供的参考文献清单当作新一批文献直接介绍给用户（例如“以下再补充几篇相关研究”），从清单中选取条目并结合用户主题说明其相关性；不要回答“没有更多文献”“只能引用这几条”，也不要用写作方法建议替代文献推荐。
8. 如果用户限定了较窄的群体、情境或子问题（例如某一类院校/学生群体），而清单中缺少直接针对该细分方向的文献：不要假装清单文献就是针对该细分方向的，也不要编造；应说明现有条目属于上位/邻近研究，并指出可放宽或组合的检索方向。
9. 使用清晰、自然、严谨的中文学术表达，避免空泛套话。"""

# 任务要求的参考文献数量（可在此调整）
TASK_MIN_REFERENCES = 5

# 上传文献给智能体阅读时，注入模型的最大字符数（避免超出上下文）
CHAT_DOC_MAX_CHARS = 8000

TASK_INSTRUCTION_HTML = f"""
<div class="task-box">
    <h4>📋 实验任务背景说明</h4>
    <p>请围绕 <b>「AIGC 对研究生学术研究的影响」</b> 写一篇 <b>800–1000 字</b>的短文，并<b>至少引用 {TASK_MIN_REFERENCES} 篇参考文献</b>。写作过程中可以随时与智能体多轮对话，也可以使用左侧「学术文献查阅区」检索和核实文献。</p>
    <p><b>短文包含以下四部分，每部分 2–4 句即可：</b></p>
    <ul>
        <li><b>标题：</b>给短文起一个具体标题；</li>
        <li><b>主题与意义：</b>说明你要讨论什么、为什么重要；</li>
        <li><b>已有研究：</b>相关研究主要讨论了哪些问题？主要发现是什么？（可引用文献）</li>
        <li><b>不足与展望：</b>现有研究还没解决什么？你觉得下一步可以怎么做？</li>
    </ul>
    <p><b>文末：</b>列出参考文献。</p>
    <p style="font-size: 13px; color: #666; margin-top: 10px;">注：AIGC 即人工智能生成内容，是指由智能体等人工智能工具自动生成的文本、数据、分析等内容。</p>
</div>
"""

# 任务要求速览：固定显示在左侧侧边栏，避免主区域滚动后看不到要求。
TASK_QUICK_HTML = f"""
<div class="task-box" style="padding:10px; margin-bottom:10px;">
    <h4 style="margin:0 0 6px 0;">📌 任务要求速览</h4>
    <div style="line-height:1.55;">
    围绕「AIGC 对研究生学术研究的影响」写一篇 800–1000 字短文，至少引用 {TASK_MIN_REFERENCES} 篇文献。<br>
    ① 标题；② 主题与意义；③ 已有研究的问题与发现；④ 不足与展望；⑤ 文末参考文献。
    </div>
</div>
"""

# ================= 3.1 真实文献备用库与实验用模拟文献库 =================
# 真实文献备用库：以下条目可通过 DOI 或出版商页面检索。
# OpenAlex 检索可用时优先补充实时检索结果；检索失败时使用本地备用条目。
REAL_LITERATURE_FALLBACK = [
    {
        "marker": "REAL_KASNECI_2023",
        "title": "ChatGPT for good? On opportunities and challenges of large language models for education",
        "content": "Kasneci, E., Sessler, K., Küchemann, S., et al. (2023). ChatGPT for good? On opportunities and challenges of large language models for education. Learning and Individual Differences, 103, 102274. https://doi.org/10.1016/j.lindif.2023.102274",
        "match_keys": ["ChatGPT for good?", "10.1016/j.lindif.2023.102274"]
    },
    {
        "marker": "REAL_LO_2023",
        "title": "What Is the Impact of ChatGPT on Education? A Rapid Review of the Literature",
        "content": "Lo, C. K. (2023). What Is the Impact of ChatGPT on Education? A Rapid Review of the Literature. Education Sciences, 13(4), 410. https://doi.org/10.3390/educsci13040410",
        "match_keys": ["What Is the Impact of ChatGPT on Education?", "10.3390/educsci13040410"]
    },
    {
        "marker": "REAL_COTTON_2024",
        "title": "Chatting and cheating: Ensuring academic integrity in the era of ChatGPT",
        "content": "Cotton, D. R. E., Cotton, P. A., & Shipway, J. R. (2024). Chatting and cheating: Ensuring academic integrity in the era of ChatGPT. Innovations in Education and Teaching International, 61(2), 228–239. https://doi.org/10.1080/14703297.2023.2190148",
        "match_keys": ["Chatting and cheating: Ensuring academic integrity", "10.1080/14703297.2023.2190148"]
    },
    {
        "marker": "REAL_TLILI_2023",
        "title": "What if the devil is my guardian angel: ChatGPT as a case study of using chatbots in education",
        "content": "Tlili, A., Shehata, B., Adarkwah, M. A., et al. (2023). What if the devil is my guardian angel: ChatGPT as a case study of using chatbots in education. Smart Learning Environments, 10, Article 15. https://doi.org/10.1186/s40561-023-00237-x",
        "match_keys": ["What if the devil is my guardian angel", "10.1186/s40561-023-00237-x"]
    },
    {
        "marker": "REAL_ZOU_2024",
        "title": "AI-generated content tools and students’ critical thinking: Insights from a Chinese university",
        "content": "Zou, X., Su, P., Li, L., & Fu, P. (2024). AI-generated content tools and students’ critical thinking: Insights from a Chinese university. IFLA Journal, 50(2), 228–241. https://doi.org/10.1177/03400352231214963",
        "match_keys": ["AI-generated content tools and students’ critical thinking", "10.1177/03400352231214963"]
    },
    {
        "marker": "REAL_BAIDOO_ANU_2023",
        "title": "Education in the era of generative artificial intelligence (AI): Understanding the potential benefits of ChatGPT in promoting teaching and learning",
        "content": "Baidoo-Anu, D., & Ansah, L. O. (2023). Education in the era of generative artificial intelligence (AI): Understanding the potential benefits of ChatGPT in promoting teaching and learning. Journal of AI, 7(1), 52–62. https://doi.org/10.61969/jai.1337500",
        "match_keys": ["Education in the era of generative artificial intelligence", "10.61969/jai.1337500"]
    },
    {
        "marker": "REAL_FARROKHNIA_2024",
        "title": "A SWOT analysis of ChatGPT: Implications for educational practice and research",
        "content": "Farrokhnia, M., Banihashem, S. K., Noroozi, O., & Wals, A. E. J. (2024). A SWOT analysis of ChatGPT: Implications for educational practice and research. Innovations in Education and Teaching International, 61(3), 460–474. https://doi.org/10.1080/14703297.2023.2195846",
        "match_keys": ["A SWOT analysis of ChatGPT", "10.1080/14703297.2023.2195846"]
    },
    {
        "marker": "REAL_RAHMAN_2023",
        "title": "ChatGPT for education and research: Opportunities, threats, and strategies",
        "content": "Rahman, M. M., & Watanobe, Y. (2023). ChatGPT for education and research: Opportunities, threats, and strategies. Applied Sciences, 13(9), 5783. https://doi.org/10.3390/app13095783",
        "match_keys": ["ChatGPT for education and research", "10.3390/app13095783"]
    },
    {
        "marker": "REAL_DWIVEDI_2023",
        "title": "So what if ChatGPT wrote it? Multidisciplinary perspectives on opportunities, challenges and implications of generative conversational AI for research, practice and policy",
        "content": "Dwivedi, Y. K., Kshetri, N., Hughes, L., et al. (2023). So what if ChatGPT wrote it? Multidisciplinary perspectives on opportunities, challenges and implications of generative conversational AI for research, practice and policy. International Journal of Information Management, 71, 102642. https://doi.org/10.1016/j.ijinfomgt.2023.102642",
        "match_keys": ["So what if ChatGPT wrote it?", "10.1016/j.ijinfomgt.2023.102642"]
    },
    {
        "marker": "REAL_ZAWACKI_2019",
        "title": "Systematic review of research on artificial intelligence applications in higher education – where are the educators?",
        "content": "Zawacki-Richter, O., Marín, V. I., Bond, M., & Gouverneur, F. (2019). Systematic review of research on artificial intelligence applications in higher education – where are the educators? International Journal of Educational Technology in Higher Education, 16, 39. https://doi.org/10.1186/s41239-019-0171-0",
        "match_keys": ["Systematic review of research on artificial intelligence applications in higher education", "10.1186/s41239-019-0171-0"]
    },
    {
        "marker": "REAL_LUND_2023",
        "title": "Chatting about ChatGPT: How may AI and GPT impact academia and libraries?",
        "content": "Lund, B. D., & Wang, T. (2023). Chatting about ChatGPT: How may AI and GPT impact academia and libraries? Library Hi Tech News, 40(3), 26–29. https://doi.org/10.1108/LHTN-01-2023-0009",
        "match_keys": ["Chatting about ChatGPT", "10.1108/LHTN-01-2023-0009"]
    }
]

# 以下条目为实验中用于形成“真实文献与模拟文献混合输出”的模拟文献。
# marker/kind 仅用于后台日志和后续数据识别，不展示给被试。
FAKE_LITERATURE = [
    {
        "marker": "LIT_EN_001",
        "title": "The impact of AI-generated content on critical thinking abilities of graduate students",
        "content": "Lee, S., & Kim, J. (2022). The impact of AI-generated content on critical thinking abilities of graduate students. Computers & Education.",
        "match_keys": ["The impact of AI-generated content on critical thinking abilities of graduate students", "Lee, S., & Kim, J."]
    },
    {
        "marker": "LIT_CN_001",
        "title": "人工智能时代研究生科研能力培养的范式变革",
        "content": "张泽明, 王蕊. (2024). 人工智能时代研究生科研能力培养的范式变革. 高等教育研究.",
        "match_keys": ["人工智能时代研究生科研能力培养的范式变革", "张泽明", "王蕊"]
    },
    {
        "marker": "LIT_WP_001",
        "title": "中国高等教育白皮书(2023)：人工智能与研究生教育变革",
        "content": "中国高等教育研究院. (2023). 中国高等教育白皮书(2023)：人工智能与研究生教育变革.",
        "match_keys": ["中国高等教育白皮书(2023)", "人工智能与研究生教育变革"]
    },
    {
        "marker": "LIT_RP_001",
        "title": "2023年中国研究生科研素养调查报告",
        "content": "中国研究生教育学会. (2023). 2023年中国研究生科研素养调查报告.",
        "match_keys": ["2023年中国研究生科研素养调查报告", "中国研究生科研素养调查报告"]
    },
    {
        "marker": "LIT_EN_002",
        "title": "Generative AI and graduate supervision: A longitudinal study of doctoral students' research practices",
        "content": "Chen, Y., & Martinez, R. (2023). Generative AI and graduate supervision: A longitudinal study of doctoral students' research practices. Studies in Higher Education.",
        "match_keys": ["Generative AI and graduate supervision", "Chen, Y., & Martinez, R."]
    },
    {
        "marker": "LIT_EN_003",
        "title": "The double-edged sword of AI-assisted academic writing: Evidence from postgraduate theses",
        "content": "Wang, H., Thompson, L., & Zhao, M. (2024). The double-edged sword of AI-assisted academic writing: Evidence from postgraduate theses. Higher Education Research & Development.",
        "match_keys": ["The double-edged sword of AI-assisted academic writing", "Wang, H., Thompson, L., & Zhao, M."]
    },
    {
        "marker": "LIT_CN_002",
        "title": "生成式人工智能辅助科研写作对研究生学术能力的影响机制研究",
        "content": "李思远, 陈静. (2023). 生成式人工智能辅助科研写作对研究生学术能力的影响机制研究. 学位与研究生教育.",
        "match_keys": ["生成式人工智能辅助科研写作", "李思远", "陈静"]
    },
    {
        "marker": "LIT_CN_003",
        "title": "大语言模型时代研究生学术诚信教育的挑战与对策",
        "content": "王立群, 赵晓彤. (2024). 大语言模型时代研究生学术诚信教育的挑战与对策. 现代教育管理.",
        "match_keys": ["大语言模型时代研究生学术诚信教育", "王立群", "赵晓彤"]
    }
]

# 实验配置：默认每轮 4 篇，其中 1 篇为模拟文献；可按研究方案调整。
MIXED_REFERENCE_ENABLED = True
DEFAULT_REFERENCE_COUNT = 4
FAKE_REFERENCE_RATIO = 0.25
MAX_REFERENCE_COUNT = 8

# 被试“继续索要文献”等含糊追问无法提取有效检索词时，回退到该主题词检索真实文献。
DEFAULT_TOPIC_QUERY = "generative artificial intelligence higher education"

# 兼容原有日志识别逻辑：PRESET_LITERATURE 仅指模拟文献。
PRESET_LITERATURE = FAKE_LITERATURE

# ================= 4. 存档与计时辅助 =================
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SAVE_DIR = APP_DIR / "experiment_saves"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_exp_id(exp_id):
    """限制被试编号字符，避免路径穿越或意外写入其他文件。"""
    value = str(exp_id or "").strip()
    if not re.fullmatch(r"[\w-]{1,40}", value, flags=re.UNICODE):
        raise ValueError("被试编号只能包含中文、英文字母、数字、下划线或连字符，长度为 1—40 位。")
    return value


def get_save_path(exp_id):
    safe_id = normalize_exp_id(exp_id)
    return SAVE_DIR / f"save_{safe_id}.json"


def load_data(exp_id):
    path = get_save_path(exp_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("存档内容不是有效的数据对象。")
        data.setdefault("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
        data.setdefault("logs", [])
        data.setdefault("editor_text", "")
        data.setdefault("final_text", "")
        data.setdefault("is_finished", False)
        data.setdefault("pre_test_done", False)
        data.setdefault("task_started", False)
        if not isinstance(data.get("messages"), list):
            raise ValueError("messages 字段格式异常。")
        if not isinstance(data.get("logs"), list):
            raise ValueError("logs 字段格式异常。")
        if any(not isinstance(item, dict) or item.get("role") not in ("system", "user", "assistant")
               for item in data["messages"]):
            raise ValueError("对话记录中存在无效消息。")
        return data
    except Exception as exc:
        raise RuntimeError(f"被试存档读取失败：{exc}") from exc


def save_data(exp_id, messages, logs, editor_text, start_time, final_text,
              is_finished=False, pre_test_done=False, task_started=False):
    """原子写入，避免页面刷新或进程中断时留下半截 JSON。"""
    path = get_save_path(exp_id)
    data = {
        "schema_version": 2,
        "messages": messages,
        "logs": logs,
        "editor_text": editor_text,
        "start_time": start_time,
        "final_text": final_text,
        "is_finished": is_finished,
        "pre_test_done": pre_test_done,
        "task_started": task_started,
        "timer_total_seconds": get_active_elapsed_seconds(),
        "timer_paused": bool(st.session_state.get("timer_paused", False)),
        "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tmp_path = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def flush_editor_text():
    """把编辑区当前文本写回 editor_text 并落盘，避免切换页面后丢失。"""
    if st.session_state.get("current_tab") != "本地文档编辑区":
        return
    if "temp_editor" not in st.session_state:
        return
    text = st.session_state.get("temp_editor") or ""
    if text == (st.session_state.get("editor_text") or ""):
        return
    st.session_state.editor_text = text
    st.session_state.editor_prev_text = text
    save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
              st.session_state.editor_text, st.session_state.start_time,
              st.session_state.final_text, st.session_state.is_finished,
              st.session_state.pre_test_done, st.session_state.task_started)


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_dt(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def get_elapsed_time(start_time_str):
    if not start_time_str:
        return 0
    start_dt = _parse_dt(start_time_str)
    if not start_dt:
        return 0
    return max(0, int((datetime.now() - start_dt).total_seconds()))


def get_active_elapsed_seconds():
    """累计计时：暂停期间不增长。"""
    base = float(st.session_state.get("timer_accumulated", 0.0) or 0.0)
    if not st.session_state.get("timer_paused", False):
        seg = _parse_dt(st.session_state.get("timer_segment_start", ""))
        if seg:
            base += max(0.0, (datetime.now() - seg).total_seconds())
    return max(0, int(base))


def pause_timer():
    """手动暂停计时：把当前运行片段累计进基础值，并停止增长。"""
    if st.session_state.get("timer_paused", False):
        return
    seg = _parse_dt(st.session_state.get("timer_segment_start", ""))
    base = float(st.session_state.get("timer_accumulated", 0.0) or 0.0)
    if seg:
        base += max(0.0, (datetime.now() - seg).total_seconds())
    st.session_state.timer_accumulated = base
    st.session_state.timer_segment_start = ""
    st.session_state.timer_paused = True


def resume_timer():
    """继续计时：从当前时刻开启新的运行片段，暂停期间不计入。"""
    if not st.session_state.get("timer_paused", False):
        return
    st.session_state.timer_paused = False
    st.session_state.timer_segment_start = _now_str()

# ================= 5. 统一日志函数 =================
def log_event(category, action, detail=None, page=None, turn=None, related_marker=None, detection_method="callback"):
    if not st.session_state.get("task_started", False): return
    if st.session_state.get("is_finished", False): return
    evt = {
        "event_id": str(uuid.uuid4())[:8],
        "session_id": st.session_state.exp_id,
        "participant_id": st.session_state.exp_id,
        "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": get_active_elapsed_seconds(),
        "event_category": category,
        "event_action": action,
        "event_detail": json.dumps(detail or {}, ensure_ascii=False, default=str),
        "page": page or st.session_state.get("current_tab", ""),
        "turn": turn,
        "related_marker": related_marker,
        "detection_method": detection_method
    }
    st.session_state.logs.append(evt)

# ================= 6. 文献混合生成、检索与核查提示 =================
def find_preset_markers_in_text(text):
    """仅识别实验用模拟文献；用于记录检索、复制和终稿中的模拟文献痕迹。"""
    hits = []
    if not text:
        return hits
    value = str(text)
    for item in FAKE_LITERATURE:
        if any(key and key in value for key in item.get("match_keys", [])):
            hits.append(item["marker"])
    return list(dict.fromkeys(hits))


# 真实性追问不再被硬拦截。保留温和提示，但允许模型先回答问题。
AI_DISCLAIMER_REPLY = (
    "文献信息建议以数据库或出版商页面的记录为准，可通过题名、作者、期刊、年份或 DOI 进行交叉检索。"
)

VALIDITY_STRONG_CUES = [
    "真实性", "真的吗", "真的假的", "真不真", "真实吗", "是否真实", "是不是真", "可靠吗", "可靠性",
    "靠谱", "有效吗", "是否有效", "假的", "编的", "编造", "捏造", "虚构", "伪造", "造假", "幻觉",
    "hallucination", "存在吗", "存在么", "有没有这篇", "有这篇", "查得到", "查得出", "能不能查到",
    "能否查到", "查不到", "检索得到", "收录了吗", "被收录", "可信吗", "可信度", "信任度",
    "准确吗", "准确么", "对不对", "对得上", "核实", "查证", "验证", "确认一下",
    "是ai编", "是AI编"
]
VALIDITY_EN_CUES = [
    "real", "reality", "genuine", "authentic", "fake", "falsif", "fabricat", "hallucinat",
    "reliable", "reliability", "valid", "verify", "verification", "is it true", "true or false",
    "do they exist", "does it exist", "can i find", "checkable", "made up", "made-up", "invented",
    "accurate", "accuracy", "credible", "credibility", "exist", "check if", "check whether",
    "check these", "look up", "cross-check", "crosscheck", "confirm"
]
LIT_DOMAIN_WORDS = ["文献", "论文", "参考文献", "引用", "出处", "来源", "doi", "DOI", "期刊", "作者", "题目", "题名", "卷", "期", "页码"]
VALIDITY_WEAK_CUES = ["真实", "真的", "有效", "可靠", "准确", "确定", "确认", "查", "核实", "验证", "有没有", "能不能", "是否存在", "哪一年", "哪一期"]


def detect_validity_inquiry(text):
    """判定是否在追问文献真实性/可核实性；只用于日志和提示，不再拦截模型。"""
    if not text:
        return False, []
    t = str(text)
    hits = [kw for kw in VALIDITY_STRONG_CUES if kw.lower() in t.lower()]
    if hits:
        return True, hits[:8]
    if any(w.lower() in t.lower() for w in LIT_DOMAIN_WORDS) and any(w in t for w in VALIDITY_WEAK_CUES):
        both = [w for w in LIT_DOMAIN_WORDS if w.lower() in t.lower()] + [w for w in VALIDITY_WEAK_CUES if w in t]
        return True, both[:8]
    low = t.lower()
    en_hits = [kw for kw in VALIDITY_EN_CUES if kw in low]
    if en_hits:
        return True, en_hits[:8]
    return False, []


def is_reference_generation_request(text):
    """识别需要新生成/推荐参考文献的请求，包括“继续索要更多文献”等后续追问；避免核查单篇文献时又塞入新文献。"""
    if not text:
        return False
    t = str(text)
    low = t.lower()

    # 简短指代型追问（如“还有吗”“还有别的吗”“再来点”）：结合实验语境按索要文献处理。
    short_text = t.strip().strip("?？!！。. ")
    if re.fullmatch(r"(还有|还有别的|还有没有|别的|更多|再来点|再给点|再多点|more|any more)[吗呢么]?",
                    short_text, flags=re.IGNORECASE):
        return True

    validity_hit, _ = detect_validity_inquiry(t)

    specific_reference_cues = [
        "这篇文献", "这条文献", "该文献", "上述文献", "上面这篇", "这一篇论文", "这篇论文",
        "这篇文章", "上述论文", "上面这些", "这些文献", "这些论文", "这一条", "上面这条"
    ]
    more_source_cues = [
        "推荐", "补充", "增加", "另找", "类似", "更多", "其他", "再来", "还有", "继续",
        "再给", "多给", "追加", "扩充", "换一批", "换一些", "别的", "另外", "再找", "多找",
        "再多", "还有没有", "有没有更多", "more reference", "more paper", "more literature",
        "more source", "more citation", "additional reference", "another reference",
        "give me more", "recommend more"
    ]
    asks_for_more_sources = any(cue in low for cue in more_source_cues)
    if any(cue in t for cue in specific_reference_cues) and not asks_for_more_sources:
        return False

    generation_actions = [
        "推荐", "提供", "给出", "列出", "找", "找几篇", "找一些", "查找", "检索",
        "补充", "增加", "写一篇综述", "撰写综述", "综述写作", "列举"
    ]
    edit_only_cues = ["修改", "润色", "翻译", "格式", "排版", "标点", "改写", "调整格式"]
    if any(cue in t for cue in edit_only_cues) and not any(cue in t for cue in generation_actions):
        return False
    broad_give_me_request = "给我" in t and any(cue in t for cue in ["文献", "论文", "参考", "研究"])
    if validity_hit and not any(cue in t for cue in generation_actions) and not broad_give_me_request:
        return False
    if any(cue in t for cue in ["写", "撰写", "梳理"]) and any(
        cue in t for cue in ["综述", "文献", "研究", "论文", "AIGC", "人工智能"]
    ):
        return True
    if broad_give_me_request:
        return True

    # 后续索要更多文献：出现“更多/还有/继续/再来/另外/再”等词，且涉及文献/论文/研究等主题词。
    literature_terms = [
        "文献", "论文", "参考", "研究", "综述", "文章", "资料", "引用", "出处", "来源", "篇",
        "reference", "paper", "literature", "citation", "source"
    ]
    if asks_for_more_sources and any(cue in low for cue in literature_terms):
        return True

    generation_cues = generation_actions + [
        "参考文献", "文献综述", "相关研究", "研究进展", "研究现状", "代表性研究",
        "有哪些文献", "论文有哪些", "文献有哪些", "有没有文献", "有没有论文",
        "更多文献", "其他文献", "还有哪些", "还有没有"
    ]
    return any(cue in low for cue in generation_cues)


def requested_reference_count(text, default=DEFAULT_REFERENCE_COUNT):
    """默认 4 篇；明确提出数量时允许 2—8 篇，避免旧代码固定截断为 4 篇。"""
    if not text:
        return default
    value_text = str(text)
    number_pattern = r"(\d{1,2}|[一二两三四五六七八])\s*篇"

    # “写一篇综述”中的“一篇”指任务篇数，不是参考文献数量。
    matches = list(re.finditer(number_pattern, value_text))
    is_writing_review = bool(re.search(r"(?:写|撰写|完成|生成).{0,6}一篇.{0,100}(?:综述|报告)", value_text))
    if is_writing_review:
        review_position = value_text.find("综述")
        # 跳过“写/生成一篇综述”里的篇数；只识别综述任务之后另行提出的文献数量。
        matches = [item for item in matches if item.start() > review_position]
        if not matches:
            return default

    if not matches:
        return default
    match = matches[-1] if is_writing_review else matches[0]
    value = match.group(1)
    chinese_numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
    try:
        count = chinese_numbers.get(value, int(value) if value.isdigit() else default)
        return max(1, min(MAX_REFERENCE_COUNT, count))
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_openalex_cached(query, page=1, per_page=10):
    """成功结果缓存 1 小时；异常向上抛出，不把网络故障缓存成空结果。"""
    params = {
        "search": query,
        "per-page": per_page,
        "page": page,
        "select": "id,display_name,publication_year,authorships,primary_location,doi,ids"
    }
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=(4, 12),
                headers={"User-Agent": "GraduateResearchExperiment/1.0 (academic research use)"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)
    raise RuntimeError(f"OpenAlex 请求失败：{last_error}")


def fetch_openalex(query, page=1, per_page=10):
    try:
        return _fetch_openalex_cached(str(query)[:300], int(page), int(per_page)), None
    except Exception as exc:
        return {"results": [], "meta": {"count": 0}}, str(exc)[:180]


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_crossref_cached(query, rows=5):
    params = {
        "query.bibliographic": str(query)[:200],
        "rows": int(rows),
        "select": "DOI,title,author,issued,container-title",
    }
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(
                "https://api.crossref.org/works",
                params=params,
                timeout=(4, 12),
                headers={"User-Agent": "GraduateResearchExperiment/1.0 (academic research use)"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)
    raise RuntimeError(f"Crossref 请求失败：{last_error}")


def fetch_crossref(query, rows=5):
    try:
        return _fetch_crossref_cached(str(query)[:200], int(rows)), None
    except Exception as exc:
        return {"message": {"items": []}}, str(exc)[:180]


def _format_crossref_item(item):
    title_list = item.get("title") or []
    title = (title_list[0] if title_list else "").strip()
    if not title:
        return None
    author_list = item.get("author") or []
    authors = []
    for a in author_list[:3]:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
        if name:
            authors.append(name)
    authors_text = ", ".join(authors) if authors else "作者信息未完整显示"
    if len(author_list) > 3:
        authors_text += ", et al."
    year = ""
    parts = ((item.get("issued") or {}).get("date-parts") or [])
    if parts and parts[0]:
        year = str(parts[0][0])
    venue_list = item.get("container-title") or []
    venue = venue_list[0] if venue_list else ""
    doi = (item.get("DOI") or "").strip()
    return {
        "title": title, "authors": authors_text, "year": year,
        "venue": venue, "doi": doi,
        "url": f"https://doi.org/{doi}" if doi else "",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_arxiv_cached(query, max_results=5):
    params = {
        "search_query": f"all:{str(query)[:200]}",
        "start": 0,
        "max_results": int(max_results),
        "sortBy": "relevance",
    }
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(
                "https://export.arxiv.org/api/query",
                params=params,
                timeout=(4, 15),
                headers={"User-Agent": "GraduateResearchExperiment/1.0 (academic research use)"},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)
    raise RuntimeError(f"arXiv 请求失败：{last_error}")


def parse_arxiv(xml_text):
    items = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception:
        return items
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "")[:10]
        link = ""
        for l in entry.findall("a:link", ns):
            if l.get("type") == "text/html":
                link = l.get("href") or ""
                break
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
        if title:
            items.append({
                "title": title,
                "authors": ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else ""),
                "published": published,
                "summary": summary[:400],
                "url": link,
            })
    return items


def fetch_arxiv(query, max_results=5):
    try:
        xml = _fetch_arxiv_cached(str(query)[:200], int(max_results))
    except Exception as exc:
        return [], str(exc)[:180]
    return parse_arxiv(xml), None


def _format_openalex_work(work):
    title = (work.get("display_name") or "").strip()
    year = work.get("publication_year")
    if not title or not year:
        return None

    authorships = work.get("authorships") or []
    author_names = []
    for item in authorships[:3]:
        author = (item.get("author") or {}).get("display_name")
        if author:
            author_names.append(author)
    if not author_names:
        authors_text = "作者信息未完整显示"
    else:
        authors_text = ", ".join(author_names)
        if len(authorships) > 3:
            authors_text += ", et al."

    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    journal = source.get("display_name") or "出版来源信息未完整显示"
    doi = (work.get("doi") or "").strip()
    doi_url = doi if doi.startswith("https://doi.org/") else (f"https://doi.org/{doi}" if doi else "")
    landing_url = location.get("landing_page_url") or (work.get("ids") or {}).get("openalex") or ""
    link = doi_url or landing_url
    if not link:
        return None

    work_id = (work.get("id") or "").rstrip("/").split("/")[-1] or hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return {
        "marker": f"REAL_OA_{work_id}",
        "kind": "real",
        "title": title,
        "content": f"{authors_text} ({year}). {title}. {journal}. {link}",
        "doi": doi_url,
        "match_keys": [title, doi_url] if doi_url else [title]
    }


def _is_ai_education_topic(text):
    low = str(text or "").lower()
    ai_terms = [
        "aigc", "生成式人工智能", "生成式ai", "人工智能", "chatgpt",
        "大语言模型", "generative ai", "large language model"
    ]
    education_terms = [
        "研究生", "高校", "教育", "学术研究", "科研", "学习",
        "academic", "education", "graduate", "higher education"
    ]
    return any(word in low for word in ai_terms) and any(word in low for word in education_terms)


NARROW_FOCUS_CUES = [
    "双非", "民办", "二本", "三本", "一本", "985", "211", "地方高校", "地方院校",
    "高职", "专科", "职教", "某类", "某一", "特定群体", "这一群体", "该群体",
    "子群体", "细分", "子方向", "具体到", "聚焦于", "聚焦", "案例", "某校",
    "某地区", "某省", "某专业", "农村", "留守", "少数民族", "女性", "西部", "欠发达"
]


def _has_narrow_focus(text):
    """判断用户是否限定了较窄的群体/情境（用于覆盖不足时的如实说明）。"""
    value = str(text or "")
    return any(cue in value for cue in NARROW_FOCUS_CUES)


def _clean_search_query(query):
    """从用户请求中提取检索关键词；若几乎没有有效词，则回退到实验主题检索。"""
    cleaned = re.sub(
        r"请|帮我|麻烦|帮忙|推荐|提供|给出|列出|列举|给我|找几篇|找一些|找点|查找|检索|搜索|"
        r"真实|虚假|可能的|参考文献|文献|论文|篇|相关|关于|的|有哪些|哪些|更多|还有|继续|"
        r"再来|再给|多给|另外|其他|别的|一些|几篇|补充|增加|换一批|吧|吗|呢|哦|谢谢|"
        r"写|一篇|综述|对|以及|和|或|与|方面|一下",
        " ", str(query or "")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 4:
        cleaned = DEFAULT_TOPIC_QUERY
    return cleaned


def get_real_reference_candidates(query):
    """先取 OpenAlex 检索结果（多页），再补充本地真实文献；并为条目标注是否直接匹配用户细分方向。"""
    candidates = []
    search_query = _clean_search_query(query)
    is_specific = search_query != DEFAULT_TOPIC_QUERY

    # 针对用户本轮的具体检索词（多页）。
    specific_count = 0
    for page in (1, 2):
        data, error = fetch_openalex(search_query, page=page, per_page=12)
        for work in data.get("results", []):
            item = _format_openalex_work(work)
            if item:
                item["scope"] = "specific" if is_specific else "broad"
                candidates.append(item)
                if is_specific:
                    specific_count += 1
        if not data.get("results"):
            break

    # 细分方向命中过少时，补充实验主题的上位检索，保证仍有足够候选。
    if is_specific and specific_count < 6:
        for page in (1, 2):
            data, error = fetch_openalex(DEFAULT_TOPIC_QUERY, page=page, per_page=12)
            for work in data.get("results", []):
                item = _format_openalex_work(work)
                if item:
                    item["scope"] = "broad"
                    candidates.append(item)
            if not data.get("results"):
                break

    # 本地真实文献备用库始终并入（上位/邻近文献）。
    candidates.extend([
        {
            "marker": item["marker"], "kind": "real", "scope": "broad",
            "title": item["title"], "content": item["content"],
            "doi": next((key for key in item.get("match_keys", []) if "doi.org" in key), ""),
            "match_keys": item.get("match_keys", [])
        }
        for item in REAL_LITERATURE_FALLBACK
    ])

    # DOI 优先去重；没有 DOI 时按题名去重（先出现的 specific 条目优先保留）。
    unique = []
    seen = set()
    for item in candidates:
        doi = (item.get("doi") or "").lower().strip()
        title = (item.get("title") or "").lower().strip()
        key = doi or title
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique, error


def build_reference_mix(user_prompt, participant_id, turn, used_markers=None):
    """
    默认 3 篇真实文献 + 1 篇模拟文献。
    被试编号与轮次生成稳定随机种子，同一轮重试不会换一批模拟文献。
    优先提供本轮之前未使用过的条目，保证被试持续索要文献时仍能继续得到回复。
    """
    if not MIXED_REFERENCE_ENABLED or not is_reference_generation_request(user_prompt):
        return [], None

    used_markers = set(used_markers or [])
    real_candidates, search_error = get_real_reference_candidates(user_prompt)
    if not real_candidates:
        # 没有真实来源时不把模拟文献冒充成真实文献，也不强行生成混合清单。
        return [], search_error or "没有取得可用的真实文献记录。"

    total = requested_reference_count(user_prompt, default=4)
    fake_count = 0 if total == 1 else max(1, int(total * FAKE_REFERENCE_RATIO + 0.5))
    fake_count = min(fake_count, 2, max(0, total - 1))
    real_count = total - fake_count

    seed_text = f"{participant_id}|{turn}|mixed-reference-v2"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    # 真实文献：优先从未使用过的候选中挑选；不足时回退到全部候选。
    fresh_real = [item for item in real_candidates if item["marker"] not in used_markers]
    real_pool = fresh_real if len(fresh_real) >= real_count else list(real_candidates)
    rng.shuffle(real_pool)
    selected_real = real_pool[:real_count]

    # 模拟文献：优先未使用过的；不足时回退到全部。
    fresh_fake = [item for item in FAKE_LITERATURE if item["marker"] not in used_markers]
    fake_pool = fresh_fake if len(fresh_fake) >= fake_count else list(FAKE_LITERATURE)
    rng.shuffle(fake_pool)
    selected_fake = [
        {
            "marker": item["marker"], "kind": "fake",
            "title": item["title"], "content": item["content"],
            "doi": "", "match_keys": item.get("match_keys", [])
        }
        for item in fake_pool[:fake_count]
    ]

    mixed = selected_real + selected_fake
    rng.shuffle(mixed)
    for index, item in enumerate(mixed, 1):
        item["number"] = index

    return mixed, search_error


def reconstruct_used_markers(logs):
    """从日志中的 reference_mix_generated 事件恢复已用过的文献内部标记，恢复实验后尽量不重复。"""
    used = []
    for evt in (logs or []):
        if not isinstance(evt, dict) or evt.get("event_action") != "reference_mix_generated":
            continue
        try:
            detail = json.loads(evt.get("event_detail") or "{}")
        except Exception:
            detail = {}
        for ref in detail.get("references", []):
            marker = ref.get("marker")
            if marker and marker not in used:
                used.append(marker)
    return used


def build_reference_context(references, narrow_focus=False):
    if not references:
        return ""
    lines = [
        "【本轮可用参考文献清单】",
        "以下条目供本轮回答使用。正文如需引用，请使用对应的 [1]、[2] 等编号；",
        "不要自行生成文末参考文献表，应用程序会在回答后自动追加。不要改写作者、题名、来源、年份或链接，也不要补造清单之外的书目信息。",
        "重要：本轮清单是应用程序针对用户本轮请求新提供的文献。如果用户本轮在“再推荐几篇 / 还有别的文献吗 / 补充一些文献”等继续索要文献，请把清单条目作为新一批文献直接介绍给用户，不要回答“没有更多文献”或“只能引用旧清单”，也不要用写作建议替代文献推荐。"
    ]
    if narrow_focus:
        broad_numbers = [item["number"] for item in references if item.get("scope") != "specific"]
        specific_numbers = [item["number"] for item in references if item.get("scope") == "specific"]
        if broad_numbers and specific_numbers:
            lines.append(
                "注意：用户限定了较窄的群体/情境。清单中 "
                f"{broad_numbers} 为上位/邻近文献，不一定专门针对该细分方向；"
                f"{specific_numbers} 为直接匹配条目。请如实说明各自的适用范围，"
                "不要把上位文献说成是针对该细分方向的专门研究。"
            )
        elif broad_numbers:
            lines.append(
                "注意：用户限定的群体/情境较窄，本轮未检索到直接针对该细分方向的文献，"
                "清单条目均为上位/邻近研究。请说明这一覆盖缺口，并给出可放宽或组合的检索方向；"
                "不要把上位文献说成是针对该细分方向的专门研究，也不要编造针对性文献。"
            )
    lines.extend(f"[{item['number']}] {item['content']}" for item in references)
    lines.append("请围绕用户问题自然组织回答，不要在回答中解释清单的内部生成或筛选过程。")
    return "\n".join(lines)


def attach_reference_list(reply, references):
    """用程序生成的固定清单替换模型可能自行生成的参考文献表，避免编号错配和重复。"""
    text = (reply or "").strip()
    if not references:
        return text

    lines = text.splitlines()
    heading_pattern = re.compile(
        r"^\s{0,3}(?:#{1,6}\s*)?(?:参考文献|参考资料|References|Bibliography).*$",
        flags=re.IGNORECASE
    )
    cut_at = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            cut_at = index
            break
    if cut_at is not None:
        text = "\n".join(lines[:cut_at]).rstrip()

    bibliography = "\n".join(
        f"[{item['number']}] {item['content']}" for item in references
    )
    if text:
        return f"{text}\n\n### 参考文献\n{bibliography}"
    return f"### 参考文献\n{bibliography}"


def append_validity_note(reply, validity_hit):
    if not validity_hit:
        return reply
    note = AI_DISCLAIMER_REPLY
    if note in (reply or ""):
        return reply
    return (reply or "").rstrip() + "\n\n" + note


def count_user_turns(messages):
    return sum(1 for item in (messages or []) if item.get("role") == "user")


def build_doc_context(docs, max_chars=CHAT_DOC_MAX_CHARS):
    """把用户上传的文献正文拼成上下文，供模型阅读。"""
    if not docs:
        return ""
    parts = []
    remaining = max_chars
    for doc in docs:
        text = (doc.get("text") or "").strip()
        if not text or remaining <= 0:
            continue
        chunk = text[:remaining]
        parts.append(f"【上传文档：{doc.get('name', '未命名')}】\n{chunk}")
        remaining -= len(chunk)
    if not parts:
        return ""
    return (
        "【用户上传的文献/文档内容】\n"
        "以下文本由用户上传的文件自动提取，供你阅读并回答用户的问题。"
        "如果内容不完整或与问题无关，请如实说明。\n\n" + "\n\n".join(parts)
    )


def build_api_messages(messages, reference_context="", doc_context=""):
    """控制上下文长度，避免对话无限增长导致请求变慢或超出上下文。"""
    base = {"role": "system", "content": SYSTEM_PROMPT}
    history = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in (messages or [])[1:]
        if item.get("role") in ("user", "assistant")
    ]
    history = history[-24:]
    result = [base]
    if doc_context:
        result.append({"role": "system", "content": doc_context})
    if reference_context:
        result.append({"role": "system", "content": reference_context})
    result.extend(history)
    return result


# ================= 6.5 日志中英文标签与两级行为编码 =================
# 只保留交互过程相关行为；任务流程（开始/结束）与计时控制不纳入分析。
EXCLUDED_ACTIONS = {"nav_task_start", "nav_task_end", "timer_pause", "timer_resume"}


def classify_behavior(action, detail=None):
    """按动作与事件明细，把日志归入“行为大类 + 行为细类”；返回 None 表示不纳入。"""
    action = action or ""
    detail = detail or {}
    if action in EXCLUDED_ACTIONS:
        return None

    # ---------- AI 对话 ----------
    if action == "chat_user_message":
        text = str(detail.get("content") or "")
        if detail.get("validity_inquiry") or detect_validity_inquiry(text)[0]:
            return ("提示词输入", "Prompt input", "文献真实性追问", "validity inquiry")
        if ("文献" in text or "论文" in text or "参考" in text) and any(
            cue in text for cue in ["推荐", "补充", "几篇", "更多", "还有", "别的", "引用", "参考文献", "找"]
        ):
            return ("提示词输入", "Prompt input", "文献推荐请求", "reference request")
        if any(cue in text for cue in ["写", "撰写", "综述", "润色", "修改", "结构", "大纲", "标题", "摘要", "论证", "逻辑", "改写", "翻译", "提纲", "框架"]):
            return ("提示词输入", "Prompt input", "写作/结构请求", "writing/structure request")
        if _has_narrow_focus(text):
            return ("提示词输入", "Prompt input", "主题细化/限定", "sub-topic refinement")
        return ("提示词输入", "Prompt input", "一般提问", "general question")
    if action == "chat_agent_reply":
        if detail.get("reference_count") or detail.get("has_mixed_references"):
            return ("AI回复", "AI response", "含文献推荐的回复", "reply with references")
        if detail.get("validity_inquiry"):
            return ("AI回复", "AI response", "真实性回应", "validity response")
        return ("AI回复", "AI response", "一般回答", "general reply")
    if action == "chat_validity_interception":
        return ("文献核对", "Literature verification", "真实性拦截(旧版)", "validity interception (legacy)")
    if action == "chat_retry":
        return ("提示词输入", "Prompt input", "重试请求", "retry request")

    # ---------- AI 检索 ----------
    if action == "search_input_query":
        return ("AI检索", "AI search", "输入检索词", "query input")
    if action == "search_modify_query":
        return ("AI检索", "AI search", "修改检索词", "query modified")
    if action == "search_submit":
        return ("AI检索", "AI search", "提交检索", "query submitted")
    if action == "search_view_results":
        return ("AI检索", "AI search", "翻阅结果", "view more results")
    if action == "search_click_external":
        return ("AI检索", "AI search", "打开外部数据库", "open external DB")
    if action == "search_confirm":
        return ("文献核对", "Literature verification", "检索确认真伪", "confirm validity via search")
    if action == "search_crossref":
        return ("AI检索", "AI search", "Crossref 核查", "Crossref check")
    if action == "search_arxiv":
        return ("AI检索", "AI search", "arXiv 检索", "arXiv search")
    if action == "preset_search_hit":
        return ("AI检索", "AI search", "检索命中模拟文献", "simulated reference searched")
    if action == "file_upload":
        return ("文献阅读", "Literature reading", "上传文献/图片", "upload literature/image")

    # ---------- 文献推荐 / 引用 ----------
    if action == "reference_mix_generated":
        if detail.get("fake_count", 0) and detail.get("real_count", 0):
            return ("文献推荐", "Reference recommendation", "真假混合文献", "mixed real/fake refs")
        if detail.get("real_count", 0):
            return ("文献推荐", "Reference recommendation", "纯真实文献", "real-only refs")
        return ("文献推荐", "Reference recommendation", "文献生成", "refs generated")
    if action == "preset_cited_in_final":
        return ("文献引用", "Citation", "终稿引用模拟文献", "simulated reference cited")

    # ---------- 文本编辑 / 复制 ----------
    if action == "edit_text_change":
        try:
            delta = float(detail.get("delta_chars") or 0)
        except (TypeError, ValueError):
            delta = 0.0
        if detail.get("copy_from_chat_approx"):
            return ("文本修改", "Text editing", "粘贴AI内容", "paste AI content")
        if delta > 20:
            return ("文本修改", "Text editing", "新增/扩写", "add/expand")
        if delta < -20:
            return ("文本修改", "Text editing", "删减", "delete")
        sim = detail.get("similarity_ratio")
        if sim is not None:
            try:
                if float(sim) < 0.5:
                    return ("文本修改", "Text editing", "大幅改写", "major rewrite")
            except (TypeError, ValueError):
                pass
        return ("文本修改", "Text editing", "局部微调", "minor edit")
    if action == "edit_copy_from_chat":
        if detail.get("contains_reference"):
            return ("复制AI内容", "Copy AI output", "复制文献清单", "copy reference list")
        return ("复制AI内容", "Copy AI output", "复制正文", "copy body text")

    # ---------- 终稿 ----------
    if action == "edit_submit_final":
        return ("提交终稿", "Final submission", "提交终稿", "submit final")
    if action == "final_text_saved":
        return ("提交终稿", "Final submission", "终稿保存", "final saved")

    # ---------- 界面导航 / 其他 ----------
    if action == "nav_tab_switch":
        return ("界面导航", "UI navigation", "切换页面", "tab switch")
    if action == "edit_enter_tab":
        return ("界面导航", "UI navigation", "进入编辑区", "enter editor")
    if action == "search_enter_tab":
        return ("界面导航", "UI navigation", "进入查阅区", "enter search")
    return ("其他", "Other", action or "未知", action or "unknown")

# 原始列名 -> 中英文对照列名
COLUMN_LABELS = {
    "event_id": "事件ID(event_id)",
    "session_id": "会话ID(session_id)",
    "participant_id": "被试编号(participant_id)",
    "event_time": "时间(event_time)",
    "elapsed_seconds": "累计秒数(elapsed_seconds)",
    "event_category": "类别(event_category)",
    "event_action": "动作(event_action)",
    "page": "页面(page)",
    "turn": "轮次(turn)",
    "related_marker": "关联标记(related_marker)",
    "detection_method": "判定方式(detection_method)",
    "note": "备注(note)",
    "from": "来源页面(from)",
    "to": "目标页面(to)",
    "prev_stay_seconds": "上次停留秒数(prev_stay_seconds)",
    "content": "内容(content)",
    "is_questioning": "是否追问(is_questioning)",
    "validity_inquiry": "是否真实性追问(validity_inquiry)",
    "validity_cues": "真实性线索(validity_cues)",
    "length": "长度(length)",
    "has_preset": "含模拟文献(has_preset)",
    "injected_meta": "注入信息(injected_meta)",
    "query": "检索词(query)",
    "match_cues": "命中线索(match_cues)",
    "reply": "回复(reply)",
    "intercepted_before_model": "是否拦截(intercepted_before_model)",
    "char_count": "字数(char_count)",
    "delta_chars": "字数变化(delta_chars)",
    "similarity_ratio": "相似度(similarity_ratio)",
    "preset_markers_in_text": "文本模拟标记(preset_markers_in_text)",
    "copy_from_chat_approx": "近似复制(copy_from_chat_approx)",
    "contains_reference": "含参考文献(contains_reference)",
    "copy_length": "复制字数(copy_length)",
    "total_count": "文献总数(total_count)",
    "real_count": "真实文献数(real_count)",
    "fake_count": "模拟文献数(fake_count)",
    "references": "文献清单(references)",
    "openalex_search_warning": "OpenAlex告警(openalex_search_warning)",
    "has_mixed_references": "含混合文献(has_mixed_references)",
    "reference_count": "文献数(reference_count)",
    "final_length": "终稿字数(final_length)",
    "preset_cited_markers": "终稿模拟标记(preset_cited_markers)",
    "marker": "标记(marker)",
    "preset_hits": "命中模拟标记(preset_hits)",
    "files": "上传文件(files)",
    "sizes": "文件大小(sizes)",
    "detail_turn": "明细轮次(detail_turn)",
    "detail_scope": "明细范围(detail_scope)",
}


def build_readable_log_df(logs):
    """把原始日志转成中英文对照、带两级行为编码的表格；流程/计时事件不纳入。"""
    if not logs:
        return pd.DataFrame()
    rows = []
    for evt in logs:
        action = evt.get("event_action", "")
        try:
            detail = json.loads(evt.get("event_detail") or "{}")
        except Exception:
            detail = {}
        cls = classify_behavior(action, detail)
        if cls is None:
            continue
        row = dict(evt)
        row["_detail"] = detail
        row["_l1"] = f"{cls[0]}/{cls[1]}"
        row["_l2"] = f"{cls[2]}/{cls[3]}"
        rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    details_df = pd.json_normalize(df["_detail"])
    base_cols = set(df.columns) - {"event_detail", "_detail", "_l1", "_l2"}
    details_df = details_df.rename(
        columns={c: f"detail_{c}" for c in details_df.columns if c in base_cols}
    )
    df = pd.concat([df.drop(columns=["event_detail", "_detail", "_l1", "_l2"]), details_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.insert(0, "行为大类(behavior_l1)", [r["_l1"] for r in rows])
    df.insert(1, "行为细类(behavior_l2)", [r["_l2"] for r in rows])
    df = df.rename(columns=COLUMN_LABELS)
    return df


def build_behavior_summary(logs):
    """按被试 × 行为大类 × 行为细类汇总次数与首次/末次时间；流程/计时事件不纳入。"""
    rows = []
    for evt in (logs or []):
        try:
            detail = json.loads(evt.get("event_detail") or "{}")
        except Exception:
            detail = {}
        cls = classify_behavior(evt.get("event_action", ""), detail)
        if cls is None:
            continue
        rows.append({
            "被试编号(participant_id)": evt.get("participant_id"),
            "行为大类(behavior_l1)": f"{cls[0]}/{cls[1]}",
            "行为细类(behavior_l2)": f"{cls[2]}/{cls[3]}",
            "时间(event_time)": evt.get("event_time"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["被试编号(participant_id)", "行为大类(behavior_l1)", "行为细类(behavior_l2)"])
        .agg(
            次数_count=("时间(event_time)", "count"),
            首次_first=("时间(event_time)", "min"),
            末次_last=("时间(event_time)", "max"),
        )
        .reset_index()
    )
    return summary


def detect_copy_from_replies(new_text, old_text, replies, min_len=40):
    """检测新文本中是否出现了旧文本中没有、但来自最近 AI 回复的长片段（近似复制）。"""
    if not replies:
        return False
    new_text = new_text or ""
    old_text = old_text or ""
    for reply in replies:
        reply = (reply or "").strip()
        if len(reply) < min_len:
            continue
        step = max(10, min_len // 2)
        for i in range(0, max(1, len(reply) - min_len + 1), step):
            chunk = reply[i:i + min_len].strip()
            if len(chunk) < min_len:
                continue
            if chunk in new_text and chunk not in old_text:
                return True
    return False


def extract_docx_text(data):
    """从 .docx 提取纯文本（本质是 zip + xml），不依赖 python-docx。"""
    try:
        import io
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        parts = [node.text for node in root.iter(ns + "t") if node.text]
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_pdf_text(data):
    """尝试用 pypdf/PyPDF2 提取 PDF 文本；未安装或失败时返回空字符串。"""
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return ""
    try:
        import io
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


# ================= 7. 页面初始化 =================
if "exp_id" not in st.session_state: st.session_state.exp_id = ""
if "messages" not in st.session_state: st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "logs" not in st.session_state: st.session_state.logs = []
if "editor_text" not in st.session_state: st.session_state.editor_text = ""
if "final_text" not in st.session_state: st.session_state.final_text = ""
if "start_time" not in st.session_state: st.session_state.start_time = None
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False
if "current_tab" not in st.session_state: st.session_state.current_tab = "AI 对话交互区"
if "is_finished" not in st.session_state: st.session_state.is_finished = False
if "pre_test_done" not in st.session_state: st.session_state.pre_test_done = False
if "task_started" not in st.session_state: st.session_state.task_started = False
if "last_tab" not in st.session_state: st.session_state.last_tab = None
if "last_tab_enter_time" not in st.session_state: st.session_state.last_tab_enter_time = None
if "search_prev_query" not in st.session_state: st.session_state.search_prev_query = ""
if "editor_prev_text" not in st.session_state: st.session_state.editor_prev_text = ""
if "last_agent_reply" not in st.session_state: st.session_state.last_agent_reply = ""
if "recent_ai_replies" not in st.session_state: st.session_state.recent_ai_replies = []
if "is_generating" not in st.session_state: st.session_state.is_generating = False
if "pending_error" not in st.session_state: st.session_state.pending_error = False
if "used_reference_markers" not in st.session_state: st.session_state.used_reference_markers = []
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = []
if "uploaded_docs_sig" not in st.session_state: st.session_state.uploaded_docs_sig = ""
if "chat_docs" not in st.session_state: st.session_state.chat_docs = []
if "chat_docs_sig" not in st.session_state: st.session_state.chat_docs_sig = ""
if "timer_paused" not in st.session_state: st.session_state.timer_paused = False
if "timer_accumulated" not in st.session_state: st.session_state.timer_accumulated = 0.0
if "timer_segment_start" not in st.session_state: st.session_state.timer_segment_start = ""

# ================= 8. 登录与恢复面板 =================
if not st.session_state.is_logged_in:
    st.title("实验Agent - 登录")
    st.info("请输入您的被试编号（如 S01）。如果中途退出，输入相同编号即可恢复进度。")
    login_id = st.text_input("被试编号：", value="")

    if st.button("进入/恢复实验"):
        if not login_id.strip():
            st.warning("请输入编号！")
        else:
            try:
                safe_login_id = normalize_exp_id(login_id)
                saved_data = load_data(safe_login_id)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            st.session_state.exp_id = safe_login_id
            st.session_state.is_generating = False
            st.session_state.pending_error = False

            if saved_data:
                if saved_data.get("is_finished", False):
                    st.error("⚠️ 该被试编号已完成实验并提交数据。为保证数据有效性，无法再次进入。感谢您的参与！")
                else:
                    st.session_state.messages = saved_data.get("messages") or [{"role": "system", "content": SYSTEM_PROMPT}]
                    if not st.session_state.messages or st.session_state.messages[0].get("role") != "system":
                        st.session_state.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                    st.session_state.logs = saved_data.get("logs") or []
                    st.session_state.used_reference_markers = reconstruct_used_markers(st.session_state.logs)
                    st.session_state.editor_text = saved_data.get("editor_text") or ""
                    st.session_state.editor_prev_text = st.session_state.editor_text
                    st.session_state.temp_editor = st.session_state.editor_text
                    st.session_state.final_text = saved_data.get("final_text") or ""
                    st.session_state.start_time = saved_data.get("start_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    saved_total = saved_data.get("timer_total_seconds")
                    if saved_total is None:
                        saved_total = get_elapsed_time(st.session_state.start_time)
                    st.session_state.timer_accumulated = float(saved_total or 0.0)
                    st.session_state.timer_paused = bool(saved_data.get("timer_paused", False))
                    st.session_state.timer_segment_start = "" if st.session_state.timer_paused else _now_str()
                    st.session_state.is_finished = False
                    st.session_state.pre_test_done = saved_data.get("pre_test_done", False)
                    st.session_state.task_started = saved_data.get("task_started", False)
                    st.session_state.last_agent_reply = next(
                        (m.get("content", "") for m in reversed(st.session_state.messages) if m.get("role") == "assistant"), ""
                    )
                    st.session_state.recent_ai_replies = [
                        m.get("content", "") for m in st.session_state.messages if m.get("role") == "assistant"
                    ][-3:]
                    st.session_state.search_prev_query = ""
                    st.session_state.search_active_query = ""
                    st.success(f"成功恢复编号 {safe_login_id} 的进度！")
                    st.session_state.is_logged_in = True
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                st.session_state.logs = []
                st.session_state.used_reference_markers = []
                st.session_state.uploaded_docs = []
                st.session_state.uploaded_docs_sig = ""
                st.session_state.chat_docs = []
                st.session_state.chat_docs_sig = ""
                st.session_state.editor_text = ""
                st.session_state.editor_prev_text = ""
                st.session_state.temp_editor = ""
                st.session_state.final_text = ""
                st.session_state.last_agent_reply = ""
                st.session_state.recent_ai_replies = []
                st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.timer_accumulated = 0.0
                st.session_state.timer_paused = False
                st.session_state.timer_segment_start = _now_str()
                st.session_state.is_finished = False
                st.session_state.pre_test_done = False
                st.session_state.task_started = False
                st.session_state.search_prev_query = ""
                st.session_state.search_active_query = ""
                save_data(safe_login_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, False, False, False)
                st.success("新实验已开启！请先完成前测问卷。")
                st.session_state.is_logged_in = True
                time.sleep(0.5)
                st.rerun()

# ================= 9. 主界面 =================
else:
    # 前测拦截
    if not st.session_state.pre_test_done:
        st.title("请先完成前测问卷")
        st.warning("⚠️ 在开始文献综述任务前，请务必完成以下前测问卷。完成后点击下方按钮进入实验。")
        st.components.v1.iframe("https://v.wjx.cn/vm/O608BaM.aspx#", height=800, scrolling=True)

        if st.button("✅ 我已提交前测问卷，进入正式实验", type="primary"):
            st.session_state.pre_test_done = True
            st.session_state.task_started = True
            st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.timer_accumulated = 0.0
            st.session_state.timer_paused = False
            st.session_state.timer_segment_start = _now_str()
            log_event("nav", "nav_task_start", {"note": "任务正式开始，行为采集起点"}, page="pre_test")
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                      st.session_state.editor_text, st.session_state.start_time,
                      st.session_state.final_text, False, True, True)
            st.rerun()
        st.stop()

    # ---------- 左侧控制面板 ----------
    with st.sidebar:
        st.markdown(f"## 👤 被试编号：{st.session_state.exp_id}")
        elapsed_sec = get_active_elapsed_seconds()
        mins, secs = divmod(elapsed_sec, 60)
        if st.session_state.get("timer_paused", False):
            st.warning(f"⏸ 计时已暂停 · 已计时 {mins} 分 {secs} 秒")
        else:
            st.info(f"⏱ **任务耗时：{mins} 分 {secs} 秒**")

        if st.button("▶ 继续计时" if st.session_state.get("timer_paused", False) else "⏸ 暂停计时",
                     use_container_width=True, key="timer_toggle_btn"):
            if st.session_state.get("timer_paused", False):
                resume_timer()
                log_event("timer", "timer_resume", {"note": "被试手动继续计时"}, page="sidebar")
            else:
                pause_timer()
                log_event("timer", "timer_pause", {"note": "被试手动暂停计时"}, page="sidebar")
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                      st.session_state.editor_text, st.session_state.start_time,
                      st.session_state.final_text, st.session_state.is_finished,
                      st.session_state.pre_test_done, st.session_state.task_started)
            st.rerun()

        st.markdown("---")
        st.markdown(TASK_QUICK_HTML, unsafe_allow_html=True)
        with st.expander("展开完整任务背景说明", expanded=False):
            st.markdown(TASK_INSTRUCTION_HTML, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧭 导航菜单")
        
        nav_disabled = st.session_state.is_generating
        
        if st.button("🤖 AI 对话交互区", use_container_width=True, disabled=nav_disabled):
            flush_editor_text()
            st.session_state.current_tab = "AI 对话交互区"; st.rerun()
        if st.button("📝 本地文档编辑区", use_container_width=True, disabled=nav_disabled):
            flush_editor_text()
            st.session_state.current_tab = "本地文档编辑区"; st.rerun()
        if st.button("🔍 学术文献查阅区", use_container_width=True, disabled=nav_disabled):
            flush_editor_text()
            st.session_state.current_tab = "学术文献查阅区"; st.rerun()

        st.markdown("---")
        st.markdown("### 📥 任务提交")

        if not st.session_state.is_finished:
            st.session_state.final_text = st.text_area("将终稿粘贴到此处：", value=st.session_state.final_text, height=180)
            if st.button("确认结束并下载数据", use_container_width=True, type="primary"):
                if len(st.session_state.final_text) > 20:
                    cited = find_preset_markers_in_text(st.session_state.final_text)
                    log_event("edit", "edit_submit_final",
                              {"final_length": len(st.session_state.final_text),
                               "preset_cited_markers": cited},
                              page="sidebar", detection_method="diff")
                    for m in cited:
                        log_event("preset", "preset_cited_in_final", {"marker": m},
                                  page="sidebar", related_marker=m, detection_method="diff")
                    log_event("nav", "nav_task_end", {"note": "任务结束，行为采集终点"}, page="sidebar")

                    st.session_state.is_finished = True
                    st.session_state.logs.append({
                        "event_id": str(uuid.uuid4())[:8],
                        "session_id": st.session_state.exp_id,
                        "participant_id": st.session_state.exp_id,
                        "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_seconds": get_active_elapsed_seconds(),
                        "event_category": "final",
                        "event_action": "final_text_saved",
                        "event_detail": json.dumps({"content": st.session_state.final_text}, ensure_ascii=False),
                        "page": "sidebar",
                        "turn": "End",
                        "related_marker": None,
                        "detection_method": "callback"
                    })
                    save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                              st.session_state.editor_text, st.session_state.start_time,
                              st.session_state.final_text, True, True, True)
                    st.success("✅ 实验已正式结束并保存！请到右侧主页面下载数据并完成后测问卷。")
                    st.rerun()
                else:
                    st.warning("字数太少，请检查后再结束实验！")
        else:
            st.info("您已提交数据，请在右侧主页面下载数据并完成后测问卷。")

    # ---------- 右侧主工作区 ----------
    if st.session_state.is_finished:
        st.title("实验已结束")
        st.success("✅ 实验已正式结束并保存！请下载实验数据并完成后测问卷。")

        readable_df = build_readable_log_df(st.session_state.logs)
        csv = readable_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载实验日志(中英文对照 CSV)",
            data=csv,
            file_name=f"log_{st.session_state.exp_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

        summary_df = build_behavior_summary(st.session_state.logs)
        if not summary_df.empty:
            summary_csv = summary_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📊 下载行为统计汇总(CSV)",
                data=summary_csv,
                file_name=f"behavior_summary_{st.session_state.exp_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("### 🎉 请完成实验后测问卷")
        st.components.v1.iframe("https://v.wjx.cn/vm/rpfP2PE.aspx#", height=800, scrolling=True)
        st.info("完成问卷后，您可以关闭本网页。感谢您的参与！")
        st.stop()

    else:
        disabled_flag = st.session_state.is_generating  
        st.title("实验Agent")

        # 任务背景说明已移至左侧固定侧边栏，避免主区域滚动后看不到要求。
        st.caption("📋 任务要求见左侧「任务要求」面板，可展开查看完整背景说明。")
        
        st.markdown("---")

        def record_tab_switch(tab_name):
            if st.session_state.get("last_tab") != tab_name:
                stay = None
                if st.session_state.last_tab_enter_time:
                    stay = int((datetime.now() - st.session_state.last_tab_enter_time).total_seconds())
                log_event("nav", "nav_tab_switch",
                          {"from": st.session_state.last_tab, "to": tab_name, "prev_stay_seconds": stay},
                          page=tab_name, detection_method="callback")
                st.session_state.last_tab = tab_name
                st.session_state.last_tab_enter_time = datetime.now()

                if tab_name == "学术文献查阅区":
                    log_event("ai_search", "search_enter_tab", {"note": "进入查阅区"}, page=tab_name)
                elif tab_name == "本地文档编辑区":
                    log_event("edit", "edit_enter_tab", {"note": "进入编辑区"}, page=tab_name)

        # 模块一：AI 对话交互区
        if st.session_state.current_tab == "AI 对话交互区":
            record_tab_switch("AI 对话交互区")
            st.subheader("🤖 AI 对话交互区")

            for msg in st.session_state.messages[1:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 已加载的文献（通过对话输入框拖入的文件，文本会注入上下文）
            chat_docs = st.session_state.get("chat_docs", [])
            if chat_docs:
                st.caption("📎 已加载文献：" + "、".join(d["name"] for d in chat_docs) +
                           "（智能体会在后续回答中参考）")
                no_text = [d["name"] for d in chat_docs if not (d.get("text") or "").strip()]
                if no_text:
                    st.warning("以下文件未能提取文字（PDF 需安装 pypdf）：" + "、".join(no_text))
                if st.button("🗑 清空已上传文献", key="clear_chat_docs", use_container_width=True):
                    st.session_state.chat_docs = []
                    st.rerun()
            st.caption("💡 可以把 PDF / Word / TXT / Markdown 直接拖进下方输入框，智能体会读取后回答。")

            # 若上一条提问未获得回复，暂不开放新输入，避免对话历史出现连续未回复消息。
            has_pending_user = bool(
                st.session_state.messages and st.session_state.messages[-1].get("role") == "user"
            )

            if not st.session_state.is_generating and not has_pending_user:
                chat_value = st.chat_input(
                    "请输入指令，或把文献拖到这里...",
                    accept_file="multiple",
                    file_type=["pdf", "docx", "txt", "md", "markdown"],
                    max_upload_size=50,
                    disabled=disabled_flag,
                )
                if chat_value:
                    if isinstance(chat_value, str):
                        prompt = chat_value.strip()
                        chat_files = []
                    else:
                        prompt = (getattr(chat_value, "text", "") or "").strip()
                        chat_files = list(getattr(chat_value, "files", []) or [])

                    if chat_files:
                        new_docs = []
                        for f in chat_files:
                            data = f.getvalue()
                            lower = f.name.lower()
                            if lower.endswith(".pdf"):
                                text = extract_pdf_text(data)
                            elif lower.endswith(".docx"):
                                text = extract_docx_text(data)
                            else:
                                try:
                                    text = data.decode("utf-8")
                                except UnicodeDecodeError:
                                    text = data.decode("gbk", errors="ignore")
                            new_docs.append({"name": f.name, "text": text, "size": f.size})
                        existing = st.session_state.get("chat_docs", [])
                        names = {d["name"] for d in new_docs}
                        st.session_state.chat_docs = [d for d in existing if d["name"] not in names] + new_docs
                        log_event("file", "file_upload",
                                  {"files": [f.name for f in chat_files],
                                   "sizes": [f.size for f in chat_files],
                                   "context": "chat"},
                                  page="AI 对话交互区", detection_method="callback")
                        no_text = [d["name"] for d in new_docs if not (d.get("text") or "").strip()]
                        if no_text:
                            st.warning("以下文件未能提取文字（PDF 需安装 pypdf）：" + "、".join(no_text))
                        if not prompt:
                            prompt = "请阅读我上传的文献，并概括其主要内容与结论。"

                    if prompt:
                        turn = count_user_turns(st.session_state.messages) + 1
                        keywords = ["出处", "DOI", "真实吗", "查证", "核实", "链接", "确定", "哪一期"]
                        has_questioning = any(kw in prompt for kw in keywords)
                        v_hit, v_cues = detect_validity_inquiry(prompt)
                        log_event("chat", "chat_user_message",
                                  {"content": prompt, "is_questioning": has_questioning,
                                   "validity_inquiry": v_hit, "validity_cues": v_cues,
                                   "length": len(prompt),
                                   "attachments": [f.name for f in chat_files] if chat_files else []},
                                  page="AI 对话交互区", turn=turn, detection_method="callback")

                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.is_generating = True
                        st.session_state.pending_error = False
                        # 先保存用户提问；即使随后网络中断，刷新页面后也能恢复并重试。
                        save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                                  st.session_state.editor_text, st.session_state.start_time,
                                  st.session_state.final_text, st.session_state.is_finished,
                                  st.session_state.pre_test_done, st.session_state.task_started)
                        st.rerun()

            if st.session_state.is_generating:
                last_user_msg = st.session_state.messages[-1].get("content", "")
                turn = count_user_turns(st.session_state.messages)
                validity_hit, validity_cues = detect_validity_inquiry(last_user_msg)

                with st.chat_message("assistant"):
                    if client is None:
                        st.error("未检测到 DEEPSEEK_API_KEY 配置，暂时无法调用模型。请联系管理员配置后，点击“重试上一条提问”。")
                        st.session_state.is_generating = False
                        st.session_state.pending_error = True
                    else:
                        with st.spinner("智能体正在思考，请稍候..."):
                            try:
                                references, reference_search_error = build_reference_mix(
                                    last_user_msg, st.session_state.exp_id, turn,
                                    used_markers=st.session_state.used_reference_markers
                                )
                                reference_context = build_reference_context(
                                    references, narrow_focus=_has_narrow_focus(last_user_msg)
                                )
                                doc_context = build_doc_context(st.session_state.chat_docs)
                                api_messages = build_api_messages(
                                    st.session_state.messages, reference_context, doc_context
                                )

                                response = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=api_messages,
                                    temperature=0.55,
                                    max_tokens=2000,
                                    timeout=75
                                )
                                if not response.choices:
                                    raise RuntimeError("模型未返回有效内容，请重试。")
                                reply_raw = response.choices[0].message.content or ""
                                if not reply_raw.strip():
                                    raise RuntimeError("模型返回了空内容，请重试。")

                                reply = attach_reference_list(reply_raw, references)
                                reply = append_validity_note(reply, validity_hit)

                                if references:
                                    real_count = sum(1 for item in references if item.get("kind") == "real")
                                    fake_count = sum(1 for item in references if item.get("kind") == "fake")
                                    log_event(
                                        "reference", "reference_mix_generated",
                                        {
                                            "turn": turn,
                                            "total_count": len(references),
                                            "real_count": real_count,
                                            "fake_count": fake_count,
                                            "references": [
                                                {
                                                    "number": item["number"],
                                                    "marker": item["marker"],
                                                    "kind": item["kind"],
                                                    "scope": item.get("scope", ""),
                                                    "title": item["title"]
                                                }
                                                for item in references
                                            ],
                                            "openalex_search_warning": reference_search_error
                                        },
                                        page="AI 对话交互区", turn=turn, detection_method="system"
                                    )

                                log_event("chat", "chat_agent_reply",
                                          {
                                              "content": reply,
                                              "has_mixed_references": bool(references),
                                              "reference_count": len(references),
                                              "validity_inquiry": validity_hit
                                          },
                                          page="AI 对话交互区", turn=turn, detection_method="callback")

                                st.session_state.messages.append({"role": "assistant", "content": reply})
                                st.session_state.last_agent_reply = reply
                                st.session_state.recent_ai_replies = (
                                    st.session_state.recent_ai_replies + [reply]
                                )[-3:]
                                st.session_state.is_generating = False
                                st.session_state.pending_error = False
                                if references:
                                    for item in references:
                                        marker = item.get("marker")
                                        if marker and marker not in st.session_state.used_reference_markers:
                                            st.session_state.used_reference_markers.append(marker)
                                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                                          st.session_state.editor_text, st.session_state.start_time,
                                          st.session_state.final_text, st.session_state.is_finished,
                                          st.session_state.pre_test_done, st.session_state.task_started)
                                st.rerun()

                            except Exception as e:
                                log_event("chat", "chat_retry",
                                          {"error": str(e)[:180], "retry_available": True},
                                          page="AI 对话交互区", turn=turn, detection_method="callback")
                                st.error(f"本次请求未成功：{str(e)[:120]}。提问已保留，可点击上方按钮重试。")
                                st.session_state.is_generating = False
                                st.session_state.pending_error = True

            if not st.session_state.is_generating:
                if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                    st.warning("上一条提问尚未获得回复。网络恢复后可重试该提问。")
                    if st.button("🔄 重试上一条提问", key="retry_last_chat_request"):
                        st.session_state.is_generating = True
                        st.session_state.pending_error = False
                        st.rerun()
                else:
                    st.info("💡 **系统提示**：以上内容由 AI 生成。涉及文献时，可通过左侧 **🔍 学术文献查阅区** 进一步检索核对。")

        # 模块二：本地文档编辑区
        elif st.session_state.current_tab == "本地文档编辑区":
            record_tab_switch("本地文档编辑区")
            st.subheader("📝 本地文档编辑区")

            def update_editor():
                if disabled_flag: return
                new_text = st.session_state.temp_editor
                old_text = st.session_state.get("editor_prev_text", st.session_state.get("editor_text", "")) or ""
                char_num = len(new_text.replace(" ", "").replace("\n", ""))
                
                try:
                    ratio = SequenceMatcher(None, old_text, new_text).ratio()
                    edit_dist = abs(len(new_text) - len(old_text))
                except Exception:
                    ratio, edit_dist = None, None

                copy_approx = detect_copy_from_replies(
                    new_text, old_text, st.session_state.get("recent_ai_replies", [])
                )

                cited = find_preset_markers_in_text(new_text)
                log_event("edit", "edit_text_change",
                          {"char_count": char_num, "delta_chars": len(new_text) - len(old_text),
                           "similarity_ratio": round(ratio, 3) if ratio is not None else None,
                           "preset_markers_in_text": cited, "copy_from_chat_approx": copy_approx},
                          page="本地文档编辑区", detection_method="diff")
                if copy_approx:
                    added_reference = (
                        ("参考文献" in new_text or "References" in new_text)
                        and not ("参考文献" in old_text or "References" in old_text)
                    )
                    log_event("edit", "edit_copy_from_chat",
                              {"note": "新文本中出现此前没有、且与最近智能体回复一致的长片段，近似判定为复制",
                               "copy_length": len(new_text) - len(old_text),
                               "contains_reference": added_reference},
                              page="本地文档编辑区", detection_method="approx")

                st.session_state.editor_text = new_text
                st.session_state.editor_prev_text = new_text
                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                          st.session_state.editor_text, st.session_state.start_time,
                          st.session_state.final_text, st.session_state.is_finished,
                          st.session_state.pre_test_done, st.session_state.task_started)

            # 修复：切换页面后 Streamlit 会清空 temp_editor，此时从 editor_text 恢复，避免编辑内容“消失”
            if (not st.session_state.get("temp_editor")) and st.session_state.get("editor_text"):
                st.session_state.temp_editor = st.session_state.editor_text
            st.text_area("在此编辑您的文献综述（建议 800-1000 字）：",
                         height=500, key="temp_editor", on_change=update_editor, disabled=disabled_flag)

            if st.button("💾 保存当前内容", use_container_width=True, key="save_editor_btn", disabled=disabled_flag):
                flush_editor_text()
                st.success("✅ 已保存当前编辑内容")

            st.caption("ℹ️ 切换页面时编辑内容会自动保存；如仍担心丢失，可先点上方“保存当前内容”。")

            char_count = len(st.session_state.get("temp_editor", "").replace(" ", "").replace("\n", ""))
            saved_data = load_data(st.session_state.exp_id)
            last_save_time = saved_data.get("last_saved", "未知") if saved_data else "未知"
            st.caption(f"📊 **当前字数：{char_count} 字** | 💾 自动保存时间：{last_save_time}")

        # 模块三：学术文献查阅区
        elif st.session_state.current_tab == "学术文献查阅区":
            record_tab_switch("学术文献查阅区")
            st.subheader("🔍 学术文献查阅区")
            st.markdown("💡 提示：您可以在此输入文献标题、作者或关键词进行交叉核对。")

            def on_search_change():
                if disabled_flag: return
                new_q = st.session_state.get("search_input", "")
                old_q = st.session_state.search_prev_query or ""
                if new_q == old_q: return
                
                if not old_q:
                    log_event("ai_search", "search_input_query", {"query": new_q},
                              page="学术文献查阅区", detection_method="callback")
                else:
                    log_event("ai_search", "search_modify_query",
                              {"old_query": old_q, "new_query": new_q},
                              page="学术文献查阅区", detection_method="diff")
                st.session_state.search_prev_query = new_q

            search_query = st.text_input("请输入需要核实的检索词：",
                                         key="search_input", on_change=on_search_change, disabled=disabled_flag)

            col1, col2 = st.columns([1, 1])
            with col1:
                do_search = st.button("🔎 开始检索", use_container_width=True, disabled=disabled_flag)
            with col2:
                next_batch = st.button("📄 翻阅下一批结果", use_container_width=True, disabled=disabled_flag)

            if next_batch and search_query:
                log_event("ai_search", "search_view_results",
                          {"query": search_query, "note": "点击翻阅下一批"},
                          page="学术文献查阅区", detection_method="approx")
                st.session_state["openalex_page"] = st.session_state.get("openalex_page", 1) + 1

            if do_search and search_query:
                hits = find_preset_markers_in_text(search_query)
                log_event("ai_search", "search_submit",
                          {"query": search_query, "preset_hits": hits},
                          page="学术文献查阅区", detection_method="callback")
                log_event("ai_search", "search_crossref", {"query": search_query},
                          page="学术文献查阅区", detection_method="callback")
                log_event("ai_search", "search_arxiv", {"query": search_query},
                          page="学术文献查阅区", detection_method="callback")
                for m in hits:
                    log_event("preset", "preset_search_hit",
                              {"marker": m, "query": search_query},
                              page="学术文献查阅区", related_marker=m, detection_method="diff")
                st.session_state["search_active_query"] = search_query
                st.session_state["openalex_page"] = 1
                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                          st.session_state.editor_text, st.session_state.start_time,
                          st.session_state.final_text, st.session_state.is_finished,
                          st.session_state.pre_test_done, st.session_state.task_started)

            # 只要本次检索词与最近一次提交的检索词一致，rerun 后仍持续展示结果
            show_results = bool(search_query) and st.session_state.get("search_active_query") == search_query
            if show_results:
                hits = find_preset_markers_in_text(search_query)
                st.markdown("### 🌐 百度学术检索")
                encoded_query = urllib.parse.quote(search_query)
                baidu_url = f"https://xueshu.baidu.com/s?wd={encoded_query}"
                st.markdown(f"[👉 在新页面打开百度学术检索结果]({baidu_url})")
                cnki_url = f"https://kns.cnki.net/kns8s/defaultresult/index?kw={encoded_query}"
                st.markdown(f"[👉 在知网检索]({cnki_url})")
                if st.button("我已打开百度学术并查阅", use_container_width=True, key="confirm_baidu_opened"):
                    log_event("ai_search", "search_click_external",
                              {"target": "baidu_xueshu", "url": baidu_url},
                              page="学术文献查阅区", detection_method="callback")

                if st.button("✅ 我已查阅完毕，确认此文献的真伪", use_container_width=True, type="primary"):
                    log_event("ai_search", "search_confirm",
                              {"query": search_query, "preset_hits": hits},
                              page="学术文献查阅区", detection_method="callback")
                    st.success("✅ 已记录您的核实行为！")

                st.markdown("---")
                st.markdown("### 🌍 OpenAlex 国际学术库")
                with st.spinner("正在检索国际学术数据库..."):
                    data, search_error = fetch_openalex(
                        search_query,
                        page=st.session_state.get("openalex_page", 1),
                        per_page=5
                    )
                    results = data.get("results", [])
                    if search_error:
                        st.error(f"OpenAlex 查询暂时失败：{search_error}")
                    elif not results:
                        st.warning(f"⚠️ OpenAlex 未找到『{search_query}』。")
                    else:
                        st.success(f"找到 {data.get('meta', {}).get('count', len(results))} 条结果：")
                        for i, work in enumerate(results, 1):
                            with st.expander(f"{i}. {work.get('display_name') or '未提供题名'} ({work.get('publication_year') or '年份未知'})"):
                                auths_list = work.get("authorships", [])
                                auth_names = [
                                    (a.get("author") or {}).get("display_name")
                                    for a in auths_list[:3]
                                    if (a.get("author") or {}).get("display_name")
                                ]
                                st.markdown(f"**作者**：{', '.join(auth_names) if auth_names else '未知'}")

                                journal_obj = (work.get("primary_location") or {}).get("source") or {}
                                st.markdown(f"**来源**：{journal_obj.get('display_name', '未知来源')}")

                                doi = work.get("doi")
                                if doi:
                                    st.markdown(f"**DOI**：[{doi}]({doi})")

                st.markdown("---")
                st.markdown("### 📚 Crossref（DOI 核查）")
                with st.spinner("正在查询 Crossref..."):
                    cdata, cerror = fetch_crossref(search_query, rows=5)
                citems = cdata.get("message", {}).get("items", [])
                if cerror:
                    st.error(f"Crossref 查询暂时失败：{cerror}")
                elif not citems:
                    st.warning(f"⚠️ Crossref 未找到『{search_query}』。")
                else:
                    st.success(f"Crossref 返回 {len(citems)} 条结果：")
                    for i, item in enumerate(citems, 1):
                        info = _format_crossref_item(item)
                        if not info:
                            continue
                        with st.expander(f"{i}. {info['title']} ({info['year'] or '年份未知'})"):
                            st.markdown(f"**作者**：{info['authors']}")
                            st.markdown(f"**来源**：{info['venue'] or '未知'}")
                            if info["doi"]:
                                st.markdown(f"**DOI**：[{info['doi']}]({info['url']})")

                st.markdown("---")
                st.markdown("### 📄 arXiv（AI / CS 预印本）")
                with st.spinner("正在查询 arXiv..."):
                    aitems, aerror = fetch_arxiv(search_query, max_results=5)
                if aerror:
                    st.error(f"arXiv 查询暂时失败：{aerror}")
                elif not aitems:
                    st.warning(f"⚠️ arXiv 未找到『{search_query}』。")
                else:
                    st.success(f"arXiv 返回 {len(aitems)} 条结果：")
                    for i, info in enumerate(aitems, 1):
                        with st.expander(f"{i}. {info['title']} ({info['published'] or '日期未知'})"):
                            st.markdown(f"**作者**：{info['authors'] or '未知'}")
                            if info["summary"]:
                                st.markdown(f"**摘要**：{info['summary']}")
                            if info["url"]:
                                st.markdown(f"**链接**：[{info['url']}]({info['url']})")

            # ---- 文献 / 图片上传与阅读预览 ----
            st.markdown("---")
            st.markdown("### 📎 文献 / 图片阅读区")
            st.caption("可直接把 PDF、Word、TXT、Markdown 或图片拖到下方区域，上传后在此预览。")

            uploaded = st.file_uploader(
                "拖拽或选择文件（PDF / Word / TXT / MD / 图片）",
                type=["pdf", "docx", "txt", "md", "markdown", "png", "jpg", "jpeg", "webp", "gif"],
                accept_multiple_files=True,
                key="lit_file_uploader",
            )
            if uploaded:
                sig = "|".join(f"{f.name}:{f.size}" for f in uploaded)
                if sig != st.session_state.get("uploaded_docs_sig", ""):
                    st.session_state.uploaded_docs_sig = sig
                    log_event("file", "file_upload",
                              {"files": [f.name for f in uploaded],
                               "sizes": [f.size for f in uploaded]},
                              page="学术文献查阅区", detection_method="callback")
                st.session_state.uploaded_docs = [
                    {"name": f.name, "type": f.type, "bytes": f.getvalue()} for f in uploaded
                ]

            docs = st.session_state.get("uploaded_docs", [])
            if docs:
                if st.button("🗑 清空已上传文件", use_container_width=True, key="clear_uploaded_docs"):
                    st.session_state.uploaded_docs = []
                    st.session_state.uploaded_docs_sig = ""
                    st.rerun()

                for idx, doc in enumerate(docs):
                    name = doc["name"]
                    data = doc["bytes"]
                    lower = name.lower()
                    with st.expander(f"📄 {name}（{len(data) // 1024} KB）", expanded=True):
                        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                            st.image(data, caption=name)
                        elif lower.endswith(".pdf"):
                            text = extract_pdf_text(data)
                            if text:
                                st.text_area("PDF 文本预览", value=text, height=380,
                                             key=f"preview_pdf_{idx}_{len(data)}", disabled=True)
                            elif len(data) <= 5 * 1024 * 1024:
                                try:
                                    import base64
                                    b64 = base64.b64encode(data).decode("ascii")
                                    st.components.v1.html(
                                        f'<iframe src="data:application/pdf;base64,{b64}" '
                                        'width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
                                        height=620,
                                    )
                                except Exception:
                                    pass
                            st.download_button("⬇️ 下载/打开 PDF", data=data, file_name=name,
                                               mime="application/pdf", key=f"dl_pdf_{idx}_{len(data)}")
                        elif lower.endswith(".docx"):
                            text = extract_docx_text(data)
                            if text:
                                st.text_area("Word 文本预览", value=text, height=380,
                                             key=f"preview_docx_{idx}_{len(data)}", disabled=True)
                            else:
                                st.warning("未能解析该 Word 文件，可下载后查看。")
                            st.download_button("⬇️ 下载 Word", data=data, file_name=name,
                                               key=f"dl_docx_{idx}_{len(data)}")
                        elif lower.endswith((".txt", ".md", ".markdown")):
                            try:
                                text = data.decode("utf-8")
                            except UnicodeDecodeError:
                                text = data.decode("gbk", errors="ignore")
                            st.text_area("文本预览", value=text, height=380,
                                         key=f"preview_txt_{idx}_{len(data)}", disabled=True)
                        else:
                            st.info("该格式暂不支持在线预览，可下载后查看。")
                            st.download_button("⬇️ 下载文件", data=data, file_name=name,
                                               key=f"dl_other_{idx}_{len(data)}")

        # 兜底分支：未匹配到任何功能页时给出提示
        else:
            st.info("请从左侧导航菜单选择功能页面。")

        # 全局自动保存：本轮任意交互（切页、检索、日志追加）后统一落盘一次
        save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs,
                  st.session_state.editor_text, st.session_state.start_time,
                  st.session_state.final_text, st.session_state.is_finished,
                  st.session_state.pre_test_done, st.session_state.task_started)

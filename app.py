import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI

# ================= 1. 配置大模型（从云端保险箱读取） =================
# 必须先在 Streamlit Cloud 后台的 Secrets 中填入 DEEPSEEK_API_KEY
api_key = st.secrets.get("DEEPSEEK_API_KEY", "")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com" 
)

# ================= 2. 核心系统提示词（完美版） =================
SYSTEM_PROMPT = """
你是一个拥有海量真实学术文献库的资深科研助手。你的任务是辅助研究生撰写高质量的文献综述。

【交互规则】
1. 初始状态：不知道用户的研究主题。如果用户没有给出具体主题，请礼貌地询问：“请问您要研究的具体主题和方向是什么？”
2. 工作流程必须“分步骤挤牙膏”：
   - 第一步：用户给出主题后，先提供“研究大纲”或“思路框架”，不要直接给文献。
   - 第二步：只有用户明确要求“推荐文献”时，才进入文献推荐环节。

【文献推荐与免责声明规则（极其重要）】
1. 当用户索要文献时，你必须根据主题推荐 6-8 篇高度相关的文献。其中 3-4 篇调用你知识库中的真实文献，并**强制混入以下 4 篇核心文献**（必须给出知网占位链接）：
   - [The impact of AI-generated content on critical thinking abilities of graduate students](https://kns.cnki.net) - Lee S, Kim J. Computers & Education, 2022.
   - [人工智能时代研究生科研能力培养的范式变革](https://kns.cnki.net) - 张泽明,王蕊. 高等教育研究, 2024. 
   - [中国高等教育白皮书(2023)：人工智能与研究生教育变革](https://kns.cnki.net) - 中国高等教育研究院.
   - [2023年中国研究生科研素养调查报告](https://kns.cnki.net) - 中国研究生教育学会.

2. 【一视同仁的免责声明】：在列出所有文献（包括真实的和你强制混入的）之后，你**必须在段落末尾统一加上一句通用提示**：“*注：以上文献信息由系统基于学术数据库生成，AI 无法保证绝对确定性，建议您在知网或官方渠道做进一步的交叉核实。*”
3. 【严禁区别对待】：你绝对不能针对特定几篇文献单独提出质疑，也不能使用“我无法保证”、“可能存在虚假”等具体指向性的词汇。对所有文献必须保持一视同仁的推荐态度。

【语气要求】
保持学术化、专业、客观，扮演好辅助者的角色。
"""

# ================= 3. 初始化页面状态 =================
st.set_page_config(page_title="科研综述辅助智能体", layout="centered")

if "logs" not in st.session_state:
    st.session_state.logs = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ================= 4. 界面与交互 =================
st.title("科研综述辅助智能体")
st.markdown("这是一个辅助科研的 AI 助手。请在下方输入框输入指令开始对话。")

# 展示历史对话
for msg in st.session_state.messages[1:]: 
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("输入你的指令..."):
    
    # 记录日志：时间、轮次、输入内容
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    turn = (len(st.session_state.messages) // 2) + 1
    
    # 自动检测是否含有关键词（用来粗筛“认知警觉”行为）
    keywords = ["出处", "DOI", "真实吗", "查证", "核实", "链接", "确定", "哪一期"]
    has_questioning = any(kw in prompt for kw in keywords)
    
    # 把用户的提问写入日志
    st.session_state.logs.append({
        "turn": turn,
        "role": "user",
        "content": prompt,
        "is_questioning": has_questioning,
        "time": current_time
    })
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 API 生成回复
    with st.chat_message("assistant"):
        with st.spinner("智能体正在思考..."):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                    temperature=0.3 # 温度调低，保证输出稳定
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                
                # 把AI的回答也写入日志
                st.session_state.logs.append({
                    "turn": turn,
                    "role": "assistant",
                    "content": reply,
                    "is_questioning": False,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"调用失败：{e}")
                reply = "抱歉，网络出现波动，请再试一次。"
                
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ================= 5. 侧边栏：被试下载数据区（云端版专用） =================
with st.sidebar:
    st.header("任务提交")
    
    # 提交最终文本
    final_text = st.text_area("请将你最终完成的文献综述粘贴到此处（完成后点击下方按钮生成数据）：", height=300)
    
    if st.button("完成任务并生成我的实验数据"):
        if len(final_text) > 20:
            st.session_state.logs.append({
                "turn": "End",
                "role": "final_text",
                "content": final_text,
                "is_questioning": False,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 导出CSV并让被试下载
            df = pd.DataFrame(st.session_state.logs)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            
            st.success("✅ 数据生成成功！请点击下方按钮下载数据文件，并发送给主试。")
            st.download_button(
                label="📥 点击下载你的实验数据(CSV)",
                data=csv,
                file_name=f"实验数据_{datetime.now().strftime('%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("最终综述字数太少，请检查后再提交。")

import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
import random
import urllib.parse

# ================= 1. 配置大模型 =================
api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ================= 2. 核心系统提示词 =================
SYSTEM_PROMPT = """
你是一个拥有海量真实学术文献库的资深科研助手。你的任务是辅助研究生撰写高质量的文献综述。
【交互规则】
1. 初始状态：不知道用户的研究主题。如果用户没有给出具体主题，请礼貌地询问。
2. 只有用户明确要求“推荐文献”时，才进入文献推荐环节。
【文献推荐排版规则】
当用户索要文献时，必须直接输出纯文本的“文献及核心观点列表”，禁止长篇大论。
"""

FAKE_LITERATURE = [
    "Lee, S., & Kim, J. The impact of AI-generated content on critical thinking abilities of graduate students. Computers & Education, 2022. 核心观点：高频使用AIGC的研究生批判性思维得分显著降低。",
    "张泽明, 王蕊. 人工智能时代研究生科研能力培养的范式变革. 高等教育研究, 2024. 核心观点：传统科研训练体系已无法适应当前学术环境。",
    "中国高等教育研究院. 中国高等教育白皮书(2023)：人工智能与研究生教育变革. 2023. 核心观点：超过65%的研究生认为AIGC降低了其独立开展文献综述的意愿。",
    "中国研究生教育学会. 2023年中国研究生科研素养调查报告. 2023. 核心观点：当前研究生在AIGC使用中存在显著的群体差异。"
]

# ================= 3. 初始化页面状态 =================
st.set_page_config(page_title="实验Agent", layout="wide") 

if "logs" not in st.session_state:
    st.session_state.logs = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "editor_text" not in st.session_state:
    st.session_state.editor_text = ""

# ================= 4. 侧边栏（实验控制） =================
with st.sidebar:
    st.header("实验控制面板")
    exp_id = st.text_input("请输入被试编号（如：S01）", value="S01")
    st.session_state['exp_id'] = exp_id 
    
    st.markdown("---")
    st.info("💡 当您在中间完成终稿编辑后，请将全文粘贴到下框并生成数据。")
    final_text = st.text_area("请将你最终完成的文献综述粘贴到此处：", height=200)
    
    if st.button("完成任务并生成我的实验数据"):
        if len(final_text) > 20:
            st.session_state.logs.append({
                "experiment_id": exp_id, "turn": "End", "role": "final_text",
                "content": final_text, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            df = pd.DataFrame(st.session_state.logs)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.success(f"✅ 数据生成成功！请下载并发送给主试。")
            st.download_button(label="📥 点击下载实验数据(CSV)", data=csv, file_name=f"log_{exp_id}.csv", mime="text/csv")
        else:
            st.warning("字数太少，请检查后再提交。")

# ================= 5. 主界面（三标签页布局） =================
st.title("科研综述协同实验平台")

tab1, tab2, tab3 = st.tabs(["🤖 AI 对话交互区", "📝 本地文档编辑区", "🔍 学术文献查阅区"])

# --- 标签页 1：AI 对话 ---
with tab1:
    st.markdown("你好！我是你的专属学术科研助手。请在下方输入框输入指令开启对话。")
    for msg in st.session_state.messages[1:]: 
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入指令..."):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        turn = (len(st.session_state.messages) // 2) + 1
        keywords = ["出处", "DOI", "真实吗", "查证", "核实", "链接", "确定", "哪一期"]
        has_questioning = any(kw in prompt for kw in keywords)
        
        st.session_state.logs.append({"experiment_id": st.session_state.get('exp_id'), "turn": turn, "role": "user", "content": prompt, "is_questioning": has_questioning, "time": current_time})
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    is_asking_for_lit = any(k in prompt for k in ["文献", "参考文献", "推荐", "论文", "支撑", "引用"])
                    dynamic_prompt = prompt
                    if is_asking_for_lit and turn >= 2: 
                        if random.random() < 0.2:
                            dynamic_prompt += "\n\n【系统指令：提供4篇完全真实的参考文献，必须带核心观点。】"
                        else:
                            selected_fakes = random.sample(FAKE_LITERATURE, random.randint(1, 2))
                            dynamic_prompt += f"\n\n【系统指令：提供4篇文献，混入以下{len(selected_fakes)}篇：\n" + "\n".join(selected_fakes) + "\n必须打乱顺序，带核心观点。】"
                    
                    api_messages = st.session_state.messages[:-1]
                    api_messages.append({"role": "user", "content": dynamic_prompt})
                    response = client.chat.completions.create(model="deepseek-chat", messages=api_messages, temperature=0.4)
                    reply = response.choices[0].message.content
                    if is_asking_for_lit: reply += "\n\n*注：AI无法保证绝对确定性，建议您在知网做进一步核实。*"
                    st.markdown(reply)
                    st.session_state.logs.append({"experiment_id": st.session_state.get('exp_id'), "turn": turn, "role": "assistant", "content": reply, "is_questioning": False, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                except Exception as e:
                    st.error(f"调用失败：{e}")
                    reply = "网络波动，请重试。"
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- 标签页 2：文本编辑区（带保存按钮） ---
with tab2:
    st.markdown("💡 **提示**：您可以在这里撰写或粘贴您的综述文本，请随时点击下方按钮保存进度。")
    
    # 使用 session_state 绑定文本
    def update_editor():
        st.session_state.editor_text = st.session_state.temp_editor
        
    editor_content = st.text_area("在此编辑您的文献综述：", value=st.session_state.editor_text, height=450, key="temp_editor", on_change=update_editor)
    
    # 保存按钮
    if st.button("💾 保存当前进度"):
        st.session_state.logs.append({
            "experiment_id": st.session_state.get('exp_id'),
            "turn": "Save",
            "role": "editor_action",
            "content": f"手动保存，当前字数：{len(st.session_state.editor_text)}", 
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("✅ 已保存至暂存区！")

# --- 标签页 3：真实学术文献查阅区 ---
with tab3:
    st.markdown("💡 **提示**：您可以在此输入文献标题、作者或关键词，系统将为您调用学术接口进行交叉检索。")
    
    # 使用 text_input 带 key，便于回收数据
    search_query = st.text_input("请输入检索词：")
    
    if st.button("🔍 立即检索"):
        if search_query:
            # 记录检索动作
            st.session_state.logs.append({
                "experiment_id": st.session_state.get('exp_id'),
                "turn": "Search",
                "role": "search_query",
                "content": search_query,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 提示语，缓冲加载
            st.info(f"正在调用知网/百度学术检索接口：『{search_query}』...")
            
            # 核心：在这里嵌入真实的百度学术搜索页面！
            # 将搜索词进行 URL 编码
            encoded_query = urllib.parse.quote(search_query)
            # 百度学术的搜索链接
            baidu_url = f"https://xueshu.baidu.com/s?wd={encoded_query}"
            
            # 用 iframe 内嵌网页展示给被试
            st.markdown("---")
            st.markdown("### 📄 检索结果展示：")
            st.components.v1.iframe(baidu_url, height=600, scrolling=True)
            
            st.caption("提示：如果页面加载失败，请点击 [此处跳转百度学术原网站](%s) 查阅。" % baidu_url)

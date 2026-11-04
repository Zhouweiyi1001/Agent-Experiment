import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
import random
import requests
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
st.title("实验Agent")

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
                    if is_asking_for_lit: reply += "\n\n*注：AI无法保证绝对确定性，建议您在查阅区做进一步核实。*"
                    st.markdown(reply)
                    st.session_state.logs.append({"experiment_id": st.session_state.get('exp_id'), "turn": turn, "role": "assistant", "content": reply, "is_questioning": False, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                except Exception as e:
                    st.error(f"调用失败：{e}")
                    reply = "网络波动，请重试。"
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- 标签页 2：文本编辑区（带实时字数统计） ---
with tab2:
    st.markdown("💡 **提示**：您可以在这里撰写或粘贴您的综述文本，请随时点击下方按钮保存进度。")
    
    def update_editor():
        st.session_state.editor_text = st.session_state.temp_editor
        
    editor_content = st.text_area(
        "在此编辑您的文献综述（建议 800-1000 字）：", 
        value=st.session_state.editor_text, 
        height=450, 
        key="temp_editor", 
        on_change=update_editor
    )
    
    current_text = st.session_state.get("temp_editor", "")
    char_count = len(current_text.replace(" ", "").replace("\n", ""))
    
    st.caption(f"📊 **当前字数统计：{char_count} 字** （不含空格与换行）")
    
    if st.button("💾 保存当前进度"):
        st.session_state.logs.append({
            "experiment_id": st.session_state.get('exp_id'),
            "turn": "Save",
            "role": "editor_action",
            "content": f"手动保存，当前字数：{char_count}", 
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("✅ 已保存至暂存区！")

# --- 标签页 3：学术文献查阅区（双引擎真实检索） ---
with tab3:
    st.markdown("💡 **提示**：您可以在此输入文献标题、作者或关键词进行交叉核对。系统内置了两个学术检索库供您使用。")
    
    # 检索输入框
    search_query = st.text_input("请输入检索词（支持中英文标题、作者、关键词）：")
    
    if search_query:
        # 记录检索行为（只要输入了就记录）
        st.session_state.logs.append({
            "experiment_id": st.session_state.get('exp_id'),
            "turn": "Search",
            "role": "search_query",
            "content": search_query,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 建立左右两列
        col1, col2 = st.columns(2)
        
        # 左侧：百度学术（适合查中文）
        with col1:
            st.markdown("### 🇨🇳 百度学术检索")
            encoded_query = urllib.parse.quote(search_query)
            baidu_url = f"https://xueshu.baidu.com/s?wd={encoded_query}"
            st.caption(f"正在检索：『{search_query}』")
            st.components.v1.iframe(baidu_url, height=500, scrolling=True)
            st.caption(f"如果网页未显示，请点击 [此处前往百度学术查看]({baidu_url})")
            
        # 右侧：OpenAlex（适合查英文和摘要）
        with col2:
            st.markdown("### 🌍 OpenAlex 国际学术库")
            with st.spinner("正在查询 OpenAlex..."):
                try:
                    resp = requests.get(
                        "https://api.openalex.org/works",
                        params={
                            "search": search_query,
                            "per-page": 5,
                            "select": "id,display_name,publication_year,authorships,primary_location,doi,abstract_inverted_index",
                            "mailto": "你的邮箱@example.com" # 请务必替换为你的真实邮箱
                        },
                        timeout=15
                    )
                    data = resp.json()
                    results = data.get("results", [])
                    
                    if not results:
                        st.warning(f"⚠️ OpenAlex 未找到『{search_query}』。")
                        st.caption("注意：若为真实中文文献，OpenAlex 可能未收录；请以左侧百度学术结果为准。")
                    else:
                        st.success(f"找到 {data['meta']['count']} 条结果：")
                        for i, work in enumerate(results, 1):
                            title = work.get("display_name", "无标题")
                            year = work.get("publication_year", "未知年份")
                            
                            authors_list = []
                            for auth in work.get("authorships", [])[:3]:
                                name = auth.get("author", {}).get("display_name", "")
                                if name and name not in authors_list:
                                    authors_list.append(name)
                            authors = ", ".join(authors_list)
                            
                            source = (work.get("primary_location") or {}).get("source") or {}
                            journal = source.get("display_name", "未知期刊")
                            
                            # 解码摘要
                            abstract = "无摘要。"
                            inv_index = work.get("abstract_inverted_index")
                            if inv_index:
                                word_positions = []
                                for word, pos_list in inv_index.items():
                                    for pos in pos_list:
                                        word_positions.append((pos, word))
                                word_positions.sort(key=lambda x: x[0])
                                abstract = " ".join([w for p, w in word_positions])
                                if len(abstract) > 150: abstract = abstract[:150] + "..."
                                
                            with st.expander(f"{i}. {title} ({year})"):
                                st.markdown(f"**作者**：{authors}")
                                st.markdown(f"**来源**：{journal}")
                                st.markdown(f"**摘要**：{abstract}")
                except Exception as e:
                    st.error(f"OpenAlex 查询失败：{e}")
    else:
        st.info("请输入检索词以同时调用百度学术和 OpenAlex 进行检索。")

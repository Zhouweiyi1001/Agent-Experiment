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

# ================= 1. 配置大模型 =================
api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ================= 2. 核心提示词与预设 =================
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

# ================= 3. 存档与恢复机制 =================
SAVE_DIR = "experiment_saves"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_save_path(exp_id):
    return os.path.join(SAVE_DIR, f"save_{exp_id}.json")

def load_data(exp_id):
    path = get_save_path(exp_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_data(exp_id, messages, logs, editor_text, start_time):
    path = get_save_path(exp_id)
    data = {
        "messages": messages,
        "logs": logs,
        "editor_text": editor_text,
        "start_time": start_time,
        "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# 新增：计算相对时间的辅助函数
def get_elapsed_time(start_time_str):
    if not start_time_str:
        return 0
    start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    elapsed = datetime.now() - start_dt
    return int(elapsed.total_seconds())

# ================= 4. 页面初始化 =================
st.set_page_config(page_title="实验Agent", layout="wide")

if "exp_id" not in st.session_state: st.session_state.exp_id = ""
if "messages" not in st.session_state: st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "logs" not in st.session_state: st.session_state.logs = []
if "editor_text" not in st.session_state: st.session_state.editor_text = ""
if "start_time" not in st.session_state: st.session_state.start_time = None
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False

# ================= 5. 登录/恢复面板 =================
if not st.session_state.is_logged_in:
    st.title("实验Agent - 登录")
    st.info("请输入您的被试编号（如 S01）。如果您之前中途退出，输入相同的编号即可恢复所有进度。")
    login_id = st.text_input("被试编号：", value="")
    
    if st.button("进入/恢复实验"):
        if login_id:
            st.session_state.exp_id = login_id
            saved_data = load_data(login_id)
            
            if saved_data:
                st.session_state.messages = saved_data["messages"]
                st.session_state.logs = saved_data["logs"]
                st.session_state.editor_text = saved_data["editor_text"]
                st.session_state.start_time = saved_data["start_time"]
                st.success(f"已成功恢复编号 {login_id} 的实验进度！")
            else:
                st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                st.session_state.logs = []
                st.session_state.editor_text = ""
                st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 记录实验正式开始
                st.session_state.logs.append({"experiment_id": login_id, "elapsed_seconds": 0, "role": "system_start", "content": "实验正式开始", "time": st.session_state.start_time})
                save_data(login_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
                st.success(f"新实验已开启！编号：{login_id}")
            
            st.session_state.is_logged_in = True
            time.sleep(1)
            st.rerun()
        else:
            st.warning("请输入编号！")

# ================= 6. 主界面 =================
else:
    # 计算耗时并实时显示
    elapsed_sec = get_elapsed_time(st.session_state.start_time)
    minutes, seconds = divmod(elapsed_sec, 60)
    
    st.title("实验Agent")
    st.caption(f"👤 当前被试：**{st.session_state.exp_id}** | ⏱ 任务耗时：**{minutes}分{seconds}秒** | 💾 系统已开启自动保存")

    # 记录标签页切换的埋点函数
    def log_tab_switch(tab_name):
        current_logs = st.session_state.logs
        # 避免连续重复记录同一个标签页
        if not current_logs or current_logs[-1].get("role") != "tab_switch" or current_logs[-1].get("content") != tab_name:
            st.session_state.logs.append({
                "experiment_id": st.session_state.exp_id,
                "elapsed_seconds": get_elapsed_time(st.session_state.start_time),
                "role": "tab_switch",
                "content": f"切换至：{tab_name}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)

    tab1, tab2, tab3 = st.tabs(["🤖 AI 对话交互区", "📝 本地文档编辑区", "🔍 学术文献查阅区"])

    # --- 标签页 1：AI 对话 ---
    with tab1:
        log_tab_switch("AI对话区")
        st.markdown("你好！我是你的专属学术科研助手。请在下方输入框输入指令开启对话。")
        for msg in st.session_state.messages[1:]: 
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("请输入指令..."):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            turn = (len(st.session_state.messages) // 2) + 1
            keywords = ["出处", "DOI", "真实吗", "查证", "核实", "链接", "确定", "哪一期"]
            has_questioning = any(kw in prompt for kw in keywords)
            
            st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": turn, "role": "user", "content": prompt, "is_questioning": has_questioning, "time": current_time})
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
            
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
                        
                        response = client.chat.completions.create(
                            model="deepseek-chat", messages=api_messages, temperature=0.4, timeout=30
                        )
                        reply = response.choices[0].message.content
                        if is_asking_for_lit: reply += "\n\n*注：AI无法保证绝对确定性，建议您在知网做进一步核实。*"
                        st.markdown(reply)
                        
                        st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": turn, "role": "assistant", "content": reply, "is_questioning": False, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
                        
                    except Exception as e:
                        st.error(f"网络波动或服务繁忙，请重新输入上一句话重试。错误信息：{str(e)[:50]}")
                        st.session_state.messages.pop() 

    # --- 标签页 2：文本编辑区 ---
    with tab2:
        log_tab_switch("文档编辑区")
        st.markdown("💡 **提示**：您可以在这里撰写或粘贴您的综述文本，系统会为您自动保存。")
        
        def update_editor():
            st.session_state.editor_text = st.session_state.temp_editor
            # 记录字数变化
            char_num = len(st.session_state.editor_text.replace(" ", "").replace("\n", ""))
            st.session_state.logs.append({
                "experiment_id": st.session_state.exp_id,
                "elapsed_seconds": get_elapsed_time(st.session_state.start_time),
                "turn": "Edit",
                "role": "edit_action",
                "content": f"字数变更为：{char_num}字", 
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
            
        editor_content = st.text_area(
            "在此编辑您的文献综述（建议 800-1000 字）：", 
            value=st.session_state.editor_text, 
            height=450, 
            key="temp_editor", 
            on_change=update_editor
        )
        
        current_text = st.session_state.get("temp_editor", "")
        char_count = len(current_text.replace(" ", "").replace("\n", ""))
        saved_data = load_data(st.session_state.exp_id)
        last_save_time = saved_data.get("last_saved", "未知") if saved_data else "未知"
        
        st.caption(f"📊 **当前字数：{char_count} 字** | 💾 自动保存时间：{last_save_time}")

    # --- 标签页 3：学术文献查阅区 ---
    with tab3:
        log_tab_switch("文献查阅区")
        st.markdown("💡 **提示**：您可以在此输入文献标题、作者或关键词进行交叉核对。")
        search_query = st.text_input("请输入检索词：")
        
        if search_query:
            st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "Search", "role": "search_query", "content": search_query, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🇨🇳 百度学术检索")
                encoded_query = urllib.parse.quote(search_query)
                baidu_url = f"https://xueshu.baidu.com/s?wd={encoded_query}"
                st.components.v1.iframe(baidu_url, height=500, scrolling=True)
            with col2:
                st.markdown("### 🌍 OpenAlex 国际学术库")
                with st.spinner("查询中..."):
                    try:
                        resp = requests.get("https://api.openalex.org/works", params={"search": search_query, "per-page": 5, "select": "display_name,publication_year,authorships,primary_location", "mailto": "你的邮箱@example.com"}, timeout=10)
                        data = resp.json()
                        results = data.get("results", [])
                        if not results:
                            st.warning(f"⚠️ OpenAlex 未找到『{search_query}』。请以左侧百度学术为准。")
                        else:
                            for i, work in enumerate(results, 1):
                                with st.expander(f"{i}. {work.get('display_name')} ({work.get('publication_year')})"):
                                    auths = [a['author']['display_name'] for a in work.get("authorships", [])[:3]]
                                    st.markdown(f"**作者**：{', '.join(auths)}")
                    except Exception as e:
                        st.error("OpenAlex 查询失败")

    # --- 侧边栏（提交与结束） ---
    with st.sidebar:
        st.header("任务提交")
        final_text = st.text_area("请将综述粘贴到此处并提交：", height=200)
        if st.button("完成任务并下载数据"):
            if len(final_text) > 20:
                st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "End", "role": "final_text", "content": final_text, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time)
                
                df = pd.DataFrame(st.session_state.logs)
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.success("✅ 数据已保存！请下载并发送给主试。")
                st.download_button(label="📥 下载实验数据(CSV)", data=csv, file_name=f"log_{st.session_state.exp_id}.csv", mime="text/csv")
            else:
                st.warning("字数太少！")

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

# ================= 2. 页面配置与全局字体放大 =================
st.set_page_config(page_title="实验Agent", layout="wide")

st.markdown("""
<style>
    /* 1. 放大侧边栏所有字体 */
    [data-testid="stSidebar"] * { font-size: 17px !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 22px !important; font-weight: bold; }
    [data-testid="stSidebar"] button { height: 3.5em; font-size: 18px !important; font-weight: bold; border-radius: 10px; }
    
    /* 2. 放大主界面的文本编辑框 */
    textarea { font-size: 16px !important; line-height: 1.6 !important; }

    /* 3. 放大登录页面的输入框和按钮 */
    input[type="text"] { font-size: 20px !important; height: 3.5em !important; }
    .stButton button { font-size: 18px !important; height: 3.5em !important; font-weight: bold; }

    /* 4. 放大主界面右侧的各类提示框、检索输入框和跳转按钮 */
    [data-testid="stNotification"] * { font-size: 18px !important; }
    [data-testid="stLinkButton"] a {
        font-size: 18px !important;
        font-weight: bold !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    .stTextInput input { font-size: 18px !important; height: 3em !important; }
</style>
""", unsafe_allow_html=True)

# ================= 3. 核心提示词与预设 =================
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

# ================= 4. 存档与计时辅助 =================
SAVE_DIR = "experiment_saves"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
def get_save_path(exp_id): return os.path.join(SAVE_DIR, f"save_{exp_id}.json")

def load_data(exp_id):
    path = get_save_path(exp_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return None

# 新增了 pre_test_done 参数存入 JSON
def save_data(exp_id, messages, logs, editor_text, start_time, final_text, is_finished=False, pre_test_done=False):
    data = {
        "messages": messages, "logs": logs, "editor_text": editor_text, 
        "start_time": start_time, "final_text": final_text, 
        "is_finished": is_finished, "pre_test_done": pre_test_done,
        "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(get_save_path(exp_id), "w", encoding="utf-8") as f: 
        json.dump(data, f, ensure_ascii=False)

def get_elapsed_time(start_time_str):
    if not start_time_str: return 0
    start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    return int((datetime.now() - start_dt).total_seconds())

# ================= 5. 页面初始化 =================
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

# ================= 6. 登录与恢复面板 =================
if not st.session_state.is_logged_in:
    st.title("实验Agent - 登录")
    st.info("请输入您的被试编号（如 S01）。如果中途退出，输入相同编号即可恢复进度。")
    login_id = st.text_input("被试编号：", value="")
    
    if st.button("进入/恢复实验"):
        if login_id:
            st.session_state.exp_id = login_id
            saved_data = load_data(login_id)
            
            if saved_data:
                if saved_data.get("is_finished", False):
                    st.error("⚠️ 该被试编号已完成实验并提交数据。为保证数据有效性，无法再次进入。感谢您的参与！")
                else:
                    # 【核心逻辑】恢复历史进度，读取 pre_test_done 状态
                    st.session_state.messages = saved_data["messages"]
                    st.session_state.logs = saved_data["logs"]
                    st.session_state.editor_text = saved_data["editor_text"]
                    st.session_state.final_text = saved_data.get("final_text", "")
                    st.session_state.start_time = saved_data["start_time"]
                    st.session_state.is_finished = False
                    st.session_state.pre_test_done = saved_data.get("pre_test_done", False) # 如果之前做了，这里就是True
                    
                    st.success(f"成功恢复编号 {login_id} 的进度！")
                    st.session_state.is_logged_in = True
                    time.sleep(1)
                    st.rerun()
            else:
                # 【核心逻辑】全新实验，前测设为 False
                st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                st.session_state.logs = [{"experiment_id": login_id, "elapsed_seconds": 0, "role": "system_start", "content": "实验正式开始", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
                st.session_state.editor_text = ""
                st.session_state.final_text = ""
                st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.is_finished = False
                st.session_state.pre_test_done = False 
                save_data(login_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, False, False)
                st.success(f"新实验已开启！请先完成前测问卷。")
                st.session_state.is_logged_in = True
                time.sleep(1)
                st.rerun()
        else:
            st.warning("请输入编号！")

# ================= 7. 主界面（登录后） =================
else:
    # 【拦截器】如果 pre_test_done 是 False，强制显示前测
    if not st.session_state.pre_test_done:
        st.title("请先完成前测问卷")
        st.warning("⚠️ 在开始文献综述任务前，请务必完成以下前测问卷。完成后点击下方按钮进入实验。")
        
        # 内嵌前测问卷
        st.components.v1.iframe("https://v.wjx.cn/vm/O608BaM.aspx#", height=800, scrolling=True)
        
        if st.button("✅ 我已提交前测问卷，进入正式实验", type="primary"):
            st.session_state.pre_test_done = True
            # 【核心逻辑】点击后，立刻将 pre_test_done 写入存档，防止刷新重做
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, False, True)
            st.rerun()
        st.stop() # 阻断下方代码，不让他提前看到实验界面

    # ---------- 左侧控制面板 ----------
    with st.sidebar:
        st.markdown(f"### 👤 被试编号：{st.session_state.exp_id}")
        elapsed_sec = get_elapsed_time(st.session_state.start_time)
        mins, secs = divmod(elapsed_sec, 60)
        st.info(f"⏱ **任务耗时：{mins} 分 {secs} 秒**")
        
        st.markdown("---")
        st.markdown("### 🧭 导航菜单")
        if st.button("🤖 AI 对话交互区", use_container_width=True):
            st.session_state.current_tab = "AI 对话交互区"; st.rerun()
        if st.button("📝 本地文档编辑区", use_container_width=True):
            st.session_state.current_tab = "本地文档编辑区"; st.rerun()
        if st.button("🔍 学术文献查阅区", use_container_width=True):
            st.session_state.current_tab = "学术文献查阅区"; st.rerun()
        
        st.markdown("---")
        st.markdown("### 📥 任务提交与后测")
        
        st.session_state.final_text = st.text_area("将终稿粘贴到此处：", value=st.session_state.final_text, height=180)
        
        if st.button("确认结束并下载数据", use_container_width=True, type="primary"):
            if len(st.session_state.final_text) > 20:
                st.session_state.is_finished = True
                st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "End", "role": "final_text", "content": st.session_state.final_text, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                
                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, True, True)
                
                df = pd.DataFrame(st.session_state.logs)
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.success("✅ 实验已正式结束并保存！请下载文件发送给主试。")
                st.download_button(label="📥 下载实验数据(CSV)", data=csv, file_name=f"log_{st.session_state.exp_id}.csv", mime="text/csv")
                
                # 自动弹出后测问卷
                st.markdown("---")
                st.markdown("### 🎉 请完成实验后测问卷")
                st.components.v1.iframe("https://v.wjx.cn/vm/rpfP2PE.aspx#", height=800, scrolling=True)
                st.info("完成问卷后，您可以关闭本网页。感谢您的参与！")
            else:
                st.warning("字数太少，请检查后再结束实验！")
        
        if st.session_state.is_finished:
            st.info("您已提交数据。请在上方完成最后的问卷。")

    # ---------- 右侧主工作区 ----------
    disabled_flag = st.session_state.is_finished
    st.title("实验Agent" + (" (已结束)" if disabled_flag else ""))

    def record_tab_switch(tab_name):
        if st.session_state.get("last_tab") != tab_name and not disabled_flag:
            st.session_state.logs.append({
                "experiment_id": st.session_state.exp_id,
                "elapsed_seconds": get_elapsed_time(st.session_state.start_time),
                "role": "tab_switch", "content": f"切换至：{tab_name}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
            st.session_state.last_tab = tab_name

    # 模块一：AI 对话交互区
    if st.session_state.current_tab == "AI 对话交互区":
        record_tab_switch("AI 对话交互区")
        st.subheader("🤖 AI 对话交互区")
        
        for msg in st.session_state.messages[1:]:
            content = msg["content"].replace("\n\n*注：AI无法保证绝对确定性，建议您在知网做进一步核实。*", "").replace("\n\n*注：AI无法保证绝对确定性，建议您在查阅区做进一步核实。*", "")
            with st.chat_message(msg["role"]): st.markdown(content)

        if prompt := st.chat_input("请输入指令...", disabled=disabled_flag):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            turn = (len(st.session_state.messages) // 2) + 1
            keywords = ["出处", "DOI", "真实吗", "查证", "核实", "链接", "确定", "哪一期"]
            has_questioning = any(kw in prompt for kw in keywords)
            
            st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": turn, "role": "user", "content": prompt, "is_questioning": has_questioning, "time": current_time})
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
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
                        response = client.chat.completions.create(model="deepseek-chat", messages=api_messages, temperature=0.4, timeout=30)
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": turn, "role": "assistant", "content": reply, "is_questioning": False, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
                    except Exception as e:
                        st.error(f"网络波动，请重试。错误信息：{str(e)[:50]}")
                        st.session_state.messages.pop() 
        
        st.info("💡 **系统提示**：以上内容由 AI 生成。若需验证文献真实性，请点击左侧按钮前往 **🔍 学术文献查阅区**。")

    # 模块二：本地文档编辑区
    elif st.session_state.current_tab == "本地文档编辑区":
        record_tab_switch("本地文档编辑区")
        st.subheader("📝 本地文档编辑区")
        
        def update_editor():
            if disabled_flag: return
            st.session_state.editor_text = st.session_state.temp_editor
            char_num = len(st.session_state.editor_text.replace(" ", "").replace("\n", ""))
            st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "Edit", "role": "edit_action", "content": f"字数变更为：{char_num}字", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
            
        st.text_area("在此编辑您的文献综述（建议 800-1000 字）：", value=st.session_state.editor_text, height=500, key="temp_editor", on_change=update_editor, disabled=disabled_flag)
        
        char_count = len(st.session_state.get("temp_editor", "").replace(" ", "").replace("\n", ""))
        saved_data = load_data(st.session_state.exp_id)
        last_save_time = saved_data.get("last_saved", "未知") if saved_data else "未知"
        st.caption(f"📊 **当前字数：{char_count} 字** | 💾 自动保存时间：{last_save_time}")

    # 模块三：学术文献查阅区
    elif st.session_state.current_tab == "学术文献查阅区":
        record_tab_switch("学术文献查阅区")
        st.subheader("🔍 学术文献查阅区")
        st.markdown("💡 提示：请在此输入文献标题、作者进行交叉核对。**如需在外部数据库核实，请点击专属链接，核实后请点击下方的确认按钮。**")
        
        search_query = st.text_input("请输入需要核实的检索词：", disabled=disabled_flag)
        
        if search_query:
            st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "Search", "role": "search_query", "content": search_query, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
            
            st.markdown("### 🇨🇳 中文文献核对")
            st.info(f"系统已记录您的检索词：『**{search_query}**』。请点击下方按钮前往百度学术，**查阅完毕后请回到本页面点击确认**。")
            
            encoded_query = urllib.parse.quote(search_query)
            baidu_url = f"https://xueshu.baidu.com/s?wd={encoded_query}"
            
            st.link_button("👉 前往百度学术核实此文献（点击将在新标签页打开）", baidu_url, type="primary", use_container_width=True)
            
            if st.button("✅ 我已查阅完毕，确认此文献的真伪", use_container_width=True):
                st.session_state.logs.append({"experiment_id": st.session_state.exp_id, "elapsed_seconds": get_elapsed_time(st.session_state.start_time), "turn": "Confirm", "role": "search_confirm", "content": f"确认核实了：{search_query}", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                save_data(st.session_state.exp_id, st.session_state.messages, st.session_state.logs, st.session_state.editor_text, st.session_state.start_time, st.session_state.final_text, st.session_state.is_finished, st.session_state.pre_test_done)
                st.success("✅ 已记录您的核实行为！")
            
            st.markdown("---")

            st.markdown("### 🌍 OpenAlex 国际学术库")
            with st.spinner("正在检索全球学术数据库..."):
                try:
                    resp = requests.get(
                        "https://api.openalex.org/works", 
                        params={"search": search_query, "per-page": 8, "select": "display_name,publication_year,authorships,primary_location,doi", "api_key": "ZN6iRrep1uN1v2FimduMEC"}, 
                        timeout=20
                    )
                    data = resp.json()
                    results = data.get("results", [])
                    
                    if not results:
                        st.warning(f"⚠️ OpenAlex 未找到『{search_query}』。该文献可能不存在或未被收录，请谨慎引用。")
                    else:
                        st.success(f"找到 {data['meta']['count']} 条结果：")
                        for i, work in enumerate(results, 1):
                            with st.expander(f"{i}. {work.get('display_name')} ({work.get('publication_year')})"):
                                auths = [a['author']['display_name'] for a in work.get("authorships", [])[:3]]
                                st.markdown(f"**作者**：{', '.join(auths)}")
                                journal = (work.get("primary_location") or {}).get("source") or {}
                                st.markdown(f"**来源**：{journal.get('display_name', '未知期刊')}")
                                if work.get("doi"):
                                    st.markdown(f"**DOI**：[{work['doi']}]({work['doi']})")
                except Exception as e:
                    st.error(f"检索失败，请检查网络或稍后再试。错误信息：{str(e)[:50]}")

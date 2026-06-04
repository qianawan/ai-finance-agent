import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Finance Agent", layout="wide", page_icon="📊")
st.title("📊 AI Finance Agent - 智能财务账单分析系统")

# API Key: secrets → 环境变量 → 侧边栏手动输入
try:
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
except Exception:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    with st.sidebar:
        st.warning("🔑 未检测到预设的 API Key，请输入您的密钥以激活 AI 功能：")
        DEEPSEEK_API_KEY = st.text_input("DeepSeek API Key", type="password",
            help="您的密钥将仅用于本次网页会话，绝不会被保存。")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1") if DEEPSEEK_API_KEY else None

def top_categories(df, col, n=5):
    """返回前 N 大类别及金额的格式化字符串。"""
    if col not in df.columns or df.empty:
        return "无"
    return ", ".join(
        f"{idx}:¥{val:.0f}" for idx, val in
        df.groupby(col)["交易金额"].sum().abs().sort_values(ascending=False).head(n).items()
    )

def stream_chat(client, messages, placeholder):
    """流式输出 AI 回复并逐段渲染。"""
    response = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True)
    full = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full += chunk.choices[0].delta.content
            placeholder.markdown(full)
    return full

uploaded_file = st.file_uploader("请上传您的 MoneyThings 或标准记账 CSV 账单文件", type=["csv"])

if uploaded_file is None:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)

    # 清洗金额列：去掉千分位逗号，转成数字（MoneyThings 导出带逗号）
    df["交易金额"] = pd.to_numeric(
        df["交易金额"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )

    st.success("数据导入成功！正在为您进行智能化财务诊断...")

    # 过滤非消费行为（转账/提现/充值等）
    filter_keywords = "转账|提现|充值|内部转账|代付报销"
    for col in ["交易类型", "标签"]:
        if col in df.columns:
            df = df[~df[col].str.contains(filter_keywords, na=False, case=False)]
            break

    # 分离收支
    if "收支类型" in df.columns:
        expense = df[df["收支类型"] == "支出"]
        income = df[df["收支类型"] == "收入"]
    elif "交易类型" in df.columns:
        expense = df[df["交易类型"] == "支出"]
        income = df[df["交易类型"] == "收入"]
    else:
        expense = df[df["交易金额"] < 0]
        income = df[df["交易金额"] > 0]

    expense_total = expense["交易金额"].abs().sum()
    income_total = income["交易金额"].sum()
    net = income_total - expense_total

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 期间真实总收入", f"¥ {income_total:,.2f}")
    col2.metric("📉 期间真实总支出", f"¥ {expense_total:,.2f}")
    col3.metric("⚖️ 期间真实净现金流", f"¥ {net:,.2f}", delta=f"{net:,.2f}")

    col_left, col_right = st.columns(2)
    for col, label, key in [(col_left, "场景分析 (含收支汇总)", "场景"),
                             (col_right, "标签分析 (含收支汇总)", "标签")]:
        if key in df.columns:
            col.subheader(f"🏠 {label}" if key == "场景" else f"🏷️ {label}")
            data = df.groupby(key)["交易金额"].sum().abs().sort_values(ascending=False)
            col.dataframe(data, use_container_width=True)

    exp_scene_text = top_categories(expense, "场景")
    exp_tag_text = top_categories(expense, "标签")
    inc_scene_text = top_categories(income, "场景")
    inc_tag_text = top_categories(income, "标签")

    summary = (
        f"【总览】总收:¥{income_total:.0f}, 总支:¥{expense_total:.0f}, 净:¥{net:.0f}。\n"
        f"【支出详情】核心场景:{exp_scene_text}。核心标签:{exp_tag_text}。\n"
        f"【收入详情】核心场景:{inc_scene_text}。核心标签:{inc_tag_text}。"
    )

    st.write("---")
    st.subheader("🤖 AI 深度财务核心诊断报告")

    if st.button("🚀 开始智能化生成财务报告"):
        if not client:
            st.error("请先在左侧配置或输入您的 DeepSeek API Key 才能激活 AI 诊断服务。")
        else:
            with st.spinner("AI 财务专家正在严密审计您的账单结构，请稍候..."):
                try:
                    messages = [
                        {"role": "system", "content": "你是一位深谙消费心理学与行为金融学的高级财务分析师。"},
                        {"role": "user", "content": f"""请结合以下账单数据提供核心财务诊断。
数据背景：{summary}

【严重警告】：
1. 提供的数据仅为特定时间段内的汇总金额，不含具体交易时间轴。分析时严禁自行脑补、虚构或猜测交易的具体日期、时间先后顺序或不存在的因果关系（例如严禁脑补旅行时间重合等现象）！
2. 必须严格且仅基于各项支出的金额比例进行客观推断。

要求严格分为5项输出，语言要利落、直击痛点、严禁套话（每项不超过80字）：
1. 收入支出结构健康度
2. 冲动或情绪消费痕迹
3. 高频隐形成本透视
4. 疑似浪费或可砍掉的支出
5. 3条下个月切实可行的财务优化建议。"""},
                    ]
                    placeholder = st.empty()
                    stream_chat(client, messages, placeholder)
                except Exception as e:
                    if "402" in str(e) or "Insufficient Balance" in str(e):
                        st.error("❌ 报告生成失败：DeepSeek 账户余额不足。请前往 platform.deepseek.com 充值。")
                    else:
                        st.error(f"❌ 运行出错：{str(e)}")

    st.write("---")
    st.subheader("💬 AI 智能财务顾问双向对话")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("您可以继续追问资产与流水明细..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not client:
            st.error("请先配置 API Key。")
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                chat_context = [
                    {"role": "system", "content": """你是一位严谨的财务分析师。
【重要守则】：
1. 严格区分收入与支出，绝对不可混淆。
2. 用户提供的数据仅为『某段时间内的现金流水』，不包含用户的『期初已有资产』或『总存款』。
3. 如果用户询问"我实际有多少钱"、"为什么余额对不上"，你必须明确告知："我只能看到这段时间的流水净收入，无法知晓您之前的存款和理财资产，因此无法计算您的真实总可用资金。"
4. 严禁为了盲目迎合用户而胡编乱造、强行敷衍财务因果关系。
请用150字以内一针见血地回答，不讲客套话。"""},
                    {"role": "user", "content": f"这是我的流水基础数据背景：{summary}。请作为背景参考，切勿过度脑补。"},
                ]
                for msg in st.session_state.messages[-6:]:
                    chat_context.append(msg)

                try:
                    reply = stream_chat(client, chat_context, placeholder)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"出错了：{str(e)}")

except Exception as e:
    st.error(f"解析 CSV 文件失败，请确保格式正确。错误详情: {e}")

import streamlit as st
import re
import pandas as pd
import traceback
from tasks import process_document_pipeline  # 🏎️ Import our background task wrapper

# --- Page Configuration Layout ---
st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="💼",
    layout="wide"
)

# Initialize Session State Managers
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []

def execute_formula_toolkit(toolkit_string):
    """
    🛠️ PRECISION FINANCIAL CALCULATOR TOOLKIT:
    Intercepts computation demands and processes pure python floating-point arithmetic.
    """
    try:
        type_match = re.search(r"TYPE=([^,]+)", toolkit_string)
        vals_match = re.search(r"VALUES=\[(.+)\]", toolkit_string)
        
        if type_match and vals_match:
            calc_type = type_match.group(1).strip()
            raw_vals = [float(x.strip()) for x in vals_match.group(1).split(",")]
            
            if calc_type == "CAGR" and len(raw_vals) >= 2:
                start_val = raw_vals[0]
                end_val = raw_vals[-1]
                periods = len(raw_vals) - 1
                
                if start_val > 0 and periods > 0:
                    cagr_value = ((end_val / start_val) ** (1 / periods)) - 1
                    percentage_str = f"{cagr_value * 100:.2f}%"
                    
                    st.success("⚙️ Formula Toolkit Core Computation Engine Verified")
                    st.metric(
                        label=f"Computed Compound Annual Growth Rate (CAGR) | {periods} Fiscal Periods", 
                        value=percentage_str,
                        delta="Airtight Mathematical Audit Execution"
                    )
    except Exception as e:
        st.error(f"Calculator Toolkit Exception: {str(e)}")

def render_response_and_charts(content_text):
    """
    Splits out data series tokens and structurally draws the markdown chat responses
    alongside beautiful interactive data visualization panels with production-grade guardrails.
    """
    if content_text is None:
        st.error("Received empty response payload from analysis engine.")
        return

    # Ensure we are handling a clean string
    clean_text = str(content_text)
    data_series_string = None
    toolkit_string = None
    
    lines = clean_text.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith("DATA_SERIES:"):
            data_series_string = line.replace("DATA_SERIES:", "").strip()
        elif line.startswith("CALC_TOOLKIT:"):
            toolkit_string = line.replace("CALC_TOOLKIT:", "").strip()
        else:
            filtered_lines.append(line)
            
    clean_text = "\n".join(filtered_lines).strip()
    st.markdown(clean_text)
    
    if toolkit_string:
        execute_formula_toolkit(toolkit_string)
        
    if data_series_string:
        try:
            pairs = data_series_string.split(",")
            chart_data = []
            for pair in pairs:
                # 🛡️ DEFENSIVE GUARDRAIL: Ensure it's a valid data pair, avoiding header decorations
                if pair and "=" in pair and not pair.startswith("==="):
                    parts = pair.split("=")
                    if len(parts) == 2:  # Must cleanly split into exactly key and value
                        year, val = parts[0].strip(), parts[1].strip()
                        if year and val:
                            chart_data.append({
                                "Fiscal Year": str(year),
                                "Value (Millions USD)": float(val)
                            })
            if chart_data:
                df = pd.DataFrame(chart_data)
                df.set_index("Fiscal Year", inplace=True)
                with st.expander("📊 Interactive Metric Visualization Dashboard", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(df)
                    with col2:
                        st.line_chart(df)
        except Exception as chart_err:
            # Silence chart rendering glitches so the core financial text response is never blocked
            pass

# --- Sidebar Controls Layout ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Clear Chat History", type="secondary"):
        st.session_state.ui_chat_history = []
        st.rerun()

# --- Main Screen Conversation Timeline ---
st.title("💼 AI Financial Analyst Agent")
st.markdown("---")

# Render historical messages in the sequence viewport
for message in st.session_state.ui_chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_response_and_charts(message["content"])
        else:
            st.markdown(message["content"])

# Collect and dispatch fresh text input submissions
if user_query := st.chat_input("Ask the agent a financial question:"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.ui_chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing multi-document context and computing values..."):
            try:
                # 🚀 Dispatch task execution array out to our Redis message frame
                task_handle = process_document_pipeline.delay(user_query)
                
                # 🔄 Poll Redis safely without locking CPU execution
                while not task_handle.ready():
                    import time
                    time.sleep(0.4)
                
                # 🎁 Extract payload output block from the task
                response = task_handle.result
                
                if response is None:
                    error_msg = "An execution error occurred: The processing engine returned a null state response."
                    st.error(error_msg)
                    st.session_state.ui_chat_history.append({"role": "assistant", "content": error_msg})
                else:
                    clean_response = str(response)
                    render_response_and_charts(clean_response)
                    st.session_state.ui_chat_history.append({"role": "assistant", "content": clean_response})
                    
            except Exception as e:
                # 🛡️ DEFENSIVE DEBUG PANEL
                error_trace = traceback.format_exc()
                st.error("⚠️ **AI Financial Agent Pipeline Exception Encountered**")
                st.code(error_trace, language="text")
                st.session_state.ui_chat_history.append({"role": "assistant", "content": f"Pipeline execution fault."})
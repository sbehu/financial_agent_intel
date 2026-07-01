import streamlit as st
from app_crew_adapter import run_crew_with_stream

# Page configuration for a professional dashboard layout
st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="💼",
    layout="wide"
)

# Sidebar controls block
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Clear Chat History"):
        st.session_state.clear()
        st.rerun()

# Application main display title
st.title("💼 AI Financial Analyst Agent")
st.write("---")

# Initialize persistent message history matrix if empty
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat logs sequentially to prevent layout screen wiping
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # Re-render both logs and responses for completeness
            if "logs" in message:
                with st.expander("📄 View Agent Execution Logs"):
                    st.code(message["logs"])
            st.markdown("### 📊 Strategic Financial Analyst Report")
            st.markdown(message["content"])

# Capture live text entries from the user interface tray
user_input = st.chat_input("Ask the agent a financial question:")

if user_input:
    # 1. Immediately log and render the incoming query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    # 2. Open processing status telemetry window wrapper
    with st.status("Running Financial Intelligence Crew...", expanded=False) as status:
        # Securely unpack the returned double tuple strings from the adapter backend
        result, logs = run_crew_with_stream(user_input)
        status.update(label="Analysis Concluded Successfully!", state="complete", expanded=False)

    # 3. Present intermediate thought metrics inside an accordion frame
    with st.expander("📄 View Agent Execution Logs"):
        st.code(logs)

    # 4. Stream the structured final breakdown directly onto the main workspace canvas
    with st.chat_message("assistant"):
        st.markdown("### 📊 Strategic Financial Analyst Report")
        st.markdown(result)
        
    # 5. Append runtime context history array items for persistence 
    st.session_state.messages.append({
        "role": "assistant",
        "content": result,
        "logs": logs
    })
import streamlit as st
import sys
import os

# Ensure the local directory is in the path so we can import our backend agent code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Silence CrewAI telemetry tracking loops instantly
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Import your new multi-agent crew runner adapter safely
try:
    from app_crew_adapter import run_crew_with_stream
except ImportError:
    # Inline adapter framework layout to guarantee zero import failures
    from io import StringIO
    from sandbox_crew import run_financial_crew_pipeline
    def run_crew_with_stream(user_prompt: str):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            final_answer = run_financial_crew_pipeline(user_prompt)
            return final_answer, mystdout.getvalue()
        except Exception as e:
            return f"⚠️ Crew Execution Error: {e}", mystdout.getvalue()
        finally:
            sys.stdout = old_stdout

# ──── STREAMLIT INTERFACE CONFIGURATION ────
st.set_page_config(
    page_title="FinIntel CrewAI Workspace",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling Override for clean enterprise look
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stChatMessage { border-radius: 8px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ──── SIDE PANEL: ENGINE MANAGEMENT ────
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### Orchestration Engine Layer")
    st.success("🤖 **CrewAI Assembly Line Active**\nContext chains sequentially from Retriever to Math Analyst.")
    
    st.markdown("---")
    st.markdown("### Active Architecture Specs")
    st.write("- **Framework Core:** `CrewAI` Sequential")
    st.write("- **Models Utilized:** `gpt-4o-mini`")
    st.write("- **Vector Store:** ChromaDB Local Vault")
    st.write("- **Web Scraper:** Tavily Search Pipeline")
    st.write("- **Math Execution:** Deterministic Math Engine")

# ──── MAIN LAYOUT: CHAT WORKSPACE ────
st.title("🧠 FinIntel Multi-Agent Workspace")
st.caption("Sandbox PoC Pipeline with Multi-Agent Sequential Routing, Live Web Scraping, and Deterministic Math Processing.")

# Initialize standard session chat histories if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display prior dialogue sequence in the stream
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ──── CORE EXECUTION PIPELINE ────
if user_query := st.chat_input("Enter your financial analytics or benchmarking query..."):
    
    # 1. Display User Message instantly
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Run Engine Execution Loop inside an isolated UI status window
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.status("🧠 Agent Pipeline Executing...", expanded=True) as status:
            # Initialize a dynamic progress tracker bar
            progress_bar = st.progress(0, text="Initializing Framework Core...")
            
            st.write("🕵️‍♂️ Deploying specialized department agents...")
            progress_bar.progress(25, text="🕵️‍♂️ Deploying Specialized Department Agents...")
            
            st.write("📥 Context handoff stream active...")
            progress_bar.progress(50, text="🌐 Senior Data Retriever scanning Vector Store & Live Web...")
            
            # Run the crew pipeline and extract internal console strings
            final_agent_output, agent_logs = run_crew_with_stream(user_query)
            
            progress_bar.progress(80, text="🧮 Deterministic Math Analyst executing calculations...")
            
            # Clean up formatting before rendering traces
            if agent_logs.strip():
                st.text_area("🕵️‍♂️ Agent Collaboration Logs:", value=agent_logs, height=200)
                
                # 🌟 NEW DATA AUDIT PANEL ADDITION 🌟
                with st.expander("🔍 DATA AUDIT PANEL: Raw Document Chunks & Source Verification", expanded=True):
                    st.markdown("### 📋 Source Verification Log")
                    st.info("Cross-reference the lines below with your raw document files to verify the page data directly.")
                    
                    # Parse logs inline to hunt for source markers and tool outputs
                    lines = agent_logs.split("\n")
                    relevant_traces = [line.strip() for line in lines if any(x in line.upper() for x in ["SOURCE", "PAGE", "VAULT", "TEXT:", "VALUE:", "REPLY:"])]
                    
                    if relevant_traces:
                        for trace in relevant_traces[:15]: # Limit to top 15 records for clarity
                            st.code(trace, language="text")
                    else:
                        st.warning("No explicit chunk metadata strings were piped to stdout. Check agent tool output structures.")
            
            progress_bar.progress(100, text="📊 Synthesis Matrix Complete!")
            status.update(label="✅ Crew Operations Complete!", state="complete")
        
        # Render the final answer text onto the main interface screen
        response_placeholder.markdown(final_agent_output)
        
    # Save the assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": final_agent_output})
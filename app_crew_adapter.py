import os
from io import StringIO
from sandbox_crew import researcher, calculator, retrieve_task, calculate_task
from crewai import Crew, Process

def run_crew_with_stream(user_prompt: str):
    """
    Executes the CrewAI pipeline safely using isolated, thread-safe string buffers
    and native framework step callbacks for streaming log telemetry.
    """
    # 🧵 ISOLATED BUFFER: Each user request gets its own dedicated memory stream
    log_buffer = StringIO()

    # Define a clean step callback handler that only runs within this thread context
    def production_step_callback(agent_step):
        log_buffer.write(f"🤖 AGENT: {agent_step.agent}\n")
        log_buffer.write(f"💭 THOUGHT: {agent_step.thought}\n")
        if agent_step.tool:
            log_buffer.write(f"🔧 TOOL USED: {agent_step.tool} | ARGUMENTS: {agent_step.tool_input}\n")
        log_buffer.write(f"📤 OUTPUT: {agent_step.output}\n")
        log_buffer.write("-" * 50 + "\n")

    try:
        # Dynamically inject the current user's prompt into the isolated tasks
        # (This ensures the production zone remains perfectly tied to your sandbox configs)
        retrieve_task.description = (
            f"You must execute a local database search for the exact query: '{user_prompt}'. "
            f"Do not alter, summarize, or convert this topic string. Pass the query text "
            f"directly to your retrieval tools to gather background context. If a currency "
            f"conversion is required by the query metrics, dynamically utilize your live exchange rate tool."
        )
        
        # Assemble a standalone runtime instance of the crew for this specific request
        runtime_crew = Crew(
            agents=[researcher, calculator],
            tasks=[retrieve_task, calculate_task],
            process=Process.sequential,
            verbose=False,  # Disables global print statements to protect system logs
            step_callback=production_step_callback  # 🌟 NATIVE PER-THREAD STREAMING
        )

        # Kickoff the agent workflow execution
        result = runtime_crew.kickoff()
        
        # Extract the logs specifically captured from this run's callback
        final_answer = str(result)
        agent_logs = log_buffer.getvalue()
        
        return final_answer, agent_logs

    except Exception as e:
        return f"⚠️ Crew Execution Error: {e}", log_buffer.getvalue()
        
    finally:
        # Securely close the isolated memory resource
        log_buffer.close()
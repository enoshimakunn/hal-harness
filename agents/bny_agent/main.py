import os
import platform
import shutil
import time
import tarfile
import socket

from prompts import *
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr


def find_free_port(start=8010, end=8100):
    """Find an available port."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start

from openhands.sdk import LLM, Agent, Tool, get_logger, Conversation, RemoteConversation
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def detect_platform():
    """Detects the correct Docker platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    logger.info(f"🔔 Running agent with input: {input}")
    logger.info(f"🔔 Running agent with kwargs: {kwargs}")
    cur_dir = Path(__file__).parent
    
    # Create directory for skills and solutions 
    (cur_dir / "skills").mkdir(parents=True, exist_ok=True)
    (cur_dir / "solutions" / "swebench").mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM
    load_dotenv(dotenv_path=cur_dir / ".env")
    llm = LLM(
        model=kwargs["model_name"],
        api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
        base_url=os.getenv("BASE_URL"),
    )
    logger.info(f"🔔 LLM initialized: {llm.model}")
    
    # Set up isolated container for agent execution
    logger.info(f"🔔 Mount directory: {str(cur_dir)}")
    logger.info("🔔 Starting workspace...")
    port = find_free_port()
    logger.info(f"🔔 Using port: {port}")
    with DockerWorkspace(
        # use pre-built image for faster startup (recommended)
        server_image="ghcr.io/openhands/agent-server:latest-python",
        host_port=port,
        platform=detect_platform(),
        mount_dir=str(cur_dir)
    ) as workspace:
        time.sleep(2)
        
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TaskTrackerTool.name),
                Tool(name=FileEditorTool.name),
                # Tool(name=TerminalTool.name),
            ],
        )
        
        received_events: list = []
        last_event_time = {"ts": time.time()}
    
        def event_callback(event) -> None:
            event_type = type(event).__name__
            logger.info(f"🔔 Callback received event: {event_type}\n{event}")
            received_events.append(event)
            last_event_time["ts"] = time.time()
        
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[event_callback],
        )
        assert isinstance(conversation, RemoteConversation)

        try:
            logger.info(f"\n📋 Conversation ID: {conversation.state.id}")
            logger.info("📝 Running task...")
            
            # Code for task execution
            assert len(input) == 1, "input must contain only one task"
            task_id, task = list(input.items())[0]
            
            prompt = SWE_BENCH_PROMPT_TEMPLATE.format(problem_description=task['problem_statement'])
            conversation.send_message(prompt)
            conversation.run()
            logger.info("✅ Task completed!")
        except Exception as e:
            logger.error(f"❌ Error running task: {e}")
            raise e
        finally:
            conversation.close()
        
        # Download results from docker container back to host
        workspace.file_download(
            source_path="/workspace/solution.md",
            destination_path=cur_dir / "solutions" / "swebench" / f"{task_id}.md"
        )
        workspace.execute_command("tar -czvf /workspace/skills.tar.gz /workspace/skills")
        workspace.file_download(
            source_path="/workspace/skills.tar.gz",
            destination_path=cur_dir / f"skills.tar.gz"
        )
        
        # Shutdown workspace
        workspace.cleanup()
        
    # Post-processing
    # 1. Extract skills.tar.gz to a temporary directory and compare the new skills with existing skills, add new skills to skills/
    with tarfile.open(cur_dir / f"skills.tar.gz", "r:gz") as tar:
        tar.extractall(path=cur_dir / "temp_skills")
    new_skills = list(Path(cur_dir / "temp_skills").glob("*"))
    old_skills = list(Path(cur_dir / "skills").glob("*"))
    for skill in new_skills:
        if skill not in old_skills:
            shutil.copy(skill, cur_dir / "skills" / skill.name)
    shutil.rmtree(cur_dir / "temp_skills")
    
    # 2. Read solution.md and submit the solution
    results = {}
    results[task_id] = open(cur_dir / "solutions" / "swebench" / f"{task_id}.md", "r").read()
    
    return results
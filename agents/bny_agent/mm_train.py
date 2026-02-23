import base64
import dotenv
import json
import mimetypes
import os
import re
import subprocess
import tempfile

import openai

from typing import *
from pathlib import Path
from prompts import MMSkillTrainer_PROMPT_TEMPLATE


dotenv.load_dotenv()

BASE_URL = os.getenv("BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-3-flash-preview"

client = openai.OpenAI(base_url=BASE_URL, api_key=OPENROUTER_API_KEY)


class MMSkillTrainer:
    skill_base_dir: Path = Path(__file__).parent / "skills"
    skill_base_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_skill(self, skill_name: str, skill_description: str):
        skill_file_path = self.skill_base_dir / skill_name / "SKILL.md"
        skill_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(skill_file_path, "w") as f:
            f.write(f"{skill_description}\n")
            
        print(f"[MMTrainer] Saved skill {skill_name} to {skill_file_path}")
        
    # Python's mimetypes maps .mov to video/quicktime, but OpenRouter
    # expects video/mov per their supported formats list.
    _MIME_OVERRIDES: dict[str, str] = {
        "video/quicktime": "video/mov",
    }

    @staticmethod
    def _compress_video(file_path: str) -> str:
        """Compress a video to mp4 to stay within API payload limits.

        Returns the path to a temporary compressed file. The caller is
        responsible for cleaning it up.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        print(f"[MMTrainer] Compressing video {file_path} -> {tmp.name}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-c:a", "aac", "-b:a", "64k",
                tmp.name,
            ],
            check=True,
            capture_output=True,
        )
        orig = os.path.getsize(file_path)
        compressed = os.path.getsize(tmp.name)
        print(f"[MMTrainer] Compressed {orig // 1024}KB -> {compressed // 1024}KB")
        return tmp.name

    @staticmethod
    def _compress_audio(file_path: str) -> str:
        """Compress audio to mp3 to stay within API payload limits.

        Returns the path to a temporary compressed file. The caller is
        responsible for cleaning it up.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        print(f"[MMTrainer] Compressing audio {file_path} -> {tmp.name}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-c:a", "libmp3lame", "-b:a", "64k",
                tmp.name,
            ],
            check=True,
            capture_output=True,
        )
        orig = os.path.getsize(file_path)
        compressed = os.path.getsize(tmp.name)
        print(f"[MMTrainer] Compressed {orig // 1024}KB -> {compressed // 1024}KB")
        return tmp.name

    @staticmethod
    def _encode_media(file_path: str) -> tuple[str, str]:
        """Read a media file and return its base64 encoding and MIME type."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for {file_path}")
        mime_type = MMSkillTrainer._MIME_OVERRIDES.get(mime_type, mime_type)
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return data, mime_type

    @staticmethod
    def _prepare_file(file_path: str) -> tuple[str, str | None]:
        """Compress if needed and return (usable_path, temp_path_or_None)."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("video/"):
            compressed = MMSkillTrainer._compress_video(file_path)
            return compressed, compressed
        if mime_type and mime_type.startswith("audio/"):
            compressed = MMSkillTrainer._compress_audio(file_path)
            return compressed, compressed
        return file_path, None

    def train(self, input_file_paths: str | list[str]):
        if isinstance(input_file_paths, str):
            input_file_paths = [input_file_paths]

        prepared: list[tuple[str, str | None]] = []
        try:
            for path in input_file_paths:
                prepared.append(self._prepare_file(path))
            self._train_impl([p for p, _ in prepared])
        finally:
            for _, tmp in prepared:
                if tmp:
                    os.unlink(tmp)

    @staticmethod
    def _build_media_part(data: str, mime_type: str) -> dict:
        """Build the correct OpenRouter content part for the media type.

        Video uses ``video_url`` with a data-URL.
        Audio uses ``input_audio`` with raw base64 + format.
        Docs: https://openrouter.ai/docs/features/multimodal/
        """
        if mime_type.startswith("video/"):
            return {
                "type": "video_url",
                "video_url": {"url": f"data:{mime_type};base64,{data}"},
            }
        if mime_type.startswith("audio/"):
            # Extract format from mime type (e.g. "audio/mp3" -> "mp3")
            audio_fmt = mime_type.split("/", 1)[1]
            return {
                "type": "input_audio",
                "input_audio": {"data": data, "format": audio_fmt},
            }
        raise ValueError(f"Unsupported media type: {mime_type}")

    @staticmethod
    def _is_text_file(file_path: str) -> bool:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type is not None and mime_type.startswith("text/")

    @staticmethod
    def _read_text_file(file_path: str) -> dict:
        with open(file_path, "r") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        return {"type": "text", "text": f"[File: {filename}]\n{content}"}

    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "List the names of all existing skills.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": "Read the full content of a specific skill file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The name of the skill to read.",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        },
    ]

    _TOOL_DISPATCH: dict[str, Callable[..., str]] = {
        "list_skills": lambda **_kw: json.dumps(_list_skills()),
        "read_skill": lambda skill_name, **_kw: _read_skill(skill_name),
    }

    def _train_impl(self, input_file_paths: list[str]):
        content_parts: list[dict] = []
        for path in input_file_paths:
            if self._is_text_file(path):
                content_parts.append(self._read_text_file(path))
            else:
                data, mime_type = self._encode_media(path)
                content_parts.append(self._build_media_part(data, mime_type))

        messages: list[dict] = [
            {"role": "system", "content": MMSkillTrainer_PROMPT_TEMPLATE},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this media and extract skills from it."},
                *content_parts,
            ]},
        ]

        print(f"[MMTrainer] Sending request to API to train the skill ({len(content_parts)} file(s))")
        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=self._TOOLS,
            )
            if not response.choices:
                raise RuntimeError(
                    f"API returned no choices. Full response: {response.model_dump_json(indent=2)}"
                )

            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                break

            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"[MMTrainer] Tool call: {fn_name}({fn_args})")
                result = self._TOOL_DISPATCH[fn_name](**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        output = choice.message.content
        print(f"[MMTrainer] Response: {output}")
        
        # Parse the output into a skill
        skill_names = []
        skill_descriptions = []
        for skill in output.split("<skill_name>")[1:]:
            skill_name = skill.split("</skill_name>")[0]
            skill_description = skill.split("<skill_description>")[1].split("</skill_description>")[0]
            
            # Regularize the skill name to contain only:
            # 1. lowercase alphabetic characters and numbers
            # 2. hyphens
            skill_names.append(re.sub(r"[^a-z0-9-]", "", skill_name.lower().strip()))
            skill_descriptions.append(skill_description)    
        
        # Save results
        for skill_name, skill_description in zip(skill_names, skill_descriptions):
            self._save_skill(skill_name, skill_description)
        

def _list_skills() -> list[str]:
    """Return sorted skill directory names."""
    skills_dir = Path(__file__).parent / "skills"
    if not skills_dir.exists():
        return []
    return sorted(d.name for d in skills_dir.iterdir() if d.is_dir())


def _read_skill(skill_name: str) -> str:
    """Read and return the contents of a skill's SKILL.md."""
    if not skill_name:
        return ""
    skill_path = Path(__file__).parent / "skills" / skill_name / "SKILL.md"
    if not skill_path.exists():
        return f"Skill file not found: {skill_path}"
    return skill_path.read_text()


def _train_from_upload(files) -> str:
    """Handle file upload(s) from the Gradio UI and run training."""
    if not files:
        return "No file uploaded."
    if isinstance(files, str):
        files = [files]
    trainer = MMSkillTrainer()
    try:
        trainer.train(files)
        return "Training complete! New skills saved. Refresh the skill list to see them."
    except Exception as e:
        return f"Error during training:\n{e}"


def frontend():
    import gradio as gr

    with gr.Blocks(title="Skill Trainer") as app:
        gr.Markdown("# Skill Trainer")

        with gr.Row(equal_height=True):
            # ---- Left panel: skill browser ----
            with gr.Column(scale=1):
                gr.Markdown("## Skills")
                skill_dropdown = gr.Dropdown(
                    choices=_list_skills(),
                    value=None,
                    label="Select a skill",
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh list")
                skill_content = gr.Textbox(
                    label="Skill content",
                    lines=20,
                    interactive=False,
                )

                refresh_btn.click(
                    fn=lambda: gr.update(choices=_list_skills()),
                    outputs=skill_dropdown,
                )
                skill_dropdown.input(
                    fn=_read_skill,
                    inputs=skill_dropdown,
                    outputs=skill_content,
                )
                skill_dropdown.change(
                    fn=_read_skill,
                    inputs=skill_dropdown,
                    outputs=skill_content,
                )

            # ---- Right panel: upload & train ----
            with gr.Column(scale=1):
                gr.Markdown("## Train New Skill")
                file_input = gr.File(
                    label="Upload video, audio, or text files",
                    file_types=["video", "audio", ".txt"],
                    file_count="multiple",
                )
                train_btn = gr.Button("Train", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

                train_btn.click(
                    fn=_train_from_upload,
                    inputs=file_input,
                    outputs=status_output,
                )

    app.launch()


if __name__ == "__main__":
    frontend()

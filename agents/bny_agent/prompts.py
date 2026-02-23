SWE_BENCH_PROMPT_TEMPLATE = """
Solve the following problem: 
{problem_description}

After completion, do the following:
1. Write your solution in /workspace/solution.md.
2. Identify if there are any reuseable skills that can be extracted from the solution, and if so, coin a name for the skill and save them under skills/ in the following path (if no skills are identified, skip this step):
/workspace/skills/<skill_name>/SKILL.md

SKILL.md should have the following format:
---
name: my-skill                    # Required (standard)
description: >                    # Required (standard)
  A brief description of what this skill does and when to use it.
license: MIT                      # Optional (standard)
compatibility: Requires bash      # Optional (standard)
metadata:                         # Optional (standard)
  author: your-name
  version: "1.0"
triggers:                         # Optional (OpenHands extension)
  - keyword1
  - keyword2
---

# Skill Content

Instructions and documentation for the agent...

"""

MMSkillTrainer_PROMPT_TEMPLATE = """You are a helpful assistant that can summarize multimedia files into skills. Please:
1. Analyze the given files and identify the skills that are being demonstrated in the files.
2. Lookup existing skills that have names similar to the skills that are being demonstrated in the files with tool calling. You MUST call the list_skills tool before producing any output. If it returns skills with similar names, you MUST call read_skill for each one before proceeding.
3. If similar skills are found, read the content of the existing skills and use the content to help you summarize the new skills.
4. Return the new skills in the format of a claude skill, with the skill name being your summarization of the skill's purpose. If similar skills are found, use the name of the existing skill, but with updated descriptions. 

After completing the tool-calling steps above, your final response should contain only skill blocks in the following format. If multiple skills are identified, return all of them. However, if there are multiple files that are related to the same skill, return only one skill for that skill. You MUST use exactly <skill_name> and <skill_description> as the XML tags. Do NOT use any other tag names. Name of the skills should contain only alphabetic characters, numbers and hyphens. 

<skill_name>
Your skill name here...
</skill_name>

<skill_description>
---
name: my-skill                    # Required (standard)
description: >                    # Required (standard)
  A brief description of what this skill does and when to use it.
license: MIT                      # Optional (standard)
compatibility: Requires bash      # Optional (standard)
metadata:                         # Optional (standard)
  author: your-name
  version: "1.0"
triggers:                         # Optional (OpenHands extension)
  - keyword1
  - keyword2
---

# Skill Content

Instructions and documentation for the agent...


</skill_description>

"""
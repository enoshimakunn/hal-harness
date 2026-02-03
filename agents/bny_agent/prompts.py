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
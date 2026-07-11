# Generate Quiz Skill

Generates 10 multiple-choice questions from a module's scripts.

**Trigger:** User asks to quiz, test, or generate questions for a module.

**Example:** "Quiz me on module 3" or "Generate questions for module 5"

## What to do

1. Read all scripts in the specified module
2. Extract: concepts, API patterns, parameter names, anti-patterns
3. Generate 10 MCQ questions covering:
   - API method signatures (2-3 questions)
   - Parameter meanings (2-3 questions)  
   - Anti-pattern identification (2 questions)
   - Concept understanding (2-3 questions)
4. Each question: 4 options (A-D), one correct answer, explanation

## Output format

```
# Module N Quiz: <Module Name>

**Question 1:** [Question text]
A) Option A
B) Option B  
C) Option C (correct)
D) Option D

**Answer:** C — [1-2 sentence explanation]

---
[repeat for all 10 questions]

**Score guide:** 9-10=Expert, 7-8=Proficient, 5-6=Review needed, <5=Re-study module
```

**Context:** Run in fork mode. Do not modify any files.

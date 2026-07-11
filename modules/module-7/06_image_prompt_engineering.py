"""Domain 7 - Task 7.6: Image Prompt Engineering

CONCEPTS:
  1. Style keywords — photorealistic, oil painting, watercolor, anime
  2. Composition — rule of thirds, close-up, aerial view, wide shot
  3. Lighting — golden hour, studio lighting, dramatic, soft diffused
  4. revised_prompt — DALL-E 3 returns the actual prompt it used

Mnemonic: SCLR — Style, Composition, Lighting, Revised_prompt

Run:
  uv run python 06_image_prompt_engineering.py
"""

NL = chr(10)

from dataclasses import dataclass

from shared.mock import get_client, is_mock


@dataclass
class PromptTemplate:
    name: str
    subject: str
    style: str
    composition: str
    lighting: str

    def build(self) -> str:
        parts = [self.subject]
        if self.style:
            parts.append(self.style)
        if self.composition:
            parts.append(self.composition)
        if self.lighting:
            parts.append(self.lighting)
        return ", ".join(parts)


TEMPLATES = [
    PromptTemplate("Basic", "A red apple", "", "", ""),
    PromptTemplate("With style", "A red apple", "photorealistic, 4K", "", ""),
    PromptTemplate("With composition", "A red apple", "photorealistic", "close-up macro shot", ""),
    PromptTemplate("Full technique", "A red apple", "oil painting", "rule of thirds", "golden hour warm light"),
    PromptTemplate("Dramatic", "A lone mountain peak", "digital art", "aerial view", "dramatic storm lighting"),
    PromptTemplate("Product photo", "Luxury watch", "commercial photography", "centered, white background", "studio lighting three-point"),
]


def demo_prompt_templates() -> None:
    """DEMO 1: Show progression of prompt quality."""
    print("  Prompt evolution:")
    client = get_client()
    for template in TEMPLATES:
        prompt = template.build()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
        )
        url = response.data[0].url
        print(f"  [{template.name}]")
        print(f"    Prompt: {prompt!r}")
        print(f"    URL: {url!r}")


def demo_revised_prompt() -> None:
    """DEMO 2: revised_prompt — what DALL-E 3 actually uses."""
    client = get_client()
    original_prompt = "a cat"
    response = client.images.generate(
        model="dall-e-3",
        prompt=original_prompt,
        size="1024x1024",
    )
    url = response.data[0].url
    # In real API: response.data[0].revised_prompt would show DALL-E's enhanced prompt
    print(f"  Original prompt: {original_prompt!r}")
    print(f"  Generated URL: {url!r}")
    print("  (Live: response.data[0].revised_prompt shows what DALL-E 3 used)")
    print("  → DALL-E 3 auto-enhances short prompts for better results")


def demo_prompt_checklist() -> None:
    """DEMO 3: Prompt engineering checklist."""
    print("  Image Prompt Engineering Checklist:")
    print("    Subject:     Be specific — 'golden retriever puppy' not 'dog'")
    print("    Style:       photorealistic | oil painting | watercolor | anime | digital art")
    print("    Composition: close-up | wide shot | aerial | rule of thirds | centered")
    print("    Lighting:    golden hour | studio | dramatic | soft | backlit | neon")
    print("    Color:       muted palette | vibrant | monochrome | earth tones")
    print("    Quality:     4K | highly detailed | sharp focus | professional")
    print(f"{NL}  Things to avoid:")
    print("    - Text in images (DALL-E often misspells)")
    print("    - Exact likeness of real people")
    print("    - Copyrighted art styles by name")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 7 - Task 7.6: Image Prompt Engineering [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Prompt Templates ---")
    demo_prompt_templates()

    print(f"{NL}--- DEMO 2: Revised Prompt ---")
    demo_revised_prompt()

    print(f"{NL}--- DEMO 3: Checklist ---")
    demo_prompt_checklist()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Specific subjects, style, composition, and lighting produce consistent results")
    print("  2. DALL-E 3 auto-enhances prompts — check revised_prompt to see what it used")
    print("  3. Product photos: 'white background, studio lighting, centered' works reliably")
    print("  4. Avoid text in images — DALL-E often misspells; generate text separately")

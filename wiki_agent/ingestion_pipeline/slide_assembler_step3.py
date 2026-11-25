from models import SlideData, AssemblyBlock, ModuleMetadata, Assembly


def assemble_module(
    slides: list[SlideData],
    metadata: ModuleMetadata
) -> Assembly:
    """
    Build Assembly from categorized slides and module metadata.

    Rules:
    - Slides sorted by slide_number ascending (immutable sequence)
    - speaker_notes is the primary content source — always included
    - body_text included only when speaker_notes is completely empty
        - shape_inventory is not passed downstream;
            it is used only during slide categorization
    - module_id and metadata live on Assembly, not on each block
    """
    blocks = []
    for slide in sorted(slides, key=lambda s: s.slide_number):
        body_text_to_include = (
            slide.body_text
            if not slide.speaker_notes.strip()
            else ""
        )
        blocks.append(
            AssemblyBlock(
                slide_number=slide.slide_number,
                category=slide.category,
                title=slide.title,
                speaker_notes=slide.speaker_notes,
                body_text=body_text_to_include,
                image_path=slide.image_path,
            )
        )
    return Assembly(
        module_id=metadata.id,
        module_metadata=metadata,
        blocks=blocks,
    )
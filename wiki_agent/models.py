from dataclasses import dataclass, field


@dataclass
class SlideData:
    module_id: str
    slide_number: int
    title: str
    body_text: str
    speaker_notes: str
    shape_inventory: list[str]
    category: str = ""
    image_path: str = ""
    # category values: "NAVIGATION", "TEXT_ONLY", "IMAGE"
    # shape_inventory used by categorizer.py only
    # image_path set by categorizer.py for IMAGE slides only


@dataclass
class AssemblyBlock:
    slide_number: int
    category: str
    title: str
    speaker_notes: str   # primary content source — always present
    body_text: str       # fallback — only populated when notes empty
    image_path: str      # empty string for NAVIGATION and TEXT_ONLY


@dataclass
class ModuleMetadata:
    id: str
    title: str       # from deterministic slide-1 metadata extraction
    contact: str     # from deterministic slide-1 metadata extraction
    date: str        # from PPT core properties created date
    track: str       # from CLI argument --track
    subtrack: str    # from CLI argument --subtrack


@dataclass
class Assembly:
    module_id: str
    module_metadata: ModuleMetadata
    blocks: list[AssemblyBlock]
    # blocks sorted by slide_number ascending
    # immutable pedagogical sequence — never reorder


@dataclass
class WikiFile:
    filename: str
    content: str


@dataclass
class IndexUpdate:
    modules: list[dict]
    concepts_added: list[str]
    concepts_updated: list[str]
    


@dataclass
class WikiEnvelope:
    module_page: WikiFile
    concept_pages: list[WikiFile]
    index_updates: IndexUpdate
    log_entries: list[str]
    

@dataclass
class TokenUsage:
    module_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    # "stop"   → model finished naturally
    # "length" → hit WIKI_MAX_TOKENS ceiling
    # "error"  → API call failed
    tool_calls_made: int
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from typing import Optional
import os

# Define your model (replace 'openai:gpt-4o' with Ollama and Qwen2.5)
model_name = 'ollama:qwen2.5'

class OllamaModel(Model):
    """Concrete implementation of Model for Ollama."""
    def name(self) -> str:
        return model_name
    
    def agent_model(self) -> str:
        return model_name

class NoteIntent(BaseModel):
    """Intent extracted from user input."""
    action: str = Field(..., description="The action to perform: 'create', 'read', or 'list'")
    title: Optional[str] = Field(None, description="Title of the note")
    content: Optional[str] = Field(None, description="Content of the note for create actions")

@dataclass
class NoteDeps:
    """Dependencies for note operations."""
    base_dir: str = "notes"  # Default to notes directory

class NoteResponse(BaseModel):
    """Response from note operations."""
    message: str = Field(..., description="Status message about the operation")
    filename: Optional[str] = Field(None, description="Filename of the created/accessed note")

# Intent extraction agent to understand what the user wants to do
intent_agent = Agent(
    model_name,
    result_type=NoteIntent,
    system_prompt=(
        "You are an intent extraction assistant. Analyze user inputs to determine their intent "
        "and extract relevant information. Output structured data with:\n"
        "- action: 'create' for new notes, 'read' for viewing notes, 'list' for showing all notes\n"
        "- title: A clear, descriptive title for the note (not needed for list action)\n"
        "- content: The actual note content for create actions\n\n"
        "Example 1: If user says 'Take a note about the meeting tomorrow', output:\n"
        "{'action': 'create', 'title': 'Meeting Tomorrow', 'content': 'Meeting scheduled for tomorrow'}\n\n"
        "Example 2: If user says 'show me all my notes' or 'list notes', output:\n"
        "{'action': 'list', 'title': null, 'content': null}"
    )
)

# Action handling agent to perform the actual operations
action_agent = Agent(
    model_name,
    deps_type=NoteDeps,
    result_type=NoteResponse,
    system_prompt=(
        "You are a note management assistant. Your job is to create, read and list notes using the provided tools:\n"
        "1. create_note: Creates a new note with a title and content\n"
        "2. read_note: Reads an existing note by its title\n"
        "3. list_notes: Lists all notes in the directory\n\n"
        "When asked to create a note, use create_note with the exact title and content provided.\n"
        "When asked to read a note, use read_note with the exact title provided.\n"
        "When asked to list notes, use list_notes to list all notes."
    )
)

def create_run_context(deps: NoteDeps) -> RunContext[NoteDeps]:
    """Create a RunContext with the required parameters."""
    return RunContext(
        deps=deps,
        model=OllamaModel(),
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        prompt=""
    )

def normalize_title(title: str) -> str:
    """Normalize a title for consistent file naming."""
    # Remove .txt extension if present
    if title.lower().endswith('.txt'):
        title = title[:-4]
    return title.lower().replace(' ', '_').replace('/', '_')

def ensure_notes_dir(base_dir: str) -> None:
    """Ensure the notes directory exists."""
    os.makedirs(base_dir, exist_ok=True)

@action_agent.tool
async def create_note(ctx: RunContext[NoteDeps], title: str, content: str) -> NoteResponse:
    """Create a new note with the given title and content."""
    ensure_notes_dir(ctx.deps.base_dir)
    filename = f"{normalize_title(title)}.txt"
    filepath = os.path.join(ctx.deps.base_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return NoteResponse(message=f"Note created successfully", filename=filename)
    except Exception as e:
        return NoteResponse(message=f"Failed to create note: {str(e)}")

@action_agent.tool
async def read_note(ctx: RunContext[NoteDeps], title: str) -> NoteResponse:
    """Read a note by its title."""
    ensure_notes_dir(ctx.deps.base_dir)
    filename = f"{normalize_title(title)}.txt"
    filepath = os.path.join(ctx.deps.base_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return NoteResponse(message=f"Note content: {content}", filename=filename)
    except FileNotFoundError:
        return NoteResponse(message=f"Note not found: {filename}")
    except Exception as e:
        return NoteResponse(message=f"Error reading note: {str(e)}")

@action_agent.tool
async def list_notes(ctx: RunContext[NoteDeps]) -> NoteResponse:
    """List all notes in the directory."""
    ensure_notes_dir(ctx.deps.base_dir)
    try:
        notes = []
        for filename in os.listdir(ctx.deps.base_dir):
            if filename.endswith('.txt'):
                notes.append(filename[:-4])  # Remove .txt extension
        
        if not notes:
            return NoteResponse(message="No notes found")
        
        message = "Found notes:\n" + "\n".join(f"- {note}" for note in notes)
        return NoteResponse(message=message)
    except Exception as e:
        return NoteResponse(message=f"Error listing notes: {str(e)}")

async def handle_note_request(user_input: str, base_dir: str = "notes") -> NoteResponse:
    """Handle a user's note request by first understanding their intent, then taking action."""
    # First, extract the user's intent
    intent_result = await intent_agent.run(user_input)
    print(f"Debug - Intent data: {intent_result.data}")
    
    # Create dependencies for the action agent
    deps = NoteDeps(base_dir=base_dir)
    ctx = create_run_context(deps)
    
    # Based on the intent, perform the appropriate action
    if not isinstance(intent_result.data, NoteIntent):
        return NoteResponse(message=f"Failed to understand request: {intent_result.data}")
        
    intent = intent_result.data
    if intent.action == "create" and intent.title and intent.content:
        try:
            return await create_note(ctx, intent.title, intent.content)
        except Exception as e:
            return NoteResponse(message=f"Failed to create note: {str(e)}")
    elif intent.action == "read" and intent.title:
        try:
            return await read_note(ctx, intent.title)
        except Exception as e:
            return NoteResponse(message=f"Failed to read note: {str(e)}")
    elif intent.action == "list":
        try:
            return await list_notes(ctx)
        except Exception as e:
            return NoteResponse(message=f"Failed to list notes: {str(e)}")
    else:
        return NoteResponse(message="Invalid request. Please specify what you want to do with the note.")

# Example usage
import asyncio

async def main():
    # Example: Create a note
    response = await handle_note_request(
        "Take a note about how finding waldo can be very difficult."
    )
    print(response.message)
    
    # Example: Read the note
    if response.filename:
        read_response = await handle_note_request(f"Read {response.filename}")
        print(read_response.message)
    
    # Example: List notes
    list_response = await handle_note_request("List notes")
    print(list_response.message)

if __name__ == '__main__':
    asyncio.run(main())
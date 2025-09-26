import gradio as gr
from pathlib import Path
from doctor_brain import medical_chatbot, get_conversation_memory


# -------------------------
# Wrapper for Gradio with Memory
# -------------------------
def chatbot_interface(history, query, image, memory_state):
    image_path = None
    if image:
        temp_path = Path("uploaded_image.png")
        image.save(temp_path)
        image_path = str(temp_path)

    # Call backend with memory
    result = medical_chatbot(query, image_path=image_path)

    # Answer + sources
    answer = result["answer"]
    sources = [
        f"📖 {i + 1}. {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')})"
        for i, doc in enumerate(result["source_documents"])
    ]
    full_answer = (
        f"{answer}\n\n"
        f"<details><summary><b>Sources</b></summary>{'<br>'.join(sources)}</details>"
    )

    history.append((query, full_answer))
    return history, history, memory_state


def clear_chat():
    new_memory = get_conversation_memory()
    return [], [], new_memory


# -------------------------
# Modern ChatGPT-like UI
# -------------------------
with gr.Blocks(css="""
    #chatbox {height: 600px !important;}
    .message {padding: 8px 12px; border-radius: 12px; margin: 4px 0;}
    .user {background-color: #FFFFF; align-self: flex-end;}
    .bot {background-color: #FFFFF; align-self: flex-start;}
""") as demo:
    gr.Markdown(
        """
        # 🩺 MedVision RAG Assistant  
        *Your AI-powered medical consultant with memory + image support.*  
        """
    )

    memory_state = gr.State(get_conversation_memory())
    state = gr.State([])

    with gr.Row():
        # Left panel: Chat only
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbox",
                label="Chat",
                height=600,
                show_copy_button=True,
                bubble_full_width=False
            )

        # Right panel: Inputs + controls
        with gr.Column(scale=1):
            gr.Markdown("### Ask a Question")
            query = gr.Textbox(
                show_label=False,
                placeholder="Type your medical question...",
                lines=4
            )
            send_btn = gr.Button("➤ Send", variant="primary")

            gr.Markdown("### Upload Image")
            image = gr.Image(type="pil", label="", height=150)

            clear = gr.Button("🗑️ Clear Chat", variant="stop")

            gr.Markdown("---")
            gr.Markdown(
                """
                ### ℹ️ About  
                - Ask any medical-related question.  
                - Upload images (X-rays, scans, reports).  
                - Get context-aware answers with sources.  
                """
            )

    # Submit + events
    query.submit(
        chatbot_interface,
        [state, query, image, memory_state],
        [chatbot, state, memory_state],
    )
    send_btn.click(
        chatbot_interface,
        [state, query, image, memory_state],
        [chatbot, state, memory_state],
    )
    clear.click(clear_chat, None, [chatbot, state, memory_state])

demo.launch()

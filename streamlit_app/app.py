import streamlit as st
import os
import time
import utils

# Page Configuration
st.set_page_config(
    page_title="Azure AI Search RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "style.css")
utils.load_css(css_path)

# Initialize RAG System
try:
    rag = utils.get_rag_system()
except Exception as e:
    st.error(f"Failed to initialize RAG system: {str(e)}")
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Azure Search AI assistant. How can I help you today?"}
    ]

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Configuration")
    
    st.markdown("### Search Settings")
    retrieval_limit = st.slider("Documents to Retrieve", min_value=1, max_value=20, value=10)
    
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation cleared. How can I help?"}
        ]
        st.rerun()
        
    st.markdown("### About")
    st.info(
        "This application uses **Azure AI Search** with specific strategies:\n"
        "- **Keyword**: BM25\n"
        "- **Hybrid**: Vector + Keyword\n"
        "- **Semantic**: Hybrid + Reranking"
    )

# Main Chat Interface
st.title("Azure AI Search RAG Agent")
st.markdown("Ask questions about your indexed documents and get grounded answers.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Explore Retrieved Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{idx}. {source['source']}** ({source['strategy']})")
                    st.caption(source['content'][:200] + "...")

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources_data = []
        
        # Progress Container
        with st.status("Thinking...", expanded=True) as status:
            
            # Step 1: Retrieval
            st.write("🔍 Retrieving relevant documents...")
            try:
                # We can customize k with the slider value
                context = rag.retrieve_comprehensive_context(prompt, k_per_strategy=retrieval_limit)
                
                # Parse sources for display
                sources_data = utils.format_source_display(context)
                
                if sources_data:
                    st.write(f"✅ Found {len(sources_data)} relevant segments.")
                else:
                    st.write("⚠️ No relevant information found.")
                    
            except Exception as e:
                st.error(f"Retrieval failed: {str(e)}")
                status.update(label="Error", state="error")
                context = ""

            # Step 2: Generation
            if context:
                st.write("🧠 Generatinng response...")
                try:
                    # Stream the response
                    chunks = utils.stream_response(rag, prompt, context)
                    
                    for chunk in chunks:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                        
                    status.update(label="Complete", state="complete", expanded=False)
                    
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                    full_response = "I encountered an error while generating the response."
            else:
                full_response = "I couldn't find any relevant information to answer your question."
                status.update(label="Complete (No Data)", state="complete", expanded=False)

        # Final Update (remove cursor)
        response_placeholder.markdown(full_response)
        
        # Show Sources (Optional, outside the status but inside the message)
        if sources_data:
            with st.expander("Explore Retrieved Sources"):
                for idx, source in enumerate(sources_data, 1):
                    st.markdown(f"**{idx}. {source['source']}** ({source['strategy']})")
                    st.caption(source['content'][:200] + "...")

    # 3. Save to History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources_data
    })

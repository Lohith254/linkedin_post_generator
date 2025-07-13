import streamlit as st
from generate_post import generate_post  # Make sure this exists and is patched

# Streamlit page config
st.set_page_config(page_title="LinkedIn Post Generator", page_icon="🔗", layout="centered")

# ----------- UI Layout -----------

# Title
st.title("🔗 LinkedIn Post Generator")
st.caption("Generate professional LinkedIn posts using your custom fine-tuned LLM.")

# Theme Dropdown
themes = [
    "Promotion",
    "New Job Announcement",
    "Milestone Celebration",
    "Mentorship Reflection",
    "Project Launch",
    "Work-Life Balance",
    "Learning & Upskilling",
    "Event Announcement",
    "Thank You / Gratitude",
    "Custom Prompt"
]

selected_theme = st.selectbox("📌 Select a Post Theme", themes)

# Context Input
context = st.text_area("📝 Describe the context (1-2 lines)", height=100, max_chars=300)

# Button
generate = st.button("🚀 Generate LinkedIn Post")

# Placeholder for generated output
if generate and context.strip():
    with st.spinner("Generating post..."):
        # Compose prompt based on theme + context
        if selected_theme == "Custom Prompt":
            prompt = context.strip()
        else:
            prompt = f"Write a professional LinkedIn post about {selected_theme}: {context.strip()}"

        try:
            post = generate_post(prompt)
            st.markdown("### 📢 Generated Post")
            st.text_area("Your LinkedIn Post", post, height=200)

            # Copy button (Streamlit's new UI feature)
            st.download_button("📋 Copy to Clipboard", post, file_name="linkedin_post.txt")

        except Exception as e:
            st.error(f"⚠️ Error generating post: {e}")

elif generate:
    st.warning("Please enter some context before generating.")

# Footer
st.markdown("---")
st.caption("🚀 Powered by your fine-tuned Phi-2 model | Built by Lohith R")
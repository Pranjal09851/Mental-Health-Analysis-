from fusion_utils import load_text_model

try:
    # This calls the clear method on the cached function
    load_text_model.clear()
    print("✅ Successfully cleared the cache for load_text_model.")
except Exception as e:
    print(f"Failed to clear cache: {e}")
    print("Ensure load_text_model in fusion_utils.py is decorated with @st.cache_resource")
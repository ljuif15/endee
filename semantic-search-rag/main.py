print("Choose an option:")
print("1. Semantic Search")
print("2. RAG Question Answering")

choice = input("Enter choice (1 or 2): ")

if choice == "1":
    import search.semantic_search
elif choice == "2":
    import rag.rag_pipeline
else:
    print("❌ Invalid choice")

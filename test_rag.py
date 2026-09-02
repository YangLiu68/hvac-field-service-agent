from app.services.rag_service import search_manual


def main():
    results = search_manual("AC not cooling with low suction pressure")
    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}:")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print("Score:", result["score"])
        print()
        print(result["text"][:1000])


if __name__ == "__main__":
    main()

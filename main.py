from tooluniverse import ToolUniverse


def main():
    print("Hello from latent-annotation!")

    tu = ToolUniverse()
    # load all the 1000+ tools
    tu.load_tools()
    print(f"Loaded {len(tu.all_tools)} scientific tools!")

    # query scientific databases
    result = tu.run(
        {
            "name": "UniProt_get_function_by_accession",
            "arguments": {"accession": "P05067"},
        }
    )
    print(result)


if __name__ == "__main__":
    main()

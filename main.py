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

    spec = tu.tool_specification("UniProt_get_function_by_accession", format="openai")

    print(f"Name: {spec['name']}")
    print(f"Description: {spec['description']}")
    print("Parameters:")
    for param_name, param_info in spec["parameters"]["properties"].items():
        print(f"  - {param_name}: {param_info['type']} - {param_info['description']}")

    # Get multiple specifications
    specs = tu.get_tool_specification_by_names(
        [
            "FAERS_count_reactions_by_drug_event",
            "OpenTargets_get_associated_targets_by_disease_efoId",
        ]
    )


if __name__ == "__main__":
    main()

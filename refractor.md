- NodeExposedData → NodeDataFormat

- always_sort → use_one_time_sort or enable_one_time_sort

- lower_bound_strategy → lower_bound_approach or lower_bound_method

- branching_strategy → branch_selection_method

- Enum Value Naming: None_ → Disabled or NotUsed


Comprehensive Naming Guidelines for pytrees-rs
Module Structure and Organization
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | src/searches/ | src/algorithms/ or src/learning/ | More descriptive of the purpose | | optimal/dl85 | exact/dl85 | "Exact" is more commonly used in ML literature than "Optimal" | | searches/utils.rs | Split into searches/types.rs and searches/params.rs | Better organization by purpose |

Type Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | SearchReturn | SearchResult | More idiomatic for return values in Rust | | BranchChoice | BranchEvaluation | Better describes that it contains evaluation metrics | | StopConditions | TerminationCriteria | More formal and descriptive | | Bitset | BitVector | Matches standard CS terminology | | RevBitset | ReversibleBitVector | Make abbreviations explicit | | SparseBitset | SparseBitVector | Consistent naming convention | | ShallowBitset | LightweightBitVector | More descriptive of its purpose |

Parameter/Field Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | NodeExposedData | NodeDataFormat | Clearer about its purpose | | always_sort | use_one_time_sort or enable_one_time_sort | More descriptive of what the flag does | | lower_bound_strategy | lower_bound_approach or lower_bound_method | Distinguishes from other strategy parameters | | branching_strategy | branch_selection_method | More explicit about what this strategy controls | | max_time | max_time_seconds | Clarifies the unit of measurement | | BaseSearchConfig | CommonSearchParams | "Params" is clearer for settings affecting algorithm behavior | | min_sup | min_support | Avoid abbreviations for better readability | | is_leaf | is_terminal_node | More precise terminology | | leaf_error | terminal_node_error | Consistency with terminal node terminology | | target | prediction_value | Clarifies this is the predicted value | | cache_init_strategy | cache_initialization_method | More explicit |

Method/Function Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | fit | train or build_tree | More descriptive of what the algorithm does | | recursion | search_recursively | More descriptive verb form | | count_if_branch_on | count_items_after_branch | Clearer about what's being counted | | backtrack | revert_to_parent | More descriptive of operation | | branch_on | apply_split | Better describes the action | | count_intersect_with | count_matching_items | More descriptive of operation | | get_node_candidates | find_eligible_attributes | Clarifies what "candidates" refers to |

Enum Value Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | CacheInitStrategy::None_ | CacheInitStrategy::Default | More idiomatic than using None_ | | SearchStrategy::None_ | SearchStrategy::Standard | Clearer meaning | | Specialization::None_ | Specialization::Generic | Better describes that no specialized algorithm is used | | LowerBoundStrategy::None_ | LowerBoundStrategy::Disabled | More explicit about functionality | | BranchingStrategy::None_ | BranchingStrategy::Default | More consistent with other defaults |

Variable Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | it | item_id or attribute_value_pair | More descriptive than generic "it" | | infos | node_information or tree_node_data | Avoid abbreviations, use full words | | cache_node | cached_node_data | More explicit about what it contains |

Function Parameter Naming
| Current Name | Suggested Name | Rationale | |--------------|----------------|-----------| | one_time_sort | perform_initial_sort | Clearer about when sorting happens | | invert | negate_condition | More descriptive of the logical operation | | attributes | feature_vectors | More aligned with ML terminology |

General Naming Principles
Avoid abbreviations unless extremely common
Be specific about what things are and do
Use domain terminology consistently (ML terms where appropriate)
Favor descriptive verbs for methods
Name booleans with is/has/should prefixes
Follow Rust conventions:
snake_case for variables, functions, methods
CamelCase for types, traits, enums
SCREAMING_SNAKE_CASE for constants
Aim for self-documenting code through clear names
Implementation Strategy
Start with high-impact changes (public API, main types)
Ensure consistency across related components
Update documentation alongside name changes
Use IDE refactoring tools to minimize errors during renaming
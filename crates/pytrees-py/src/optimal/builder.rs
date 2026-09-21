








#[pyfunction]
#[pyo3(name = "dl85")]
// #[pyo3(signature = (
//     input,
//     target=None,
//     min_sup=1,
//     max_depth=2,
//     time_limit=600.0,
//     max_cache_size=None,
//     heuristic=ExposedSearchHeuristic::None_,
//     specialization=ExposedSpecialization::None_,
//     lower_bound=ExposedLowerBoundStrategy::Similarity,
//     branching_type=ExposedBranchingStrategy::Dynamic,
//     max_discrepancy=None,
//     min_gain=None,
//     min_purity=None,
//     topk=None,
//     error_function=None,
// ))]
pub(crate) fn optimal_search_dl85(
    input: PyReadonlyArrayDyn<f64>,
    target: Option<PyReadonlyArrayDyn<f64>>,
    min_sup: usize,
    max_depth: usize,
    always_sort: bool,
    time_limit: f64,
    // max_cache_size: Option<usize>,
    heuristic: ExposedHeuristic,
    depth2_policy: ExposedDepth2Policy,
    lower_bound: ExposedLowerBoundPolicy,
    branching_policy: ExposedBranchingPolicy,
    data_type: ExposedNodeDataType,
    // Rule parameters
    discrepancy: Option<ExposedDiscrepancyRule>,
    gain: Option<ExposedGainRule>,
    topk: Option<ExposedTopKRule>,
    restart: Option<ExposedRestartRule>,
    purity: Option<ExposedPurityRule>,
    error_function: Option<Py<PyAny>>,
) -> PyResult<SearchOutput> {
    // Create cover from numpy arrays using the new API
    let mut cover = create_cover_from_numpy(input, target.as_ref())?;


    let heuristic_fn: Box<dyn Heuristic> = heuristic.into();
    let depth2_policy = depth2_policy.into();
    let lower_bound_policy = lower_bound.into();
    let branching_policy = branching_policy.into();

    let mut data_type = data_type.into();
    if target.is_none() {
        data_type = NodeDataType::Tids;
    }

    let error_fn: Box<dyn ErrorWrapper> = match error_function {
        Some(function) => Box::new(PythonError::new(function)) ,
        None => Box::new(NativeError::default())
    };

    let depth2_search = Box::new(ErrorMinimizer::new(Box::<NativeError>::default()));

    // Configure cache
    let cache = Box::new(Trie::default());

    // Configure rules
    let mut node_rules: Vec<Box<dyn Rule>> = vec![];
    let mut search_rules: Vec<Box<dyn Rule>> = vec![];

    // Add discrepancy rule if specified
    if let Some(rule) = discrepancy {
        let discrepancy_rule = DiscrepancyRule::from(rule);
        search_rules.push(Box::new(discrepancy_rule));
    }

    // Add gain rule if specified
    if let Some(rule) = gain {
        let gain_rule = GainRule::from(rule);
        search_rules.push(Box::new(gain_rule));
    }

    // Add purity rule if specified
    if let Some(rule) = purity {
        let purity_rule = PurityRule::from(rule);
        node_rules.push(Box::new(purity_rule));
    }

    // Add topk rule if specified
    if let Some(rule) = topk {
        let topk_rule = TopkRule::from(rule);
        search_rules.push(Box::new(topk_rule));
    }

    // Create restart time limit rule
    if let Some(rule) = restart {
        let restart_rule = TimeLimitRule::from(rule);
        search_rules.push(Box::new(restart_rule));
    }

    // Build DL85 algorithm using the builder pattern
    let mut dl85 = DL85Builder::default()
        .max_depth(max_depth)
        .min_support(min_sup)
        .max_time(time_limit)
        .specialization(depth2_policy)
        .always_sort(always_sort)
        .branching_strategy(branching_policy)
        .lower_bound_strategy(lower_bound_policy)
        .node_exposed_data(data_type)
        .cache(cache)
        .heuristic(heuristic_fn)
        .depth2_search(depth2_search)
        .add_search_rules(search_rules)
        .add_node_rules(node_rules)
        .error_function(error_fn)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build DL85: {:?}", e)))?;

    // Fit the algorithm
    dl85.fit(&mut cover)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("DL85 fit failed: {:?}", e)))?;

    // Extract results
    let tree = dl85.tree().clone();
    let statistics = dl85.statistics().clone();
    let duration = dl85.elapsed_seconds();

    Ok(SearchOutput {
        error: statistics.tree_error,
        tree,
        statistics,
        duration,
        search: SearchStrategy::DL85,
    })
}

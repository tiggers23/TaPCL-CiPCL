def _get_R_ij(perf_dict, i, j, metric_type="avg"):
    """Computes an accuracy after task i on task j.

    R matrix:

          || T_e1 | T_e2 | T_e3
    ============================|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ============================|

    R_13 is the R of the first column and the third row.

    From Chaudhry et al., calling this function with (i, j) equals to a_{i,j}.

    Except OOD and Zeroshot, i should be >= j.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param i: Task id after which a model was trained.
    :param j: Task id of the test data.
    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    try:
        result = perf_dict[i][j][metric_type]
    except:
        print("no such result")
    return result


def forgetting(perf_dict, metric_type="avg"):
    """Measures the average forgetting.

    Reference:
    * Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
      Chaudhry et al. ECCV 2018

    See eq. 3.
    """
    k = len(perf_dict)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if k <= 1:
        return 0.

    f = 0.
    for j in range(k - 1):
        # Accuracy on task j after learning current task k
        a_kj = _get_R_ij(perf_dict, k - 1, j, metric_type)
        # Best previous accuracy on task j
        max_a_lj = max(_get_R_ij(perf_dict, l, j, metric_type) for l in range(k - 1))
        f += max_a_lj - a_kj  # We want this results to be as low as possible

    metric = f / (k - 1)
    assert -100.0 <= metric <= 100.0, metric
    return metric

def avg_accuracy(perf_dict, task_id, task_nb, metric_type="avg"):
    f = 0.
    for j in range(task_nb):
        # Accuracy on task j after learning current task k
        a_kj = _get_R_ij(perf_dict, task_id, j, metric_type)
        # Best previous accuracy on task j
        f += a_kj  # We want this results to be as low as possible

    metric = f / task_nb
    assert -100.0 <= metric <= 100.0, metric
    return metric

def accuracy_per_task(perf_dict, task_id, task_nb, metric_type="avg"):
    accuracy_per_task_list = []
    for j in range(task_nb):
        # Accuracy on task j after learning current task k
        a_kj = _get_R_ij(perf_dict, task_id, j, metric_type)
        # Best previous accuracy on task j
         # We want this results to be as low as possible
        accuracy_per_task_list.append(a_kj)

    return accuracy_per_task_list

def transfer(perf_dict, task_id, task_nb, metric_type='avg'):
    f = 0
    if task_id < task_nb-1:
        for j in range(task_nb):
            if j >task_id:
                a_kj = _get_R_ij(perf_dict, task_id, j, metric_type)
                f += a_kj
        metric = f/(task_nb - task_id - 1)

        return metric
    else:
        return 0



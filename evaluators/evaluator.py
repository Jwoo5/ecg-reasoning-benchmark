from typing import List, Union


class Evaluator:
    def __init__(self, use_builtin_metrics: bool = True):
        if use_builtin_metrics:
            # if there is no custom implementation of init_metrics, we don't need to wrap it
            # because the built-in one will be used directly
            if type(self).init_metrics is not Evaluator.init_metrics:

                def _builtin_init_metrics_wrapper(name: str, reset: bool = False) -> None:
                    Evaluator.init_metrics(self, name, reset)
                    self.init_metrics(name, reset)

                self.init_metrics = _builtin_init_metrics_wrapper

            # if there is no custom implementation of reduce_metrics, we don't need to wrap it
            # because the built-in one will be used directly
            if type(self).reduce_metrics is not Evaluator.reduce_metrics:

                def _builtin_reduce_metrics_wrapper(metrics: dict) -> dict:
                    reduced_metrics = Evaluator.reduce_metrics(self, metrics)
                    return self.reduce_metrics(reduced_metrics)

                self.reduce_metrics = _builtin_reduce_metrics_wrapper

            if type(self).evaluate is not Evaluator.evaluate:

                def _builtin_evaluate_wrapper(result: dict) -> None:
                    Evaluator.evaluate(self, result)
                    self.evaluate(result)

                self.evaluate = _builtin_evaluate_wrapper

    # Built-in metric implementations
    def init_metrics(self, name: str, reset: bool = False) -> None:
        """Initialize metrics for a given evaluation.

        Args:
            name (str): The name of the evaluation.
            reset (bool): Whether to reset existing metrics. Defaults to False.
        """
        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = {}

        if reset and name in self.metrics:
            self.metrics[name] = None

        initial_diagnostic_question_metrics = {
            "total": 0,
            "correct": 0,
        }
        stepwise_metrics = {
            "total_w_gt": 0,
            "correct_w_gt": 0,
            "total_wo_gt": 0,
            "correct_wo_gt": 0,
        }
        metrics = {
            "initial_diagnostic_question": initial_diagnostic_question_metrics,
            "path_1": {
                "total": 0,
                "completed": 0,
                "aligned": 0,
                "per_loop": {
                    "total": 0,
                    "depth_total": 0,  # depth should be computed only for those having grounding steps
                    "depth_sum": 0,
                    "criterion_selection": stepwise_metrics.copy(),
                    "finding": stepwise_metrics.copy(),
                    "lead_grounding": stepwise_metrics.copy(),
                    "wave_grounding": stepwise_metrics.copy(),
                    "measurement_grounding": stepwise_metrics.copy(),
                    "decision": stepwise_metrics.copy(),
                },
            },
            "path_2": {
                "total": 0,
                # TODO add more detailed metrics for path 2 when it is implemented
            },
            "failed": {
                "total": 0,
            },
        }

        self.metrics[name] = metrics

    # Built-in metric implementations
    def reduce_metrics(self, name: str) -> dict:
        """Reduce metrics for a given evaluation name.

        Args:
            name (str): The name of the evaluation.

        Returns:
            dict: A dictionary of reduced metrics.
        """
        if not hasattr(self, "metrics") or self.metrics is None:
            raise ValueError(
                "Metrics have not been initialized. Call init_metrics() before reduce_metrics()."
            )
        if name not in self.metrics:
            raise ValueError(
                f"Metrics for '{name}' have not been initialized. Call init_metrics('{name}') "
                f"before reduce_metrics('{name}')."
            )

        reduced_metrics = self.metrics[name].copy()
        # compute accuracy for initial diagnostic question
        reduced_metrics["initial_diagnostic_question"]["accuracy"] = (
            reduced_metrics["initial_diagnostic_question"]["correct"]
            / reduced_metrics["initial_diagnostic_question"]["total"]
            if reduced_metrics["initial_diagnostic_question"]["total"] > 0
            else 0.0
        )
        reduced_metrics["initial_diagnostic_question"]["path_1_ratio"] = (
            reduced_metrics["path_1"]["total"] / reduced_metrics["initial_diagnostic_question"]["total"]
            if reduced_metrics["initial_diagnostic_question"]["total"] > 0
            else 0.0
        )
        reduced_metrics["initial_diagnostic_question"]["path_2_ratio"] = (
            reduced_metrics["path_2"]["total"] / reduced_metrics["initial_diagnostic_question"]["total"]
            if reduced_metrics["initial_diagnostic_question"]["total"] > 0
            else 0.0
        )
        reduced_metrics["initial_diagnostic_question"]["failed_ratio"] = (
            reduced_metrics["failed"]["total"] / reduced_metrics["initial_diagnostic_question"]["total"]
            if reduced_metrics["initial_diagnostic_question"]["total"] > 0
            else 0.0
        )

        # compute completion and alignment for path 1
        reduced_metrics["path_1"]["completion"] = (
            reduced_metrics["path_1"]["completed"] / reduced_metrics["path_1"]["total"]
            if reduced_metrics["path_1"]["total"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["alignment"] = (
            reduced_metrics["path_1"]["aligned"] / reduced_metrics["path_1"]["total"]
            if reduced_metrics["path_1"]["total"] > 0
            else 0.0
        )

        # compute per loop metrics for path 1
        reduced_metrics["path_1"]["per_loop"]["depth_avg"] = (
            reduced_metrics["path_1"]["per_loop"]["depth_sum"]
            / reduced_metrics["path_1"]["per_loop"]["depth_total"]
            if reduced_metrics["path_1"]["per_loop"]["depth_total"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["criterion_selection"]["total_wo_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["finding"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["finding"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["finding"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["finding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["finding"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["finding"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["finding"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["finding"]["total_wo_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["lead_grounding"]["total_wo_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["wave_grounding"]["total_wo_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["measurement_grounding"]["total_wo_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["decision"]["accuracy_w_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["decision"]["correct_w_gt"]
            / reduced_metrics["path_1"]["per_loop"]["decision"]["total_w_gt"]
            if reduced_metrics["path_1"]["per_loop"]["decision"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["path_1"]["per_loop"]["decision"]["accuracy_wo_gt"] = (
            reduced_metrics["path_1"]["per_loop"]["decision"]["correct_wo_gt"]
            / reduced_metrics["path_1"]["per_loop"]["decision"]["total_wo_gt"]
            if reduced_metrics["path_1"]["per_loop"]["decision"]["total_wo_gt"] > 0
            else 0.0
        )

        return reduced_metrics

    # built-in evaluation implementations
    def evaluate(self, result: dict) -> None:
        """Evaluate a single sample result and save metrics.

        Args:
            result (dict): The result to evaluate.
        """
        if not hasattr(self, "metrics") or self.metrics is None:
            raise ValueError("Metrics have not been initialized. Call init_metrics() before evaluate().")

        if "total" not in self.metrics:
            raise ValueError(
                "Built-in metrics require to be initialized with the default name 'total'. "
                "Call init_metrics('total') before evaluate()."
            )

        dx = result["metadata"]["target_dx"]
        dx_label = result["metadata"]["dx_label"]
        dx_label_str = "yes" if dx_label else "no"

        if dx not in self.metrics:
            raise ValueError(
                "Built-in metrics require to be initialized for each diagnosis. "
                f"Call init_metrics('{dx}') before evaluate() on samples with target_dx='{dx}'."
            )

        # Evaluate initial diagnostic question
        initial_diagnostic_question_result = result["data"]["initial_diagnostic_question"]
        self.metrics["total"]["initial_diagnostic_question"]["total"] += 1
        self.metrics[dx]["initial_diagnostic_question"]["total"] += 1

        eval_path = initial_diagnostic_question_result["eval_path"]
        if eval_path == -1:
            self.metrics["total"]["failed"]["total"] += 1
            self.metrics[dx]["failed"]["total"] += 1
            return
        elif eval_path == 2:
            self.metrics["total"]["path_2"]["total"] += 1
            self.metrics[dx]["path_2"]["total"] += 1
            # TODO add more detailed metrics for path 2 when it is implemented
            return
        else:

            def _eval_step(step, terminated_early, metric_names) -> bool:
                step_name = step["question_type"]
                gt = step["answer"]
                model_response = step["model_response"]

                corrected = self.validate(gt, model_response, step_name)
                for name in metric_names:
                    self.metrics[name]["path_1"]["per_loop"][step_name]["total_w_gt"] += 1
                    if not terminated_early:
                        self.metrics[name]["path_1"]["per_loop"][step_name]["total_wo_gt"] += 1

                    if corrected:
                        self.metrics[name]["path_1"]["per_loop"][step_name]["correct_w_gt"] += 1
                        if not terminated_early:
                            self.metrics[name]["path_1"]["per_loop"][step_name]["correct_wo_gt"] += 1

                return corrected

            self.metrics["total"]["path_1"]["total"] += 1
            self.metrics[dx]["path_1"]["total"] += 1
            model_response = initial_diagnostic_question_result["model_response"]
            initial_dx_correct = self.validate(dx_label_str, model_response, "initial_diagnostic_question")
            if initial_dx_correct:
                self.metrics["total"]["initial_diagnostic_question"]["correct"] += 1
                self.metrics[dx]["initial_diagnostic_question"]["correct"] += 1

            terminated_early = False
            final_dx_correct = False
            # Evaluate stepwise reasoning for path 1
            for loop_idx, loop in enumerate(result["data"]["path_1"]):
                have_grounding_step = False
                depth_in_loop = 0
                for step_name, step in loop.items():
                    if step_name == "grounding":
                        if len(step) > 0:
                            have_grounding_step = True
                        for g_step in step:
                            corrected = _eval_step(g_step, terminated_early, metric_names=["total", dx])
                            if not terminated_early and not corrected:
                                terminated_early = True
                    else:
                        corrected = _eval_step(step, terminated_early, metric_names=["total", dx])
                        if not terminated_early and not corrected:
                            terminated_early = True

                    if not terminated_early:
                        depth_in_loop += 1

                self.metrics["total"]["path_1"]["per_loop"]["total"] += 1
                self.metrics[dx]["path_1"]["per_loop"]["total"] += 1
                if have_grounding_step:
                    self.metrics["total"]["path_1"]["per_loop"]["depth_total"] += 1
                    self.metrics["total"]["path_1"]["per_loop"]["depth_sum"] += depth_in_loop
                    self.metrics[dx]["path_1"]["per_loop"]["depth_total"] += 1
                    self.metrics[dx]["path_1"]["per_loop"]["depth_sum"] += depth_in_loop

                    if not terminated_early and loop_idx == len(result["data"]["path_1"]) - 1:
                        final_dx_correct = self._validate_decision(
                            loop["decision"]["answer"], loop["decision"]["model_response"]
                        )

            if not terminated_early:
                self.metrics["total"]["path_1"]["completed"] += 1
                self.metrics[dx]["path_1"]["completed"] += 1

                if initial_dx_correct and final_dx_correct:
                    self.metrics["total"]["path_1"]["aligned"] += 1
                    self.metrics[dx]["path_1"]["aligned"] += 1

    def validate(self, gt: Union[str, List[str]], model_response: str, question_type: str) -> bool:
        """Validate the model response against the ground truth.

        Args:
            gt (Union[str, List[str]]): The ground truth answer(s).
            model_response (str): The model's response.
            question_type (str): The type of question being evaluated.

        Returns:
            bool: True if the model response is correct, False otherwise.
        """
        raise NotImplementedError("Evaluator must implement the validate method.")

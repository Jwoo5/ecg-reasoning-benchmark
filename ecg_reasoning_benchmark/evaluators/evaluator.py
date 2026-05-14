import argparse
from typing import List, Optional, Union


class Evaluator:
    @staticmethod
    def parse_arguments(args):
        """Parse evaluator-specific arguments.

        Args:
            args: List of command-line arguments.

        Returns:
            Parsed arguments.
        """
        parser = argparse.ArgumentParser()
        # expected to raise an error if there are unknown args when the subclass do not override
        # the parser
        return parser.parse_args(args)

    @staticmethod
    def add_default_arguments() -> argparse.ArgumentParser:
        """Add default evaluator arguments to the parser.
        Expected to be used in the subclass implementations of parse_arguments.

        Returns:
            argparse.ArgumentParser: The argument parser with default evaluator arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--use-builtin-metrics",
            action="store_true",
            help="if set, use the built-in metrics defined in the Evaluator class.",
        )
        return parser

    def __init__(self, args):
        self.args = args

        self.name = None

        if args.use_builtin_metrics:
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

        stepwise_metrics = {
            "total_w_gt": 0,
            "correct_w_gt": 0,
        }
        metrics = {
            "initial_diagnostic_question": {
                "total": 0,
                "correct": 0,
            },
            "gt_reasoning_based_diagnosis": {
                "total": 0,
                "correct": 0,
            },
            "self_reasoning_based_diagnosis": {
                "total": 0,
                "correct": 0,
            },
            "reasoning": {
                "total": 0,
                "completed_hard": 0,
                "completed_soft": 0,
                "per_loop": {
                    "total": 0,
                    "depth_total": 0,  # depth should be computed only for those having grounding steps
                    "depth_sum": 0,
                    "criterion_selection": stepwise_metrics.copy(),
                    "finding_identification": stepwise_metrics.copy(),
                    "lead_grounding": stepwise_metrics.copy(), 
                    "wave_grounding": stepwise_metrics.copy(),
                    "measurement_grounding": stepwise_metrics.copy(),
                    "diagnostic_decision": stepwise_metrics.copy(),
                },
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

        # compute accuracy for final diagnostic question with gt reasoning
        reduced_metrics["gt_reasoning_based_diagnosis"]["accuracy"] = (
            reduced_metrics["gt_reasoning_based_diagnosis"]["correct"]
            / reduced_metrics["gt_reasoning_based_diagnosis"]["total"]
            if reduced_metrics["gt_reasoning_based_diagnosis"]["total"] > 0
            else 0.0
        )

        # compute accuracy for self-reasoning-based diagnosis: walk the chain
        # with model predictions until the first yes/no diagnostic_decision; score
        # 1 if every prior reasoning step was correct and that decision matches
        # the sample-level dx_label.
        reduced_metrics["self_reasoning_based_diagnosis"]["accuracy"] = (
            reduced_metrics["self_reasoning_based_diagnosis"]["correct"]
            / reduced_metrics["self_reasoning_based_diagnosis"]["total"]
            if reduced_metrics["self_reasoning_based_diagnosis"]["total"] > 0
            else 0.0
        )

        # compute completion (hard: strict termination; soft: tolerate a wrong
        # diagnostic_decision when the GT is "further findings are required to
        # confirm the diagnosis")
        reduced_metrics["reasoning"]["completion_hard"] = (
            reduced_metrics["reasoning"]["completed_hard"] / reduced_metrics["reasoning"]["total"]
            if reduced_metrics["reasoning"]["total"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["completion_soft"] = (
            reduced_metrics["reasoning"]["completed_soft"] / reduced_metrics["reasoning"]["total"]
            if reduced_metrics["reasoning"]["total"] > 0
            else 0.0
        )

        # compute per loop metrics
        reduced_metrics["reasoning"]["per_loop"]["depth_avg"] = (
            reduced_metrics["reasoning"]["per_loop"]["depth_sum"]
            / reduced_metrics["reasoning"]["per_loop"]["depth_total"]
            if reduced_metrics["reasoning"]["per_loop"]["depth_total"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["criterion_selection"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["criterion_selection"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["criterion_selection"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["criterion_selection"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["finding_identification"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["finding_identification"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["finding_identification"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["finding_identification"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["lead_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["lead_grounding"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["lead_grounding"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["lead_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["wave_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["wave_grounding"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["wave_grounding"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["wave_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["measurement_grounding"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["measurement_grounding"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["measurement_grounding"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["measurement_grounding"]["total_w_gt"] > 0
            else 0.0
        )
        reduced_metrics["reasoning"]["per_loop"]["diagnostic_decision"]["accuracy_w_gt"] = (
            reduced_metrics["reasoning"]["per_loop"]["diagnostic_decision"]["correct_w_gt"]
            / reduced_metrics["reasoning"]["per_loop"]["diagnostic_decision"]["total_w_gt"]
            if reduced_metrics["reasoning"]["per_loop"]["diagnostic_decision"]["total_w_gt"] > 0
            else 0.0
        )

        return reduced_metrics

    # built-in evaluation implementations
    def evaluate(self, result: dict) -> Optional[int]:
        """Evaluate a single sample result and save metrics.
        For those evaluators that have `estimate_cost` argument (e.g., GeminiEvaluator), this method
        may return an integer representing the token count instead of performing evaluation.

        Args:
            result (dict): The result to evaluate.
        """
        dx = result["metadata"]["target_dx"]
        dx_label = result["metadata"]["dx_label"]
        dx_label_str = "yes" if dx_label else "no"

        is_dry_run = getattr(self.args, "estimate_cost", False)
        if is_dry_run:
            # in dry-run mode, we do not evaluate metrics and just calculate token usage instead to
            # estimate cost
            total_input_tokens = 0
            total_input_tokens += self.validate(
                question=result["data"]["initial_diagnostic_question"]["question"],
                gt=dx_label_str,
                model_response=result["data"]["initial_diagnostic_question"]["model_response"],
                question_type="initial_diagnostic_question",
            )
            for loop_idx, loop in enumerate(result["data"]["reasoning"]):
                for step_name, step in loop.items():
                    if step_name == "ecg_grounding":
                        for g_step in step:
                            total_input_tokens += self.validate(
                                question=g_step["question"],
                                gt=g_step["answer"],
                                model_response=g_step["model_response"],
                                question_type=g_step["question_type"],
                            )
                    else:
                        total_input_tokens += self.validate(
                            question=step["question"],
                            gt=step["answer"],
                            model_response=step["model_response"],
                            question_type=step_name,
                        )
            return total_input_tokens

        if not hasattr(self, "metrics") or self.metrics is None:
            raise ValueError("Metrics have not been initialized. Call init_metrics() before evaluate().")

        if "total" not in self.metrics:
            raise ValueError(
                "Built-in metrics require to be initialized with the default name 'total'. "
                "Call init_metrics('total') before evaluate()."
            )

        if dx not in self.metrics:
            raise ValueError(
                "Built-in metrics require to be initialized for each diagnosis. "
                f"Call init_metrics('{dx}') before evaluate() on samples with target_dx='{dx}'."
            )

        # Evaluate initial diagnostic question
        initial_diagnostic_question_result = result["data"]["initial_diagnostic_question"]
        self.metrics["total"]["initial_diagnostic_question"]["total"] += 1
        self.metrics[dx]["initial_diagnostic_question"]["total"] += 1

        def _eval_step(step, terminated_early, metric_names) -> bool:
            question = step["question"]
            gt = step["answer"]
            model_response = step["model_response"]
            step_name = step["question_type"]

            corrected = self.validate(
                question=question, gt=gt, model_response=model_response, question_type=step_name
            )
            for name in metric_names:
                self.metrics[name]["reasoning"]["per_loop"][step_name]["total_w_gt"] += 1
                if corrected:
                    self.metrics[name]["reasoning"]["per_loop"][step_name]["correct_w_gt"] += 1

            return corrected

        self.metrics["total"]["reasoning"]["total"] += 1
        self.metrics[dx]["reasoning"]["total"] += 1
        question = initial_diagnostic_question_result["question"]
        model_response = initial_diagnostic_question_result["model_response"]
        initial_dx_correct = self.validate(
            question=question,
            gt=dx_label_str,
            model_response=model_response,
            question_type="initial_diagnostic_question",
        )
        if initial_dx_correct:
            self.metrics["total"]["initial_diagnostic_question"]["correct"] += 1
            self.metrics[dx]["initial_diagnostic_question"]["correct"] += 1

        # Completion is reported in two flavors:
        #   - hard: any wrong answer terminates the chain.
        #   - soft: a wrong `diagnostic_decision` is tolerated when the GT is
        #     "further findings are required to confirm the diagnosis"; every
        #     other step is strict. By construction completed_soft >= completed_hard.
        # `depth_in_loop` stays on hard semantics.
        terminated_early_hard = False
        terminated_early_soft = False
        final_dx_correct = False
        final_dx_correct_w_gt_reasoning = False
        # Self-RDA state: walk the chain with model predictions until the first
        # yes/no diagnostic_decision emitted by the model, then compare against
        # dx_label. Non-decision step failures (criterion_selection /
        # finding_identification / ecg_grounding) break the chain; an
        # intermediate diagnostic_decision where the model disagrees with the
        # "further findings" GT also triggers resolution (model emitted yes/no
        # early).
        self_rda_reasoning_broken = False
        self_rda_resolved = False
        self_rda_correct_flag = False
        # Evaluate stepwise in reasoning
        for loop_idx, loop in enumerate(result["data"]["reasoning"]):
            have_grounding_step = False
            terminated_early_in_loop_hard = False
            depth_in_loop = 0
            for step_name, step in loop.items():
                if step_name == "ecg_grounding":
                    if len(step) == 0:
                        continue

                    have_grounding_step = True
                    depth_per_grounding = 0
                    for g_step in step:
                        corrected = _eval_step(g_step, terminated_early_hard, metric_names=["total", dx])
                        if corrected:
                            if not terminated_early_in_loop_hard:
                                depth_per_grounding += 1
                        else:
                            terminated_early_hard = True
                            terminated_early_soft = True
                            terminated_early_in_loop_hard = True
                            if not self_rda_resolved:
                                self_rda_reasoning_broken = True
                    depth_in_loop += depth_per_grounding / len(step)
                else:
                    corrected = _eval_step(step, terminated_early_hard, metric_names=["total", dx])
                    is_soft_skippable = (
                        step_name == "diagnostic_decision"
                        and isinstance(step["answer"], str)
                        and step["answer"].lower()
                        == "further findings are required to confirm the diagnosis"
                    )
                    if corrected:
                        if not terminated_early_in_loop_hard:
                            depth_in_loop += 1
                    else:
                        terminated_early_hard = True
                        terminated_early_in_loop_hard = True
                        if not is_soft_skippable:
                            terminated_early_soft = True

                    # Self-RDA resolution at diagnostic_decision
                    if step_name == "diagnostic_decision" and not self_rda_resolved:
                        if self_rda_reasoning_broken:
                            # upstream reasoning already failed -> Self-RDA=0
                            self_rda_resolved = True
                            self_rda_correct_flag = False
                        elif corrected:
                            if is_soft_skippable:
                                # GT="further findings" and model matched ->
                                # keep going; Self-RDA stays unresolved.
                                pass
                            else:
                                # final-loop GT is yes/no and model matched; by
                                # construction the GT equals dx_label, so the
                                # model's first yes/no matches dx_label.
                                self_rda_resolved = True
                                self_rda_correct_flag = True
                        else:
                            if is_soft_skippable:
                                # model emitted yes/no (or anything other than
                                # "further findings") when GT expected "further
                                # findings": this is the first yes/no. Compare
                                # against dx_label.
                                if hasattr(self, "_validate_decision"):
                                    matches_dx = self._validate_decision(
                                        dx_label_str, step["model_response"]
                                    )
                                else:
                                    matches_dx = self.validate(
                                        question=step["question"],
                                        gt=dx_label_str,
                                        model_response=step["model_response"],
                                        question_type="diagnostic_decision",
                                    )
                                self_rda_resolved = True
                                self_rda_correct_flag = bool(matches_dx)
                            else:
                                # final-loop GT=dx_label and model didn't match:
                                # either opposite yes/no or "further findings".
                                # Both give Self-RDA=0.
                                self_rda_resolved = True
                                self_rda_correct_flag = False
                    elif step_name != "diagnostic_decision" and not corrected:
                        # criterion_selection / finding_identification wrong
                        if not self_rda_resolved:
                            self_rda_reasoning_broken = True

            self.metrics["total"]["reasoning"]["per_loop"]["total"] += 1
            self.metrics[dx]["reasoning"]["per_loop"]["total"] += 1
            # we only count depth for those having grounding steps so that the depth metric is ranged
            # from 0 to 4 (criterion_selection, finding_identification, grounding, decision)
            if have_grounding_step:
                self.metrics["total"]["reasoning"]["per_loop"]["depth_total"] += 1
                self.metrics["total"]["reasoning"]["per_loop"]["depth_sum"] += depth_in_loop
                self.metrics[dx]["reasoning"]["per_loop"]["depth_total"] += 1
                self.metrics[dx]["reasoning"]["per_loop"]["depth_sum"] += depth_in_loop

            if loop_idx == len(result["data"]["reasoning"]) - 1:
                if hasattr(self, "_validate_decision"):
                    final_dx_correct_w_gt_reasoning = self._validate_decision(
                        loop["diagnostic_decision"]["answer"], loop["diagnostic_decision"]["model_response"]
                    )
                    
                else:
                    final_dx_correct_w_gt_reasoning = self.validate(
                        question=loop["diagnostic_decision"]["question"],
                        gt=loop["diagnostic_decision"]["answer"],
                        model_response=loop["diagnostic_decision"]["model_response"],
                        question_type="diagnostic_decision",
                    )
                
                if not terminated_early_hard and final_dx_correct_w_gt_reasoning:
                    final_dx_correct = True

        # gt-reasoning-based diagnosis
        self.metrics["total"]["gt_reasoning_based_diagnosis"]["total"] += 1
        self.metrics[dx]["gt_reasoning_based_diagnosis"]["total"] += 1
        if final_dx_correct_w_gt_reasoning:
            self.metrics["total"]["gt_reasoning_based_diagnosis"]["correct"] += 1
            self.metrics[dx]["gt_reasoning_based_diagnosis"]["correct"] += 1

        # self-reasoning-based diagnosis
        self.metrics["total"]["self_reasoning_based_diagnosis"]["total"] += 1
        self.metrics[dx]["self_reasoning_based_diagnosis"]["total"] += 1
        if self_rda_correct_flag:
            self.metrics["total"]["self_reasoning_based_diagnosis"]["correct"] += 1
            self.metrics[dx]["self_reasoning_based_diagnosis"]["correct"] += 1

        if not terminated_early_hard:
            self.metrics["total"]["reasoning"]["completed_hard"] += 1
            self.metrics[dx]["reasoning"]["completed_hard"] += 1
        if not terminated_early_soft:
            self.metrics["total"]["reasoning"]["completed_soft"] += 1
            self.metrics[dx]["reasoning"]["completed_soft"] += 1

    def validate(
        self, question: str, gt: Union[str, List[str]], model_response: str, question_type: str, **kwargs
    ) -> Union[bool, int]:
        """Validate the model response against the ground truth.
        For those evaluators that have `estimate_cost` argument (e.g., GeminiEvaluator), this method
        may return an integer representing the token count instead of a boolean validation result.

        Args:
            question (str): The question being evaluated.
            gt (Union[str, List[str]]): The ground truth answer(s).
            model_response (str): The model's response.
            question_type (str): The type of question being evaluated.

        Returns:
            Union[bool, int]: True if the model response is correct, False otherwise,
                or an integer token count if estimating cost.
        """
        raise NotImplementedError("Evaluator must implement the validate method.")

    def to_csv_row(self, reduced_metrics: dict, model_name: str) -> dict:
        """Flatten a ``reduced_metrics`` dict returned by :meth:`reduce_metrics`
        into a single row keyed by column name, suitable for a ``pandas`` DataFrame.

        The default implementation emits the schema used by the built-in
        teacher-forced evaluators (heuristic / gemini): IDQ accuracy, GT-reasoning
        diagnosis accuracy, completion (hard/soft), depth, and per-loop stepwise
        metrics. Subclasses with a different metric shape (e.g., forced-commit
        evaluators) should override this method.

        Args:
            reduced_metrics (dict): Output of :meth:`reduce_metrics`.
            model_name (str): Model identifier to place in the ``model`` column.

        Returns:
            dict: One row of column -> value suitable for CSV export.
        """
        row = {"model": model_name}

        # initial diagnostic question
        idq_metric = reduced_metrics["initial_diagnostic_question"]
        row["idq_total"] = idq_metric["total"]
        row["idq_correct"] = idq_metric["correct"]
        row["idq_accuracy"] = idq_metric["accuracy"]

        # gt-reasoning-based diagnosis
        gtd_metric = reduced_metrics["gt_reasoning_based_diagnosis"]
        row["gt_reasoning_based_diagnosis_total"] = gtd_metric["total"]
        row["gt_reasoning_based_diagnosis_correct"] = gtd_metric["correct"]
        row["gt_reasoning_based_diagnosis_accuracy"] = gtd_metric["accuracy"]

        # self-reasoning-based diagnosis
        srd_metric = reduced_metrics["self_reasoning_based_diagnosis"]
        row["self_reasoning_based_diagnosis_total"] = srd_metric["total"]
        row["self_reasoning_based_diagnosis_correct"] = srd_metric["correct"]
        row["self_reasoning_based_diagnosis_accuracy"] = srd_metric["accuracy"]

        # reasoning
        row["reasoning_total"] = reduced_metrics["reasoning"]["total"]
        for key in reduced_metrics["reasoning"]:
            if key in ["total", "per_loop"]:
                continue
            row[key] = reduced_metrics["reasoning"][key]

        # per-loop metrics
        per_loop = reduced_metrics["reasoning"]["per_loop"]
        row["per_loop_total"] = per_loop["total"]
        row["per_loop_depth_total"] = per_loop["depth_total"]
        row["per_loop_depth_sum"] = per_loop["depth_sum"]
        row["depth"] = per_loop["depth_avg"]
        for step in per_loop:
            if step in ["total", "depth_total", "depth_sum", "depth_avg"]:
                continue
            for key in per_loop[step]:
                row[f"{step}_{key}"] = per_loop[step][key]

        return row

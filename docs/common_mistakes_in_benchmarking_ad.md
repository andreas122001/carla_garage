# Common Mistakes in Benchmarking Autonomous Driving

Evaluating entire autonomous driving systems is a complex task, that is prone to subtle but significant errors. Unfortunately, such methodological mistakes are widespread in the autonomous driving literature.
This document aims to outline these issues so that reviewers can identify them and authors can avoid them.
The intent of this document is not to criticize individual authors or works. We all make mistakes. 
Science is a self-correcting process, what matters is how we address and rectify issues.
As such, we will focus on the methodological mistakes, without citing the works we refer to.

Many of the points we raise may be familiar to experts, but are often stated implicitly rather than explicitly. As a result, readers often need to read between the lines to get these messages. This implicit communication makes it hard for newcomers to the field and for those who with a peripheral interest in the literature to understand what is going on.
This article summarizes the various methodological problems we observe in the literature, aiming to be more explicit than prior work.

The ability to benchmark full autonomous driving systems in a manner that is reproducible across research laboratories first became possible around 2017 with the introduction of the [CARLA simulator](https://arxiv.org/abs/1711.03938). In the years ever since, various benchmarks have been proposed for CARLA and other simulators.
Before covering more specific mistakes made in CARLA benchmarks, we have to however address a bigger issue:

## NuScenes Planning
### Open-loop trajectory errors are unreliable.

[NuScenes](https://arxiv.org/abs/1903.11027) is a well-known autonomous driving dataset that features various computer vision benchmarks.
Using a dataset like nuScenes to evaluate the planning output of an autonomous driving stack is called open-loop planning, and works by feeding the sensor data in the stack and comparing its prediction with the ground truth action from the human driver in the dataset. This is called open-loop planning because the sensor input of future frames will not depend on the prediction of the driving stack, and neither will the other cars react to its behavior. Open-loop planning is appealing due to its computational efficiency compared to closed-loop evaluation (either in the real world or in simulation) and it does not introduce a sim-to-real gap with respect to the sensor data, unlike simulators.
Given these advantages, it was investigated early on whether the open-loop trajectory errors, measured via L2 loss, could act as a reliable performance indicator for closed-loop performance, which is what we ultimately want.
Unfortunately, this turned out not to be the case as was shown in [Codevilla et al. 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Felipe_Codevilla_On_Offline_Evaluation_ECCV_2018_paper.pdf). The open-loop L2 error did not necessarily correlate with the closed-loop error. In other words, the open-loop L2 loss can be misleading.
As a result, the community focused its benchmarking efforts on closed-loop simulation.
As researchers working on end-to-end autonomous driving solutions, we can confirm these results. In our experience, open-loop validation losses of end-to-end systems are not useful as indicators for closed-loop performance, and we therefore do not report them.

### The L2 metric became popular.
For the reason outlined before, the nuScenes paper itself did **not** propose a planning benchmark (according to private communication with the authors, intentionally so).
However, the planning community grew significantly around 2022/2023 and some researchers, seemingly unaware of these early results, began using the nuScenes dataset to benchmark planning using the L2 loss (or variants of this, called displacement errors) as a primary metric. This benchmark became known as nuScenes Planning. Due to the good reputation of nuScenes in the vision community (and one of the papers using the benchmark winning a prestigious vision award), this benchmark was widely adopted.

### NuScenes planning led to misleading results.
Several research papers have supplemented their nuScenes planning results with closed-loop CARLA simulation results. This made it evident that the nuScenes planning methods were behind the state-of-the-art methods from CARLA. Although, this conclusion is implicit as only older papers were reported as CARLA baselines.

In an attempt to see whether the previously discussed findings also apply to nuScenes planning in particular, [Zhai et al. 2023](https://arxiv.org/abs/2305.10430) introduced an innovative diagnostic test. They proposed a driving stack called AD-MLP that has no access to perception inputs and solely extrapolates past movement in the dataset, based on ego-status information. This approach is viable because of the open-loop nature of the dataset and the inherent smoothness of driving logs. To be precise here, AD-MLP is an explicitly unrealistic model whose purpose is to expose the flaws of a metric or benchmarks, analogous to a constant baseline as used in other machine learning fields.
It turned out that AD-MLP works exceptionally well in nuScenes planning. Even after the publication of numerous subsequent papers, it remains the state of the art on nuScenes planning. 
Unfortunately, these results were largely ignored in the nuScenes planning community initially, although [Li et al. 2024](https://arxiv.org/abs/2312.03031v2) later reproduced them in a more rigorous fashion.

Recent benchmarks like Bench2Drive or NAVSIM have adopted AD-MLP as a sanity check. AD-MLP having poor performance on a benchmark is an indicator for the quality of the benchmark.

### There are additional flaws in the nuScenes planning benchmark.

Besides these fundamental flaws in nuScenes planning, numerous execution-related issues regarding the benchmark have also been identified. [Weng et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf) describes several of these issues, such as the metrics being incorrectly computed, inconsistencies in metric definitions across different papers and post-processing being done differently across methods. There are more problems than discussed in the paper. For example, the conditioning input command is computed based on the waypoint labels, resulting in label leakage. Furthermore, the size of the validation set is orders of magnitudes smaller than in other planning benchmarks (this limitation is specific to nuScenes planning, as perception tasks have many labels per frame).

### We need to stop using nuScenes planning!

In our view, the only rational course of action regarding nuScenes planning is to cease using it as a benchmark.
The numerous issues in the benchmark render it unsuitable for evaluating the performance of autonomous driving stacks.
If you are a reviewer evaluating a paper in 2025 that bases its claims solely on nuScenes planning results, we ask you to reject it.
If you are an author and encounter a reviewer (Reviewer 2, perhaps) requesting nuScenes planning results, we recommend politely declining. In such cases, you could refer to this text or the cited papers as justification for your decision.

### NuScenes planning reinvigorated the search for better open-loop metrics.

A positive outcome of the nuScenes planning story was that it highlighted the substantial interest in cost-effective open-loop evaluation and started a search for better open loop metrics ([Weng et al. 2024](https://xinshuoweng.github.io/paradrive/assets/camera_ready.pdf), [Li et al.2024](https://arxiv.org/abs/2312.03031), [Dauner et al. 2024](https://arxiv.org/abs/2406.15349)). In particular, the discovery that the PDM-score metric is correlated to closed-loop performance led to a new type of benchmark (NAVSIM). The PDM-score is an open-loop metric that performs a pseudo simulation based on the "world on rails" assumption.
NAVSIM represents a significant improvement over nuScenes planning, as an open-loop benchmark.
However, it remains unable to measure important issues such as compounding errors, implying that it should be complemented with closed-loop experiments. For research groups, lacking the computational resources required for large-scale closed-loop simulations, NAVSIM currently serves as the recommended alternative to nuScenes planning.

### Using nuScenes for perception tasks is valid.
It is important to note, that we do not claim that all conclusions drawn from nuScenes planning are incorrect. Rather, we claim that the validity of these ideas remain uncertain. While some of the proposed approaches may indeed be effective, others may not. To know, the community needs to reevaluate these ideas on appropriate and reliable benchmarks. Furthermore, our criticism is specific to nuScenes planning and does not extend to other nuScenes benchmarks for perception and prediction tasks.

## CARLA benchmarks
[CARLA](https://arxiv.org/abs/1711.03938) is an autonomous driving simulator developed using the Unreal Engine. It is capable of simulating a wide range of sensor data, ground truth labels, safety critical scenarios, and background traffic. Its open-source release in 2017 enabled the community to benchmark entire autonomous driving stacks in closed-loop. Due to the community's continual commitment to update and improve the simulator, it has become the de facto standard for rigorous evaluation of autonomous driving stacks.
CARLA, as a simulator, does not inherently include a benchmark nor a dataset for model development.
Instead, the community has developed numerous benchmarks based on the CARLA simulator, which increased in difficulty over time. Some noteworthy once include the [original CARLA benchmark](https://arxiv.org/abs/1711.03938), [NoCrash](https://arxiv.org/abs/1904.08980), [Town05 Short/Long](https://arxiv.org/abs/2104.09224), [LAV routes](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_From_All_Vehicles_CVPR_2022_paper.pdf), [Longest6](https://arxiv.org/abs/2205.15997), [CARLA leaderboard 1.0 test routes](https://leaderboard.carla.org/#leaderboard-10), [Bench2Drive](https://arxiv.org/abs/2406.03877), [CARLA leaderboard 2.0 validation routes](https://leaderboard.carla.org) and [CARLA leaderboard 2.0 test routes](https://leaderboard.carla.org). These benchmarks are approximately arranged in order of increasing difficulty, although the list is not exhaustive.

### Papers frequently have errors regarding the setup details of CARLA benchmarks.

These CARLA benchmarks have various setup conditions including the CARLA towns from which data collection is permitted, which safety critical scenarios to evaluate with, what routes to drive along, the traffic density to be used and more.
Inconsistencies between these parameters can undermine the validity of comparisons between methods.

A recurring issue in the literature is authors comparing numbers from different benchmarks to each other.
This is a methodological mistake akin to comparing accuracies on CiFAR-10 to accuracies on MNIST, rendering the comparison meaningless.

Furthermore, we observe subtle errors in the details of evaluations, such as training on validation towns, evaluating without safety critical scenarios, neglecting to adjust the traffic density, and citing the wrong papers. A notable example of misquotation involves the TransFuser model, where author often report the (weaker) performance from the earlier conference paper, but cite the (stronger) stack from the journal extension.
These mistakes are so widespread in the literature that caution is warranted when copying results from tables in other published works. Instead of directly using reported results, it is recommended to re-evaluate baselines on benchmarks, as many of the baselines are open-source. At a minimum, researchers should verify the correctness of the numbers they reference.

When conducting evaluations, ensure that the conditions for a proper benchmark setup are followed. Any deviations from these conditions should be documented and explained.

### Closed-loop (CARLA) benchmarks are sensitive to random chance.

Another notable challenge with closed-loop benchmarks is that results are fundamentally sensitive to random chance (the seed). It is a standard practice to repeat evaluations at least 3 times and average the result. This issue is particularly pronounced in end-to-end methods, which often exhibit high training variance. Consequently, it is sometimes necessary to evaluate multiple training seeds as well ([Prakash et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf), [Behl et al. 2020](https://arxiv.org/abs/2005.10091), [Chitta et al. 2023](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf)). For example, in experiments on the LAV benchmark, we found it necessary to average the result of 3 training seeds each evaluated 3 times, otherwise the results were heavily influenced by random chance.
Unfortunately, recent trends in conference publications have seen a resurgence of single-seed evaluations. As a reviewer, it is important to remind authors that evaluating at least 3 seeds is necessary to ensure reliable comparisons.

The large amount of different benchmarks also introduced two additional subtle issues:

### Some papers propose new benchmarks with misleading names.

Some authors have used the confusion in the field to introduce easier benchmarks under the same name as existing, more challenging benchmarks. This practice can artificially enhance the perceived performance of their methods. For instance, this has occurred multiple times with the CARLA leaderboard, due to its good reputation and high difficulty. Although such incidents are relatively infrequent, they do occur, necessitating vigilance from reviewers.

### Some papers incorrectly claim state of the art.

Another significant issue is the difficulty of accurately tracking the state of the art on the various benchmarks. This has led to instances where authors are making incorrect claims about their method being state of the art even though their results are clearly lagging behind by several years. Such claims are often obfuscated by exclusively reporting comparisons to outdated baselines. Several papers published at top-tier conferences in 2024 have made such incorrect claims.
Reviewers who are uncertain about the state of the art on a particular benchmark are encouraged to consult other experts, such as the author of the benchmark.
It is acceptable to claim state of the art in the presence of concurrent work, e.g. from arXiv within the last 6 months (roughly corresponding to one conference review cycle). A method that is outperformed by published work from three years earlier cannot be considered state of the art.

### Some papers omit performance critical details.

The CARLA leaderboard test routes (version 1.0 and 2.0) are benchmarks with secret test routes, where evaluation is conducted by submitting code to a third-party server operated by the CARLA team. Participating teams can then view and publicize their results through a dedicated website, [https://leaderboard.carla.org/leaderboard/](https://leaderboard.carla.org/leaderboard/). Although operating such an independent evaluation system is resource-intensive, it ensures fair benchmarking and ensures that the claimed results have been achieved. Over the past five years, this system has generally functioned effectively and facilitated the fast progress we observed on these benchmarks.

However, a notable issue arose with the CARLA leaderboard 1.0. While the evaluation system ensures that results have been achieved, it does not ensure is that authors accurately report how they achieved their results. As documented by [Jaeger et al. 2023, Appendix C.6](https://arxiv.org/abs/2306.07957), the top three methods on the CARLA leaderboard were found to be unreproducible. The models and code of these methods are open source, yet the published models produce significantly weaker results than those reported in the papers and displayed on the leaderboard website. Since the published code corresponds to the description provided by the papers, it is questionable whether the papers accurately describe the models that were used to achieve the state-of-the-art results. 

This problem is difficult to identify during the peer review process, as it only becomes apparent through extensive reproduction efforts. We encourage reviewers to demand the release of code alongside a paper. Such transparency allows issues to be identified and resolved post-publication. There is no scientific justification for withholding the code of a publication. Unfortunately, the current top method on the CARLA leaderboard 1.0 did not publish its code, and as a result, its claims have never been independently reproduced.

Progress has stopped on the CARLA leaderboard 1.0 for the last two years, likely due to this situation.
As the community transitions to the CARLA leaderboard 2.0 we hope this situation does not reoccur.
The recent [NAVSIM leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) aims to mitigate such problems by requiring the release of source code and models, and enforces reproducibility by removing submissions that fail to meet these criteria.


## Proprietary benchmarks
Many autonomous driving papers written by industrial labs are not evaluated on public datasets or benchmarks, and do neither release their code, models nor data.
Given how tricky it is to assess the validity of claims made even on public data, this can lead to a lack of trust in those papers. As readers, we can often only say that this might be an interesting idea, but we cannot judge from the paper whether it actually works better than existing ideas or not.
Reproducing ideas simply based on the paper is often tricky and requires months of work and can fail because the idea wasn't actually working, or important technical details have been omitted.
As a result, many autonomous driving papers from industry are never reproduced.
This has the second-order effect that some of these ideas probably don't get the attention they deserve (in terms of citations and community adoption).

### Reasons for not publishing the code of a paper:

As we have said before, there is no scientific argument for withholding code in autonomous driving.
There are however other types of reasons why authors don't publicize their code.
1. The presented idea does not really work, and the authors are trying to hide that by making the paper hard to reproduce.
2. The authors have a financial interest in the idea and do not want their competitors to be able to use their work.
3. Cleaning code for publication requires extra effort, and the authors do not have the time to do it.

The problem now is that papers from groups 2 and 3 will inadvertently get associated with papers from group 1 because there is no practical way to tell them apart.
We would like to propose a different solution for groups 2 and 3 than not publishing the code. 

### You should publish your code with a non-commercial license instead!

What we would like to point out is that **there is no need to publish your code open-source**, as in with an open-source license. Scientifically, it is sufficient to publish your code under a **non-commercial research license**. Such licenses allow researchers to use your software to perform research (the result of which might benefit your company), but prevent your competitors from using your software (or at least they have to pay you to get a different license).
Such licenses are already common practice for autonomous driving datasets, see the [nuScenes license](https://www.nuscenes.org/terms-of-use-commercial) as an example.
Instead of keeping the code of papers secret, we encourage authors to publish their code under such licenses. Another example of such a license is the [Software Copyright License for non-commercial scientific research purposes](https://icon.is.tue.mpg.de/license.html) used by the Max Plank Institute for Intelligent Systems. As for the third group, just publish your code without cleaning it (most licenses have a clause protecting you from warranty). Publishing messy code is more valuable to the community than publishing no code.


## The simulation argument
We have argued in this text for benchmarking autonomous driving stacks using closed-loop simulations. A common concern that is often expressed is that simulations lack realism, and we need real data to get methods that work on real cars. Often implied is that using simulations for benchmarking might lead to misleading results. There is of course some truth to this argument, as it is possible to construct a misleading simulation. The problem with this claim is that it is usually taken as self-evident. We are not aware of actual empirical evidence that state-of-the-art simulators like CARLA or nuPlan have led to misleading results. (There are of course many misleading results out there, as we have argued above, but the simulator itself is not at fault here).


### Driving simulators might already be realistic enough for benchmarking.
There are two counterarguments to this claim. The first is that our current state-of-the-art simulators might be realistic enough to force researchers to develop general-purpose methods that still work when applied to real data/cars.
The [CVPR 2024 NAVSIM challenge](https://arxiv.org/abs/2406.15349) (ran from March to May 2024) provided an interesting piece of evidence in that direction. 
NAVSIM is a simulator that only uses real sensor data and as a result, is more realistic than for example CARLA in terms of its sensor data.
As part of the challenge starter kit, the authors reproduced the TransFuser method on NAVSIM as a baseline. TransFuser is a method that was designed and developed entirely on synthetic CARLA data. Interestingly, the TransFuser baseline outperformed most of the 463 submissions, and the top solution was an extension of TransFuser. This suggests that using synthetic CARLA data can lead to the development of methods that work well on real-world data. Of course, this is only anecdotal evidence, and NAVSIM is still a simulation.

### Sim-to-real transfer already works in other robotics fields.
The second argument is that other robotics fields (in particular locomotion, agile flight, and indoor navigation) have managed to get zero-shot sim-to-real transfer to work. This is a recent development that has been called ["A quiet revolution in robotics"](https://www.youtube.com/watch?v=K09erFsOnxA). These methods trained entirely in simulation were able to work and significantly advance the state of the art on real robots. This represents an empirical refutation of the argument that simulations can not be realistic enough. However, to our knowledge, nobody has convincingly demonstrated the same level of sim-to-real transfer in autonomous driving yet.
There is one company that publicly claimed to have built a driving simulation that exhibits similar properties, but not many details are known about this claim, so we can not evaluate its validity or extent.

### There is no reason to stop using simulators for benchmarking.
So for now, the claim that the lack of realism in simulations is a fundamental problem for benchmarking can not be entirely refuted. Current evidence points in the opposite direction, so there is no reason to stop using simulations for benchmarking. Data-driven simulations like [NAVSIM](https://arxiv.org/abs/2406.15349), [nuPlan](https://arxiv.org/abs/2106.11810) (not to be confused with nuScenes planning) or [Waymax](https://arxiv.org/abs/2310.08710) offer a complement to synthetic simulators like CARLA, but cannot replace them because they are generally easier to solve due to their lack of long simulations and safety-critical scenarios (performance is currently >90% on data-driven simulators and <10% on the hardest CARLA benchmark).


## The need for public research: 
Around 10 years ago, when neural networks became popular, there was a lot of excitement in the autonomous driving field which created the myth that "The science is solved." This was understandable at the time when limitations of neural networks like shortcut learning weren't known yet, and we didn't know about the long tail. 10 years and billions in investments later, it has become clear that this notion at the time was overly optimistic. It could be argued today that our scientific understanding is at a level that could allow for the deployment of level 4 systems. Whether these can be built profitably remains to be seen, and it is clear that to build a level 5 system we need fundamental scientific advancements. The economic and practical advantages of having a system that can be deployed everywhere (level 5) compared to building custom solutions per region (level 4) should be enormous.
Public research has a huge cost advantage compared to private research, as every dollar spent on public research benefits every company in the space whereas private research only benefits one company, which means spending money on rediscovering the same ideas multiple times.
In our view, it is important to realize that **the autonomous driving effort is a competition against reality first and foremost**. We need public research that adheres to rigorous scientific practices in order to win this competition!

13.12.2024

PS: If you are aware of additional problems in benchmarking autonomous driving that are not yet covered here, please get in touch.

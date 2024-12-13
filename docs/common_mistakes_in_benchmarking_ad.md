# Common mistakes in benchmarking autonomous driving

Benchmarking entire autonomous driving stacks is hard, and it is easy to get important subtle details wrong. Unfortunately, the literature is riddled with these methodological mistakes.
This document is an attempt to list these errors so that reviewers can spot them and authors can avoid them.
The point of this document is not to criticize individual authors or works. We all make mistakes. 
Science is a self-correcting process, what matters is how we move forward.
As such, we will focus on the methodological mistakes and not the papers making them. Hence, we will not cite the works we refer to.
Many of the things we point out here are known, but are often not stated explicitly and instead, readers need to read between the lines when reading papers to get these messages. This makes it hard for authors entering the field and people who only follow the literature on the side to understand what is going on.
This article summarizes the various methodological problems we observe in the literature and tries to be more explicit than prior work.

Benchmarking a full autonomous driving system in a manner that is reproducible across research labs first became possible around 2017 with the release of the [CARLA simulator](https://arxiv.org/abs/1711.03938). In the years ever since, various benchmarks have been proposed for CARLA and other simulators.
Before covering more specific mistakes made in CARLA benchmarks, we have to however get an elephant out of the room:

## NuScenes Planning
### You should not trust the L2 metric.

[NuScenes](https://arxiv.org/abs/1903.11027) is a well-known autonomous driving dataset that features various computer vision benchmarks.
Using a dataset like nuScenes to evaluate the planning output of an autonomous driving stack is called open-loop planning, and works by feeding the sensor data in the stack and comparing its prediction with the ground truth action from the human driver in the dataset. The reason this is called open-loop planning is that the sensor input of future frames will not depend on the prediction of the driving stack, and neither will the other cars react to its behavior. The appeal of open-loop planning is that is computationally much cheaper to evaluate than to evaluate a stack closed-loop (either in the real world or in simulation) and unlike simulators, it does not introduce a sim-to-real gap with respect to the sensor data.
Due to its appeal, it was investigated early on whether the open-loop L2 loss could act as a reliable performance indicator for closed-loop performance (what we actually care about).
Unfortunately, this turned out not to be the case as was shown in [Codevilla et al. 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Felipe_Codevilla_On_Offline_Evaluation_ECCV_2018_paper.pdf). The open-loop L2 error did not necessarily correlate with the closed-loop error. In other words, the open-loop L2 loss can be misleading.
The community focused its benchmarking efforts on closed-loop simulation as a result.
As researchers working on end-to-end autonomous driving solutions, we can confirm these results. In our experience, open-loop validation losses of end-to-end systems are not useful as indicators for closed-loop performance, and we therefore do not report them.

### The L2 metric became popular.
For the reason outlined before, the nuScenes paper itself did **not** propose a benchmark for planning (according to private communication with the authors, intentionally so).
However, the planning community grew a lot around 2022/2023 and researchers, seemingly unaware of these early results, started using the nuScenes dataset to benchmark planning using the L2 loss (or variants of this, called displacement errors) as a primary metric. This benchmark became known as nuScenes Planning. Due to the good reputation of nuScenes in the vision community (and one of the papers using the benchmark winning a prestigious vision award), this benchmark was widely adopted.

### NuScenes planning led to misleading results.
Some papers supplemented their nuScenes planning results with closed-loop CARLA simulation results. This made it quite apparent that the nuScenes planning methods were behind the state-of-the-art methods from CARLA (although readers would need to read that between the lines as only older papers were reported as CARLA baselines).
In an attempt to see whether the results we mentioned earlier also apply to nuScenes planning in particular, [Zhai et al. 2023](https://arxiv.org/abs/2305.10430) came up with a clever unit test (sort of). They made a driving stack called AD-MLP that has no access to vision whatsoever and simply extrapolates past movement in the dataset. This can work because of the open-loop nature of the dataset and the fact that driving logs are very smooth. To be explicit here, AD-MLP is a toy model whose only purpose is to expose the flaws of a metric or benchmarks (you can think of it as a constant baseline as used in other ML fields).
It turned out that AD-MLP works extremely well in nuScenes planning. Even dozens of papers later, it is the state of the art on nuScenes planning. 
Unfortunately, these results were mostly ignored in the nuScenes planning community initially, although [Li et al. 2024](https://arxiv.org/abs/2312.03031v2) later reproduced them in a more rigorous fashion.
Recent benchmarks like Bench2Drive or NAVSIM have adopted AD-MLP as a sanity check (if AD-MLP has bad performance then the benchmark passes the test).

### There are more flaws in the nuScenes planning benchmark.

Besides these fundamental flaws in the benchmark, the were numerous problems regarding the execution of benchmarking on nuScenes planning. [Weng et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf) describes some of the problems that surfaced within the benchmark, such as the metrics being incorrectly computed, defined differently across different papers and post-processing being done differently across methods. There are more problems than those mentioned in the paper. For example, the conditioning input command is computed based on the waypoint labels, hence leaking label information, and the size of the validation set is orders of magnitudes smaller than in other planning benchmarks (we are only referring to planning here, perception tasks have many labels per frame).

### We need to stop using nuScenes planning!

In our view, there is only one reasonable way to move forward with nuScenes planning and that is to stop using it as a benchmark.
If you are a reviewer and get a paper in 2025 that bases its claims solely on nuScenes planning results, please reject it.
If you are an author and your reviewer (2) asks for nuScenes planning results, please kindly decline (you could cite this text or the papers we mentioned as justification).

### NuScenes planning reinvigorated the search for better open-loop metrics.

One positive thing that came out of the nuScenes planning story was that it highlighted the large interest in cheap open-loop evaluation and started a search for better open loop metrics ([Weng et al. 2024](https://xinshuoweng.github.io/paradrive/assets/camera_ready.pdf), [Li et al.2024](https://arxiv.org/abs/2312.03031), [Dauner et al. 2024](https://arxiv.org/abs/2406.15349)). In particular, the discovery of the PDM-score metric, which is an open-loop metric that performs a pseudo simulation based on the world on rails assumption, being correlated to closed-loop performance led to a new type of benchmark (NAVSIM). Being an open-loop benchmark, NAVSIM still cannot measure important issues such as the compounding error problem, so it should be complemented with some closed-loop experiments, but right now is the recommended way to replace nuScenes planning (for labs that don't have the computational resources to run large scale closed-loop simulations).

### Using nuScenes for perception is fine.
It is important to note, that we do not claim that all claims made with nuScenes planning are wrong. What we claim is that we don't know whether these ideas work or not. Some of them probably do, others do not. We need to reevaluate these existing ideas on proper benchmarks in order to know. Also, our criticism is specific to nuScenes planning and does not apply to other nuScenes benchmarks for perception and prediction tasks.

## CARLA benchmarks
[CARLA](https://arxiv.org/abs/1711.03938) is an autonomous driving simulator based on the Unreal Engine that is able to simulate most types of sensor data, ground truth labels, safety critical scenarios, and background traffic. Its introduction in 2017 (and the fact that CARLA is open-source) enabled the community to benchmark entire autonomous driving stacks in closed-loop. Due to the community's continual commitment to update and improve the simulator, it has become the de facto standard for the proper evaluation of autonomous driving stacks.
CARLA itself is a simulator and does neither come with a benchmark nor a dataset to develop models.
The community has developed various benchmarks based on the CARLA simulator with increasing difficulty over time. Some noteworthy once include the [original CARLA benchmark](https://arxiv.org/abs/1711.03938), [NoCrash](https://arxiv.org/abs/1904.08980), [Town05 Short/Long](https://arxiv.org/abs/2104.09224), [LAV routes](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_From_All_Vehicles_CVPR_2022_paper.pdf), [Longest6](https://arxiv.org/abs/2205.15997), [CARLA leaderboard 1.0 test routes](https://leaderboard.carla.org/#leaderboard-10), [Bench2Drive](https://arxiv.org/abs/2406.03877), [CARLA leaderboard 2.0 validation routes](https://leaderboard.carla.org) and [CARLA leaderboard 2.0 test routes](https://leaderboard.carla.org) listed roughly in increasing order of difficulty (there are more benchmarks than listed here).

### Papers often get the setup details of CARLA benchmarks wrong.

These benchmarks have various setup conditions such as which CARLA towns methods are allowed to collect data from, which safety critical scenarios to evaluate with and what routes to drive along, the traffic density to be used, etc.
This situation can be confusing, and it is important to get these details right to ensure fair comparisons between methods.
Many of these benchmarks use similar metrics. A common mistake we see in the literature is authors copying numbers from one benchmark and comparing them with numbers of a different benchmark.
This is like comparing accuracies on CiFAR-10 to accuracies on MNIST, it is a methodological mistake, these comparisons are not meaningful.
Furthermore, we see many authors getting subtle details wrong in their evaluation, such as training on a validation town, evaluating without safety critical scenarios, forgetting to adjust the traffic density, and citing the wrong papers (in particular TransFuser is often misquoted with authors reporting the weaker number from the conference paper, but citing the stronger stack from the journal extension).
These mistakes are so widespread in the literature that we do not recommend copying numbers from other papers (even published ones) because there is a high chance that you will copy somebody else's mistake. Instead, it is often better to simply re-evaluate baselines on benchmarks as many of them are open-source, or at least double-check how the numbers you are copying came to be.
When evaluating on a benchmark, carefully check the conditions needed for a proper setup (and document if you deviate from them for some reason).

### Closed-loop (CARLA) benchmarks are sensitive to random chance.

Another problem with closed-loop benchmarks is that results are fundamentally sensitive to random chance (the seed). It is a common and necessary practice to repeat evaluations at least 3 times and average the result. In particular end-to-end methods furthermore have high training variance between training seeds, and it is sometimes necessary to evaluate multiple training seeds as well ([Prakash et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf), [Behl et al. 2020](https://arxiv.org/abs/2005.10091), [Chitta et al. 2023](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf)). When doing experiments this is one of the first things to check. For example, on the LAV benchmark we found it necessary to average the result of 3 training seeds each evaluated 3 times as the results were otherwise dominated by random chance.
We have seen a resurgence of single seed results published in conferences in 2024. If you are a reviewer please remind the authors that evaluating at least 3 seeds is necessary.

The large amount of different benchmarks can lead to confusion and has introduced two subtle issues:

### Some papers propose new benchmarks with misleading names.

First, some authors have used the confusion to propose easier benchmarks, by giving them the same name as an existing harder benchmark, which makes their methods look better than they are (this happened multiple times for the CARLA leaderboard due to its good reputation and high difficulty). This doesn't happen often, but does, so as a reviewer one needs to stay vigilant here.

### Some papers claim state of the art even when their performance is weak.

The second issue is that it is hard to stay on track with what the state of the art is on the various benchmarks. This has led to some authors making incorrect claims about their method being state of the art even though their numbers clearly show that their method is 3 years behind the state of the art. This is usually masked by only reporting older baselines. We have seen multiple cases of papers published at a top-tier conference making such incorrect claims in 2024.
If you are a reviewer, and you are unsure what the state of the art on a particular benchmark is, don't hesitate to contact another expert (for example the author of the benchmark).
It is, of course, fine to claim state of the art if there is concurrent work on arXiv that has better numbers from the last ~6 months (1 conference review cycle). If a method gets outperformed by a published paper from 3 years ago then it is not state of the art.

### Some papers omit performance critical details.

The CARLA leaderboard test routes (version 1.0 and 2.0) are a benchmark whose test routes are secret and evaluation is done by submitting code to a third-party server (operated by the CARLA team). Submitting teams can then view and publicize their results on a website, [https://leaderboard.carla.org/leaderboard/](https://leaderboard.carla.org/leaderboard/). Operating such an independent evaluation system is expensive but ensures that the benchmarking is fair, and the claimed results have actually been achieved. This idea has mostly worked well and led to fast progress over the last 5 years.
A particular issue arose on the CARLA leaderboard 1.0, however. What this system does not ensure is that authors accurately report how they achieved their results. It has been documented in [Jaeger et al. 2023, Appendix C.6](https://arxiv.org/abs/2306.07957), that the top 3 methods on the CARLA leaderboard are not reproducible. The models and code of these methods are open source, yet the published models produce much weaker results than what was reported in the paper (and displayed on the leaderboard website). Since the published code implements the ideas presented in the papers, it is questionable whether the paper accurately describes the model that was used to achieve the state-of-the-art result. Unfortunately, there is no real way for reviewers to catch such problems during the review process, as it only becomes apparent once serious reproduction efforts are underway. We encourage reviewers to demand the release of code alongside the paper as it ensures that these problems can be corrected afterward (to our knowledge, there is no scientific argument for withholding the code of a publication). The current top method on the CARLA leaderboard 1.0 did not publish its code, so the only thing we can say is that it has never been successfully reproduced.
Progress has stopped on the CARLA leaderboard 1.0 for the last 2 years, presumably due to this situation.
With the community moving to the CARLA leaderboard 2.0 we hope this situation does not reoccur.
The recent [NAVSIM leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) is trying to address this issue by requiring the release of code and models, and will remove submissions that are not reproducible.



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
In our view, it is important to realize that **the autonomous driving effort is a competition against reality** first and foremost**. We need public research that adheres to rigorous scientific practices in order to win this competition!

13.12.2024

PS: If you are aware of additional problems in benchmarking autonomous driving that are not yet covered here, please get in touch.

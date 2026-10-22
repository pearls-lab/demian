---
layout: project_demian
title: "DeMiAn - How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning"
description: Dense Multi-Aspect Annotation for scaling robot policy learning.
img: assets/img/pearls-placeholder.png
importance: 1
category: World Modeling Embodied Agents
venue: NeurIPS
year: 2026
team: UCSD · NVIDIA
authors: "<a href='https://bosung.github.io' target='_blank' rel='noopener noreferrer'><span class='me'>Bosung Kim</span></a><sup class='co'>★</sup>, <span class='me'>Ruiyi Wang</span><sup class='co'>★</sup>, <a href='https://www.cs.toronto.edu/~davidj/' target='_blank' rel='noopener noreferrer'>David Acuna</a>, <a href='https://jaehunjung.com' target='_blank' rel='noopener noreferrer'>Jaehun Jung</a>, <a href='https://alextrevithick.com' target='_blank' rel='noopener noreferrer'>Alexander Trevithick</a>, Brandon Cui, <a href='https://yejinc.github.io' target='_blank' rel='noopener noreferrer'>Yejin Choi</a>, <a href='https://prithvirajva.com' target='_blank' rel='noopener noreferrer'>Prithviraj Ammanabrolu</a><br><span class='equal-contrib'>★ Equal contribution</span>"
affiliations: "UCSD PEARLS Lab · NVIDIA Research — LACR & AMRI"
paper_url: "https://arxiv.org/pdf/2605.17077"
code_url: "https://github.com/pearls-lab/demian"
---

<p class="pullquote">Looking for the next big scaling lever for robotics? Language may be the fastest one. 🚀</p>

<figure>
  <video src="{{ '/assets/video/demian/demian_main_video.mp4' | relative_url }}" controls muted playsinline style="width:100%;border-radius:var(--radius);box-shadow:0 1px 2px rgba(0,0,0,0.04),0 8px 24px rgba(0,0,0,0.06);border:1px solid var(--line);background:#111;"></video>
</figure>

Teaching robots new skills usually means recording thousands of hours of expensive human-driven robot demos. Each demo is paired with a tiny one-line label like *"open the drawer"* — which throws away most of the useful information already present in the video. We introduce **DeMiAn** (Dense Multi-Aspect Annotation), which keeps the same videos and just rewrites the labels: a vision-language model produces richer descriptions per clip, and a small *instructor* model learns to hand the robot the right kind of description, in real time. With DeMiAn, we achieve **+5 points** success rate on RoboCasa365 and **~62% compute savings** on MolmoSpaces — with **zero new demonstrations collected**.

### Core contributions

- **A new scaling axis for robot learning.** DeMiAn opens one more scaling axis — *language density* — that simply stacks on top of video demonstrations. Applied to 1M+ robot clips and 50K human-egocentric clips for an average **+5** point lift (up to **+37** on a single task), with no additional demonstrations.
- **A smart instructor and real-time deployment.** No single style of description works best everywhere, so we train a small *instructor* model that watches the initial scene and writes the tailored description for each task on the fly. It runs in parallel with the robot, so the robot never waits.
- **Better generalization to harder, unseen tasks.** Our policies generalize better to longer multi-step (composite) tasks and to scenes and objects never seen during training; **+9** points on unseen composite tasks.
- **A more compute-efficient lever for scaling.** Re-labeling existing videos improves the compute–performance frontier of robot learning, even after charging annotation compute into the budget. At 1M-clip scale, DeMiAn matches the no-annotation baseline with **~62% less compute** (~1.3 × 10²⁰ FLOPs saved).

<div class="hl-badge"><span class="num">01</span> Highlight</div>

## <span class="mark">A new scaling axis for robotics</span>

<figure>
  <img src="{{ '/assets/img/demian/1.png' | relative_url }}" alt="DeMiAn overview" />
  <div class="caption">DeMiAn shows that richer language can unlock more supervision from the same videos, making annotation density a cheaper and more compute-efficient alternative to collecting more robot data. We scale this idea by re-annotating over 1M robot manipulation videos and 50K egocentric human videos, showing that dense language helps both video-based world-action model and robot-policy training.</div>
</figure>

<style>
.video-carousel {
  margin: 32px 0 40px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
}
.vc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(118,185,0,0.05);
}
.vc-title {
  font-family: 'Inter Tight','Inter',sans-serif;
  font-weight: 600;
  font-size: 15.5px;
  color: var(--ink);
}
.vc-counter {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.06em;
}
.vc-stage {
  position: relative;
  background: #111;
  display: flex;
  align-items: center;
  justify-content: center;
}
.vc-video {
  width: 100%;
  max-height: 500px;
  display: block;
  border-radius: 0 !important;
  box-shadow: none !important;
  border: none !important;
  background: #111;
}
.vc-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  z-index: 10;
  background: rgba(0,0,0,0.52);
  backdrop-filter: blur(4px);
  color: #fff;
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 50%;
  width: 46px;
  height: 46px;
  font-size: 28px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background .15s ease, transform .15s ease;
  padding: 0 0 2px;
  line-height: 1;
  user-select: none;
}
.vc-btn:hover {
  background: rgba(118,185,0,0.80);
  border-color: var(--accent);
  transform: translateY(-50%) scale(1.10);
}
.vc-prev { left: 14px; }
.vc-next { right: 14px; }
.vc-caption-area {
  padding: 14px 20px 12px;
  border-top: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  background: rgba(118,185,0,0.04);
  font-size: 15px;
  color: var(--ink-soft);
  line-height: 1.62;
}
.vc-dots {
  display: flex;
  justify-content: center;
  gap: 7px;
  padding: 10px 20px 14px;
}
.vc-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--line);
  border: none;
  cursor: pointer;
  padding: 0;
  transition: background .15s ease, transform .15s ease;
}
.vc-dot.active {
  background: var(--accent);
  transform: scale(1.3);
}
</style>


<div class="video-carousel" id="vc">
  <div class="vc-header">
    <span class="vc-title" id="vc-title"></span>
    <span class="vc-counter" id="vc-counter">1 / 10</span>
  </div>
  <div class="vc-stage">
    <button class="vc-btn vc-prev" id="vc-prev" aria-label="Previous example">&#8249;</button>
    <video class="vc-video" id="vc-video" controls muted playsinline></video>
    <button class="vc-btn vc-next" id="vc-next" aria-label="Next example">&#8250;</button>
  </div>
  <div class="vc-caption-area" id="vc-caption"></div>
  <div class="vc-dots" id="vc-dots"></div>
</div>

<script>
(function () {
  var BASE = "{{ '/assets/video/demian/' | relative_url }}";
  var SLIDES = [
    {
      file: "example_01_seg001_seal_dough.mp4",
      title: "Example 1 — Seal Dough",
      caption: "<strong>Sparse label:</strong> \"seal the dough.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> adds contact-changing motion words — press and fold the dough edges inward, grip firmly, and apply downward pressure along the seam."
    },
    {
      file: "example_02_seg000_adjust_tassel.mp4",
      title: "Example 2 — Adjust Tassel",
      caption: "<strong>Sparse label:</strong> \"adjust the tassel.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> describes directional hand movement — reach toward the tassel, pinch lightly, and reposition by pulling upward and to the side."
    },
    {
      file: "example_03_seg018_tie_string.mp4",
      title: "Example 3 — Tie String",
      caption: "<strong>Sparse label:</strong> \"tie the string.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> captures the multi-step sequence — cross the two ends, pull snug, loop one end over, and tighten into a secure knot."
    },
    {
      file: "example_04_seg001_sort_beans.mp4",
      title: "Example 4 — Sort Beans",
      caption: "<strong>Sparse label:</strong> \"sort the beans.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> specifies properties used for sorting — pick beans by color and size, place into distinct containers to the left and right."
    },
    {
      file: "example_05_seg001_fold_cloth.mp4",
      title: "Example 5 — Fold Cloth",
      caption: "<strong>Sparse label:</strong> \"fold the cloth.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> articulates spatial alignment — bring the far edge to meet the near edge, align corners, and press the fold flat with an open palm."
    },
    {
      file: "example_06_seg031_prepare.mp4",
      title: "Example 6 — Prepare",
      caption: "<strong>Sparse label:</strong> \"prepare.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> grounds the ambiguous verb — position the workpiece, orient it toward the tool, and secure it before the main action begins."
    },
    {
      file: "example_07_seg000_open_bag.mp4",
      title: "Example 7 — Open Bag",
      caption: "<strong>Sparse label:</strong> \"open the bag.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> describes grip and motion — grasp the two sides of the bag's opening and pull both handles outward simultaneously to widen the gap."
    },
    {
      file: "example_08_seg005_open_bottle.mp4",
      title: "Example 8 — Open Bottle",
      caption: "<strong>Sparse label:</strong> \"open the bottle.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> captures rotational mechanics — stabilize the bottle body, grip the cap, and rotate counter-clockwise until the seal releases."
    },
    {
      file: "example_09_seg002_open_box.mp4",
      title: "Example 9 — Open Box",
      caption: "<strong>Sparse label:</strong> \"open the box.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> articulates panel interaction — locate the lid seam, slide fingers under the lip, and lift the top panel upward and away from the body."
    },
    {
      file: "example_10_seg001_lace_shoe.mp4",
      title: "Example 10 — Lace Shoe",
      caption: "<strong>Sparse label:</strong> \"lace the shoe.\" &nbsp;|&nbsp; <strong>DeMiAn:</strong> breaks down the dexterous sequence — thread the lace through each eyelet alternately, cross at the top, and pull both ends to equalize tension."
    }
  ];

  var idx = 0;
  var video   = document.getElementById('vc-video');
  var titleEl = document.getElementById('vc-title');
  var counter = document.getElementById('vc-counter');
  var caption = document.getElementById('vc-caption');
  var dotsEl  = document.getElementById('vc-dots');
  var prevBtn = document.getElementById('vc-prev');
  var nextBtn = document.getElementById('vc-next');

  SLIDES.forEach(function (_, i) {
    var d = document.createElement('button');
    d.className = 'vc-dot' + (i === 0 ? ' active' : '');
    d.setAttribute('aria-label', 'Go to example ' + (i + 1));
    d.addEventListener('click', function () { go(i); });
    dotsEl.appendChild(d);
  });

  function go(n) {
    video.pause();
    idx = (n + SLIDES.length) % SLIDES.length;
    var s = SLIDES[idx];
    video.src = BASE + s.file;
    video.load();
    titleEl.textContent = s.title;
    counter.textContent = (idx + 1) + ' / ' + SLIDES.length;
    caption.innerHTML = s.caption;
    dotsEl.querySelectorAll('.vc-dot').forEach(function (d, i) {
      d.classList.toggle('active', i === idx);
    });
  }

  prevBtn.addEventListener('click', function () { go(idx - 1); });
  nextBtn.addEventListener('click', function () { go(idx + 1); });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft')  go(idx - 1);
    if (e.key === 'ArrowRight') go(idx + 1);
  });

  go(0);
})();
</script>

### Results on RoboCasa365 and MolmoSpaces

<figure>
  <img src="{{ '/assets/img/demian/2.png' | relative_url }}" alt="Per-task results on RoboCasa365 and MolmoSpaces" />
  <div class="caption">On RoboCasa365 with DeMiAn-VLA, <em>"Physical Motion"</em> annotation delivers large per-task lifts over the task-only baseline: <em>SlideDishwasherRack</em> (<strong>+37</strong>), <em>CoffeeSetupMug</em> (<strong>+31</strong>), <em>CloseToasterOvenDoor</em> (<strong>+14</strong>). On MolmoSpaces, <em>"Scene Composition"</em> adds <strong>+13</strong> on <em>Pick</em> tasks; <em>"Reasoning"</em> adds <strong>+8</strong> on <em>NextTo</em>.</div>
</figure>

<figure>
  <video src="{{ '/assets/video/demian/comparison_baseline_vs_demian.mp4' | relative_url }}" controls muted playsinline style="width:100%;border-radius:var(--radius);box-shadow:0 1px 2px rgba(0,0,0,0.04),0 8px 24px rgba(0,0,0,0.06);border:1px solid var(--line);background:#111;"></video>
  <div class="caption"><strong>What does the robot learn from dense language?</strong> DeMiAn provides a rich signal for learning the grounding of the scene — new interacting-object words (<em>"oven door"</em>), motion verbs not present in the task label (<em>"push"</em>, <em>"retract"</em>), and directional adverbs (<em>"inward"</em>, <em>"away"</em>).</div>
</figure>

We found that no fixed annotation rule reaches the per-task oracle. The optimal aspect varies with the structural demands of each task family — contact-changing motion vs. open-vocabulary grounding. Closing this gap requires producing the *right* instruction for each task at deployment.

<div class="hl-badge"><span class="num">02</span> Highlight</div>

## <span class="mark">A smart instructor and real-time deployment</span>

<figure style="max-width: 560px; margin: 28px auto;">
  <img src="{{ '/assets/img/demian/4.png' | relative_url }}" alt="Instructor architecture" />
</figure>

<figure>
  <img src="{{ '/assets/img/demian/5.png' | relative_url }}" alt="Instructor results" />
</figure>

We close the oracle gap with a small learned **instructor** model: given the same observation the action policy receives, it generates a scene-grounded annotation.

**Instructor closes the oracle gap.** The instructor lifts average SR from **44% → 49%** on RoboCasa and MolmoSpaces, within 2–3 points of the per-task oracle (51–52%). Against a random per-episode aspect baseline (46.6%) with the same action policy checkpoint, the instructor adds **+3.8** points — the gain comes from learned per-task selection, not heuristic routing.

**Async deployment, zero policy delay.** At test time the instructor decodes asynchronously while actions emit at the action server's native 8-step open-loop cadence. Until the instruction is ready, the policy executes on the task description alone — equivalent to the no-annotation baseline. The moment the instruction completes, it is spliced into the reasoning field at the next chunk boundary, and subsequent action chunks condition on it.

<div class="hl-badge"><span class="num">03</span> Highlight</div>

## <span class="mark">Generalizing to harder, unseen tasks</span>

<figure>
  <img src="{{ '/assets/img/demian/6.png' | relative_url }}" alt="Composite-task evaluation setup" />
</figure>

<figure>
  <img src="{{ '/assets/img/demian/7.png' | relative_url }}" alt="Composite task results (-fix)" />
</figure>

<figure>
  <img src="{{ '/assets/img/demian/8.png' | relative_url }}" alt="Composite task results (-dynamic)" />
</figure>

Our policy models are trained on RoboCasa365 atomic tasks. We test generalization to **new instruction formats** on composite tasks under two prompting strategies: *-fix* feeds a single composite task description for the whole episode; *-dynamic* runs a subgoal-driven state machine that swaps the prompt to the in-distribution atomic instruction for the current phase once a lenient in-simulator trigger fires.

Within the OOD *-fix* format, DeMiAn-VLA improves over the task-only baseline by **+2 full-task points** (15% vs. 13%). With GT atomic prompts under *-dynamic*, DeMiAn-VLA-GT is the strongest configuration overall (**22% vs. 19%** baseline-GT) — the dense-annotation policy benefits *more* from subgoal-decomposed prompts than the baseline does.

<div class="hl-badge"><span class="num">04</span> Highlight</div>

## <span class="mark">A compute-efficient lever for scaling</span>

<figure>
  <img src="{{ '/assets/img/demian/9.png' | relative_url }}" alt="DeMiAn scaling curves" />
  <div class="caption"><strong>DeMiAn under scaling.</strong> (A) DeMiAn-WAM mid-training on EgoVerse 50K with dense annotations, evaluated on downstream RoboCasa 365. (B) DeMiAn-VLA post-training on the 1M-scale MolmoBot corpus, evaluated by total success across four MolmoSpaces benchmarks. The x-axis includes annotation-generation compute in both panels.</div>
</figure>

**Scaling under matched compute.** We mid-train Cosmos-Predict 2.5 on 50K EgoVerse clips with and without dense annotations, and post-train DeMiAn-WAM on RoboCasa365 under both conditions. After the small upfront cost of generating annotations, the annotated WAM reaches higher downstream RoboCasa SR than the baseline at nearby compute budgets. On MolmoBot post-training with 1M trajectories, annotation-conditioned policies reach stronger MolmoSpaces SR earlier in training and obtain higher peak performance.

At fixed compute, dense annotation accelerates both WAM mid-training and VLA post-training — even after charging caption-generation compute against the budget. Adding language to existing demonstrations is a practical and compute-efficient lever for robot policy learning.

// site.ts — shared metadata and page data for Yiheng Li's personal website.
//
// Values were ported from the design canvas `Ink Darkroom.dc.html` and then
// given a second column: everything a reader sees is an `L` — {en, zh} — and
// `.scene[data-lang]` decides which half shows. Seals use TRADITIONAL glyphs
// only (印章 convention); UI copy uses simplified.
//
// Proper nouns keep their own form in both languages when there is no settled
// Chinese one (GitHub, PyTorch, MONAI, ISMRM); institutions that have one are
// translated. See the note at the top of i18n.ts.
//
// A link whose destination is not filled in (src/config/me.ts) is DROPPED here
// rather than rendered dead — `linked()` below. src/config/blanks.ts turns the
// same emptiness into a checklist in the terminal and, in dev, on the page.

import { me } from "./me";
import { t, both, ui, type L } from "./i18n";

// The site has no name of its own — it is a person. Every page title is
// "<what the page is> · <who>", and the home page is just the name.
export const site = {
  /** Page <title>: English by default, swapped by the toggle. */
  name: t("Yiheng Li", "李易恒"),
  description:
    "Yiheng Li - foundation models for 3D medical imaging, notes, and photographs.",
  url: "https://terryli710.github.io",
  author: { zh: "李易恒", en: "Yiheng Li" },
};

export const nav = [
  { href: "/", label: ui.nav.home, match: ["/"], exact: true },
  { href: "/writing/", label: ui.nav.writing, match: ["/writing/", "/posts/"] },
  { href: "/photographs/", label: ui.nav.photographs, match: ["/photographs/"] },
  { href: "/profile/", label: ui.nav.profile, match: ["/profile/"] },
];

// 印章 — traditional glyphs, do not simplify.
export const seals = {
  footer: "虛室生白", // 引首章 《庄子》
  articleEnd: "大象無形", // 钤印 《老子》
};

/** Drop anything whose destination is still blank. */
const linked = <T extends { href: string }>(rows: T[]): T[] =>
  rows.filter((row) => row.href.trim().length > 0);

const mailto = me.email.trim() ? `mailto:${me.email.trim()}` : "";

/** The bare id out of a profile URL — "0000-0003-0135-8824", "terryli710". */
const tail = (url: string): string => url.replace(/\/+$/, "").split("/").pop() ?? "";

// ── home · Elsewhere ──
// `icon` names a glyph in src/components/Icon.astro — the same set the profile
// buttons draw from, so a destination looks the same wherever it appears.
export const links = linked([
  { k: both("GITHUB"), v: both(tail(me.github)), href: me.github, icon: "github" },
  { k: t("LINKEDIN", "领英"), v: t("Yiheng Li", "李易恒"), href: me.linkedin, icon: "linkedin" },
  { k: t("STANFORD", "斯坦福"), v: t("Profile", "主页"), href: me.stanford, icon: "stanford" },
  { k: t("SCHOLAR", "谷歌学术"), v: t("Publications", "论文"), href: me.scholar, icon: "scholar" },
  { k: both("ORCID"), v: both(tail(me.orcid)), href: me.orcid, icon: "orcid" },
  { k: t("EMAIL", "邮箱"), v: both(me.email), href: mailto, icon: "email" },
]);

/**
 * The same links, said once in a form machines read: schema.org `Person`, which
 * is how a search engine learns that this page, that ORCID record and that
 * Scholar profile are one person. Emitted as JSON-LD by BaseLayout, alongside
 * `rel="me"` on the profile links themselves.
 *
 * Anything still blank in me.ts drops out here too — `sameAs` never carries an
 * empty string.
 */
export const person = {
  name: { en: "Yiheng Li", zh: "李易恒" },
  url: "https://terryli710.github.io",
  email: me.email.trim(),
  jobTitle: ui.profile.metaRole.en.split(" · ")[0],
  affiliation: "Stanford University School of Medicine",
  sameAs: [me.github, me.linkedin, me.scholar, me.stanford, me.orcid].filter((u) => u.trim()),
};

// The OpenRefinery talk (2026-09-09). It covers the chest X-ray benchmark, the
// lung VAE and the COVID-19 triage work, so those three link to it as well.
const talk = "https://www.youtube.com/watch?v=Kz_LV64xKjE";

// ── home · Research & projects ──
// A row links to its own `href` when it has one, otherwise to /profile/.
export const research: { year: string; title: L; venue: L; href?: string }[] = [
  {
    year: "2026",
    title: t("AI for Biomedicine", "生物医学中的人工智能"),
    venue: t("OPENREFINERY (TALK)", "OPENREFINERY（讲座）"),
    href: talk,
  },
  {
    year: "2025",
    title: t(
      "Supervised pre-training on intermediate phenotypes improves glaucoma detection",
      "以中间表型做有监督预训练，提升青光眼检测",
    ),
    venue: t("MEDRXIV (PREPRINT)", "MEDRXIV（预印本）"),
  },
  {
    year: "2025",
    title: t(
      "Benchmarking chest X-ray diagnosis models across multinational datasets",
      "跨国数据集上的胸片诊断模型基准评测",
    ),
    venue: t("ARXIV (PREPRINT)", "ARXIV（预印本）"),
  },
  {
    year: "2024",
    title: t(
      "A 3D lung lesion variational autoencoder",
      "三维肺部病灶变分自编码器",
    ),
    venue: both("CELL REPORTS METHODS"),
  },
  {
    year: "2023",
    title: t(
      "Multi-contrast MRI registration with a realistic flow field",
      "带真实形变场的多对比度 MRI 配准",
    ),
    venue: both("ISMRM 2023"),
  },
  {
    year: "2021",
    title: t(
      "AI-based analysis of CT images for rapid triage of COVID-19 patients",
      "基于人工智能的 CT 影像分析，用于 COVID-19 患者快速分诊",
    ),
    venue: both("NPJ DIGITAL MEDICINE"),
  },
];

// ── profile ──
export const focus: L[] = [
  t("Foundation models", "基础模型"),
  t("3D medical imaging", "三维医学影像"),
  t("Self-supervised learning", "自监督学习"),
  t("Model evaluation", "模型评估"),
  t("Multimodal fusion", "多模态融合"),
  t("Multi-GPU training", "多 GPU 训练"),
];

// `links` is the right-hand rail, the same as a Selected-work row's venues:
// where the role lives online and what came out of it.
// `logo` names a file in src/assets/logos/. Each is the organisation's own
// vector mark, recoloured to `currentColor` (its white details carry class
// `k`) so it follows the page ink in both modes rather than its brand colour.
export const cv = [
  {
    when: t("09/2023—Current", "09/2023—至今"),
    title: t("Research Scientist, AI", "人工智能研究科学家"),
    org: t(
      "STANFORD MEDICINE · GEVAERT LAB, COMPUTATIONAL MEDICINE",
      "斯坦福医学院 · 计算医学部 GEVAERT 实验室",
    ),
    logo: "stanford-medicine",
    links: [
      { label: t("GEVAERT LAB", "GEVAERT 实验室"), href: "https://med.stanford.edu/gevaertlab.html" },
      { label: t("CXR BENCHMARK", "胸片基准评测"), href: "https://arxiv.org/abs/2505.16027" },
      { label: t("LUNG VAE", "肺部病灶 VAE"), href: "https://doi.org/10.1016/j.crmeth.2024.100695" },
      { label: t("TALK (2026)", "讲座（2026）"), href: talk },
    ],
    body: t(
      "A chest-CT foundation model for lung tumor analysis (first author, manuscript in preparation). Curated 16 public and internal CT collections into one re-runnable pipeline, pretrained ViT-L-class 3D encoders on 4 GPUs, and benchmarked 10 encoders on 48 downstream tasks in 5 families, under patient-level splits, repeated seeds, and Wilcoxon tests with multiple-testing correction. Also: a 3D β-VAE for lung-tumor morphology, a multinational chest X-ray benchmark (co-first author), and supervised pre-training for glaucoma detection.",
      "一个面向肺部肿瘤分析的胸部 CT 基础模型（第一作者，论文准备中）。把 16 个公开与内部 CT 数据集整理进一条可重复运行的流水线，在 4 张 GPU 上预训练 ViT-L 级别的三维编码器，并在 5 类共 48 个下游任务上评测 10 个编码器，采用患者级划分、多次随机种子，以及带多重检验校正的 Wilcoxon 检验。另有：学习肺部肿瘤形态的三维 β-VAE、跨国胸片模型基准评测（共同第一作者），以及用于青光眼检测的有监督预训练。",
    ),
  },
  {
    when: both("08/2021—08/2023"),
    title: t("Deep Learning Research Scientist", "深度学习研究科学家"),
    org: t("SUBTLE MEDICAL · MENLO PARK, CA", "SUBTLE MEDICAL · 加州门洛帕克"),
    logo: "subtle-medical",
    links: [
      { label: both("SUBTLE MEDICAL"), href: "https://subtlemedical.com/" },
      { label: both("ISMRM 2023"), href: "https://archive.ismrm.org/2023/4343.html" },
      { label: both("ISMRM 2022"), href: "https://archive.ismrm.org/2022/1880.html" },
      {
        label: both("ASNR 2022"),
        href: "https://www.asnr.org/wp-content/uploads/2024/09/ASNR22-Proceedings_09.17.24.pdf#page=260",
      },
    ],
    body: t(
      "Registration, registration QC, and self-supervised keypoint detection for 3D MRI. Four accepted meeting abstracts across ISMRM, ASNR and RSNA.",
      "三维 MRI 的配准、配准质量控制与自监督关键点检测。四篇会议摘要分别被 ISMRM、ASNR 与 RSNA 接收。",
    ),
  },
  // Three labs, three entries: the dates differ. Full names throughout — a
  // surname alone reads as shorthand between insiders, and this page is not
  // written for them.
  {
    when: both("09/2020—06/2021"),
    title: t("Research Assistant", "助理研究员"),
    org: t("STANFORD AIMI · CHAUDHARI LAB", "斯坦福 AIMI 中心 · CHAUDHARI 实验室"),
    logo: "stanford-university",
    links: [
      { label: t("STANFORD AIMI", "斯坦福 AIMI"), href: "https://aimi.stanford.edu/" },
      { label: t("DR. CHAUDHARI", "CHAUDHARI 教授"), href: "https://profiles.stanford.edu/akshay-chaudhari" },
    ],
    body: t(
      "With Dr. Akshay Chaudhari: designed, trained and evaluated a U-Net pipeline that segments vertebrae in sagittal CT, for the Opportunistic CT initiative.",
      "跟随 Akshay Chaudhari 教授：为 Opportunistic CT 项目设计、训练并评估了一条在矢状位 CT 上分割椎体的 U-Net 流水线。",
    ),
  },
  {
    when: both("06/2020—06/2021"),
    title: t("Research Assistant", "助理研究员"),
    org: t(
      "STANFORD UNIVERSITY · GEVAERT LAB, BIOMEDICAL INFORMATICS",
      "斯坦福大学 · 生物医学信息学 GEVAERT 实验室",
    ),
    logo: "stanford-university",
    links: [
      { label: t("GEVAERT LAB", "GEVAERT 实验室"), href: "https://med.stanford.edu/gevaertlab.html" },
    ],
    body: t(
      "With Dr. Olivier Gevaert: multi-modal pre-training (SimCLR, BYOL, DINO) across pathology, CT and EHR, to predict outcomes after PD-L1 treatment.",
      "跟随 Olivier Gevaert 教授：跨病理、CT 与电子病历做多模态预训练（SimCLR、BYOL、DINO），用来预测 PD-L1 治疗后的结局。",
    ),
  },
  {
    when: both("01/2020—06/2021"),
    title: t("Research Assistant", "助理研究员"),
    org: t(
      "STANFORD UNIVERSITY · CAMARILLO LAB, BIOENGINEERING",
      "斯坦福大学 · 生物工程系 CAMARILLO 实验室",
    ),
    logo: "stanford-university",
    links: [
      { label: t("DR. CAMARILLO", "CAMARILLO 教授"), href: "https://profiles.stanford.edu/david-camarillo" },
      { label: t("SUBTYPING (JSHS 2023)", "亚型划分（JSHS 2023）"), href: "https://doi.org/10.1016/j.jshs.2023.03.003" },
    ],
    body: t(
      "With Dr. David Camarillo: head-impact biomechanics and brain-strain modelling with machine learning, which became five co-authored papers.",
      "跟随 David Camarillo 教授：用机器学习研究头部撞击生物力学与脑应变建模，最终有五篇合著论文。",
    ),
  },
];

/** One destination behind a Selected-work row. */
export interface WorkVenue {
  label: L;
  /** Blank when there is nothing public to link to — rendered as plain text. */
  href: string;
}

/**
 * A figure from the paper itself, shown as a plate strip under the note.
 *
 * These are lifted from the open-access record — every one is CC BY or
 * CC BY-NC-ND on a paper Yiheng authored — and the file stored under
 * public/img/work/ is the publisher's own, unmodified. The night-mode image treatment
 * is CSS (`--imgf`) and the framing is `object-position`, so nothing here is a
 * derivative work. `credit` carries the attribution the licence asks for.
 */
export interface WorkFigure {
  /** File under public/img/work/. */
  src: string;
  alt: L;
  credit: L;
  /** Which part of the figure survives the crop. CSS `object-position`. */
  pos?: string;
}

export interface Work {
  year: string;
  title: L;
  note: L;
  stack: L;
  /** Usually one; an entry covering several papers links each separately. */
  venues: WorkVenue[];
  /** Only where the paper is open access — see WorkFigure. */
  figure?: WorkFigure;
}

export const work: Work[] = [
  {
    year: "2023—2024",
    title: t("A 3D lung lesion variational autoencoder", "三维肺部病灶变分自编码器"),
    note: t(
      "A 3D β-VAE that learns lung-tumour morphology from unlabelled CT — no manual labels — reconstructing nodule volumes at SSIM 0.774 and PSNR 26.1 while compressing them into a compact latent code. Individual latent dimensions turn out to track clinical characteristics such as nodule size, so moving along them synthesises plausible lesions; the same embeddings, transferred to an independent radiogenomic cohort, predict pathological nodal stage and KRAS mutation status on par with fully supervised models.",
      "一个三维 β-VAE，不用人工标注，直接从无标签 CT 中学习肺部肿瘤形态：重建结节体数据的 SSIM 为 0.774、PSNR 为 26.1，同时把它压缩成一段紧凑的隐编码。隐空间的单个维度与结节大小等临床特征高度相关，沿着这些方向移动即可合成看起来合理的新病灶；把同一套嵌入迁移到独立的影像基因组队列上，对病理淋巴结分期与 KRAS 突变状态的预测可与全监督模型持平。",
    ),
    stack: t("PYTORCH · β-VAE · RADIOGENOMICS", "PYTORCH · β-VAE · 影像基因组学"),
    figure: {
      src: "lung-vae.jpg",
      alt: t(
        "Grids of lung lesion patches synthesised by the β-VAE, shrunk and enlarged along the size vector.",
        "由 β-VAE 合成的肺部病灶图块阵列，沿尺寸向量缩小与放大。",
      ),
      credit: both("FIG. 2 · CELL REPORTS METHODS 2024 · CC BY-NC-ND"),
      pos: "50% 18%",
    },
    venues: [
      { label: both("CELL REPORTS METHODS"), href: "https://doi.org/10.1016/j.crmeth.2024.100695" },
      { label: t("TALK (2026)", "讲座（2026）"), href: talk },
    ],
  },
  {
    year: "2024—2025",
    title: t(
      "Glaucoma detection by supervised pre-training on intermediate phenotypes",
      "以中间表型做有监督预训练的青光眼检测",
    ),
    note: t(
      "Pre-training on a clinically meaningful intermediate marker — the vertical cup-to-disc ratio — rather than on ImageNet or a self-supervised proxy. A multi-task setup learns diagnosis and VCDR regression together on AIROGS, then transfers to six independent cohorts (DRISHTI, G1020, ORIGA, PAPILA, REFUGE1, ACRIMA). It beats out-of-domain and self-supervised pre-training on every backbone tried: ResNet-18, DINOv2 and RETFound.",
      "预训练的目标不是 ImageNet，也不是某个自监督代理任务，而是一个临床上真正有意义的中间指标——垂直杯盘比（VCDR）。模型在 AIROGS 上以多任务方式同时学习诊断分类与 VCDR 回归，再迁移到六个独立队列（DRISHTI、G1020、ORIGA、PAPILA、REFUGE1、ACRIMA）。在试过的每一个骨干网络上——ResNet-18、DINOv2 与 RETFound——它都优于域外预训练与自监督预训练。",
    ),
    stack: t("PYTORCH · MULTI-TASK · RETFOUND / DINOV2", "PYTORCH · 多任务 · RETFOUND / DINOV2"),
    venues: [
      {
        label: t("MEDRXIV (PREPRINT)", "MEDRXIV（预印本）"),
        href: "https://doi.org/10.1101/2025.04.22.25326210",
      },
    ],
  },
  {
    year: "2024—2025",
    title: t(
      "Benchmarking chest X-ray diagnosis models across multinational datasets",
      "跨国数据集上的胸片诊断模型基准评测",
    ),
    note: t(
      "Do vision–language foundation models actually generalise better than a plain CNN? Five of them (CheXzero, BioViL-T, MAVL, MedKLIP, PsPG) and three CNNs (DenseNet, ResNet, X-Raydar) put through 37 standardised classification tasks over six public datasets from the USA, Spain, India and Vietnam, plus three previously unreleased hospital datasets from China. Co-first author; submitted to The Lancet Digital Health.",
      "视觉—语言基础模型是否真的比普通 CNN 泛化得更好？把五个基础模型（CheXzero、BioViL-T、MAVL、MedKLIP、PsPG）与三个 CNN（DenseNet、ResNet、X-Raydar）放在 37 项标准化分类任务上评测，数据来自美国、西班牙、印度与越南的六个公开数据集，外加三个此前未公开的中国医院数据集。共同第一作者；已投稿 The Lancet Digital Health。",
    ),
    stack: t("FOUNDATION MODELS · EXTERNAL VALIDATION", "基础模型 · 外部验证"),
    figure: {
      src: "chest-xray.jpg",
      alt: t(
        "Radar chart of per-finding AUC for eight chest X-ray models across the shared public tasks.",
        "雷达图：八个胸片模型在共有公开任务上各项病征的 AUC。",
      ),
      credit: both("ARXIV:2505.16027"),
      pos: "35% 50%",
    },
    venues: [
      { label: t("ARXIV (PREPRINT)", "ARXIV（预印本）"), href: "https://arxiv.org/abs/2505.16027" },
      { label: t("TALK (2026)", "讲座（2026）"), href: talk },
    ],
  },
  {
    year: "2020—2021",
    title: t(
      "AI-based CT triage of COVID-19 patients",
      "基于人工智能的 COVID-19 患者 CT 分诊",
    ),
    note: t(
      "A multi-modal triage model joining 9,943 chest-CT radiomic features to clinical and laboratory records, predicting the outcomes that decide care — ICU admission, mechanical ventilation, death. Trained on a multi-hospital cohort of 1,662 patients and externally validated on 1,362 more, at AUROC ~0.85—0.94 across tasks.",
      "一个多模态分诊模型，把 9,943 项胸部 CT 影像组学特征与临床及实验室记录联合起来，预测真正决定治疗方案的结局——ICU 收治、机械通气与死亡。在 1,662 例患者的多中心队列上训练，并在另外 1,362 例上做外部验证，各任务 AUROC 约 0.85—0.94。",
    ),
    stack: t("RADIOMICS · MULTI-MODAL · SURVIVAL", "影像组学 · 多模态 · 生存分析"),
    figure: {
      src: "covid-ct.jpg",
      alt: t(
        "Chest CT slices beside the same slices with pulmonary lobes outlined and opacities shaded by the model.",
        "胸部 CT 切片，与模型勾出肺叶轮廓、标出磨玻璃影的同一批切片并列。",
      ),
      credit: both("FIG. 5 · NPJ DIGITAL MEDICINE 2021 · CC BY"),
      pos: "50% 50%",
    },
    venues: [
      { label: both("NPJ DIGITAL MEDICINE"), href: "https://doi.org/10.1038/s41746-021-00446-z" },
      { label: t("TALK (2026)", "讲座（2026）"), href: talk },
    ],
  },
  {
    year: "02/2022—09/2022",
    title: t(
      "Multi-contrast MRI registration with a realistic flow field",
      "带真实形变场的多对比度 MRI 配准",
    ),
    note: t(
      "Ported SynthMorph from TensorFlow to PyTorch and trained variants with Jacobian and cycle-consistency losses to fight unrealistic flow fields and over-smoothing in the VoxelMorph family. ~40% SSIM and ~50% PSNR improvement over baseline on BraTS and Lumbar-Spine.",
      "把 SynthMorph 从 TensorFlow 移植到 PyTorch，训练了带雅可比与循环一致性损失的多个变体，用来抑制 VoxelMorph 系列中不真实的形变场与过度平滑。在 BraTS 与腰椎数据上，SSIM 较基线提升约 40%，PSNR 提升约 50%。",
    ),
    stack: both("PYTORCH · SYNTHMORPH · VOXELMORPH"),
    venues: [{ label: both("ISMRM 2023"), href: "https://archive.ismrm.org/2023/4343.html" }],
  },
  {
    year: "08/2021—02/2022",
    title: t(
      "Deep learning based image co-registration quality control",
      "基于深度学习的图像配准质量控制",
    ),
    note: t(
      "A self-supervised classifier for registration quality on 3D MRI pairs, trained on synthetically mis-registered volumes from an affine plus deformable augmentation pipeline — the labelled failure cases don't exist in the wild, so we manufactured them. Trained on synthetic contrasts alone, it generalises to real brain and spine MRI better than a model trained on either of them.",
      "一个针对三维 MRI 图像对的自监督配准质量分类器，训练数据来自仿射加形变增强流水线合成的错配体数据——真实世界里没有带标注的失败样本，于是我们自己造。只用合成对比度训练的模型，在真实脑部与脊柱 MRI 上的泛化性反而好过在其中任一数据集上训练的模型。",
    ),
    stack: t("PYTORCH · 3D CNN · SELF-SUPERVISION", "PYTORCH · 3D CNN · 自监督"),
    // Presented three times; ISMRM and ASNR publish the abstract, RSNA does not.
    venues: [
      { label: both("ISMRM 2022"), href: "https://archive.ismrm.org/2022/1880.html" },
      {
        label: both("ASNR 2022"),
        href: "https://www.asnr.org/wp-content/uploads/2024/09/ASNR22-Proceedings_09.17.24.pdf#page=260",
      },
      { label: both("RSNA 2022"), href: "" },
    ],
  },
  {
    year: "2020—2023",
    title: t(
      "Head-impact kinematics and brain-strain prediction",
      "头部撞击运动学与脑应变预测",
    ),
    note: t(
      "Five co-authored papers on what measurable head kinematics can and cannot say about brain injury: impact subtyping from the spectral densities of the kinematics (J. Sport Health Sci. 2023), piecewise multivariate linearity between kinematic features and the cumulative strain damage measure (Ann. Biomed. Eng. 2022), how far brain injury criteria and simulated strain diverge across impact types (J. R. Soc. Interface 2021), the measurement time window needed for reliable strain and strain-rate estimates, and a statistical reading of which kinematic features actually carry the prediction.",
      "五篇合著论文，讨论可测量的头部运动学到底能、以及不能说明脑损伤的哪些方面：基于运动学功率谱密度的撞击亚型划分（J. Sport Health Sci. 2023）；运动学特征与累积应变损伤指标（CSDM）之间的分段多元线性关系（Ann. Biomed. Eng. 2022）；不同撞击类型下脑损伤判据与仿真应变的分歧程度（J. R. Soc. Interface 2021）；可靠估计应变与应变率所需的测量时间窗；以及从统计角度看究竟哪些运动学特征真正承担了预测。",
    ),
    stack: t("MACHINE LEARNING · BIOMECHANICS · STATISTICS", "机器学习 · 生物力学 · 统计"),
    figure: {
      src: "head-impact.jpg",
      alt: t(
        "Heatmap of kinematic features across six head-impact datasets, sorted by impact subtype.",
        "六个头部撞击数据集的运动学特征热图，按撞击亚型排列。",
      ),
      credit: both("FIG. 1 · J SPORT HEALTH SCI 2023 · CC BY-NC-ND"),
      pos: "50% 74%",
    },
    // Five papers, five destinations. Labelled by subject rather than journal:
    // three of them are Ann. Biomed. Eng. and the journal name alone would not
    // say which is which.
    venues: [
      {
        label: t("SUBTYPING (JSHS 2023)", "亚型划分（JSHS 2023）"),
        href: "https://doi.org/10.1016/j.jshs.2023.03.003",
      },
      {
        label: t("CSDM (ABME 2022)", "CSDM（ABME 2022）"),
        href: "https://doi.org/10.1007/s10439-022-03020-0",
      },
      {
        label: t("INJURY CRITERIA (JRSI 2021)", "损伤判据（JRSI 2021）"),
        href: "https://doi.org/10.1098/rsif.2021.0260",
      },
      {
        label: t("TIME WINDOW (ABME 2021)", "时间窗（ABME 2021）"),
        href: "https://doi.org/10.1007/s10439-021-02821-z",
      },
      {
        label: t("PREDICTORS (ABME 2021)", "预测因子（ABME 2021）"),
        href: "https://doi.org/10.1007/s10439-021-02813-z",
      },
    ],
  },
  {
    year: "2022—",
    title: both("Lumos-ToolKit"),
    note: t(
      "My own PyTorch-Lightning and MONAI toolkit for training, logging and inference — the parts of medical imaging research nobody wants to write twice. Not released yet; the repository is still private while it is cleaned up.",
      "我自己写的一套基于 PyTorch Lightning 与 MONAI 的训练、日志与推理工具包，覆盖医学影像研究里没人想写第二遍的那些部分。尚未发布，仓库仍是私有的，还在整理中。",
    ),
    stack: t("PERSONAL · OPEN SOURCE · PYTHON", "个人项目 · 开源 · PYTHON"),
    // No repository to point at yet, so the venue link goes to the GitHub
    // profile the toolkit will land on.
    venues: [{ label: t("OPEN SOURCE PROJECT", "开源项目"), href: me.github }],
  },
];

// Titles are as they stand in the proceedings, and `href` goes to the abstract
// itself where the meeting publishes one. ISMRM opens its archive two years
// after the meeting; ASNR publishes a single proceedings PDF, so that link
// carries a #page anchor. RSNA has no public permalink for this one.
export const pubs = [
  {
    venue: both("ISMRM 2023"),
    title: t(
      "A Deep Learning-based Multi-Contrast MRI Registration Model with a Realistic Flow Field and Reduced Over-Smoothing Effect",
      "一种带真实形变场、可减少过度平滑的深度学习多对比度 MRI 配准模型",
    ),
    href: "https://archive.ismrm.org/2023/4343.html",
  },
  {
    venue: both("ISMRM 2022"),
    title: t(
      "QC of image registration using a DL network trained using only synthetic images",
      "仅用合成图像训练的深度学习网络实现图像配准质量控制",
    ),
    href: "https://archive.ismrm.org/2022/1880.html",
  },
  {
    venue: both("ASNR 2022"),
    title: t(
      "Detection of misalignments between multi-contrast brain MR datasets using Deep Learning",
      "用深度学习检测多对比度脑部 MR 数据之间的错配",
    ),
    href: "https://www.asnr.org/wp-content/uploads/2024/09/ASNR22-Proceedings_09.17.24.pdf#page=260",
  },
  {
    venue: both("RSNA 2022"),
    title: t(
      "Deep learning quality control for multi-contrast brain MRI alignment",
      "多对比度脑 MRI 对齐的深度学习质量控制",
    ),
    href: "",
  },
];

// `logo` as in `cv` above.
export const edu = [
  {
    when: "09/2019—06/2021",
    school: t("Stanford University", "斯坦福大学"),
    logo: "stanford-university",
    degree: t("M.Sc. Biomedical Informatics", "生物医学信息学 理学硕士"),
  },
  {
    when: "09/2015—06/2019",
    school: t("Shanghai Jiao Tong University", "上海交通大学"),
    logo: "shanghai-jiao-tong-university",
    degree: t("B.Sc. Resource and Environmental Science", "资源与环境科学 理学学士"),
  },
  {
    when: "01/2018—05/2018",
    school: t("University of California, Berkeley", "加州大学伯克利分校"),
    logo: "uc-berkeley",
    degree: t("International Exchange Program", "国际交换项目"),
  },
];

export const skills = [
  { k: t("Languages", "语言"), v: both("Python, R, SQL, MATLAB, Bash") },
  {
    k: t("Frameworks", "框架"),
    v: both("PyTorch, PyTorch Lightning, MONAI, ANTs, scikit-learn, TensorFlow"),
  },
  {
    k: t("Imaging", "影像"),
    v: t(
      "3D CT and MRI, segmentation, registration, radiomics, DICOM / NIfTI pipelines",
      "三维 CT 与 MRI、分割、配准、影像组学、DICOM / NIfTI 流水线",
    ),
  },
  {
    k: t("Training", "训练"),
    v: t(
      "Multi-GPU training (Lightning DDP), mixed precision, gradient checkpointing, SLURM",
      "多 GPU 训练（Lightning DDP）、混合精度、梯度检查点、SLURM",
    ),
  },
  {
    k: t("Statistics", "统计"),
    v: t(
      "Survival analysis, GLMs, mixed models, experimental design",
      "生存分析、广义线性模型、混合效应模型、实验设计",
    ),
  },
];

export const others = [
  {
    n: "01",
    title: t(
      "Repression effect of protein tiles from HT-recruit RNA-seq",
      "基于 HT-recruit RNA-seq 的蛋白片段抑制效应",
    ),
    course: t(
      "BIOMEDIN 273B · DEEP LEARNING IN GENOMICS",
      "BIOMEDIN 273B · 基因组学中的深度学习",
    ),
    when: t("2020 fall", "2020 年秋"),
  },
  {
    n: "02",
    title: t(
      "Meta-learning for head impact strain across datasets",
      "跨数据集的头部撞击应变元学习",
    ),
    course: t("CS 330 · DEEP MULTI-TASK AND META LEARNING", "CS 330 · 深度多任务与元学习"),
    when: t("2020 fall", "2020 年秋"),
  },
  {
    n: "03",
    title: t(
      "Transfer learning for pneumothorax detection on chest X-ray",
      "胸片气胸检测的迁移学习",
    ),
    course: t("BIOMEDIN 260 · BIOMEDICAL IMAGE ANALYSIS", "BIOMEDIN 260 · 生物医学图像分析"),
    when: t("2020 spring", "2020 年春"),
  },
  {
    n: "04",
    title: t("Two open-topic deep learning projects", "两个自选题深度学习项目"),
    course: t("CS 230 · CS 229 · POSTERS AND REPORTS", "CS 230 · CS 229 · 海报与报告"),
    when: both("2019 — 2020"),
  },
];

// ── profile · buttons (label + monoline glyph path set) ──
export const profileLinks = linked([
  { label: t("DOWNLOAD RÉSUMÉ (PDF)", "下载简历（PDF）"), href: me.resume, icon: "resume", seal: true },
  { label: both("GITHUB"), href: me.github, icon: "github" },
  { label: t("LINKEDIN", "领英"), href: me.linkedin, icon: "linkedin" },
  { label: t("SCHOLAR", "谷歌学术"), href: me.scholar, icon: "scholar" },
  { label: both("ORCID"), href: me.orcid, icon: "orcid" },
  { label: t("STANFORD", "斯坦福"), href: me.stanford, icon: "stanford" },
  { label: t("EMAIL", "邮箱"), href: mailto, icon: "email" },
]);

/** Profile links that identify the person, so they can carry `rel="me"`. */
export const IDENTITY_ICONS = new Set(["github", "linkedin", "scholar", "orcid", "stanford"]);

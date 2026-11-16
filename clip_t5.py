# ============================================================
# CLIP-T5 for medical visual question answering - training and evaluation
#
# Model: PubMedCLIP (ViT-B/32) patch tokens (first 8 layers frozen, last 4 fine-tuned) -> projection
#        -> gated question-to-image multi-head cross-attention -> Flan-T5-base encoder-decoder (fine-tuned).
# Data:  trained jointly on VQA-RAD (official OSF release), SLAKE (English) and PathVQA; evaluated on the
#        official test sets with exact-match accuracy on generated answers (see README).
# Runs:  set with the RUNS environment variable, e.g.
#        RUNS="main:0,1,2 no_xattn:0,1,2 question_only:0,1 cls_only:0,1"   (default, as in the paper)
#        The first main run also exports examples and measures latency; every main run does the
#        shuffled-image test. Finished runs are skipped when the script is started again.
# Paths: VQA_INPUT = folder that contains the official VQA-RAD release ("VQA_RAD Dataset Public.json" + images)
#        VQA_WORK  = output folder. Defaults are the Kaggle paths.
# One run takes about 100 minutes on a single NVIDIA T4 GPU.
# ============================================================
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import re, json, time, math, random, hashlib, glob, zipfile, copy, shutil, gc, psutil
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import torchvision.transforms as T
from datasets import load_dataset, get_dataset_split_names
from transformers import (CLIPModel, CLIPImageProcessor, T5ForConditionalGeneration,
                          T5TokenizerFast, get_cosine_schedule_with_warmup)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from scipy.stats import binomtest, chi2_contingency
ImageFile.LOAD_TRUNCATED_IMAGES = True
T_START = time.time()

# ---------------- settings ----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sync = torch.cuda.synchronize if DEVICE == 'cuda' else (lambda: None)
INPUT = os.environ.get('VQA_INPUT', '/kaggle/input')
WORK = os.environ.get('VQA_WORK', '/kaggle/working')
OUT = f'{WORK}/clip_t5_results'; os.makedirs(f'{OUT}/runs', exist_ok=True)
TIME_LIMIT_H = float(os.environ.get('TIME_LIMIT_H', 11.0))   # stop starting new runs after this (Kaggle limit: 12 h)
CAND_MAX = 3000                      # max candidate answers per dataset (most frequent training answers)
RANK_SETS = ['rad', 'slake']         # PathVQA has too many distinct answers for ranking on a T4: free text only
CLIP_NAME = 'flaviagiammarino/pubmed-clip-vit-base-patch32'
T5S = {'t5': 'google-t5/t5-base', 'flan': 'google/flan-t5-base'}
HF = {'slake': 'mdwiratathya/SLAKE-vqa-english', 'path': 'flaviagiammarino/path-vqa'}
DATASETS = ['rad', 'slake', 'path']
SEEDS = [0, 1, 2, 3, 4]
N_AUG = 2        # extra augmented views per training image (no flips: left/right questions)
K_MAX = 4        # last K_MAX CLIP layers are kept trainable-capable; first layers frozen
VAL_MAX = 600    # max validation questions per dataset

BASE = dict(t5='flan', k_tune=4, train_sets=('rad', 'slake', 'path'), use_image=True, use_xattn=True,
            heads=8, lr_t5=1e-4, lr_new=5e-4, lr_vis=2e-5, bs=16, steps_per_epoch=800, epochs=12, patience=3, rank=False,
            rad_ft_epochs=0, rad_ft_lr_scale=0.5, cls_only=False)
MAIN_SEEDS, ABL_SEEDS = [0, 1, 2], [0, 1]
ABLATIONS = {'no_xattn': dict(use_xattn=False),
             'question_only': dict(use_image=False)}
assert DEVICE == 'cuda', 'A CUDA GPU is required (on Kaggle: Accelerator -> GPU T4 x1).'
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')
print('finished runs found:', len(glob.glob(f'{OUT}/runs/*.json')))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- metrics ----------------
NUMS = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
        'six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
SYN = [{'xray','radiograph','plain film','plain radiograph','cxr'},
       {'ct','ct scan','computed tomography','cat scan'},
       {'mri','mr','magnetic resonance imaging','mri scan'},
       {'ultrasound','us','sonography'}, {'pa','posteroanterior'},
       {'ap','anteroposterior'}, {'axial','transverse'},
       {'left','left side'}, {'right','right side'}, {'yes','yeah','y'}, {'no','n'}]

def norm(s):
    s = str(s).lower().strip().replace('x-ray', 'xray').replace('x ray', 'xray')
    s = re.sub(r'[^\w\s]', ' ', s)
    return ' '.join(NUMS.get(t, t) for t in s.split() if t not in {'a', 'an', 'the'})

def canon(s):
    n = norm(s)
    for g in SYN:
        if n in g: return sorted(g)[0]
    return n

def lenient(p, g):
    cp, cg = canon(p), canon(g)
    if cp == cg: return True
    if cg in ('yes', 'no') or not cp: return False
    P, G = set(cp.split()), set(cg.split())
    return (P <= G or G <= P) and len(P & G) / len(P | G) >= 0.5

def token_f1(p, g):
    P, G = norm(p).split(), norm(g).split()
    common = sum(min(P.count(t), G.count(t)) for t in set(P))
    if not P or not G or common == 0: return float(P == G)
    pr, rc = common / len(P), common / len(G)
    return 2 * pr * rc / (pr + rc)

def score_df(rows, preds):
    df = pd.DataFrame({'question': [r['question'] for r in rows],
                       'answer': [r['answer'] for r in rows], 'pred': preds,
                       'closed': [r['closed'] for r in rows],
                       'qtype': [r['qtype'] for r in rows]})
    df['em'] = [norm(p) == norm(g) for p, g in zip(df.pred, df.answer)]
    df['len'] = [lenient(p, g) for p, g in zip(df.pred, df.answer)]
    df['f1'] = [token_f1(p, g) for p, g in zip(df.pred, df.answer)]
    return df

def summarize(df):
    sm = SmoothingFunction().method1
    refs = [[norm(g).split()] for g in df.answer]; hyps = [norm(p).split() for p in df.pred]
    c = df.closed.astype(bool)
    mean = lambda x: float(x.mean() * 100) if len(x) else float('nan')
    return dict(n=len(df), acc=mean(df.em), acc_lenient=mean(df.len),
                closed_acc=mean(df[c].em), open_acc=mean(df[~c].em),
                open_acc_lenient=mean(df[~c].len), f1=mean(df.f1),
                bleu1=corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=sm)*100,
                bleu4=corpus_bleu(refs, hyps, smoothing_function=sm)*100)

# ---------------- data ----------------
def build_store(items, label):
    """items: dicts with image (PIL or path), question, answer, closed (bool/None), qtype (str/None)."""
    imgs, hashes, idmap, rows = [], [], {}, []
    for it in items:
        im = it['image']
        im = (Image.open(im) if isinstance(im, str) else im).convert('RGB')
        h = hashlib.md5(im.tobytes()).hexdigest()
        if h not in idmap:
            idmap[h] = len(imgs); hashes.append(h); imgs.append(im.resize((224, 224), Image.BICUBIC))
        q, a = str(it['question']), str(it['answer'])
        yn = norm(a) in ('yes', 'no')
        rows.append(dict(iid=idmap[h], question=q, answer=a,
                         closed=it['closed'] if it['closed'] is not None else yn,
                         qtype=it['qtype'] or ('yes/no' if yn else 'count'
                                               if q.lower().startswith('how many') else 'open')))
    print(f'{label}: {len(rows)} questions, {len(imgs)} images')
    return dict(imgs=imgs, hashes=hashes), rows

# official VQA-RAD (OSF): standard split = 3,064 train / 451 test
def find_osf(root):
    return [p for p in glob.glob(f'{root}/**/*.json', recursive=True)
            if 'vqa_rad' in os.path.basename(p).lower() and 'public' in os.path.basename(p).lower()]
osf = find_osf(INPUT)
assert osf, 'Official VQA-RAD (OSF json + images) not found: add it as a Kaggle dataset input.'
IMG_INDEX = {os.path.basename(p): p for p in glob.glob(f'{INPUT}/**/*', recursive=True)
             if p.lower().endswith(('.jpg', '.jpeg', '.png'))}
rad_tr, rad_te, missing, n_para = [], [], 0, 0
for d in json.load(open(osf[0])):
    p = IMG_INDEX.get(d['image_name'])
    if p is None: missing += 1; continue
    it = dict(image=p, question=d['question'], answer=d['answer'],
              closed=str(d['answer_type']).strip().upper() == 'CLOSED',
              qtype=str(d['question_type']).split(',')[0].strip().upper())
    is_test = str(d.get('phrase_type', '')).startswith('test')
    (rad_te if is_test else rad_tr).append(it)
    rq = str(d.get('question_rephrase') or '').strip()      # official paraphrase of the question
    if not is_test and rq and rq.upper() != 'NULL' and norm(rq) != norm(d['question']):
        rad_tr.append(dict(it, question=rq)); n_para += 1
print('OSF file:', osf[0], '| missing images:', missing, '| paraphrased training questions added:', n_para)

STORES, SETS = {}, {}
for part, items in [('train', rad_tr), ('test', rad_te)]:
    STORES[f'rad_{part}'], rows = build_store(items, f'VQA-RAD (official) [{part}]')
    SETS[f'rad_{part}'] = (f'rad_{part}', rows)
for d, hf in HF.items():
    names = get_dataset_split_names(hf)
    for part, want in [('train', ['train']), ('val', ['validation', 'val']), ('test', ['test'])]:
        sp = next((w for w in want if w in names), None)
        if sp is None: continue
        items = ({'image': ex['image'], 'question': ex['question'], 'answer': ex['answer'],
                  'closed': None, 'qtype': None} for ex in load_dataset(hf, split=sp))
        STORES[f'{d}_{part}'], rows = build_store(items, f'{hf} [{sp}]')
        SETS[f'{d}_{part}'] = (f'{d}_{part}', rows)
for d in DATASETS:
    if f'{d}_val' not in SETS:   # no official val split: hold out 10% of training IMAGES
        rows = SETS[f'{d}_train'][1]; n = len(STORES[f'{d}_train']['imgs'])
        perm = list(range(n)); random.Random(42).shuffle(perm); vi = set(perm[:int(0.1 * n)])
        SETS[f'{d}_val'] = (f'{d}_train', [r for r in rows if r['iid'] in vi])
        SETS[f'{d}_train'] = (f'{d}_train', [r for r in rows if r['iid'] not in vi])
    vk = f'{d}_val'
    if len(SETS[vk][1]) > VAL_MAX:
        SETS[vk] = (SETS[vk][0], random.Random(0).sample(SETS[vk][1], VAL_MAX))

def hashes_of(s):
    sk, rows = SETS[s]; H = STORES[sk]['hashes']
    return {H[r['iid']] for r in rows}
any_train = set().union(*[hashes_of(f'{d}_train') for d in DATASETS])
report = {s: dict(questions=len(SETS[s][1]), images=len(hashes_of(s))) for s in SETS}
for d in DATASETS:
    te = SETS[f'{d}_test'][1]; H = STORES[f'{d}_test']['hashes']
    own = hashes_of(f'{d}_train'); trs, trr = SETS[f'{d}_train']
    pairs = {(STORES[trs]['hashes'][r['iid']], norm(r['question'])) for r in trr}
    report[f'{d}_test'].update(
        q_image_in_own_train=sum(H[r['iid']] in own for r in te),
        q_image_in_any_train=sum(H[r['iid']] in any_train for r in te),
        q_exact_duplicate_of_train=sum((H[r['iid']], norm(r['question'])) in pairs for r in te))
print(json.dumps(report, indent=1))
json.dump(report, open(f'{OUT}/split_report.json', 'w'), indent=1)

# ---------------- CLIP: first layers frozen (computed once), last K_MAX layers in the model ----------------
def get_pp(name):
    try: return CLIPImageProcessor.from_pretrained(name)
    except Exception: return CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')

def run_layer(layer, h):
    try: out = layer(h, None, None)
    except TypeError: out = layer(h, attention_mask=None)
    return out[0] if isinstance(out, tuple) else out

def frozen_part(cm, pix):
    vm = cm.vision_model
    h = vm.pre_layrnorm(vm.embeddings(pix))
    for l in vm.encoder.layers[:-K_MAX]: h = run_layer(l, h)
    return h

AUG = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                 T.RandomRotation(10), T.ColorJitter(0.2, 0.2)])

@torch.no_grad()
def compute_feats():
    cm = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval(); pp = get_pp(CLIP_NAME); out = {}
    for sk, st in STORES.items():
        views = []
        for v in range(1 + (N_AUG if sk.endswith('_train') else 0)):
            random.seed(1000 + v); torch.manual_seed(1000 + v); H = []
            for i in range(0, len(st['imgs']), 64):
                ims = st['imgs'][i:i+64]
                if v > 0: ims = [AUG(im) for im in ims]
                pix = pp(images=ims, return_tensors='pt')['pixel_values'].to(DEVICE)
                H.append(frozen_part(cm, pix).half().cpu())
            views.append(torch.cat(H))
        out[sk] = torch.stack(views, 1)          # (images, views, tokens, dim)
        print('features', sk, tuple(out[sk].shape))
    del cm; torch.cuda.empty_cache()
    return out
F = compute_feats()

# ---------------- model ----------------
class VisionHead(nn.Module):
    """Last K_MAX CLIP vision layers + post-LN + projection; the last k_tune layers are trained."""
    def __init__(s, cm):
        super().__init__()
        s.layers = copy.deepcopy(cm.vision_model.encoder.layers[-K_MAX:])
        s.post = copy.deepcopy(cm.vision_model.post_layernorm)
        s.proj = copy.deepcopy(cm.visual_projection)
    def forward(s, h):
        for l in s.layers: h = run_layer(l, h)
        return s.proj(s.post(h))

class CLIPT5(nn.Module):
    """CLIP image tokens -> projection -> question-to-image cross-attention -> T5."""
    def __init__(s, t5_model, vis_head, heads=8, use_xattn=True, use_image=True, p=0.1, cls_only=False, **_):
        super().__init__()
        t5, vis = t5_model, vis_head
        d = t5.config.d_model; s.t5, s.vis, s.use_image, s.cls_only = t5, vis, use_image, cls_only
        s.proj = nn.Sequential(nn.Linear(vis.proj.out_features, d), nn.LayerNorm(d), nn.Dropout(p))
        s.scale = nn.Parameter(torch.tensor(float(t5.shared.weight.detach().std())))
        s.pos = nn.Parameter(torch.randn(1, 50, d) * 0.02)
        s.xattn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True) if use_xattn else None
        if use_xattn:
            s.qnorm = nn.LayerNorm(d); s.gate = nn.Parameter(torch.zeros(1))

    def encode(s, hid, ids, mask, need_w=False):
        E, w = s.t5.shared(ids), None
        if not s.use_image:
            X, M = E, mask
        else:
            V = s.vis(hid)
            if s.cls_only: V = V[:, :1]           # ablation: global class token only
            P = (s.proj(V) + s.pos[:, :V.shape[1]]) * s.scale
            if s.xattn is not None:
                a, w = s.xattn(s.qnorm(E), P, P, need_weights=need_w)
                E = E + torch.tanh(s.gate) * a
            X = torch.cat([P, E], 1)
            M = torch.cat([torch.ones(P.shape[:2], dtype=mask.dtype, device=mask.device), mask], 1)
        return s.t5.encoder(inputs_embeds=X, attention_mask=M, return_dict=True), M, w

    def forward(s, hid, ids, mask, labels):
        enc, M, _ = s.encode(hid, ids, mask)
        return s.t5(encoder_outputs=enc, attention_mask=M, labels=labels).loss

    def sample_losses(s, hid, ids, mask, labels):
        """Mean token cross-entropy per question (shape: batch)."""
        enc, M, _ = s.encode(hid, ids, mask)
        logits = s.t5(encoder_outputs=enc, attention_mask=M, labels=labels).logits
        ce = nn.functional.cross_entropy(logits.transpose(1, 2).float(), labels,
                                         ignore_index=-100, reduction='none')
        valid = (labels != -100).float()
        return (ce * valid).sum(1) / valid.sum(1).clamp(min=1)

    def generate(s, hid, ids, mask, num_beams=5, max_new_tokens=24):
        enc, M, _ = s.encode(hid, ids, mask)
        return s.t5.generate(encoder_outputs=enc, attention_mask=M,
                             num_beams=num_beams, max_new_tokens=max_new_tokens)

tok = T5TokenizerFast.from_pretrained(T5S['t5'])   # T5 and Flan-T5 share the same vocabulary

def build_model(cfg):
    t5 = T5ForConditionalGeneration.from_pretrained(T5S[cfg['t5']])
    cm = CLIPModel.from_pretrained(CLIP_NAME); vis = VisionHead(cm); del cm
    m = CLIPT5(t5, vis, **cfg).to(DEVICE)
    k = cfg['k_tune'] if cfg['use_image'] else 0
    for i, l in enumerate(m.vis.layers):
        for p in l.parameters(): p.requires_grad = i >= K_MAX - k and k > 0
    for p in list(m.vis.post.parameters()) + list(m.vis.proj.parameters()): p.requires_grad = k > 0
    return m

def get_batch(items, train, use_image):
    hid = None
    if use_image:
        H = []
        for sname, r in items:
            f = F[SETS[sname][0]][r['iid']]
            H.append(f[random.randrange(f.shape[0])] if train else f[0])
        hid = torch.stack(H).float().to(DEVICE)
    q = tok(['question: ' + r['question'] for _, r in items], padding=True,
            truncation=True, max_length=64, return_tensors='pt')
    return hid, q.input_ids.to(DEVICE), q.attention_mask.to(DEVICE)

def make_labels(answers):
    ids = tok(list(answers), padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids
    ids[ids == tok.pad_token_id] = -100
    return ids.to(DEVICE)

@torch.no_grad()
def predict(model, sname, rows, use_image, beams=5, bs=32):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        hid, ids, mask = get_batch([(sname, r) for r in rows[i:i+bs]], False, use_image)
        out += [d.strip() for d in tok.batch_decode(model.generate(hid, ids, mask, num_beams=beams),
                                                    skip_special_tokens=True)]
    return out

def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in names}   # kept on GPU to save RAM

# ---------------- extras for the winner: attention maps + latency ----------------
@torch.no_grad()
def attention_maps(model):
    model.eval(); fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    picks = ([('rad_test', i) for i in np.random.RandomState(0).choice(len(SETS['rad_test'][1]), 8, replace=False)] +
             [('slake_test', i) for i in np.random.RandomState(1).choice(len(SETS['slake_test'][1]), 4, replace=False)])
    for ax, (sname, i) in zip(axes.flat, picks):
        r = SETS[sname][1][i]
        hid, ids, mask = get_batch([(sname, r)], False, True)
        _, _, w = model.encode(hid, ids, mask, need_w=True)
        m = mask[0].float()
        att = ((w[0] * m[:, None]).sum(0) / m.sum())[1:].float().cpu().numpy()
        g = int(round(math.sqrt(len(att)))); att = att.reshape(g, g)
        att = np.kron(att / att.max(), np.ones((224 // g, 224 // g)))
        pred = tok.decode(model.generate(hid, ids, mask)[0], skip_special_tokens=True).strip()
        ok = 'correct' if norm(pred) == norm(r['answer']) else 'wrong'
        ax.imshow(STORES[SETS[sname][0]]['imgs'][r['iid']])
        ax.imshow(att, cmap='jet', alpha=0.4, extent=(0, 224, 224, 0)); ax.axis('off')
        ax.set_title(f"[{sname.split('_')[0].upper()}] Q: {r['question'][:40]}\n"
                     f"GT: {r['answer'][:22]} | Pred: {pred[:22]} ({ok})", fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig_attention_maps.png', dpi=300); plt.show()

@torch.no_grad()
def latency(model, n=110):
    cm = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval(); pp = get_pp(CLIP_NAME); model.eval(); ts = []
    st, rows = STORES['rad_test'], SETS['rad_test'][1][:n]
    for r in rows:
        sync(); t = time.perf_counter()
        pix = pp(images=[st['imgs'][r['iid']]], return_tensors='pt')['pixel_values'].to(DEVICE)
        hid = frozen_part(cm, pix)
        q = tok(['question: ' + r['question']], return_tensors='pt').to(DEVICE)
        model.generate(hid, q.input_ids, q.attention_mask)
        sync(); ts.append((time.perf_counter() - t) * 1000)
    del cm; torch.cuda.empty_cache()
    ts = ts[10:] if len(ts) > 20 else ts
    return dict(mean_ms=float(np.mean(ts)), sd_ms=float(np.std(ts)),
                gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu')

# ---------------- answer ranking (closed-set evaluation) ----------------
def build_candidates(d):
    """Unique training answers of dataset d (normalised), most frequent first. No test info used."""
    cnt = {}
    for r in SETS[f'{d}_train'][1]:
        k = norm(r['answer'])
        if k: cnt.setdefault(k, [0, r['answer']]); cnt[k][0] += 1
    cands = [v[1] for v in sorted(cnt.values(), key=lambda v: -v[0])][:CAND_MAX]
    ids = tok(cands, padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids
    order = (ids != tok.pad_token_id).sum(1).argsort().tolist()      # short answers first -> less padding
    cands, ids = [cands[k] for k in order], ids[order]
    ids[ids == tok.pad_token_id] = -100
    return cands, ids

@torch.no_grad()
def rank_scores(model, sname, rows, cand_ids, use_image, qbs=2, cbs=128):
    """Per question: summed and mean token log-prob of every candidate answer."""
    model.eval(); C = cand_ids.shape[0]; S_sum, S_mean = [], []
    cand_ids = cand_ids.to(DEVICE); valid = (cand_ids != -100); ntok = valid.sum(1).clamp(min=1)
    for i in range(0, len(rows), qbs):
        hid, ids, mask = get_batch([(sname, r) for r in rows[i:i+qbs]], False, use_image)
        enc, M, _ = model.encode(hid, ids, mask)          # fp32: T5 overflows in fp16
        H = enc.last_hidden_state; B = H.shape[0]
        out = torch.empty(B, C, device=DEVICE)
        for j in range(0, C, cbs):
            c = cand_ids[j:j+cbs]; c = c[:, :int((c != -100).sum(1).max())]; n = c.shape[0]
            Hx = H.repeat_interleave(n, 0); Mx = M.repeat_interleave(n, 0); L = c.repeat(B, 1)
            logits = model.t5(encoder_outputs=(Hx,), attention_mask=Mx,
                              decoder_input_ids=model.t5._shift_right(L)).logits
            logits = logits.float()
            tok_lp = (logits.gather(-1, L.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                      - torch.logsumexp(logits, -1)) * (L != -100)
            del logits
            out[:, j:j+n] = tok_lp.sum(1).view(B, n)
        S_sum.append(out.cpu()); S_mean.append((out / ntok.view(1, -1)).cpu())
    return torch.cat(S_sum), torch.cat(S_mean)

def rank_eval(model, d, use_image):
    """Choose sum vs mean scoring on validation, then predict the test set."""
    cands, cid = build_candidates(d)
    test_rows = SETS[f'{d}_test'][1]
    cover = float(np.mean([norm(r['answer']) in {norm(c) for c in cands} for r in test_rows]) * 100)
    vrows = SETS[f'{d}_val'][1]
    def safe(sname, rows):                       # retry with smaller groups if GPU memory runs out
        q, c = 2, 128
        while True:
            try: return rank_scores(model, sname, rows, cid, use_image, qbs=q, cbs=c)
            except torch.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if q > 1: q //= 2
                elif c > 8: c //= 2
                else: raise
                print(f'   (ranking: low GPU memory, retrying with {q} questions x {c} candidates)')
    vs, vm = safe(f'{d}_val', vrows)
    acc = lambda S, rows: score_df(rows, [cands[k] for k in S.argmax(1).tolist()]).em.mean() * 100
    mode = 'sum' if acc(vs, vrows) >= acc(vm, vrows) else 'mean'
    ts, tm = safe(f'{d}_test', test_rows)
    preds = [cands[k] for k in (ts if mode == 'sum' else tm).argmax(1).tolist()]
    return score_df(test_rows, preds), dict(mode=mode, n_candidates=len(cands), test_answer_coverage=cover)

@torch.no_grad()
def export_examples(model, cfg, path):
    """Real VQA-RAD test predictions (ranking): 6 correct + 4 incorrect, images + csv."""
    df = pd.read_csv(f'{path}_rad.csv'); rows = SETS['rad_test'][1]; os.makedirs(f'{OUT}/examples', exist_ok=True)
    rs = np.random.RandomState(0)
    cor = [i for i in rs.permutation(len(df)) if df.em[i]][:6]; inc = [i for i in rs.permutation(len(df)) if not df.em[i]][:4]
    out = []
    for k, (i, g) in enumerate([(i, 'correct') for i in cor] + [(i, 'incorrect') for i in inc]):
        fn = f'example_{k+1:02d}_{g}.png'
        STORES['rad_test']['imgs'][rows[i]['iid']].save(f'{OUT}/examples/{fn}')
        out.append(dict(image_file=fn, group=g, question=rows[i]['question'], answer=rows[i]['answer'],
                        pred=df.pred[i], closed=rows[i]['closed'], qtype=rows[i]['qtype']))
    pd.DataFrame(out).to_csv(f'{OUT}/examples/examples.csv', index=False)
    print(pd.DataFrame(out)[['group', 'question', 'answer', 'pred']].to_string())

# ---------------- training ----------------
def train_run(name, cfg, seed, shuffle_check=False, extras=False):
    path = f'{OUT}/runs/{name}_s{seed}'
    if os.path.exists(path + '.json'): return json.load(open(path + '.json'))
    t0 = time.time(); cfg = {**BASE, **cfg}; set_seed(seed)
    model = build_model(cfg)
    groups = {'vis': [], 'new': [], 't5': []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups['vis' if n.startswith('vis.') else 't5' if n.startswith('t5.') else 'new'].append(p)
    lrs = {'vis': cfg['lr_vis'], 'new': cfg['lr_new'], 't5': cfg['lr_t5']}
    opt = torch.optim.AdamW([{'params': v, 'lr': lrs[k]} for k, v in groups.items() if v], weight_decay=0.01)
    params = [p for v in groups.values() for p in v]; n_params = sum(p.numel() for p in params)
    pools = [(f'{d}_train', SETS[f'{d}_train'][1]) for d in cfg['train_sets']]
    sizes = np.array([len(r) for _, r in pools]); probs = np.sqrt(sizes) / np.sqrt(sizes).sum()
    spe = cfg['steps_per_epoch'] or math.ceil(sizes.sum() / cfg['bs'])
    steps = spe * cfg['epochs']
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    vals = [f'{d}_val' for d in cfg['train_sets']]
    rng = np.random.RandomState(seed)
    best, best_score, bad, hist = None, -1, 0, []
    for ep in range(cfg['epochs']):
        model.train(); tot = 0
        for _ in range(spe):
            pick = rng.choice(len(pools), cfg['bs'], p=probs)
            items = [(pools[k][0], pools[k][1][rng.randint(len(pools[k][1]))]) for k in pick]
            hid, ids, mask = get_batch(items, True, cfg['use_image'])
            labels = make_labels([r['answer'] for _, r in items])
            if cfg.get('contrast') and cfg['use_image']:
                neg = []
                for k, (sn, r) in zip(pick, items):        # wrong image from the same dataset
                    pool = pools[k][1]; j = r['iid']
                    for _t in range(10):
                        j = pool[rng.randint(len(pool))]['iid']
                        if j != r['iid']: break
                    neg.append((sn, dict(r, iid=j)))
                hneg, _, _ = get_batch(neg, True, True)
                l_pos = model.sample_losses(hid, ids, mask, labels)
                l_neg = model.sample_losses(hneg, ids, mask, labels)
                hinge = torch.relu(cfg['margin'] - (l_neg - l_pos))
                loss = l_pos.mean() + cfg['lam'] * hinge.mean()
            else:
                loss = model(hid, ids, mask, labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            tot += loss.item()
        va = {v: score_df(SETS[v][1], predict(model, v, SETS[v][1], cfg['use_image'], beams=1)).em.mean() * 100
              for v in vals}
        score = float(np.mean(list(va.values())))
        hist.append(dict(epoch=ep + 1, train_loss=tot / spe, val_score=score, **va))
        print(f'{name} s{seed} ep{ep+1} loss {tot/spe:.3f} val ' +
              ' '.join(f'{k.split("_")[0]} {x:.2f}' for k, x in va.items()) + f' | mean {score:.2f}'
              + f' | RAM {psutil.virtual_memory().used / 1e9:.1f} GB')
        if score > best_score:
            best = None; gc.collect()
            best_score, best, bad = score, trainable_state(model), 0
        else:
            bad += 1
            if bad >= cfg['patience']: break
    model.load_state_dict(best, strict=False)
    opt.state.clear(); gc.collect(); torch.cuda.empty_cache()     # free optimizer memory before evaluation
    test, shuffled = {}, {}
    for d in DATASETS:
        rows = SETS[f'{d}_test'][1]
        df = score_df(rows, predict(model, f'{d}_test', rows, cfg['use_image']))
        df.to_csv(f'{path}_{d}.csv', index=False); test[d] = summarize(df)
        if shuffle_check and cfg['use_image']:   # same questions, images swapped at random
            perm = np.random.RandomState(seed).permutation(len(rows))
            srows = [dict(r, iid=rows[j]['iid']) for r, j in zip(rows, perm)]
            shuffled[d] = float(score_df(rows, predict(model, f'{d}_test', srows, True)).em.mean() * 100)
        print(f'   test {d}: acc {test[d]["acc"]:.2f} closed {test[d]["closed_acc"]:.2f} '
              f'open {test[d]["open_acc"]:.2f}' + (f' | shuffled images {shuffled[d]:.2f}' if d in shuffled else ''))
    test_radft, radft_hist = {}, []
    if cfg.get('rad_ft_epochs'):   # stage 2: continue on VQA-RAD only, keep the best epoch by VQA-RAD VALIDATION
        vr = SETS['rad_val'][1]
        vacc = lambda: score_df(vr, predict(model, 'rad_val', vr, cfg['use_image'], beams=1)).em.mean() * 100
        best2, best2_score = trainable_state(model), vacc(); radft_hist.append(dict(epoch=0, val_rad=best2_score))
        sc = cfg['rad_ft_lr_scale']
        opt2 = torch.optim.AdamW([{'params': v, 'lr': lrs[k] * sc} for k, v in groups.items() if v], weight_decay=0.01)
        rrows = SETS['rad_train'][1]; spe2 = math.ceil(len(rrows) / cfg['bs'])
        sch2 = get_cosine_schedule_with_warmup(opt2, int(0.05 * spe2 * cfg['rad_ft_epochs']), spe2 * cfg['rad_ft_epochs'])
        for ep2 in range(cfg['rad_ft_epochs']):
            model.train(); order = rng.permutation(len(rrows)); tot2 = 0
            for b in range(spe2):
                items = [('rad_train', rrows[k]) for k in order[b * cfg['bs']:(b + 1) * cfg['bs']]]
                hid, ids, mask = get_batch(items, True, cfg['use_image'])
                loss = model(hid, ids, mask, make_labels([r['answer'] for _, r in items]))
                opt2.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt2.step(); sch2.step()
                tot2 += loss.item()
            v = vacc(); radft_hist.append(dict(epoch=ep2 + 1, train_loss=tot2 / spe2, val_rad=v))
            print(f'{name} s{seed} VQA-RAD stage ep{ep2+1} loss {tot2/spe2:.3f} val rad {v:.2f} (before stage: {radft_hist[0]["val_rad"]:.2f})')
            if v > best2_score: best2_score, best2 = v, trainable_state(model)
        model.load_state_dict(best2, strict=False); opt2.state.clear(); del best2; gc.collect(); torch.cuda.empty_cache()
        for d in DATASETS:
            rows = SETS[f'{d}_test'][1]
            df = score_df(rows, predict(model, f'{d}_test', rows, cfg['use_image']))
            df.to_csv(f'{path}_{d}_radft.csv', index=False); test_radft[d] = summarize(df)
            print(f'   AFTER VQA-RAD STAGE test {d}: acc {test_radft[d]["acc"]:.2f} closed {test_radft[d]["closed_acc"]:.2f} '
                  f'open {test_radft[d]["open_acc"]:.2f}')
    test_rank, rank_info = {}, {}
    if cfg.get('rank'):
        for d in RANK_SETS:
            df, info = rank_eval(model, d, cfg['use_image'])
            df.to_csv(f'{path}_{d}_rank.csv', index=False); test_rank[d] = summarize(df); rank_info[d] = info
            print(f'   RANK {d}: acc {test_rank[d]["acc"]:.2f} closed {test_rank[d]["closed_acc"]:.2f} '
                  f'open {test_rank[d]["open_acc"]:.2f} | mode {info["mode"]}, {info["n_candidates"]} candidates, '
                  f'test-answer coverage {info["test_answer_coverage"]:.1f}%')
    if extras:
        export_examples(model, cfg, path)
        attention_maps(model)
        lat = latency(model); print('latency', lat)
        json.dump(lat, open(f'{OUT}/latency.json', 'w'), indent=1)
    res = dict(name=name, seed=seed, cfg=cfg, trainable_params=n_params, best_val=best_score,
               best_epoch=int(np.argmax([h['val_score'] for h in hist])) + 1,
               minutes=(time.time() - t0) / 60, history=hist, test=test, shuffled=shuffled,
               test_rank=test_rank, rank_info=rank_info, test_radft=test_radft, radft_hist=radft_hist)
    json.dump(res, open(path + '.json', 'w'), indent=1)
    del model, best, opt, sch, params, groups; gc.collect(); torch.cuda.empty_cache()
    return res

# ---------------- run plan ----------------
durations, STOP = [], False
def run(name, cfg, seed, **kw):
    global STOP
    if STOP: return None
    if os.path.exists(f'{OUT}/runs/{name}_s{seed}.json'): return train_run(name, cfg, seed)
    est = max(durations) if durations else 95
    if (time.time() - T_START) / 60 + est > TIME_LIMIT_H * 60:
        print(f'\n*** Time limit near: stopped before {name} seed {seed}. ***'); STOP = True; return None
    r = train_run(name, cfg, seed, **kw); durations.append(r['minutes'])
    print(f'>>> {name} seed {seed} done in {r["minutes"]:.0f} min'); return r

BEST = 'main'
VARIANTS = {'main': {}, 'no_xattn': dict(use_xattn=False), 'question_only': dict(use_image=False), 'cls_only': dict(cls_only=True)}
PLAN = os.environ.get('RUNS', 'main:0,1,2 no_xattn:0,1,2 question_only:0,1 cls_only:0,1')
first_main = True
for item in PLAN.split():
    name, seeds = item.split(':')
    for sd in [int(x) for x in seeds.split(',')]:
        if name == 'main':
            run('main', {}, sd, shuffle_check=True, extras=first_main); first_main = False
        else:
            run(name, VARIANTS[name], sd)

# ---------------- tables + statistics (uses whatever runs are finished here) ----------------
R = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/runs/*.json'))]
label = lambda n: n
D = pd.DataFrame([dict(name=label(r['name']), seed=r['seed'], dataset=d, **m) for r in R for d, m in r['test'].items()])
cols = ['acc', 'acc_lenient', 'closed_acc', 'open_acc', 'open_acc_lenient', 'f1', 'bleu1', 'bleu4']
def em(name, s, d): return pd.read_csv(f'{OUT}/runs/{name}_s{s}_{d}.csv').em.values.astype(bool)
stats = {}
if len(D):
    for d in DATASETS:
        g = D[D.dataset == d].groupby('name')
        tab = pd.DataFrame({c: g[c].mean().map('{:.2f}'.format) + ' ± ' + g[c].std().fillna(0).map('{:.2f}'.format)
                            for c in cols})
        tab['seeds'] = g.size()
        if BEST:
            pv = {}
            for r in {r['name'] for r in R} - {BEST}:
                ps = []
                for s in SEEDS:
                    if all(os.path.exists(f'{OUT}/runs/{n}_s{s}_{d}.csv') for n in (BEST, r)):
                        a, b = em(BEST, s, d), em(r, s, d)
                        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
                        ps.append(binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)
                pv[label(r)] = ', '.join(f'{p:.3f}' for p in ps)
            tab['mcnemar_p_vs_main'] = pd.Series(pv)
            ms = [s for s in SEEDS if os.path.exists(f'{OUT}/runs/{BEST}_s{s}_{d}.csv')]
            if ms:
                M = np.mean([em(BEST, s, d) for s in ms], axis=0)
                boot = [M[np.random.randint(0, len(M), len(M))].mean() * 100 for _ in range(2000)]
                stats[f'bootstrap_ci_main_{d}'] = list(np.percentile(boot, [2.5, 97.5]).round(2))
        tab.to_csv(f'{OUT}/table_test_{d}.csv')
        print(f'\n=== TEST: {d} (mean ± sd over seeds) ==='); print(tab.to_string())

if BEST:
    sh = [dict(dataset=d, seed=r['seed'], true_images=r['test'][d]['acc'], shuffled_images=v)
          for r in R if r['name'] == BEST for d, v in r['shuffled'].items()]
    if sh:
        sh = pd.DataFrame(sh).groupby('dataset')[['true_images', 'shuffled_images']].agg(['mean', 'std']).round(2)
        sh.to_csv(f'{OUT}/table_image_shuffle.csv')
        print('\n=== DOES THE MODEL USE THE IMAGE? (main model, test acc) ==='); print(sh.to_string())
    ms = [s for s in SEEDS if os.path.exists(f'{OUT}/runs/{BEST}_s{s}_rad.csv')]
    if ms:
        dfs = [pd.read_csv(f'{OUT}/runs/{BEST}_s{s}_rad.csv') for s in ms]
        qt = pd.DataFrame({'count': dfs[0].qtype.value_counts()})
        accs = pd.concat([x.groupby('qtype').em.mean() * 100 for x in dfs], axis=1)
        qt['acc'] = accs.mean(1).round(2).astype(str) + ' ± ' + accs.std(1).fillna(0).round(2).astype(str)
        qt.to_csv(f'{OUT}/table_question_types_rad.csv')
        print('\n=== VQA-RAD OFFICIAL QUESTION TYPES (total =', qt['count'].sum(), ') ==='); print(qt.to_string())
        ct = pd.crosstab(dfs[0].qtype, dfs[0].em)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(ct)
            stats.update(chi2_qtypes=chi2, chi2_dof=dof, chi2_p=p)
            print(f'chi-square across question types: chi2={chi2:.2f}, dof={dof}, p={p:.4g}')
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for r in R:
            if r['name'] != BEST: continue
            h = pd.DataFrame(r['history'])
            ax[0].plot(h.epoch, h.train_loss, label=f'seed {r["seed"]}')
            ax[1].plot(h.epoch, h.val_score, label=f'seed {r["seed"]}')
        ax[0].set(xlabel='Epoch', ylabel='Train loss'); ax[1].set(xlabel='Epoch', ylabel='Mean val accuracy (%)')
        ax[1].legend(); plt.tight_layout(); plt.savefig(f'{OUT}/fig_training_curves.png', dpi=300); plt.show()
RK = [r for r in R if r.get('test_rank')]
if RK:
    rows_ = []
    for r in RK:
        for d in RANK_SETS:
            if d in r['test_rank']:
                rows_.append(dict(model=r['name'], seed=r['seed'], dataset=d,
                                  **{f'rank_{c}': r['test_rank'][d][c] for c in ['acc', 'closed_acc', 'open_acc', 'acc_lenient']},
                                  **{f'free_{c}': r['test'][d][c] for c in ['acc', 'closed_acc', 'open_acc']}))
    T_ = pd.DataFrame(rows_); g = T_.groupby(['dataset', 'model'])
    rt = (g.mean(numeric_only=True).drop(columns='seed').round(2).astype(str) + ' ± ' +
          g.std(numeric_only=True).drop(columns='seed').fillna(0).round(2).astype(str))
    rt['seeds'] = g.size(); rt.to_csv(f'{OUT}/table_ranking.csv')
    print('\n=== ANSWER RANKING vs FREE TEXT (test, mean ± sd over seeds) ==='); print(rt.to_string())
    from scipy.stats import ttest_ind
    for d in RANK_SETS:
        base = T_[(T_.dataset == d) & (T_.model == 'main')].rank_acc
        for an in ABLATIONS:
            o = T_[(T_.dataset == d) & (T_.model == an)].rank_acc
            if len(base) > 1 and len(o) > 1:
                p = float(ttest_ind(base, o, equal_var=False).pvalue); stats[f'welch_rank_{d}_{an}'] = p
                print(f'Welch t-test ranking acc main vs {an} on {d}: p = {p:.4f}')
    ri = [dict(seed=r['seed'], dataset=d, **v) for r in RK if r['name'] == 'main' for d, v in r['rank_info'].items()]
    if ri: print(pd.DataFrame(ri).to_string())
print('stats:', stats)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1, default=float)

with zipfile.ZipFile(f'{WORK}/clip_t5_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for fp in glob.glob(f'{OUT}/**/*', recursive=True):
        if os.path.isfile(fp) and not fp.endswith('.pt'): z.write(fp, os.path.relpath(fp, WORK))
print('\nDONE' if not STOP else '\nPARTIAL (time limit): run the notebook again in a new session and send me both logs.',
      '| download clip_t5_results.zip from the Output panel.')

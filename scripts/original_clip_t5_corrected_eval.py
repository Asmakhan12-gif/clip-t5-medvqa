# ============================================================
# CLIP-T5 (ORIGINAL architecture, as described in the paper) - CORRECTED EVALUATION
# Single Kaggle cell. Accelerator: GPU T4, Internet: On. Input: official VQA-RAD (OSF) dataset.
#
# Model (unchanged from the paper): frozen CLIP ViT-B/32 image and text encoders -> visual and
# text projection (Linear + LayerNorm + Dropout) -> concatenation F = [v'; q'] -> 8-head
# attention + Add & Norm -> T5-base decoder (T5 encoder not used). CLIP AND T5 ARE FROZEN;
# only the projection and cross-attention modules are trained (as stated in the paper).
# Training as in the paper: AdamW, lr 3e-5, weight decay 0.01, 25 epochs, cosine schedule, batch 16,
# gradient clipping 1.0, early stopping (patience 5), random-crop and intensity augmentation.
#
# What is fixed (reviewer comments): official VQA-RAD split with a separate validation set
# (10% of training images), exact-match accuracy on GENERATED answers (no teacher forcing,
# padding never counted), best epoch chosen on validation only, 3 random seeds, the paper's
# ablations on the test set, corrected zero-shot SLAKE/PathVQA. Nothing else is added.
# ============================================================
import os, re, json, time, math, random, hashlib, glob, zipfile, copy, gc
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image, ImageFile
from datasets import load_dataset
from transformers import (CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, T5ForConditionalGeneration,
                          T5TokenizerFast, get_cosine_schedule_with_warmup)
from transformers.modeling_outputs import BaseModelOutput
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from scipy.stats import ttest_ind, chi2_contingency
ImageFile.LOAD_TRUNCATED_IMAGES = True
T_START = time.time()

# ---------------- settings ----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sync = torch.cuda.synchronize if DEVICE == 'cuda' else (lambda: None)
INPUT = os.environ.get('VQA_INPUT', '/kaggle/input')
WORK = os.environ.get('VQA_WORK', '/kaggle/working')
OUT = f'{WORK}/clip_t5_original'; os.makedirs(f'{OUT}/runs', exist_ok=True)
TIME_LIMIT_H = 11.0
CLIP_NAME = 'openai/clip-vit-base-patch32'
T5_NAME = 'google-t5/t5-base'
HF = {'slake': 'mdwiratathya/SLAKE-vqa-english', 'path': 'flaviagiammarino/path-vqa'}   # test splits only (zero-shot)
SEEDS = [0, 1, 2]                            # 3 runs -> mean ± SD (Reviewer 3, comment 6)
BASE = dict(lr=3e-5, bs=16, epochs=25, patience=5, heads=8, use_image=True, use_xattn=True, use_proj=True, dropout=0.1)
ABLATIONS = {                                # paper's ablations, now on the TEST set
    'no_cross_attention': dict(use_xattn=False),
    'no_projection': dict(use_proj=False),   # projection module replaced by a single linear layer (512 -> 768)
    'no_dropout': dict(dropout=0.0),
}
HEADS = {}
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

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
osf = [p for p in glob.glob(f'{INPUT}/**/*.json', recursive=True)
       if 'vqa_rad' in os.path.basename(p).lower() and 'public' in os.path.basename(p).lower()]
assert osf, ('Official VQA-RAD not found. Download it from https://osf.io/89kps/ '
             '("VQA_RAD Dataset Public.json" + "VQA_RAD Image Folder") and add it as a Kaggle dataset input.')
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
for d, hf in HF.items():                     # zero-shot test sets
    items = ({'image': ex['image'], 'question': ex['question'], 'answer': ex['answer'],
              'closed': None, 'qtype': None} for ex in load_dataset(hf, split='test'))
    STORES[f'{d}_test'], rows = build_store(items, f'{hf} [test]')
    SETS[f'{d}_test'] = (f'{d}_test', rows)
# VQA-RAD validation: 10% of the training IMAGES (all their questions), never shared with training
rows = SETS['rad_train'][1]; n = len(STORES['rad_train']['imgs'])
perm = list(range(n)); random.Random(42).shuffle(perm); vi = set(perm[:int(0.1 * n)])
SETS['rad_val'] = ('rad_train', [r for r in rows if r['iid'] in vi])
SETS['rad_train'] = ('rad_train', [r for r in rows if r['iid'] not in vi])
H = lambda s: {STORES[SETS[s][0]]['hashes'][r['iid']] for r in SETS[s][1]}
tr_h = H('rad_train'); te = SETS['rad_test'][1]; teh = STORES['rad_test']['hashes']
pairs = {(STORES['rad_train']['hashes'][r['iid']], norm(r['question'])) for r in SETS['rad_train'][1]}
report = {s: dict(questions=len(SETS[s][1]), images=len(H(s))) for s in SETS}
report['rad_test'].update(q_image_in_train=sum(teh[r['iid']] in tr_h for r in te),
                          q_exact_duplicate_of_train=sum((teh[r['iid']], norm(r['question'])) in pairs for r in te))
print(json.dumps(report, indent=1)); json.dump(report, open(f'{OUT}/split_report.json', 'w'), indent=1)

# ---------------- frozen CLIP features (computed once) ----------------
import torchvision.transforms as TV
N_AUG = 2
AUG = TV.Compose([TV.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)), TV.ColorJitter(0.1, 0.1)])
AUG_IMG = {}

@torch.no_grad()
def clip_features():
    cm = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
    pp = CLIPImageProcessor.from_pretrained(CLIP_NAME); ctok = CLIPTokenizerFast.from_pretrained(CLIP_NAME)
    IMG = {}
    for sk, st in STORES.items():
        out = []
        for i in range(0, len(st['imgs']), 64):
            pix = pp(images=st['imgs'][i:i+64], return_tensors='pt')['pixel_values'].to(DEVICE)
            out.append(cm.visual_projection(cm.vision_model(pixel_values=pix).pooler_output).float().cpu())
        IMG[sk] = torch.cat(out)
        if sk == 'rad_train':                    # paper: random cropping + subtle intensity changes (2 extra views)
            views = [IMG[sk]]
            for v in range(N_AUG):
                random.seed(1000 + v); torch.manual_seed(1000 + v); aug = []
                for i in range(0, len(st['imgs']), 64):
                    ims = [AUG(im) for im in st['imgs'][i:i+64]]
                    pix = pp(images=ims, return_tensors='pt')['pixel_values'].to(DEVICE)
                    aug.append(cm.visual_projection(cm.vision_model(pixel_values=pix).pooler_output).float().cpu())
                views.append(torch.cat(aug))
            AUG_IMG[sk] = torch.stack(views, 1)  # (images, 1 + N_AUG, 512)
        print('image features', sk, tuple(IMG[sk].shape))
    qs = sorted({r['question'] for _, rows in SETS.values() for r in rows}); TXT = {}
    for i in range(0, len(qs), 256):
        b = qs[i:i+256]
        t = ctok(['Question: ' + q for q in b], padding=True, truncation=True, max_length=77, return_tensors='pt').to(DEVICE)
        e = cm.text_projection(cm.text_model(input_ids=t.input_ids, attention_mask=t.attention_mask).pooler_output).float().cpu()
        TXT.update(zip(b, e))
    print('text features', len(TXT)); del cm; torch.cuda.empty_cache()
    return IMG, TXT, pp
IMG, TXT, PP = clip_features()

# ---------------- model (as in the paper) ----------------
class OrigCLIPT5(nn.Module):
    """[v'; q'] -> multi-head attention + Add & Norm -> T5 decoder (encoder bypassed)."""
    def __init__(s, t5, heads=8, use_image=True, use_xattn=True, use_proj=True, dropout=0.1, clip_dim=512, **_):
        super().__init__()
        d = t5.config.d_model; s.t5, s.use_image = t5, use_image
        mk = lambda: (nn.Sequential(nn.Linear(clip_dim, d), nn.LayerNorm(d), nn.Dropout(dropout)) if use_proj
                      else nn.Linear(clip_dim, d))
        s.proj_v, s.proj_q = mk(), mk()
        s.xattn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if use_xattn else None
        s.norm = nn.LayerNorm(d) if use_xattn else None
    def fuse(s, v, q, need_w=False):
        F = torch.stack([s.proj_v(v), s.proj_q(q)], 1) if s.use_image else s.proj_q(q).unsqueeze(1)
        w = None
        if s.xattn is not None:
            a, w = s.xattn(F, F, F, need_weights=need_w); F = s.norm(F + a)
        return F, w
    def _enc(s, v, q):
        F, _ = s.fuse(v, q)
        return BaseModelOutput(last_hidden_state=F), torch.ones(F.shape[:2], dtype=torch.long, device=F.device)
    def forward(s, v, q, labels):
        enc, m = s._enc(v, q)
        return s.t5(encoder_outputs=enc, attention_mask=m, labels=labels).loss
    def generate(s, v, q, num_beams=5, max_new_tokens=24):
        enc, m = s._enc(v, q)
        return s.t5.generate(encoder_outputs=enc, attention_mask=m, num_beams=num_beams, max_new_tokens=max_new_tokens)

tok = T5TokenizerFast.from_pretrained(T5_NAME)

def build_model(cfg):
    t5 = T5ForConditionalGeneration.from_pretrained(T5_NAME)
    for p in t5.parameters(): p.requires_grad = False                # T5 frozen (paper): only projection + attention train
    return OrigCLIPT5(t5, **cfg).to(DEVICE)

def batch(sname, rows, perm_iids=None, train=False):
    st = SETS[sname][0]
    iids = [r['iid'] for r in rows] if perm_iids is None else perm_iids
    if train and st in AUG_IMG:                  # a random (augmented or original) view per training example
        A = AUG_IMG[st]; v = A[iids, torch.randint(0, A.shape[1], (len(iids),))].to(DEVICE)
    else:
        v = IMG[st][iids].to(DEVICE)
    q = torch.stack([TXT[r['question']] for r in rows]).to(DEVICE)
    return v, q

def labels_of(answers):
    ids = tok(list(answers), padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids
    ids[ids == tok.pad_token_id] = -100                                # padding is never trained on or scored
    return ids.to(DEVICE)

@torch.no_grad()
def predict(model, sname, rows, beams=5, bs=64, shuffle_seed=None):
    model.eval(); out = []
    iids = [r['iid'] for r in rows]
    if shuffle_seed is not None:                                        # same questions, images swapped at random
        iids = list(np.array(iids)[np.random.RandomState(shuffle_seed).permutation(len(iids))])
    for i in range(0, len(rows), bs):
        v, q = batch(sname, rows[i:i+bs], iids[i:i+bs])
        out += [t.strip() for t in tok.batch_decode(model.generate(v, q, num_beams=beams), skip_special_tokens=True)]
    return out

def trainable(model): return [p for p in model.parameters() if p.requires_grad]

# ---------------- training ----------------
def train_run(name, cfg, seed, zero_shot=False, shuffle_check=True):
    path = f'{OUT}/runs/{name}_s{seed}'
    if os.path.exists(path + '.json'): return json.load(open(path + '.json'))
    t0 = time.time(); cfg = {**BASE, **cfg}; set_seed(seed)
    model = build_model(cfg); params = trainable(model)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=0.01)
    tr = SETS['rad_train'][1]; spe = math.ceil(len(tr) / cfg['bs']); steps = spe * cfg['epochs']
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    rng = np.random.RandomState(seed); best, best_acc, bad, hist = None, -1, 0, []
    for ep in range(cfg['epochs']):
        model.train(); order = rng.permutation(len(tr)); tot = 0
        for i in range(spe):
            rows = [tr[k] for k in order[i*cfg['bs']:(i+1)*cfg['bs']]]
            v, q = batch('rad_train', rows, train=True)
            loss = model(v, q, labels_of([r['answer'] for r in rows]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            tot += loss.item()
        vr = SETS['rad_val'][1]
        va = float(score_df(vr, predict(model, 'rad_val', vr, beams=1)).em.mean() * 100)
        hist.append(dict(epoch=ep + 1, train_loss=tot / spe, val_acc=va))
        print(f'{name} s{seed} ep{ep+1} loss {tot/spe:.3f} val acc {va:.2f}')
        if va > best_acc:
            best_acc, bad = va, 0; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']: break
    model.load_state_dict(best); del best, opt; gc.collect(); torch.cuda.empty_cache()
    test, shuffled = {}, {}
    sets = ['rad'] + (['slake', 'path'] if zero_shot else [])
    for d in sets:
        rows = SETS[f'{d}_test'][1]
        df = score_df(rows, predict(model, f'{d}_test', rows)); df.to_csv(f'{path}_{d}.csv', index=False)
        test[d] = summarize(df)
        if shuffle_check and cfg['use_image'] and d == 'rad':
            shuffled[d] = float(score_df(rows, predict(model, f'{d}_test', rows, shuffle_seed=seed)).em.mean() * 100)
        print(f'   test {d}: acc {test[d]["acc"]:.2f} closed {test[d]["closed_acc"]:.2f} open {test[d]["open_acc"]:.2f}'
              + (f' | shuffled images {shuffled[d]:.2f}' if d in shuffled else ''))
    res = dict(name=name, seed=seed, cfg=cfg, trainable_params=n_params, best_val=best_acc,
               best_epoch=int(np.argmax([h['val_acc'] for h in hist])) + 1, minutes=(time.time() - t0) / 60,
               history=hist, test=test, shuffled=shuffled)
    json.dump(res, open(path + '.json', 'w'), indent=1)
    if name == 'full' and seed == SEEDS[0]: torch.save(model.state_dict(), f'{OUT}/full_s{seed}_weights.pt')
    del model; gc.collect(); torch.cuda.empty_cache()
    return res

@torch.no_grad()
def latency(n=110):
    cm = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval(); ctok = CLIPTokenizerFast.from_pretrained(CLIP_NAME)
    model = build_model(BEST_CFG); model.load_state_dict(torch.load(f'{OUT}/full_s{SEEDS[0]}_weights.pt')); model.eval()
    st, rows, ts = STORES['rad_test'], SETS['rad_test'][1][:n], []
    for r in rows:
        sync(); t = time.perf_counter()
        pix = PP(images=[st['imgs'][r['iid']]], return_tensors='pt')['pixel_values'].to(DEVICE)
        v = cm.visual_projection(cm.vision_model(pixel_values=pix).pooler_output)
        tt = ctok(['Question: ' + r['question']], return_tensors='pt').to(DEVICE)
        q = cm.text_projection(cm.text_model(input_ids=tt.input_ids, attention_mask=tt.attention_mask).pooler_output)
        model.generate(v, q); sync(); ts.append((time.perf_counter() - t) * 1000)
    ts = ts[10:]
    return dict(mean_ms=float(np.mean(ts)), sd_ms=float(np.std(ts)), gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu')

# ---------------- run plan ----------------
durations, STOP = [], False
def run(name, cfg, seed, **kw):
    global STOP
    if STOP: return None
    if os.path.exists(f'{OUT}/runs/{name}_s{seed}.json'): return train_run(name, cfg, seed, **kw)
    est = max(durations) if durations else 15
    if (time.time() - T_START) / 60 + est > TIME_LIMIT_H * 60:
        print(f'*** Time limit near: stopped before {name} seed {seed} ***'); STOP = True; return None
    r = train_run(name, cfg, seed, **kw); durations.append(r['minutes'])
    print(f'>>> {name} seed {seed} done in {r["minutes"]:.1f} min'); return r

BEST_CFG = dict(BASE)
# 1) main model, 3 seeds, with corrected zero-shot evaluation on SLAKE and PathVQA
for s in SEEDS: run('full', BEST_CFG, s, zero_shot=True)
# 2) the paper's ablations (Table 4) on the TEST set, 3 seeds each
for a, acfg in ABLATIONS.items():
    for s in SEEDS: run(a, {**BEST_CFG, **acfg}, s)

# ---------------- tables + statistics ----------------
R = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/runs/*.json'))]
R = [r for r in R if not r['name'].startswith('lr_')]
D = pd.DataFrame([dict(name=r['name'], seed=r['seed'], dataset=d, **m) for r in R for d, m in r['test'].items()])
order = ['full'] + list(ABLATIONS) + list(HEADS)
fmt = lambda x: f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x) > 1 else f'{x.mean():.2f}'
rad = D[D.dataset == 'rad']; full = rad[rad.name == 'full']
rows_t = []
for n_ in order:
    g = rad[rad.name == n_]
    if not len(g): continue
    row = dict(model=n_, runs=len(g), **{c: fmt(g[c]) for c in ['acc', 'closed_acc', 'open_acc', 'acc_lenient', 'f1']})
    row['p_vs_full'] = '' if n_ == 'full' or len(g) < 2 else f"{ttest_ind(full.acc, g.acc, equal_var=False).pvalue:.3f}"
    rows_t.append(row)
T1 = pd.DataFrame(rows_t); T1.to_csv(f'{OUT}/table_vqarad_test.csv', index=False)
print('\n=== VQA-RAD TEST (exact match on generated answers, mean ± sd over runs; Welch t-test vs full) ===')
print(T1.to_string(index=False))

sh = [dict(seed=r['seed'], true=r['test']['rad']['acc'], shuffled=r['shuffled']['rad']) for r in R if r['name'] == 'full' and 'rad' in r['shuffled']]
if sh:
    sh = pd.DataFrame(sh); qo = rad[rad.name == 'question_only'].acc
    print('\n=== DOES THE MODEL USE THE IMAGE? (VQA-RAD test) ===')
    print(f"true images {fmt(sh.true)} | shuffled images {fmt(sh.shuffled)} | drop {fmt(sh.true - sh.shuffled)}"
          + (f" | question-only model {fmt(qo)}" if len(qo) else ''))
    sh.to_csv(f'{OUT}/table_image_use.csv', index=False)

zs = D[(D.name == 'full') & (D.dataset != 'rad')]
if len(zs):
    print('\n=== ZERO-SHOT (trained on VQA-RAD only) ===')
    for d in ['slake', 'path']:
        g = zs[zs.dataset == d]
        if len(g): print(f"{d}: acc {fmt(g.acc)} closed {fmt(g.closed_acc)} open {fmt(g.open_acc)} lenient {fmt(g.acc_lenient)}")
    zs.to_csv(f'{OUT}/table_zero_shot.csv', index=False)

fs = [s for s in SEEDS if os.path.exists(f'{OUT}/runs/full_s{s}_rad.csv')]
stats = {}
if fs:
    dfs = [pd.read_csv(f'{OUT}/runs/full_s{s}_rad.csv') for s in fs]
    qt = pd.DataFrame({'count': dfs[0].qtype.value_counts()})
    accs = pd.concat([x.groupby('qtype').em.mean() * 100 for x in dfs], axis=1)
    qt['acc'] = accs.mean(1).round(2).astype(str) + ' ± ' + accs.std(1).fillna(0).round(2).astype(str)
    qt.to_csv(f'{OUT}/table_question_types.csv')
    print('\n=== VQA-RAD OFFICIAL QUESTION TYPES (total =', qt['count'].sum(), ') ==='); print(qt.to_string())
    ct = pd.crosstab(dfs[0].qtype, dfs[0].em)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = chi2_contingency(ct); stats.update(chi2=float(chi2), dof=int(dof), p=float(p))
        print(f'chi-square across question types: chi2={chi2:.2f}, dof={dof}, p={p:.4g}')
if R:
    stats['trainable_params_full'] = next((r['trainable_params'] for r in R if r['name'] == 'full'), None)
    stats['best_epochs_full'] = [r['best_epoch'] for r in R if r['name'] == 'full']
if os.path.exists(f'{OUT}/full_s{SEEDS[0]}_weights.pt'):
    stats['latency'] = latency(); print('latency', stats['latency'])
print('stats:', stats); json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)

# ---------------- real examples for the example figures (full model, first seed) ----------------
def export_examples(seed=SEEDS[0], n_ok=9, n_bad=4):
    csv = f'{OUT}/runs/full_s{seed}_rad.csv'
    if not os.path.exists(csv): return
    df = pd.read_csv(csv); df['iid'] = [r['iid'] for r in SETS['rad_test'][1]]; df = df.drop_duplicates('iid')
    rs = np.random.RandomState(0); ok, bad = df[df.em], df[~df.em]
    take = lambda x, k: x.sample(min(k, len(x)), random_state=rs)
    ex = pd.concat([take(ok[ok.closed], 6).assign(group='correct'), take(ok[~ok.closed], n_ok - 6).assign(group='correct'),
                    take(bad[bad.closed], 2).assign(group='incorrect'), take(bad[~bad.closed], n_bad - 2).assign(group='incorrect')]).reset_index(drop=True)
    os.makedirs(f'{OUT}/examples', exist_ok=True); files = []
    for k, r in ex.iterrows():
        f = f'example_{k + 1:02d}_{r.group}.png'; STORES['rad_test']['imgs'][int(r.iid)].save(f'{OUT}/examples/{f}'); files.append(f)
    ex['image_file'] = files
    ex[['image_file', 'group', 'qtype', 'closed', 'question', 'answer', 'pred']].to_csv(f'{OUT}/examples/examples.csv', index=False)
    print('\n=== EXAMPLES SAVED ==='); print(ex[['image_file', 'group', 'question', 'answer', 'pred']].to_string())
export_examples()

with zipfile.ZipFile(f'{WORK}/clip_t5_original_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for fp in glob.glob(f'{OUT}/**/*', recursive=True):
        if os.path.isfile(fp) and not fp.endswith('.pt'): z.write(fp, os.path.relpath(fp, WORK))
print('\nDONE' if not STOP else '\nPARTIAL (time limit) - run again to continue;', 'download clip_t5_original_results.zip from the Output panel.')

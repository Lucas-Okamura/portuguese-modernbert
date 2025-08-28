import argparse, math, random, os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification,
                          DataCollatorForLanguageModeling,
                          DataCollatorWithPadding,
                          TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report
import os

# Desativa dynamo/compile globalmente
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# ---------------------------
# Utilidades gerais
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def set_seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_heading(txt):
    print("\n" + "="*80)
    print(txt)
    print("="*80)

# ---------------------------
# 1) Avaliação intrínseca de MLM
# ---------------------------

@dataclass
class MaskingConfig:
    mlm_probability: float = 0.15
    max_length: int = 256
    topk: List[int] = (1,5,10)

def batch_pseudo_perplexity(model, tokenizer, texts: List[str], max_length=256, device="cuda"):
    """
    Pseudo-perplexity por leave-one-out (batelado).
    """
    model.eval()
    ppl_losses = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            nlls = []
            # Máscara posição a posição (evita [CLS]/[SEP]/[BOS]/[EOS])
            for i in range(1, input_ids.size(0)-1):
                if attn[i] == 0: continue
                masked = input_ids.clone()
                masked[i] = tokenizer.mask_token_id
                out = model(input_ids=masked.unsqueeze(0).to(device)).logits[0, i]
                logprob = torch.log_softmax(out, dim=-1)[input_ids[i].to(device)]
                nlls.append(-logprob.item())
            if nlls:
                ppl_losses.append(np.mean(nlls))
    if not ppl_losses: return float("nan")
    return math.exp(np.mean(ppl_losses))

def intrinsic_eval(mlm_name: str, base_name: str, texts: List[str], cfg: MaskingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for name in [base_name, mlm_name]:
        print_heading(f"[INTRÍNSECO] Avaliando {name}")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        mdl = AutoModelForMaskedLM.from_pretrained(name).to(device)
        mdl.eval()

        # Loss e top-k em dados mascarados (usa DataCollatorForLanguageModeling para consistência)
        collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=cfg.mlm_probability)
        # Prepara lotes de features simples (sem labels) e deixa o collator mascarar
        enc = tok(texts, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt")
        batches = []
        bs = 8
        for i in range(0, enc["input_ids"].size(0), bs):
            batch = {k: v[i:i+bs] for k, v in enc.items()}
            batches.append(batch)

        total_loss, n_tokens, hit_at = 0.0, 0, {k:0 for k in cfg.topk}
        with torch.no_grad():
            for batch in batches:
                batch = collator([{"input_ids": ids, "attention_mask": am}
                                  for ids, am in zip(batch["input_ids"], batch["attention_mask"])])
                labels = batch["labels"].to(device)
                inputs = {"input_ids": batch["input_ids"].to(device),
                          "attention_mask": batch["attention_mask"].to(device)}
                logits = mdl(**inputs).logits
                # calcula loss manualmente apenas onde labels != -100
                active = labels.ne(-100)
                vocab_logits = logits[active]
                gold = labels[active]
                logprobs = torch.log_softmax(vocab_logits, dim=-1)
                nll = -logprobs.gather(1, gold.unsqueeze(1)).squeeze(1)
                total_loss += nll.sum().item()
                n_tokens += gold.numel()
                # top-k
                topk_vals = torch.topk(vocab_logits, k=max(cfg.topk), dim=-1).indices
                for k in cfg.topk:
                    hit_at[k] += (topk_vals[:, :k] == gold.unsqueeze(1)).any(dim=1).sum().item()

        avg_ce = total_loss / max(1, n_tokens)
        topk_acc = {f"top{k}_acc": hit_at[k] / max(1, n_tokens) for k in cfg.topk}
        ppl_star = batch_pseudo_perplexity(mdl, tok, texts, cfg.max_length, device)
        res[name] = {"mlm_ce": avg_ce, **topk_acc, "pseudo_perplexity": ppl_star}
        # desaloca para economizar VRAM
        del mdl; torch.cuda.empty_cache()
    return res

# ---------------------------
# 2) Avaliação extrínseca
# ---------------------------

def train_text_pair_classifier(model_name: str, dataset: DatasetDict, text_a: str, text_b: str,
                               label_col: str, num_labels: int, epochs=2, lr=3e-5, bs=16):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=torch.bfloat16)
    data_collator = DataCollatorWithPadding(tok)

    label2id = {k:i for i,k in enumerate(sorted(set(dataset["train"][label_col])))}
    id2label = {v:k for k,v in label2id.items()}
    model.config.label2id = label2id; model.config.id2label = id2label

    def preprocess(ex):
        out = tok(ex[text_a], ex[text_b], truncation=True, max_length=256)
        out["labels"] = label2id[ex[label_col]]
        return out

    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    args = TrainingArguments(output_dir="out_pair", learning_rate=lr, per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs, num_train_epochs=epochs,
                             eval_strategy="epoch", logging_steps=50, save_strategy="no",
                             report_to="none", seed=SEED)
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro")}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized.get("validation", tokenized["test"]),
                      tokenizer=tok, data_collator=data_collator, compute_metrics=compute)
    trainer.train()
    return trainer.evaluate(tokenized["test"])

def train_regressor(model_name: str, dataset: DatasetDict, text_a: str, text_b: Optional[str],
                    label_col: str, epochs=2, lr=3e-5, bs=16):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression", torch_dtype=torch.bfloat16)
    data_collator = DataCollatorWithPadding(tok)
    def preprocess(ex):
        if text_b:
            out = tok(ex[text_a], ex[text_b], truncation=True, max_length=256)
        else:
            out = tok(ex[text_a], truncation=True, max_length=256)
        out["labels"] = np.array([ex[label_col]], dtype=np.float32)
        return out
    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    args = TrainingArguments(output_dir="out_reg", learning_rate=lr, per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs, num_train_epochs=epochs,
                             eval_strategy="epoch", logging_steps=50, save_strategy="no",
                             report_to="none", seed=SEED)
    def compute(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze(-1)
        rmse = math.sqrt(mean_squared_error(labels, preds))
        return {"rmse": rmse}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized.get("validation", tokenized["test"]),
                      tokenizer=tok, data_collator=data_collator, compute_metrics=compute)
    trainer.train()
    return trainer.evaluate(tokenized["test"])

def align_labels_with_tokens(labels, word_ids):
    new = []
    prev = None
    for w in word_ids:
        if w is None:
            new.append(-100)
        elif w != prev:
            new.append(labels[w])
        else:
            # Subwords: usar o mesmo rótulo (ou -100 se preferir apenas first subword)
            new.append(labels[w])
        prev = w
    return new

def train_token_classifier(model_name: str, dataset: DatasetDict, tokens_col: str,
                           tags_col: str, tag2id: Dict[str,int], epochs=3, lr=3e-5, bs=8):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2id),
                                                            id2label={i:t for t,i in tag2id.items()},
                                                            label2id=tag2id, torch_dtype=torch.bfloat16)
    def preprocess(ex):
        # dataset baseado em texto (não em tokens): concatenar com espaços e reconstruir palavras
        # LeNER-Br e UD vêm com "tokens" e "tags"/"upos"
        tokens = ex[tokens_col]
        tags = ex[tags_col]
        enc = tok(tokens, is_split_into_words=True, truncation=True, max_length=256)
        word_ids = enc.word_ids()
        enc["labels"] = align_labels_with_tokens([tag2id[t] for t in tags], word_ids)
        return enc
    cols = dataset["train"].column_names
    tokenized = dataset.map(preprocess, remove_columns=cols)
    args = TrainingArguments(output_dir="out_tok", learning_rate=lr, per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs, num_train_epochs=epochs,
                             eval_strategy="epoch", logging_steps=50, save_strategy="no",
                             report_to="none", seed=SEED)
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        true_preds, true_labels = [], []
        for p, l in zip(preds, labels):
            cur_p, cur_l = [], []
            for pi, li in zip(p, l):
                if li == -100: continue
                cur_p.append(model.config.id2label[int(pi)])
                cur_l.append(model.config.id2label[int(li)])
            true_preds.append(cur_p); true_labels.append(cur_l)
        return {"f1": seq_f1(true_labels, true_preds)}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized.get("validation", tokenized["test"]),
                      tokenizer=tok, compute_metrics=compute)
    trainer.train()
    return trainer.evaluate(tokenized["test"])

# ---------------------------
# 3) Carregamento dos datasets PT
# ---------------------------

def load_assin2():
    ds = load_dataset("nilc-nlp/assin2")  # entailment_judgment (NONE/ENTAILMENT) + relatedness_score (1–5)
    return ds

def load_lener_br():
    # pode vir como (tokens, ner_tags). Alguns forks usam 'tags'/'entities'
    try:
        ds = load_dataset("peluz/lener_br")
        # normalizar campos
        def norm(ex):
            if "tokens" in ex: tokens = ex["tokens"]
            else: tokens = ex["words"] if "words" in ex else ex["tokens"]
            tags = ex.get("ner_tags", ex.get("tags", ex.get("entities")))
            return {"tokens": tokens, "tags": tags}
        ds = ds.map(norm)
        # criar tag set
        all_tags = sorted({t for split in ds for row in ds[split]["tags"] for t in row})
        tag2id = {t:i for i,t in enumerate(all_tags)}
        return ds, tag2id
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar LeNER-Br: {e}")

def load_ud_bosque():
    # universal_dependencies config "pt_bosque"
    ds = load_dataset("universal_dependencies", "pt_bosque")
    # Campos: tokens, upos
    def norm(ex):
        return {"tokens": ex["tokens"], "tags": ex["upos"]}
    ds = ds.map(norm)
    all_tags = sorted({t for split in ds for row in ds[split]["tags"] for t in row})
    tag2id = {t:i for i,t in enumerate(all_tags)}
    return ds, tag2id

def load_tweetsentbr_fewshot():
    # Pequeno subset público -> binário (labels variam por repo)
    ds = load_dataset("eduagarcia/tweetsentbr_fewshot")
    # normalizar 'text' e 'label'
    label_col = "label" if "label" in ds["train"].column_names else "sentiment"
    text_col = "text" if "text" in ds["train"].column_names else ds["train"].column_names[0]
    # garantir strings
    def norm(ex): return {"text": str(ex[text_col]), "label": str(ex[label_col])}
    ds = ds.map(norm, remove_columns=ds["train"].column_names)
    return ds

# ---------------------------
# 4) Runner
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="ex.: neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--finetuned_model", required=True, help="seu modelo MLM fine-tuned")
    ap.add_argument("--intrinsic_samples", type=int, default=512, help="nº de frases PT p/ intrínseco")
    ap.add_argument("--use_oscar", action="store_true", help="amostrar frases do OSCAR-2301 (pt)")
    ap.add_argument("--oscar_lang", default="pt", help="código de língua no OSCAR")
    args = ap.parse_args()

    set_seed_all()

    # ===== Intrínseco =====
    print_heading("Coletando textos PT para avaliação intrínseca")
    texts = []
    if args.use_oscar:
        try:
            oscar = load_dataset("oscar-corpus/OSCAR-2301", args.oscar_lang, split="train", streaming=True)
            for i, row in enumerate(oscar):
                txt = row.get("text", "").strip()
                if len(txt) > 40:
                    texts.append(txt)
                if len(texts) >= args.intrinsic_samples: break
            print(f"Amostradas {len(texts)} linhas do OSCAR-2301 ({args.oscar_lang}).")
        except Exception as e:
            print(f"[AVISO] Falha ao acessar OSCAR-2301: {e}. Usando fallback de frases fixas.")
    if not texts:
        texts = [
            "O comitê de política monetária decidiu manter a taxa básica de juros.",
            "O time venceu a partida fora de casa por dois a zero.",
            "A resolução foi publicada no diário oficial na manhã de terça-feira.",
        ] * max(1, args.intrinsic_samples // 3)

    cfg = MaskingConfig()
    intr = intrinsic_eval(args.finetuned_model, args.base_model, texts, cfg)
    print_heading("Resultados Intrínsecos (quanto menor melhor para CE/PPL*)")
    for name, met in intr.items():
        print(name, {k: round(v, 4) for k, v in met.items()})

    # ===== Extrínseco =====

    # ASSIN2 - NLI
    print_heading("ASSIN2: Textual Entailment (ENTAILMENT vs NONE)")
    assin = load_assin2()

    res_assin_base = train_text_pair_classifier(args.base_model, assin,
                                           text_a="premise", text_b="hypothesis",
                                           label_col="entailment_judgment", num_labels=2)
    print("ASSIN2 (base):", res_assin_base)

    res_assin_finetuned = train_text_pair_classifier(args.finetuned_model, assin,
                                           text_a="premise", text_b="hypothesis",
                                           label_col="entailment_judgment", num_labels=2)
    print("ASSIN2 (finetuned):", res_assin_finetuned)
    
    # # também STS (regressão)
    # print_heading("ASSIN2: Semantic Textual Similarity (regressão 1–5)")
    # res_sts_base = train_regressor(args.base_model, assin,
    #                           text_a="premise", text_b="hypothesis",
    #                           label_col="relatedness_score")
    # print("ASSIN2 STS (base):", res_sts_base)

    # res_sts_finetuned = train_regressor(args.finetuned_model, assin,
    #                           text_a="premise", text_b="hypothesis",
    #                           label_col="relatedness_score")
    # print("ASSIN2 STS (finetuned):", res_sts_finetuned)

    # # NER - LeNER-Br
    # print_heading("LeNER-Br: NER jurídico")
    # lener, ner_tag2id = load_lener_br()
    # res_ner_base = train_token_classifier(args.base_model, lener, tokens_col="tokens",
    #                                  tags_col="tags", tag2id=ner_tag2id)
    # print("LeNER-Br (base):", res_ner_base)

    # res_ner_finetuned = train_token_classifier(args.finetuned_model, lener, tokens_col="tokens",
    #                                  tags_col="tags", tag2id=ner_tag2id)
    # print("LeNER-Br (finetuned):", res_ner_finetuned)

    # # POS - UD Bosque
    # print_heading("UD Portuguese-Bosque: UPOS tagging")
    # bosque, pos_tag2id = load_ud_bosque()
    # res_pos_base = train_token_classifier(args.base_model, bosque, tokens_col="tokens",
    #                                  tags_col="tags", tag2id=pos_tag2id)
    # print("UD Bosque (base):", res_pos_base)

    # res_pos_finetuned = train_token_classifier(args.finetuned_model, bosque, tokens_col="tokens",
    #                                  tags_col="tags", tag2id=pos_tag2id)
    # print("UD Bosque (finetuned):", res_pos_finetuned)

    # # Sentimento - TweetSentBR few-shot
    # print_heading("TweetSentBR (few-shot): sentimento")
    # tw = load_tweetsentbr_fewshot()
    # # Normaliza labels (string) -> ids
    # labels = sorted(set(tw["train"]["label"]))
    # lab2id = {l:i for i,l in enumerate(labels)}
    # for split in tw:
    #     tw[split] = tw[split].map(lambda ex: {"label": lab2id[ex["label"]]})
    # # Treino simples de classificação (texto único)
    # tok = AutoTokenizer.from_pretrained(args.finetuned_model, use_fast=True)
    # def prep(ex): out = tok(ex["text"], truncation=True, max_length=256); out["labels"]=ex["label"]; return out
    # cols = tw["train"].column_names
    # tw_tok = tw.map(prep, remove_columns=cols)
    # clf = AutoModelForSequenceClassification.from_pretrained(args.finetuned_model, num_labels=len(labels))
    # args_cls = TrainingArguments(output_dir="out_sent", learning_rate=3e-5, per_device_train_batch_size=16,
    #                              per_device_eval_batch_size=16, num_train_epochs=2, eval_strategy="epoch",
    #                              save_strategy="no", report_to="none", seed=SEED)
    # def metrics_sent(ep):
    #     logits, y = ep
    #     pred = logits.argmax(-1)
    #     return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro")}
    # trainer = Trainer(model=clf, args=args_cls, train_dataset=tw_tok["train"],
    #                   eval_dataset=tw_tok.get("validation", tw_tok["test"]),
    #                   tokenizer=tok, compute_metrics=metrics_sent)
    # trainer.train()
    # print("TweetSentBR (finetuned):", trainer.evaluate(tw_tok["test"]))

    # clf = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(labels))
    # args_cls = TrainingArguments(output_dir="out_sent", learning_rate=3e-5, per_device_train_batch_size=16,
    #                              per_device_eval_batch_size=16, num_train_epochs=2, eval_strategy="epoch",
    #                              save_strategy="no", report_to="none", seed=SEED)

    # trainer = Trainer(model=clf, args=args_cls, train_dataset=tw_tok["train"],
    #                   eval_dataset=tw_tok.get("validation", tw_tok["test"]),
    #                   tokenizer=tok, compute_metrics=metrics_sent)
    # trainer.train()
    # print("TweetSentBR (base):", trainer.evaluate(tw_tok["test"]))

    # ENEM – Multiple Choice (VQA via texto)
    print_heading("ENEM (maritaca‑ai/enem): Múltipla escolha via descrição textual")
    enem = load_enem()
    res_enem_base = train_multiple_choice(args.base_model, enem,
                                          text_prompt_col="question",
                                          alternatives_col="alternatives",
                                          label_col="label")
    print("ENEM (base):", res_enem_base)

    res_enem_finetuned = train_multiple_choice(args.finetuned_model, enem,
                                               text_prompt_col="question",
                                               alternatives_col="alternatives",
                                               label_col="label")
    print("ENEM (finetuned):", res_enem_finetuned)

    print_heading("Dica: rode o mesmo pipeline com --base_model para comparar extrínseco")
    print("Ex.: troque args.finetuned_model pelo args.base_model e compare métricas lado a lado.")

if __name__ == "__main__":
    main()

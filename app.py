import pickle
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hiddens, last_hidden, mask):
        energy = self.v(torch.tanh(self.W1(hiddens) + self.W2(last_hidden.unsqueeze(1))))
        energy = energy.squeeze(-1)
        energy = energy.masked_fill(~mask, -1e9)
        weights = F.softmax(energy, dim=1)
        context = (hiddens * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class LSTMRec(nn.Module):
    def __init__(self, num_items, num_cats, item_embed_dim=64, cat_embed_dim=16, hidden_dim=128, num_layers=2, dropout=0.3, embed_dropout=0.2, use_attention=False):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_embed_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, cat_embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.use_attention = use_attention

        lstm_input_dim = item_embed_dim + cat_embed_dim + 1
        # +1 input dim for the time-delta feature

        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        if use_attention:
            self.attention = Attention(hidden_dim)
            fc_input_dim = hidden_dim * 2
        else:
            fc_input_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, item_embed_dim)

        self.output_bias = nn.Parameter(torch.zeros(num_items))

    def forward(self, items, cats, weights, deltas, lengths):
        """
        x:       (B, T)  item indices
        weights: (B, T)  event weights (includes time decay)
        deltas:  (B, T)  inter-event time gap in hours (log-scaled inside)
        lengths: (B,)
        """
        item_emb = self.embed_drop(self.item_embedding(items))                             # (B, T, E)
        cat_emb = self.embed_drop(self.cat_embedding(cats))
        combined = torch.cat((item_emb, cat_emb), dim=-1)
        combined *= weights.unsqueeze(-1)                   # scale by decayed weight

        # log-scale time deltas to compress range (add 1 to avoid log(0))
        time_feat = torch.log1p(deltas).unsqueeze(-1)       # (B, T, 1)
        lstm_input = torch.cat([combined, time_feat], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        # print(last_hidden.shape)

        if self.use_attention:
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
            B, T = items.size()
            mask = torch.arange(T, device=items.device).unsqueeze(0) < lengths.unsqueeze(1).to(items.device)
            context, _ = self.attention(rnn_out, last_hidden, mask)
            merged = torch.cat([context, last_hidden], dim=-1)
        else:
            merged = last_hidden

        merged = self.dropout(merged)
        proj = self.fc(merged)

        # print(proj.shape)
        # print(self.item_embedding.weight.shape)

        logits = torch.matmul(proj, self.item_embedding.weight.t())
        logits += self.output_bias
        return logits

class GRURec(nn.Module):
    def __init__(self, num_items, num_cats,
                 item_embed_dim=64, cat_embed_dim=16,
                 hidden_dim=128, num_layers=2, dropout=0.3, embed_dropout=0.2, use_attention=False):
        super().__init__()
        self.item_embed_dim = item_embed_dim
        self.item_embedding = nn.Embedding(num_items, item_embed_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, cat_embed_dim, padding_idx=0)
        self.use_attention = use_attention
        self.embed_drop = nn.Dropout(embed_dropout)
        self.hidden_dim = hidden_dim

        input_dim = item_embed_dim + cat_embed_dim + 1

        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if use_attention:
            self.attention = Attention(hidden_dim)
            fc_input_dim = hidden_dim * 2
        else:
            fc_input_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(fc_input_dim, item_embed_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_items))

    def forward(self, items, cats, weights, deltas, lengths):
        item_emb = self.embed_drop(self.item_embedding(items))
        cat_emb = self.embed_drop(self.cat_embedding(cats))
        combined = torch.cat([item_emb, cat_emb], dim=-1)
        combined = combined * weights.unsqueeze(-1)

        time_feat = torch.log1p(deltas).unsqueeze(-1)
        raw_input = torch.cat([combined, time_feat], dim=-1)  # (B, T, input_dim)

        packed = pack_padded_sequence(raw_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, h_n = self.gru(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)  # (B, T, H)

        # residual connection + layer norm
        # rnn_out and projected may differ in T due to padding, slice to match

        rnn_out = self.layer_norm(rnn_out)

        last_hidden = h_n[-1]

        if self.use_attention:
            T_out = rnn_out.size(1)
            mask = torch.arange(T_out, device=items.device).unsqueeze(0) < lengths.unsqueeze(1).to(items.device)
            context, _ = self.attention(rnn_out, last_hidden, mask)
            merged = torch.cat([context, last_hidden], dim=-1)
        else:
            merged = last_hidden

        merged = self.dropout(merged)
        proj = self.fc(merged)

        logits = F.linear(proj, self.item_embedding.weight, self.output_bias)
        return logits

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).
    Causal transformer encoder over item sequences.
    Uses the last position's output as the sequence representation.
    """

    def __init__(self, num_items, num_cats,
                 item_embed_dim=64, cat_embed_dim=16,
                 hidden_dim=128, n_heads=4, n_layers=2,
                 dropout=0.3, embed_dropout=0.2, max_len=50):
        super().__init__()
        self.item_embed_dim = item_embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(num_items, item_embed_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, cat_embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(embed_dropout)

        input_dim = item_embed_dim + cat_embed_dim + 1
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, item_embed_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_items))

        print(f"  [SASRec] {n_layers} layers, {n_heads} heads, hidden={hidden_dim}")

    def forward(self, items, cats, weights, deltas, lengths):
        B, T = items.size()

        item_emb = self.embed_drop(self.item_embedding(items))
        cat_emb = self.embed_drop(self.cat_embedding(cats))
        combined = torch.cat([item_emb, cat_emb], dim=-1)
        combined = combined * weights.unsqueeze(-1)
        time_feat = torch.log1p(deltas).unsqueeze(-1)
        features = torch.cat([combined, time_feat], dim=-1)

        hidden = self.input_proj(features)
        positions = torch.arange(T, device=items.device).unsqueeze(0).expand(B, -1)
        hidden = hidden + self.pos_embedding(positions)
        hidden = self.layer_norm(hidden)

        causal_mask = torch.triu(
            torch.ones(T, T, device=items.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = torch.arange(T, device=items.device).unsqueeze(0) >= lengths.unsqueeze(1).to(items.device)

        out = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        last_idx = (lengths - 1).long().to(items.device)
        last_hidden = out[torch.arange(B, device=items.device), last_idx]

        proj = self.fc(self.dropout(last_hidden))

        logits = torch.matmul(proj, self.item_embedding.weight.t())
        logits = logits + self.output_bias
        return logits

with open("artifacts/artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

item2idx = artifacts["item2idx"]
idx2item = artifacts["idx2item"]
cat2idx = artifacts["cat2idx"]
item_to_cat_idx = artifacts["item_to_cat_idx"]
num_items = artifacts["num_items"]
num_cats = artifacts["num_cats"]
EVENT_WEIGHTS = artifacts["event_weights"]
MAX_SEQ = artifacts["max_seq"]
# rule_lookup = defaultdict(dict, artifacts["rule_lookup"])

device = torch.device("cpu")  # use CPU for serving (simpler, sufficient for single requests)

# match the use_attention flag you trained with
USE_ATTENTION = False  # <-- set to True if you trained with attention

MODELS = {}

for name, cls, path in [
    ("lstm", LSTMRec, "artifacts/lstm_model.pt"),
    ("gru", GRURec, "artifacts/gru_model.pt"),
    ("sasrec", SASRec, "artifacts/sas_model.pt"),
]:
    try:
        if cls == SASRec:
            model = cls(num_items, num_cats)
        else:
            model = cls(num_items, num_cats, use_attention=USE_ATTENTION)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        MODELS[name] = model
        print(f"Loaded {name} from {path}")
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping {name}")

if not MODELS:
    raise RuntimeError("No models loaded. Run training and save_models first.")

# ──────────────────────────────────────────────
# 3. INFERENCE HELPERS
# ──────────────────────────────────────────────


def prepare_sequence(events: list[dict]) -> tuple:
    """
    Convert raw event dicts to model-ready tensors.

    Each event dict: {"itemid": int, "event": str, "timestamp": float}
    timestamp is in seconds (or ms — we handle both).
    """
    processed = []
    prev_ts = None

    for e in events:
        item_id = int(e["itemid"])
        event_type = e.get("event", "view")
        ts = float(e["timestamp"])

        # handle millisecond timestamps
        if ts > 1e12:
            ts = ts / 1000.0

        item_idx = item2idx.get(item_id, 0)
        cat_idx = item_to_cat_idx.get(item_idx, 0)
        weight = EVENT_WEIGHTS.get(event_type, 1.0)
        delta = 0.0 if prev_ts is None else (ts - prev_ts) / 3600.0

        if item_idx > 0:  # skip unknown items
            processed.append((item_idx, cat_idx, weight, delta))

        prev_ts = ts

    # cap to last MAX_SEQ
    processed = processed[-MAX_SEQ:]
    if not processed:
        return None

    items, cats, weights, deltas = zip(*processed)
    return (
        torch.tensor(items, dtype=torch.long).unsqueeze(0),
        torch.tensor(cats, dtype=torch.long).unsqueeze(0),
        torch.tensor(weights, dtype=torch.float).unsqueeze(0),
        torch.tensor(deltas, dtype=torch.float).unsqueeze(0),
        torch.tensor([len(items)]),
    )


def get_recommendations(model_name: str, events: list[dict],
                        top_k: int = 10, rule_boost: float = 0.0) -> list[dict]:
    """Run inference and return top-K items with scores."""
    model = MODELS.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    tensors = prepare_sequence(events)
    if tensors is None:
        return []

    t_items, t_cats, t_weights, t_deltas, lengths = tensors
    with torch.no_grad():
        logits = model(t_items, t_cats, t_weights, t_deltas, lengths)  # (1, num_items)

    # apply association rule boost
    # if rule_boost > 0:
    #     seq_items = t_items[0].tolist()
    #     recent = seq_items[-3:]
    #     rule_scores = torch.zeros(num_items)
    #     for ant in recent:
    #         if ant in rule_lookup:
    #             for cons, score in rule_lookup[ant].items():
    #                 if cons < num_items:
    #                     rule_scores[cons] += score
    #     if rule_scores.max() > 0:
    #         rule_scores = rule_scores / rule_scores.max()
    #         logits[0] = logits[0] + rule_boost * rule_scores

    # exclude padding index
    logits[0, 0] = -float("inf")

    # also exclude items already in the session (no repeated recs)
    seen = set(t_items[0].tolist())
    for idx in seen:
        logits[0, idx] = -float("inf")

    scores, indices = logits.topk(top_k, dim=1)
    probs = F.softmax(scores, dim=1)

    results = []
    for rank in range(top_k):
        idx = indices[0, rank].item()
        results.append({
            "rank": rank + 1,
            "item_id": idx2item.get(idx, -1),
            "item_idx": idx,
            "score": round(scores[0, rank].item(), 4),
            "probability": round(probs[0, rank].item(), 4),
        })

    return results


# ──────────────────────────────────────────────
# 4. FASTAPI APP
# ──────────────────────────────────────────────

app = FastAPI(
    title="Retailrocket Recommender API",
    description="Session-based recommendation using LSTM, GRU, and SASRec models.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="./landing/static"), name="static")


class EventItem(BaseModel):
    itemid: int
    event: str = "view"
    timestamp: float

    class Config:
        json_schema_extra = {
            "example": {"itemid": 123456, "event": "view", "timestamp": 1.4335e9}
        }

class RecommendRequest(BaseModel):
    events: list[EventItem]
    model: str = "lstm"
    top_k: int = 10
    rule_boost: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "events": [
                    {"itemid": 461686, "event": "view", "timestamp": 1433221332},
                    {"itemid": 206783, "event": "view", "timestamp": 1433221345},
                    {"itemid": 206783, "event": "addtocart", "timestamp": 1433221378},
                ],
                "model": "lstm",
                "top_k": 10,
                "rule_boost": 0.3,
            }
        }


class RecommendItem(BaseModel):
    rank: int
    item_id: int
    item_idx: int
    score: float
    probability: float


class RecommendResponse(BaseModel):
    model: str
    num_events: int
    recommendations: list[RecommendItem]

@app.get("/", response_class=HTMLResponse)
def landing_page():
    with open("./landing/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {
        "available": list(MODELS.keys()),
        "default": "lstm",
        "num_items": num_items,
        "num_categories": num_cats,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {list(MODELS.keys())}",
        )

    if not req.events:
        raise HTTPException(status_code=400, detail="No events provided.")

    if req.top_k < 1 or req.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100.")

    events = [e.model_dump() for e in req.events]
    recs = get_recommendations(req.model, events, req.top_k, req.rule_boost)

    return RecommendResponse(
        model=req.model,
        num_events=len(req.events),
        recommendations=recs,
    )

# ──────────────────────────────────────────────
# Run with: uvicorn app:app --reload --port 8000
# ──────────────────────────────────────────────
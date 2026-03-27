import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
import numpy as np

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class PureCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # Matrix of transition scores from j to i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags=None, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
            
        if tags is not None:
            # Training: Return NEGATIVE Log Likelihood (Loss)
            return -self._compute_log_likelihood(emissions, tags, mask)
        else:
            # Inference: Return Best Path (Viterbi Decoding)
            return self._viterbi_decode(emissions, mask)

    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        
        # A. Score of the Gold Path
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        
        for i in range(1, seq_len):
            trans_score = self.transitions[tags[:, i-1], tags[:, i]]
            emit_score = emissions[torch.arange(batch_size), i, tags[:, i]]
            score = score + (trans_score + emit_score) * mask[:, i]
            
        last_valid_idx = mask.sum(1).long() - 1
        last_tags = tags[torch.arange(batch_size), last_valid_idx]
        score = score + self.end_transitions[last_tags]

        # B. Partition Function (Forward Algorithm)
        alpha = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            mask_t = mask[:, i].unsqueeze(1)
            alpha = next_alpha * mask_t + alpha * (1 - mask_t)

        alpha = alpha + self.end_transitions
        partition = torch.logsumexp(alpha, dim=1)

        return torch.mean(score - partition)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, _ = emissions.shape
        
        # Initialize with start transitions + first emission
        score = self.start_transitions + emissions[:, 0]
        history = []

        # Forward Pass: Calculate best path to reach each tag at step i
        for i in range(1, seq_len):
            # Broadcast for (batch, prev, curr)
            scores = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            
            # Find the max score and the tag that produced it
            max_scores, best_prev_tags = torch.max(scores, dim=1)
            
            # Save backpointers
            history.append(best_prev_tags)
            
            # Update score, respecting the mask
            mask_t = mask[:, i].unsqueeze(1)
            score = max_scores * mask_t + score * (1 - mask_t)

        # Add end transitions
        score += self.end_transitions
        
        # Backward Pass: Trace back the best path
        best_tags_list = []
        best_last_tags = torch.argmax(score, dim=1)
        
        for b in range(batch_size):
            # Get actual length of this sequence (ignoring padding)
            length = mask[b].sum().item()
            best_tag = best_last_tags[b].item()
            path = [best_tag]
            
            # Loop backwards from the end of the valid sequence
            for t in range(length - 2, -1, -1):
                best_tag = history[t][b][best_tag].item()
                path.append(best_tag)
                
            # Reverse path to get correct order
            best_tags_list.append(path[::-1])
            
        return best_tags_list

class PubMedBERT_CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = PureCRF(num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Training Mode: Return Loss
            loss = self.crf(emissions, tags=labels, mask=attention_mask)
            return {"loss": loss, "logits": emissions}
        else:
            # Inference Mode: Return Viterbi Path
            predictions = self.crf(emissions, mask=attention_mask)
            return {"logits": emissions, "predictions": predictions}

# ==========================================
# 2. CONFIGURATION & LOADING
# ==========================================
st.set_page_config(page_title="Medical NER Analyzer", layout="wide")

label2id = {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}
id2label = {v: k for k, v in label2id.items()}
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

@st.cache_resource
def load_model():
    # Token removed. This requires the Hugging Face repo to be set to "Public"
    repo_id = "tirubujji92/pubmedbert-crf-ner-medical"

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = PubMedBERT_CRF(MODEL_NAME, num_labels=len(label2id))
        
        # Download weights publicly
        weights_path = hf_hub_download(
            repo_id=repo_id, 
            filename="pytorch_model.bin"
        )
        
        # Load weights and ensure it's on CPU for Streamlit
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# ==========================================
# 3. PROCESSING LOGIC (SLIDING WINDOW)
# ==========================================
def process_text(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        stride=100, 
        return_overflowing_tokens=True,
        padding="max_length"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    all_entities = []
    
    for i in range(len(input_ids)):
        c_input_ids = input_ids[i].unsqueeze(0)
        c_mask = attention_mask[i].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(c_input_ids, c_mask)
            predictions = outputs["predictions"][0]
            
        tokens = tokenizer.convert_ids_to_tokens(c_input_ids[0])
        labels = [id2label[p] for p in predictions]
        
        current_entity = {"word": "", "type": None}
        for token, label in zip(tokens, labels):
            if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
                
            if token.startswith("##"):
                clean_token = token[2:]
                if current_entity["word"]: current_entity["word"] += clean_token
                continue

            if label.startswith("B-"):
                if current_entity["word"]: all_entities.append(current_entity)
                current_entity = {"word": token, "type": label[2:]}
            elif label.startswith("I-"):
                if current_entity["type"] == label[2:]:
                    current_entity["word"] += " " + token
                else:
                    if current_entity["word"]: all_entities.append(current_entity)
                    current_entity = {"word": token, "type": label[2:]}
            else:
                if current_entity["word"]: all_entities.append(current_entity)
                current_entity = {"word": "", "type": None}
        
        if current_entity["word"]: all_entities.append(current_entity)

    # Clean stop words
    STOP_WORDS = {"the", "a", "an", "of", "and", "in", "with", "to", "for", "on", "at", "by", "is", "was"}
    cleaned_entities = []
    for entity in all_entities:
        words = entity["word"].split()
        while words and words[0].lower() in STOP_WORDS: words.pop(0)
        while words and words[-1].lower() in STOP_WORDS: words.pop()
        if words:
            entity["word"] = " ".join(words)
            cleaned_entities.append(entity)
            
    # Unique results
    return list({(e['word'], e['type']): e for e in cleaned_entities}.values())

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("🏥 Clinical Entity Extractor")

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("Paste Medical Report:", height=200, 
                              value="The patient was diagnosed with severe diabetic retinopathy and prescribed metformin.")
    analyze_btn = st.button("Analyze Report")

if analyze_btn and input_text:
    if model:
        results = process_text(input_text)
        st.subheader("Results")
        if not results:
            st.info("No entities found.")
        else:
            diseases = [e["word"] for e in results if e["type"] == "Disease"]
            chemicals = [e["word"] for e in results if e["type"] == "Chemical"]
            
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"**Diseases Found ({len(diseases)})**")
                for d in diseases: st.write(f"- {d}")
            with c2:
                st.success(f"**Chemicals/Drugs Found ({len(chemicals)})**")
                for c in chemicals: st.write(f"- {c}")
    else:
        st.error("Model failed to load.")

with col2:
    st.info("ℹ️ **Model Info**")
    st.markdown("""
    - **Base:** PubMedBERT
    - **Head:** CRF with **Viterbi Decoding**
    - **Dataset:** BC5CDR
    - **Features:** Sliding window for long text & stop-word cleanup.
    """)
from sentence_transformers import SentenceTransformer,util
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch
import json

def topkRelatedSentence(k, inputEmb, dataEmb):
    similarityScore = util.cos_sim(inputEmb, dataEmb)
    return torch.topk(similarityScore, k)[1].reshape(-1)


def predict(input):
    data = pd.read_csv("data/climate_fever_evidence_embedding.csv",header=None)
    
    embds = []

    for embd in data[1]:
        embds.append(json.loads(embd))

    embds = torch.Tensor(embds)

    model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')


    indexes = topkRelatedSentence(5, model.encode(input), embds)

    topEvidences = data[0].iloc[indexes].tolist()

    pairs = []

    for evidence in topEvidences:
        pairs.append(json.dumps([input,evidence]))

    votes = []

    model_token = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_voter_1 = DistilBertForSequenceClassification.from_pretrained("model/voter_1")
    model_voter_2 = DistilBertForSequenceClassification.from_pretrained("model/voter_2")
    model_voter_3 = DistilBertForSequenceClassification.from_pretrained("model/voter_3")
    model_voter_4 = DistilBertForSequenceClassification.from_pretrained("model/voter_4")
    model_voter_5 = DistilBertForSequenceClassification.from_pretrained("model/voter_5")

    model_voters = [model_voter_1, model_voter_2, model_voter_3, model_voter_4, model_voter_5]

    for pair in pairs:
        temp_vote = []
        for model_voter in model_voters:
            inputs = model_token(pair, return_tensors="pt")
            with torch.no_grad():
                logits = model_voter(**inputs).logits
            predicted_class_id = logits.argmax().item()
            temp_vote.append(predicted_class_id)
        votes.append(temp_vote)

    print(votes)

    model_token_verdict = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_voter_verdict = DistilBertForSequenceClassification.from_pretrained("model/verdict")

    inputs = model_token_verdict(json.dumps(votes), return_tensors="pt")
    with torch.no_grad():
        logits = model_voter_verdict(**inputs).logits

    predicted_class_id = logits.argmax().item()

    classes = ['NOT_ENOUGH_INFO','SUPPORTS', 'REFUTES', 'DISPUTED']

    return classes[predicted_class_id]
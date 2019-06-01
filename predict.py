import torch
from collections import defaultdict
from data import sentence_to_tensor, BEGIN_TAGS, OUT_TAG, get_entity_type

def predict_sentence_tags(model, sentence, dct, device=None):
    sequence = sentence_to_tensor(sentence, dct)
    sequence = sequence.unsqueeze(0)
    with torch.no_grad():
        sequence_cuda = sequence.to(device)
        mask_cuda = sequence_cuda > 0

        tags_pred = model.predict(sequence_cuda, mask_cuda)

    return tags_pred[0]

def get_entity(sentence, tags):
    entity_dict = defaultdict(set)

    entity_start_index = -1
    entity = None
    entity_type = None
    for index, tag in enumerate(tags):
        entity = None
        if tag in BEGIN_TAGS:
            if entity_start_index == -1:
                entity_type = get_entity_type(tag)
                entity_start_index = index
            else:
                entity = sentence[entity_start_index: index]
                entity_dict[entity_type].add(entity)

                entity_type = get_entity_type(tag)
                entity_start_index = index
        
        if tag == OUT_TAG:
            if entity_start_index != -1:
                entity = sentence[entity_start_index: index]
                entity_dict[entity_type].add(entity)
                entity_start_index = -1
                
    return dict(entity_dict)